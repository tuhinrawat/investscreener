"""
ws_manager.py — WebSocket price feed manager.

Wraps kite_client.py's ticker with:
  • Auto-reconnect: a background keeper thread watches _TICKER_RUNNING and
    restarts the ticker on silent disconnects with 5-second exponential back-off.
  • Batch subscription: subscribes all tokens in batches of 500 (Kite hard
    limit) and merges new tokens without a full restart.
  • Health metrics: tick_rate_5s, subscription_count, reconnect_count,
    last_tick_age_s — exposed for the sidebar debug panel.

All price state lives in kite_client module-level vars (_TICKER_PRICES,
_TICKER_SYM_MAP).  This module only adds the orchestration layer.
No Streamlit imports anywhere in this file.
"""
from __future__ import annotations

import threading
import time
import logging

import kite_client as _kc

_log = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────────
_keeper_thread: threading.Thread | None = None
_keeper_stop = threading.Event()

_reconnect_count: int = 0
_start_api_key: str = ""
_start_access_token: str = ""
_start_token_map: dict = {}          # {token: symbol}

# Rolling tick counter for rate metric
_tick_bucket: list[float] = []       # epoch timestamps of recent ticks
_TICK_WINDOW_S = 5.0                 # measure ticks in last 5 s

_KITE_SUB_BATCH = 500                # Kite WS hard limit per subscribe() call
_RECONNECT_BACKOFF_S = 5             # seconds between reconnect attempts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record_tick() -> None:
    """Called by the patched on_ticks callback to update tick rate metrics."""
    now = time.time()
    _tick_bucket.append(now)
    # Trim old entries outside the rolling window
    cutoff = now - _TICK_WINDOW_S
    while _tick_bucket and _tick_bucket[0] < cutoff:
        _tick_bucket.pop(0)


def _subscribe_all(token_map: dict) -> None:
    """Subscribe tokens in batches of 500, updating _TICKER_SYM_MAP first."""
    if not _kc._TICKER_OBJ or not _kc._TICKER_RUNNING:
        return
    tokens = list(token_map.keys())
    for i in range(0, len(tokens), _KITE_SUB_BATCH):
        batch = tokens[i: i + _KITE_SUB_BATCH]
        try:
            _kc._TICKER_OBJ.subscribe(batch)
            _kc._TICKER_OBJ.set_mode(_kc._TICKER_OBJ.MODE_LTP, batch)
        except Exception as exc:
            _log.warning("ws_manager: subscribe batch failed: %s", exc)


def _keeper_loop() -> None:
    """
    Background thread: watches _TICKER_RUNNING.  If it goes False (disconnect
    or error), waits _RECONNECT_BACKOFF_S and restarts the ticker using the
    last-known credentials.
    """
    global _reconnect_count
    while not _keeper_stop.is_set():
        time.sleep(2)   # poll interval
        if _keeper_stop.is_set():
            break
        if not _kc._TICKER_RUNNING and _start_api_key and _start_access_token and _start_token_map:
            _log.info("ws_manager: ticker disconnected — reconnecting in %ss", _RECONNECT_BACKOFF_S)
            time.sleep(_RECONNECT_BACKOFF_S)
            if _keeper_stop.is_set():
                break
            ok = _kc.start_ticker(_start_api_key, _start_access_token, {})
            if ok:
                _subscribe_all(_start_token_map)
                _reconnect_count += 1
                _log.info("ws_manager: reconnected (attempt %d)", _reconnect_count)
            else:
                _log.warning("ws_manager: reconnect failed")


# ── Public API ────────────────────────────────────────────────────────────────

def start(api_key: str, access_token: str, token_symbol_map: dict) -> bool:
    """
    Start the KiteTicker WebSocket and the keeper thread.

    token_symbol_map: {instrument_token (int): display_name (str)}
      e.g. {738561: "RELIANCE", 256265: "NIFTY 50", …}

    Safe to call multiple times — stops the old connection before starting
    a new one.  Returns True if the ticker started, False on error.
    """
    global _start_api_key, _start_access_token, _start_token_map
    global _keeper_thread, _reconnect_count

    if not api_key or not access_token or not token_symbol_map:
        return False

    _start_api_key    = api_key
    _start_access_token = access_token
    _start_token_map  = dict(token_symbol_map)

    # Patch on_ticks to also record tick timestamps for rate metric
    _orig_on_ticks = _kc._ticker_on_ticks

    def _instrumented_on_ticks(ws, ticks):
        _record_tick()
        _orig_on_ticks(ws, ticks)

    _kc._ticker_on_ticks = _instrumented_on_ticks  # type: ignore[attr-defined]

    # Update sym map and start — subscribe in batches after connect fires
    ok = _kc.start_ticker(api_key, access_token, {})
    if not ok:
        return False

    # Bulk-subscribe all tokens in batches of 500 once connected
    # Give the socket 1 s to establish before subscribing
    def _deferred_subscribe():
        time.sleep(1.5)
        with _kc._TICKER_LOCK:
            _kc._TICKER_SYM_MAP.update(token_symbol_map)
        _subscribe_all(token_symbol_map)

    threading.Thread(target=_deferred_subscribe, daemon=True, name="ws_deferred_sub").start()

    # Start keeper thread (only one)
    _keeper_stop.clear()
    if _keeper_thread is None or not _keeper_thread.is_alive():
        _keeper_thread = threading.Thread(
            target=_keeper_loop, daemon=True, name="ws_keeper"
        )
        _keeper_thread.start()

    _log.info(
        "ws_manager: started — %d tokens queued for subscription",
        len(token_symbol_map),
    )
    return True


def stop() -> None:
    """Stop the ticker and the keeper thread cleanly."""
    _keeper_stop.set()
    _kc.stop_ticker()
    _log.info("ws_manager: stopped")


def add_symbols(token_symbol_map: dict) -> None:
    """
    Add new instrument tokens to a running ticker without restarting.
    Already-subscribed tokens are ignored.
    """
    if not token_symbol_map:
        return
    with _kc._TICKER_LOCK:
        new = {t: s for t, s in token_symbol_map.items() if t not in _kc._TICKER_SYM_MAP}
    if not new:
        return
    with _kc._TICKER_LOCK:
        _kc._TICKER_SYM_MAP.update(new)
    # Also update keeper's reference map so reconnects include them
    _start_token_map.update(new)
    tokens = list(new.keys())
    for i in range(0, len(tokens), _KITE_SUB_BATCH):
        batch = tokens[i: i + _KITE_SUB_BATCH]
        try:
            if _kc._TICKER_OBJ and _kc._TICKER_RUNNING:
                _kc._TICKER_OBJ.subscribe(batch)
                _kc._TICKER_OBJ.set_mode(_kc._TICKER_OBJ.MODE_LTP, batch)
        except Exception as exc:
            _log.warning("ws_manager: add_symbols subscribe failed: %s", exc)


def get_prices() -> dict[str, float]:
    """
    Returns {display_name: last_price} for all subscribed instruments that
    have received at least one tick.  Thread-safe snapshot.
    """
    return _kc.get_all_ticker_prices()


def get_price(symbol: str) -> float | None:
    """Return latest price for a single display-name symbol, or None."""
    return _kc.get_ticker_ltp(symbol)


def is_alive() -> bool:
    """True if the WebSocket is running AND received a tick in the last 10 s."""
    return _kc.is_ticker_alive()


def is_started() -> bool:
    """True if the WebSocket thread has been started (regardless of ticks)."""
    return _kc.is_ticker_started()


def subscription_count() -> int:
    """Number of tokens currently in the subscription map."""
    with _kc._TICKER_LOCK:
        return len(_kc._TICKER_SYM_MAP)


def get_health() -> dict:
    """
    Returns a health snapshot dict suitable for display in a debug panel:
      is_alive, is_started, subscription_count, reconnect_count,
      last_tick_age_s, tick_rate_5s
    """
    now = time.time()
    tick_ts = _kc.get_ticker_ts()
    age = round(now - tick_ts, 1) if tick_ts else None

    # Count ticks in the last 5 s for rate
    cutoff = now - _TICK_WINDOW_S
    rate = sum(1 for t in _tick_bucket if t >= cutoff) / _TICK_WINDOW_S

    return {
        "is_alive":          is_alive(),
        "is_started":        is_started(),
        "subscription_count": subscription_count(),
        "reconnect_count":   _reconnect_count,
        "last_tick_age_s":   age,
        "tick_rate_5s":      round(rate, 1),
    }
