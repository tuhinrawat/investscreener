"""
kite_client.py — Wraps the Kite SDK with two things it doesn't give you:
  1. Persistent access token (so you don't re-login every script run)
  2. Rate limiting that respects Kite's actual caps

Why this matters: KiteConnect's SDK is a thin REST wrapper. It will
happily fire 100 requests/sec at the API and get you 429-ed and
temp-banned. We add the safety rails.

Auth flow (frontend-driven):
  1. app.py calls client.get_login_url() and opens it in the browser.
  2. Zerodha redirects back to the app with ?request_token=xxx.
  3. app.py calls client.complete_auth(request_token) which exchanges it
     for an access_token and caches it to data/access_token.json.
  4. Subsequent runs load the cached token until it expires (~6 AM IST).
"""
import json
import os as _kc_os
import time
from datetime import datetime, date
from pathlib import Path

from kiteconnect import KiteConnect
import config

# On Streamlit Cloud the container filesystem is SHARED between all users.
# data/access_token.json must never be read or written there — each user's
# token must come exclusively from their own browser session (session_state).
_KC_ON_CLOUD: bool = _kc_os.environ.get("HOME", "").rstrip("/").endswith("appuser")


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.
    Lock prevents two concurrent Streamlit threads from both skipping sleep
    and firing simultaneous API calls that would trigger a 429.
    """
    def __init__(self, calls_per_sec: float):
        import threading
        self.min_interval = 1.0 / calls_per_sec
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self.last_call
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.last_call = time.time()


class KiteClient:
    """
    Wrapper around KiteConnect. On init, tries to load a cached access token.
    If none exists or it's stale, `self.authenticated` is False — the caller
    (app.py) is responsible for driving the OAuth flow via get_login_url() /
    complete_auth(). No blocking CLI prompts.
    """
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        access_token: str = "",
    ):
        """
        Create a KiteClient.

        Params (all optional):
          api_key      — Kite API key.  Falls back to env / screener_keys.json.
          api_secret   — Kite API secret.  Same fallback chain.
          access_token — Pre-existing OAuth token for this session.  When
                         supplied the client is immediately authenticated without
                         touching the disk cache.  Pass "" to attempt the disk
                         cache (useful for local single-user dev).

        Multi-user usage (Streamlit Cloud):
          Pass api_key, api_secret, and access_token from st.session_state so
          each browser session has fully isolated credentials.
        """
        _key    = api_key    or config._get_secret("KITE_API_KEY")
        _secret = api_secret or config._get_secret("KITE_API_SECRET")

        self._api_key    = _key
        self._api_secret = _secret
        self.missing_keys = not (_key and _secret)

        self.hist_limiter  = RateLimiter(config.HIST_REQUESTS_PER_SEC)
        self.quote_limiter = RateLimiter(config.QUOTE_REQUESTS_PER_SEC)

        if self.missing_keys:
            self.authenticated = False
            self.kite = None
            return

        self.kite = KiteConnect(api_key=_key)
        self._patch_session()

        if access_token:
            # Caller supplied a session token — use it directly, skip disk cache
            self.kite.set_access_token(access_token)
            self.authenticated = True
        elif _KC_ON_CLOUD:
            # On Cloud the disk cache is shared across all users — never read it.
            # Authentication must come from the per-user browser session only.
            self.authenticated = False
        else:
            # Local dev: try today's token from disk (single-user convenience)
            self.authenticated = self._try_load_cached_token()

    # ----------------------------------------------------------
    # SESSION PATCH — replace requests with curl_cffi so that
    # Cloudflare's TLS fingerprint check sees Chrome, not Python.
    # We preserve all headers the SDK already set (X-Kite-Version,
    # User-Agent, etc.) by copying them onto the new session.
    # ----------------------------------------------------------
    def _patch_session(self):
        try:
            from curl_cffi.requests import Session as CurlSession
            old_headers = dict(self.kite.reqsession.headers)
            new_session = CurlSession(impersonate="chrome120")
            new_session.headers.update(old_headers)
            self.kite.reqsession = new_session
        except ImportError:
            # Fallback: at least stop leaking the Python User-Agent
            self.kite.reqsession.trust_env = False
            self.kite.reqsession.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })

    # ----------------------------------------------------------
    # AUTH
    # ----------------------------------------------------------
    def _try_load_cached_token(self) -> bool:
        """
        Load today's access token from disk. Returns True if a valid
        token was found and set, False otherwise.
        Kite tokens expire daily at ~6 AM IST, so we key on today's date.
        """
        token_path = Path(config.TOKEN_CACHE)
        if not token_path.exists():
            return False
        try:
            data = json.loads(token_path.read_text())
            if date.fromisoformat(data["date"]) == date.today():
                self.kite.set_access_token(data["access_token"])
                return True
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        return False

    def get_login_url(self) -> str:
        """Return the Zerodha OAuth login URL to open in the browser."""
        if self.kite is None:
            return ""
        return self.kite.login_url()

    def complete_auth(self, request_token: str) -> str:
        """
        Exchange a request_token (from Kite's OAuth redirect) for an
        access_token.  Also writes a local disk cache (for single-user local
        dev convenience — ignored on Streamlit Cloud where the caller stores
        the token in st.session_state).
        Returns the access_token string.
        """
        session = self.kite.generate_session(
            request_token, api_secret=self._api_secret
        )
        access_token = session["access_token"]
        self.kite.set_access_token(access_token)
        self.authenticated = True

        # Write disk cache only on local dev — never on Cloud (shared file)
        if not _KC_ON_CLOUD:
            try:
                token_path = Path(config.TOKEN_CACHE)
                token_path.parent.mkdir(parents=True, exist_ok=True)
                token_path.write_text(json.dumps({
                    "access_token": access_token,
                    "date": date.today().isoformat(),
                }))
            except Exception:
                pass
        return access_token

    def get_profile(self) -> dict:
        """Fetch the logged-in user's profile (user_id, user_name, email)."""
        if self.kite is None or not self.authenticated:
            return {}
        try:
            return self.kite.profile() or {}
        except Exception:
            return {}

    # ----------------------------------------------------------
    # INSTRUMENTS — full master list of NSE securities
    # ----------------------------------------------------------
    def get_instruments(self, exchange: str = "NSE") -> list:
        """Returns full instruments list. ~10K rows for NSE."""
        return self.kite.instruments(exchange)

    # ----------------------------------------------------------
    # HISTORICAL — single instrument, rate-limited
    # ----------------------------------------------------------
    def get_historical(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day"
    ) -> list:
        """
        Returns list of candles: [{date, open, high, low, close, volume}]
        Rate-limited to stay under 3 req/sec.
        """
        self.hist_limiter.wait()
        return self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )

    # ----------------------------------------------------------
    # QUOTE — batch OHLC for many instruments at once
    # ----------------------------------------------------------
    def get_ohlc_batch(self, instruments: list[str]) -> dict:
        """
        instruments format: ['NSE:RELIANCE', 'NSE:TCS', ...]
        Returns dict keyed by 'NSE:SYMBOL' with last_price + ohlc.
        Max 1000 per call per Kite docs; we chunk at 250 for safety.
        """
        result = {}
        chunk_size = config.QUOTE_BATCH_SIZE
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            result.update(self.kite.ohlc(chunk))
        return result

    def get_full_quote_batch(self, instruments: list[str]) -> dict:
        """
        Like get_ohlc_batch but returns full quote (volume, depth, etc).
        Limit is 500 per call.
        """
        result = {}
        chunk_size = 250
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            result.update(self.kite.quote(chunk))
        return result

    # ----------------------------------------------------------
    # ORDER MANAGEMENT — place, query, cancel orders
    # ----------------------------------------------------------
    def place_order(
        self,
        tradingsymbol: str,
        qty: int,
        transaction_type: str,          # "BUY" | "SELL"
        order_type: str = "LIMIT",      # "MARKET" | "LIMIT" | "SL" | "SL-M"
        product: str = "CNC",           # "CNC" | "MIS" | "NRML"
        price: float = None,            # required for LIMIT / SL
        trigger_price: float = None,    # required for SL / SL-M
        variety: str = "regular",
        tag: str = None,                # max 20 chars, alphanumeric
    ) -> str:
        """
        Place an equity order on NSE.  Returns the Kite order_id string.

        order_type mapping:
          LIMIT   — buy/sell at exactly `price` (or better)
          MARKET  — immediate fill at market price
          SL-M    — stop-market: triggers when price crosses `trigger_price`
          SL      — stop-limit: triggers at `trigger_price`, fills at `price`

        For intraday BUY_ABOVE signals, use order_type="SL-M" with
        trigger_price=entry so the order activates when price crosses entry.
        """
        kwargs = dict(
            variety=variety,
            exchange="NSE",
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=qty,
            product=product,
            order_type=order_type,
            validity="DAY",
        )
        if price is not None and order_type in ("LIMIT", "SL"):
            kwargs["price"] = price
        if trigger_price is not None and order_type in ("SL", "SL-M"):
            kwargs["trigger_price"] = trigger_price
        if tag:
            kwargs["tag"] = str(tag)[:20]   # Kite max tag length
        return str(self.kite.place_order(**kwargs))

    def get_orders(self) -> list:
        """Return all orders (open + executed) for today."""
        return self.kite.orders() or []

    def cancel_order(self, order_id: str, variety: str = "regular") -> str:
        """Cancel a pending order.  Returns order_id on success."""
        return str(self.kite.cancel_order(variety=variety, order_id=order_id))

    def get_positions(self) -> dict:
        """Return current net + day positions with unrealized P&L."""
        return self.kite.positions()

    def get_holdings(self) -> list:
        """
        Return long-term equity holdings from DEMAT.
        Each item has: tradingsymbol, quantity, average_price, last_price,
        close_price, pnl, day_change, day_change_percentage.
        """
        return self.kite.holdings() or []

    def get_margins(self, segment: str = "equity") -> dict:
        """
        Return fund/margin details for the given segment ('equity' or 'commodity').
        Key fields under the segment key:
          net                  — total account balance
          available.live_balance — cash available to trade right now
          available.cash       — cash component
          used.debits          — total debit (used margin + MTM losses)
        """
        return self.kite.margins(segment) or {}

    def get_ltp_batch(self, instruments: list[str]) -> dict[str, float]:
        """
        Fastest possible price fetch — returns only the last traded price.

        Kite's ltp() endpoint is lighter than ohlc() or quote():
          • Single field per symbol (last_price only)
          • Up to 1000 instruments per call
          • Typically responds in 100–300 ms for 50–100 symbols

        Returns: {"NSE:RELIANCE": 2451.5, "NSE:TCS": 3842.0, ...}
        """
        result = {}
        chunk_size = 500   # ltp() supports up to 1000; keep 500 for headroom
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            raw = self.kite.ltp(chunk)
            # raw format: {"NSE:RELIANCE": {"instrument_token": ..., "last_price": 2451.5}}
            for key, val in raw.items():
                result[key] = float(val["last_price"])
        return result

    # ----------------------------------------------------------
    # INTRADAY CANDLES — today's 5-min (or other interval) candles
    # ----------------------------------------------------------
    def get_today_candles(
        self,
        instrument_token: int,
        interval: str = "5minute",
    ) -> "pd.DataFrame":
        """
        Fetch today's intraday candles for one instrument.

        Returns a DataFrame with columns:
          date, open, high, low, close, volume
        sorted ascending by time.  Returns an empty DataFrame on error or
        if the market hasn't opened yet.

        interval choices (Kite):
          "minute"   — 1-min candles (up to 60 days of history)
          "3minute"  — 3-min candles
          "5minute"  — 5-min candles  ← default for ORB/VWAP
          "15minute" — 15-min candles
          "30minute" — 30-min candles
          "60minute" — 60-min candles

        Rate-limited to 3 req/sec (shared hist_limiter).
        """
        import pandas as pd  # noqa: PLC0415
        if not self.authenticated:
            return pd.DataFrame()
        try:
            from datetime import datetime, timezone, timedelta  # noqa: PLC0415
            _IST = timezone(timedelta(hours=5, minutes=30))
            now_ist    = datetime.now(_IST)
            from_dt    = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
            to_dt      = now_ist
            self.hist_limiter.wait()
            candles = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_dt,
                to_date=to_dt,
                interval=interval,
                continuous=False,
                oi=False,
            )
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles)
            df = df.rename(columns={"date": "date"})
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    def get_today_open(self, instruments: list[str]) -> dict[str, float]:
        """
        Returns today's open price for a list of instruments using the OHLC API.
        Format: {"NSE:RELIANCE": 2440.0, ...}
        This is used for gap-up / gap-down detection at signal trigger time.
        """
        result = {}
        chunk_size = config.QUOTE_BATCH_SIZE
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            try:
                raw = self.kite.ohlc(chunk)
                for key, val in raw.items():
                    ohlc = val.get("ohlc", {})
                    if "open" in ohlc:
                        result[key] = float(ohlc["open"])
            except Exception:
                pass
        return result


# ======================================================================
# KITE TICKER — WebSocket real-time LTP (100–250 ms latency vs 500–1000 ms REST)
#
# Architecture:
#   • One module-level KiteTicker instance shared across all Streamlit
#     render threads (fragments share the same Python process).
#   • Prices are written by the WebSocket thread into _TICKER_PRICES under
#     a threading.Lock, and read lock-free from render threads (dict reads
#     in CPython are effectively atomic for simple key lookups, but we
#     still protect bulk reads with the lock for consistency).
#   • start_ticker() is idempotent — calling it while already running
#     disconnects the old socket cleanly before starting a new one.
#   • is_ticker_alive() checks both _TICKER_RUNNING flag and that a tick
#     arrived within the last 10 s (stale guard for silent disconnects).
# ======================================================================
import threading as _kc_threading

_TICKER_LOCK: "_kc_threading.Lock" = _kc_threading.Lock()
_TICKER_PRICES: dict = {}          # instrument_token (int) → last_price (float)
_TICKER_SYM_MAP: dict = {}         # instrument_token (int) → display name (str)
_TICKER_TS: float = 0.0            # epoch-seconds of last tick received
_TICKER_OBJ = None                 # KiteTicker instance (or None)
_TICKER_RUNNING: bool = False      # set False on disconnect/error


def _ticker_on_ticks(ws, ticks):
    global _TICKER_TS
    with _TICKER_LOCK:
        for tick in ticks:
            token = tick.get("instrument_token")
            ltp   = tick.get("last_price")
            if token is not None and ltp is not None:
                _TICKER_PRICES[token] = float(ltp)
        _TICKER_TS = time.time()


def _ticker_on_connect(ws, response):
    tokens = list(_TICKER_SYM_MAP.keys())
    if tokens:
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_LTP, tokens)


def _ticker_on_error(ws, code, reason):
    global _TICKER_RUNNING
    _TICKER_RUNNING = False


def _ticker_on_close(ws, code, reason):
    global _TICKER_RUNNING
    _TICKER_RUNNING = False


def start_ticker(api_key: str, access_token: str, token_symbol_map: dict) -> bool:
    """
    Start (or restart) the KiteTicker WebSocket in a background thread.

    token_symbol_map: {instrument_token: display_name}
      e.g. {738561: "RELIANCE", 256265: "NIFTY 50", 260105: "NIFTY BANK", …}

    Returns True if the socket was started successfully, False on error.
    Safe to call multiple times — cleans up the previous connection first.
    """
    global _TICKER_OBJ, _TICKER_SYM_MAP, _TICKER_RUNNING, _TICKER_PRICES

    stop_ticker()   # disconnect any existing socket cleanly

    if not api_key or not access_token or not token_symbol_map:
        return False

    with _TICKER_LOCK:
        _TICKER_SYM_MAP.clear()
        _TICKER_SYM_MAP.update(token_symbol_map)
        _TICKER_PRICES.clear()

    try:
        from kiteconnect import KiteTicker as _KT  # noqa: PLC0415
        kt = _KT(api_key, access_token)
        kt.on_ticks   = _ticker_on_ticks
        kt.on_connect = _ticker_on_connect
        kt.on_error   = _ticker_on_error
        kt.on_close   = _ticker_on_close
        _TICKER_OBJ     = kt
        _TICKER_RUNNING = True
        kt.connect(threaded=True)
        return True
    except Exception:
        _TICKER_RUNNING = False
        return False


def stop_ticker() -> None:
    """Disconnect the WebSocket and reset state."""
    global _TICKER_OBJ, _TICKER_RUNNING
    if _TICKER_OBJ is not None:
        try:
            _TICKER_OBJ.close()
        except Exception:
            pass
        _TICKER_OBJ = None
    _TICKER_RUNNING = False


def is_ticker_alive() -> bool:
    """
    True if the WebSocket connection is active AND a tick arrived in the last 10 s.
    The 10-s recency guard catches silent disconnects where the socket flag
    stays True but the feed has stalled.
    """
    if not _TICKER_RUNNING:
        return False
    return (time.time() - _TICKER_TS) < 10.0


def get_all_ticker_prices() -> dict:
    """
    Returns {display_name: last_price} for every subscribed instrument
    that has received at least one tick.
    Thread-safe snapshot under the module lock.
    """
    with _TICKER_LOCK:
        return {
            _TICKER_SYM_MAP[t]: p
            for t, p in _TICKER_PRICES.items()
            if t in _TICKER_SYM_MAP
        }


def get_ticker_ltp(symbol: str) -> "float | None":
    """Return the latest WebSocket price for a symbol (by display name), or None."""
    with _TICKER_LOCK:
        for token, name in _TICKER_SYM_MAP.items():
            if name == symbol:
                return _TICKER_PRICES.get(token)
    return None


def get_ticker_ts() -> float:
    """Epoch seconds of the most recent tick received (0 if none yet)."""
    return _TICKER_TS


def update_ticker_subscriptions(new_token_symbol_map: dict) -> None:
    """
    Add new instrument tokens to the running ticker without restarting.
    Already-subscribed tokens are left unchanged.
    """
    global _TICKER_SYM_MAP
    if _TICKER_OBJ is None or not _TICKER_RUNNING:
        return
    with _TICKER_LOCK:
        new_tokens = [t for t in new_token_symbol_map if t not in _TICKER_SYM_MAP]
        if not new_tokens:
            return
        _TICKER_SYM_MAP.update(new_token_symbol_map)
    try:
        _TICKER_OBJ.subscribe(new_tokens)
        _TICKER_OBJ.set_mode(_TICKER_OBJ.MODE_LTP, new_tokens)
    except Exception:
        pass
