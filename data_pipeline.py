"""
data_pipeline.py — Orchestration layer.

Three jobs:
  1. refresh_universe()        — pull NSE instruments master, filter to EQ
  2. full_rescan()             — pull 400 days of OHLCV for filtered universe,
                                 then compute all metrics
  3. quick_refresh()           — hit OHLC batch API, update LTP + today's stats,
                                 re-rank composite without re-pulling history

Why split? Because they have wildly different latency profiles:
  - Universe refresh: ~5 sec (one API call), run weekly
  - Full rescan:      ~3-5 min, run daily post-market
  - Quick refresh:    ~10 sec, run on every dashboard click during market hours
"""
import io
import json
import pickle
import time
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Optional

_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Current datetime in IST, timezone-naive (safe for TIMESTAMP columns)."""
    return datetime.now(_IST).replace(tzinfo=None)

import pandas as pd
from tqdm import tqdm

# ── Scan checkpoint ───────────────────────────────────────────────────────────
# After every stock fetch the candle rows are appended here.  If the DB push
# fails (SSL drop, timeout, etc.) the data is NOT lost — it survives in this
# file and can be re-pushed via push_checkpoint() without re-fetching Kite.
_CHECKPOINT_PATH = Path("/tmp/_ohlcv_scan_checkpoint.pkl")


def checkpoint_exists() -> bool:
    return _CHECKPOINT_PATH.exists() and _CHECKPOINT_PATH.stat().st_size > 0


def checkpoint_row_count() -> int:
    if not checkpoint_exists():
        return 0
    try:
        data = pickle.loads(_CHECKPOINT_PATH.read_bytes())
        return len(data)
    except Exception:
        return 0


def push_checkpoint(progress_callback=None) -> int:
    """
    Load candle rows saved by the last (possibly interrupted) scan and push
    them to the DB in chunks.  Returns the number of rows pushed.
    Clears the checkpoint file on success.
    """
    if not checkpoint_exists():
        return 0
    rows = pickle.loads(_CHECKPOINT_PATH.read_bytes())
    if not rows:
        _CHECKPOINT_PATH.unlink(missing_ok=True)
        return 0
    df = pd.DataFrame(rows)
    chunk = 2000
    total = len(df)
    for i in range(0, total, chunk):
        if progress_callback:
            progress_callback(i, total, "pushing to DB…")
        db.upsert_ohlcv(df.iloc[i: i + chunk])
    _CHECKPOINT_PATH.unlink(missing_ok=True)
    return total


def _checkpoint_append(rows: list) -> None:
    """Append a list of candle-row dicts to the checkpoint file."""
    existing: list = []
    if checkpoint_exists():
        try:
            existing = pickle.loads(_CHECKPOINT_PATH.read_bytes())
        except Exception:
            existing = []
    existing.extend(rows)
    _CHECKPOINT_PATH.write_bytes(pickle.dumps(existing))


def clear_checkpoint() -> None:
    _CHECKPOINT_PATH.unlink(missing_ok=True)

import config
import db
from kite_client import KiteClient
from indicators import compute_all
from signals import compute_all_signals


# ============================================================
# NSE INDEX CONSTITUENT FETCH — narrows universe before history pull
# ============================================================
def fetch_nse_index_symbols(index_name: str) -> set[str]:
    """
    Downloads NSE's official constituent CSV for the given index and returns
    a set of trading symbols (e.g., {"RELIANCE", "TCS", "INFY"}).

    Falls back to a locally cached copy if the download fails. Returns empty
    set only if both the download AND cache fail — in that case the caller
    falls back to the full EQ universe.
    """
    cache_path = config.NSE_INDEX_CACHE
    url = config.NSE_INDEX_URLS.get(index_name)
    if not url:
        return set()

    # ── Load from local cache ────────────────────────────────────────────────
    # Fresh cache (< 7 days): use directly without hitting NSE.
    # Stale cache (≥ 7 days): attempt a refresh, but fall back to the stale
    # copy rather than blowing up to the full 2700-stock universe.
    stale_fallback: set[str] = set()
    if cache_path.exists():
        try:
            cached      = json.loads(cache_path.read_text())
            cached_date = date.fromisoformat(cached.get("date", "2000-01-01"))
            if cached.get("index") == index_name and cached.get("symbols"):
                age_days = (date.today() - cached_date).days
                if age_days < 7:
                    print(f"Using cached {index_name} list ({cached_date})")
                    return set(cached["symbols"])
                else:
                    # Keep as fallback — will be used if download fails
                    stale_fallback = set(cached["symbols"])
                    print(f"Cache stale ({age_days}d) — attempting refresh of {index_name}")
        except Exception:
            pass

    # ── Download fresh ───────────────────────────────────────────────────────
    # NSE requires a warm-up homepage hit to set cookies before it serves CSVs.
    try:
        try:
            from curl_cffi.requests import Session as CurlSession
            session = CurlSession(impersonate="chrome120")
        except ImportError:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })

        session.get("https://www.nseindia.com", timeout=10)
        r = session.get(url, headers={"Referer": "https://www.nseindia.com"}, timeout=15)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        symbols = set(df["Symbol"].str.strip().tolist())
        if len(symbols) < 10:
            raise ValueError(f"Suspiciously small symbol list ({len(symbols)}) — likely a bad response")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "index":   index_name,
            "date":    date.today().isoformat(),
            "symbols": sorted(symbols),
        }))
        print(f"✓ Downloaded {index_name}: {len(symbols)} constituents")
        return symbols

    except Exception as e:
        if stale_fallback:
            print(f"⚠ NSE download failed ({e}) — using stale cache ({len(stale_fallback)} symbols)")
            return stale_fallback
        print(f"⚠ NSE download failed and no cache ({e}) — using full EQ universe")
        return set()


# ============================================================
# UNIVERSE REFRESH
# ============================================================
def refresh_universe(client: KiteClient) -> pd.DataFrame:
    """
    Pulls NSE instrument master, filters to EQ series stocks.
    Stores in DB. Returns the filtered DataFrame.
    """
    print("Fetching NSE instruments...")
    raw = client.get_instruments(config.EXCHANGE)
    df = pd.DataFrame(raw)

    # 1. Filter to equity stocks only.
    # NOTE: Kite's instruments API does NOT return a `series` column, so we
    # cannot filter by series directly. Instead we use a structural pattern:
    # every bond, SGB, surveillance stock, and ETF series variant has a hyphen
    # in its tradingsymbol (e.g., OMFURN-ST, 10IIFL28A-NE, SGBMAY29I-GB,
    # 672KL27-SG). Regular NSE EQ stocks never have hyphens (RELIANCE, TCS,
    # 3MINDIA, 5PAISA — digit-starting stocks are fine, hyphen ones are not).
    df = df[df["instrument_type"] == "EQ"]
    df = df[df["segment"] == "NSE"]
    df = df[~df["tradingsymbol"].str.contains("-", na=False)]
    df = df.drop_duplicates(subset="instrument_token")

    # 2. Optionally narrow to a specific NSE index (e.g., Nifty 500).
    # This is the biggest speed lever: Nifty 500 covers 96% of market cap
    # but cuts the pull from ~1800 stocks → 504, saving ~9 minutes per scan.
    if config.UNIVERSE_INDEX:
        index_symbols = fetch_nse_index_symbols(config.UNIVERSE_INDEX)
        if index_symbols:
            before = len(df)
            df = df[df["tradingsymbol"].isin(index_symbols)]
            print(f"✓ {config.UNIVERSE_INDEX} filter: {before} → {len(df)} stocks")
        # If fetch failed, index_symbols is empty and we fall through to full universe

    # Keep only the columns we need
    keep_cols = [
        "instrument_token", "tradingsymbol", "name",
        "exchange", "segment", "instrument_type",
        "tick_size", "lot_size"
    ]
    df = df[keep_cols].copy()
    df["last_updated"] = _now_ist()

    db.upsert_instruments(df)
    print(f"✓ Stored {len(df)} instruments in universe")
    return df


# ============================================================
# HISTORICAL PULL — the slow part
# ============================================================
def fetch_historical_for_universe(
    client: KiteClient,
    universe_df: pd.DataFrame,
    days: int = None,
    progress_callback=None,
) -> int:
    """
    Loops through universe, pulls daily candles for last N days each.
    Incremental: stocks already cached get only their missing tail (today –
    last_cached_date), so daily reruns finish in ~1-2 min vs 10+ min.
    Stores in DB. Returns count of stocks successfully fetched.

    progress_callback: optional fn(idx, total, symbol) for Streamlit UI updates
    """
    days = days or config.HISTORICAL_LOOKBACK_DAYS
    to_date = datetime.now()
    full_from_date = to_date - timedelta(days=days)

    # Load latest cached date per token for incremental refresh
    conn = db.get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT instrument_token, MAX(date) AS latest FROM daily_ohlcv GROUP BY instrument_token"
    )
    rows = cur.fetchall()
    cur.close()
    db.release_conn(conn)
    latest_map: dict[int, date] = {int(r[0]): r[1] for r in rows}

    # Decide from_date per stock: full lookback for new stocks, delta for cached
    today = to_date.date()
    skipped = 0
    for token in latest_map:
        last = latest_map[token]
        if hasattr(last, "date"):
            last = last.date()
        if last >= today:
            skipped += 1
    if skipped:
        print(f"  {skipped} stocks already up to date — skipping")

    success_count = 0
    fail_count = 0
    skip_count  = 0   # stocks already up to date — counted to distinguish "nothing to do" from "all failed"
    first_error = ""
    error_samples: list[str] = []   # first 5 failing symbols + error for diagnostics
    all_candles = []

    iterator = enumerate(universe_df.itertuples(index=False))
    if progress_callback is None:
        iterator = enumerate(tqdm(list(universe_df.itertuples(index=False)),
                                  desc="Pulling history"))

    for idx, row in iterator:
        token = int(row.instrument_token)
        symbol = row.tradingsymbol

        if progress_callback:
            progress_callback(idx, len(universe_df), symbol)

        # Incremental: skip stocks already up to date; use delta from_date for cached
        cached_last = latest_map.get(token)
        if cached_last is not None:
            if hasattr(cached_last, "date"):
                cached_last = cached_last.date()
            if cached_last >= today:
                skip_count += 1
                continue  # nothing to fetch — NOT a failure
            from_date = datetime.combine(cached_last + timedelta(days=1), datetime.min.time())
        else:
            from_date = full_from_date

        try:
            candles = client.get_historical(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day",
            )
            if not candles:
                # Empty list = no new candles (holiday / new listing) — not a hard error
                skip_count += 1
                continue
            new_rows = [
                {
                    "instrument_token": token,
                    "date": c["date"].date() if hasattr(c["date"], "date") else c["date"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                }
                for c in candles
            ]
            all_candles.extend(new_rows)
            # Persist to checkpoint immediately — survives DB failures
            _checkpoint_append(new_rows)
            success_count += 1

            # Flush to DB every ~20 stocks and clear that portion from checkpoint
            if len(all_candles) > 5_000:
                db.upsert_ohlcv(pd.DataFrame(all_candles))
                all_candles = []
                # Checkpoint still holds the full run; clear only after full success

        except Exception as e:
            fail_count += 1
            exc_detail = f"[{type(e).__name__}] {e}"
            if fail_count == 1:
                first_error = exc_detail
            if len(error_samples) < 5:
                error_samples.append(f"{symbol}: {exc_detail}")
            continue

    # Final DB flush
    if all_candles:
        db.upsert_ohlcv(pd.DataFrame(all_candles))

    # All data committed — checkpoint no longer needed
    clear_checkpoint()

    print(f"✓ History pulled: {success_count} fetched, {skip_count} skipped (cached), {fail_count} failed")
    if error_samples:
        print(f"  First {len(error_samples)} failures: {error_samples}")

    # Only fatal if we have ZERO usable data at all — i.e., nothing was fetched AND
    # nothing was cached from a prior run.  If some stocks are already in the DB
    # (skip_count > 0) we can proceed even if today's incremental fetch failed.
    if success_count == 0 and skip_count == 0 and fail_count > 0:
        raise RuntimeError(
            f"All {fail_count} historical data fetches failed — DB is empty. "
            f"First error: {first_error}. "
            f"Sample failures: {error_samples[:3]}. "
            "Possible causes: (1) expired/invalid Kite access token — re-authenticate; "
            "(2) Historical Data add-on not active on your Kite Connect app; "
            "(3) API key does not have permission for historical data. "
            "Use the 'Test Kite Token' button in the sidebar to diagnose."
        )

    if success_count == 0 and fail_count > 0 and skip_count > 0:
        # Partial failure: cached data exists but today's update failed.
        # Warn but don't abort — compute_metrics will use yesterday's close prices.
        print(
            f"⚠ {fail_count} stocks failed to update today "
            f"({skip_count} were already cached). "
            f"First error: {first_error}. "
            "Proceeding with cached data — signals may use yesterday's close."
        )

    # Return tuple so callers can surface warnings without crashing
    return success_count, fail_count, first_error, error_samples


# ============================================================
# LIQUIDITY PRE-FILTER — runs BEFORE full historical pull
# ============================================================
def apply_liquidity_filter(client: KiteClient, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Before pulling 400 days × 2000 stocks (= 13 min @ 2.5 req/sec),
    use a cheap pre-filter to drop obvious illiquids.

    Strategy: pull TODAY's OHLC for entire universe (5 batches × 250 = 25 sec),
    drop anything with price < ₹50 or zero volume. Cuts universe ~50%.

    This is a quick-and-dirty filter. The real liquidity filter (20-day avg
    turnover) happens after we have history.
    """
    print(f"Pre-filtering universe (currently {len(universe_df)} stocks)...")
    instruments = [f"NSE:{s}" for s in universe_df["tradingsymbol"]]
    quotes = client.get_ohlc_batch(instruments)

    keep = []
    for _, row in universe_df.iterrows():
        key = f"NSE:{row.tradingsymbol}"
        q = quotes.get(key)
        if not q:
            continue
        ltp = q.get("last_price", 0)
        if ltp < config.MIN_PRICE:
            continue
        keep.append(row)

    filtered = pd.DataFrame(keep)
    print(f"✓ Pre-filter: {len(filtered)} stocks pass price floor (₹{config.MIN_PRICE})")
    return filtered


# ============================================================
# METRICS COMPUTATION — runs AFTER history is in DB
# ============================================================
def compute_metrics_for_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock in universe with cached OHLCV, computes all indicators
    and the composite score. Returns DataFrame ready for DB upsert.
    """
    # First, get Nifty 50's 3M return for relative strength calc
    nifty_df = db.load_ohlcv(config.NIFTY_50_TOKEN)
    nifty_3m = None
    if not nifty_df.empty and len(nifty_df) > 63:
        from indicators import pct_return
        nifty_3m = pct_return(nifty_df.sort_values("date").reset_index(drop=True), 63)
        print(f"Nifty 50 3M return: {nifty_3m:.2f}%")
    else:
        print("⚠ Nifty 50 history not cached — relative strength will be NULL")

    # Bulk-load ALL OHLCV in one query — eliminates N individual DB round-trips
    all_tokens = universe_df["instrument_token"].dropna().astype(int).tolist()
    print(f"Bulk-loading OHLCV for {len(all_tokens)} tokens in one query…")
    ohlcv_map = db.load_ohlcv_bulk(all_tokens)
    print(f"Loaded OHLCV for {len(ohlcv_map)} tokens")

    rows = []
    for row in tqdm(universe_df.itertuples(index=False), total=len(universe_df),
                    desc="Computing indicators"):
        token = int(row.instrument_token)
        symbol = row.tradingsymbol

        ohlcv = ohlcv_map.get(token, pd.DataFrame())
        if ohlcv.empty or len(ohlcv) < 30:
            continue

        metrics = compute_all(ohlcv, nifty_3m_return=nifty_3m)
        if metrics is None:
            continue

        # Apply liquidity gate AFTER computation (we have the real numbers now)
        if (metrics["avg_turnover_cr"] is None or
            metrics["avg_turnover_cr"] < config.MIN_AVG_TURNOVER_CR):
            continue
        if (metrics["avg_volume"] is None or
            metrics["avg_volume"] < config.MIN_AVG_VOLUME):
            continue

        # Compute trade signals (entry/stop/target for all three setups)
        sigs = compute_all_signals(ohlcv, metrics)
        metrics.update(sigs)

        metrics["instrument_token"] = token
        metrics["tradingsymbol"] = symbol
        metrics["last_updated"] = _now_ist()
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Order columns to match DB schema
    cols = [
        "instrument_token", "tradingsymbol", "ltp",
        "avg_turnover_cr", "avg_volume",
        "ret_5d", "ret_1m", "ret_3m", "ret_6m", "ret_1y",
        "rs_vs_nifty_3m", "vol_expansion_ratio",
        "rsi_14", "ema_20", "ema_50", "ema_200", "atr_14",
        "high_52w", "low_52w", "dist_from_52w_high_pct",
        "dist_from_50ema_pct", "support_20d", "resistance_20d",
        "trend_score", "composite_score",
        # Swing signals
        "swing_signal", "swing_setup",
        "swing_entry", "swing_stop", "swing_t1", "swing_t2",
        "swing_rr", "swing_quality", "swing_reason",
        # Intraday signals
        "intraday_signal", "intraday_pivot",
        "intraday_r1", "intraday_r2", "intraday_r3",
        "intraday_s1", "intraday_s2", "intraday_s3",
        "intraday_entry", "intraday_stop", "intraday_t1", "intraday_t2",
        "intraday_confidence", "intraday_gap_flag", "intraday_nifty_gate",
        "intraday_reason",
        # Scaling signals
        "scale_signal", "scale_setup",
        "scale_entry_1", "scale_stop", "scale_trailing_stop",
        "scale_target", "scale_quality", "scale_reason",
        "last_updated",
    ]
    # Only keep columns that exist (graceful if signals module returned partial data)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    print(f"✓ Computed metrics for {len(df)} stocks (post liquidity gate)")
    return df


# ============================================================
# MAIN ORCHESTRATORS — these are what app.py calls
# ============================================================
def full_rescan(progress_callback=None, client: "KiteClient | None" = None) -> dict:
    """
    The big one. End-to-end refresh.
    Pass `client` from st.session_state["kite_client"] for multi-user isolation.
    Returns summary dict for UI display.
    """
    t0 = time.time()
    db.init_schema()
    if client is None:
        client = KiteClient()

    # Step 1: Universe (only if stale, or if DB has non-EQ instruments from a
    # previous run that lacked the series filter)
    age = db.get_instruments_age_days()
    needs_refresh = age < 0 or age > config.INSTRUMENTS_REFRESH_DAYS
    if not needs_refresh:
        conn = db.get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM instruments")
        universe = db._df_from_cursor(cur)
        cur.close()
        db.release_conn(conn)
        # Bonds/SGBs/surveillance stocks have hyphens in tradingsymbol
        # (e.g., OMFURN-ST, 672KL27-SG). If any exist, the DB predates the
        # hyphen filter — force a clean pull.
        if universe["tradingsymbol"].str.contains("-", na=False).any():
            print("⚠ Universe contains non-EQ instruments — refreshing with series filter")
            needs_refresh = True
    if needs_refresh:
        universe = refresh_universe(client)

    # Step 2: Pre-filter REMOVED.
    # The previous OHLC-based pre-filter sent 250 instruments per URL to
    # api.kite.trade, which triggered Cloudflare's managed challenge because
    # URLs with 250+ query params look like bot traffic. The real liquidity
    # gate (avg_turnover_cr, avg_volume) runs in compute_metrics_for_universe
    # after we have historical data — that filter is more accurate anyway.
    filtered = universe

    # Step 3: Add Nifty 50 to the fetch list (for relative strength)
    nifty_row = pd.DataFrame([{
        "instrument_token": config.NIFTY_50_TOKEN,
        "tradingsymbol": "NIFTY 50",
        "name": "NIFTY 50",
        "exchange": "NSE", "segment": "INDICES",
        "instrument_type": "EQ", "tick_size": 0.05, "lot_size": 1,
    }])
    fetch_list = pd.concat([filtered, nifty_row], ignore_index=True)

    # Step 4: Pull historical (the slow part)
    n_fetched, n_fetch_failed, _fetch_first_err, _fetch_err_samples = fetch_historical_for_universe(
        client, fetch_list, progress_callback=progress_callback
    )

    # Verify OHLCV landed in DB before spending time on indicators
    conn = db.get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM daily_ohlcv")
    ohlcv_rows = cur.fetchone()[0]
    cur.close()
    db.release_conn(conn)
    print(f"  daily_ohlcv total rows after fetch: {ohlcv_rows:,}")
    if ohlcv_rows == 0:
        raise RuntimeError(
            "Historical fetch completed but daily_ohlcv is empty — "
            "no candle data was written. "
            "Check that your Kite access token is valid and try again."
        )

    # Step 5: Compute everything
    metrics_df = compute_metrics_for_universe(filtered)

    # Step 4.5: Pre-select intraday candidates for tomorrow's 6-pillar scan.
    # Runs after Step 5 so that computed metrics (atr_14, composite_score, ema_200)
    # are available. Persists a shortlist to the intraday_candidates table.
    if not metrics_df.empty:
        try:
            candidates_df = select_intraday_candidates(metrics_df)
            db.replace_intraday_candidates(candidates_df)
            print(f"✓ Step 4.5: {len(candidates_df)} intraday candidates saved for tomorrow's 6-pillar scan")
        except Exception as _cand_err:
            print(f"⚠ Step 4.5 (candidate pre-selection) failed: {_cand_err} — continuing")

    # Step 6: Persist
    if not metrics_df.empty:
        db.replace_metrics(metrics_df)
    else:
        print("⚠ compute_metrics_for_universe returned 0 rows — "
              "check liquidity gate thresholds and OHLCV data quality")

    elapsed = time.time() - t0
    result: dict = {
        "universe_size":    len(universe),
        "post_pre_filter":  len(filtered),
        "ohlcv_rows":       ohlcv_rows,
        "history_fetched":  n_fetched,
        "history_failed":   n_fetch_failed,
        "fetch_first_error": _fetch_first_err,
        "fetch_err_samples": _fetch_err_samples,
        "metrics_computed": len(metrics_df),
        "elapsed_sec":      round(elapsed, 1),
    }
    return result


def refresh_intraday_signal_statuses(live_ltp: dict) -> int:
    """
    Compare live LTP against each active 6-pillar signal's entry/SL/T1/T2/T3
    and update the status column in intraday_signals.

    Status lifecycle:
        WATCHING    → generated, waiting for entry level to be hit
        APPROACHING → LTP within 0.3% of entry
        TRIGGERED   → LTP crossed entry (for BUY: ltp >= entry; SHORT: ltp <= entry)
        AT_T1       → LTP reached T1
        AT_T2       → LTP reached T2
        RUNNER      → T2 hit, runner position tracking
        HIT_SL      → stop loss hit → closed
        REVERSED    → price re-crossed back through ORH/ORL invalidating the setup
        EXPIRED     → 15:15 reached

    Called from quick_refresh() every ~15 s during market hours.
    Also respects already-terminal statuses (HIT_SL, EXPIRED) — won't overwrite.
    Returns count of status changes made.
    """
    IST  = timezone(timedelta(hours=5, minutes=30))
    now  = datetime.now(tz=IST)

    # After 15:15, mark all non-terminal WATCHING/APPROACHING as EXPIRED
    market_closed = now.hour > 15 or (now.hour == 15 and now.minute >= 15)

    try:
        sigs = db.load_intraday_signals(include_no_signal=False)
    except Exception:
        return 0
    if sigs.empty:
        return 0

    changes = 0
    TERMINAL = {"HIT_SL", "RUNNER", "AT_T2", "EXPIRED", "REVERSED"}

    for _, row in sigs.iterrows():
        sym     = row["tradingsymbol"]
        signal  = row.get("signal", "")        # BUY or SHORT
        status  = row.get("status") or "WATCHING"
        entry   = row.get("entry")
        sl      = row.get("stop_loss")
        t1      = row.get("t1")
        t2      = row.get("t2")
        orh     = row.get("orh")
        orl     = row.get("orl")

        if status in TERMINAL:
            # Already done — only apply market-close EXPIRED to WATCHING/APPROACHING
            continue
        if not entry or signal not in ("BUY", "SHORT"):
            continue

        ltp = live_ltp.get(sym)
        if not ltp or ltp <= 0:
            continue

        entry = float(entry)
        sl    = float(sl)   if sl   else None
        t1    = float(t1)   if t1   else None
        t2    = float(t2)   if t2   else None
        orh   = float(orh)  if orh  else None
        orl   = float(orl)  if orl  else None
        ltp   = float(ltp)

        new_status = status

        if market_closed:
            if status in ("WATCHING", "APPROACHING", "TRIGGERED", "AT_T1"):
                new_status = "EXPIRED"
        elif signal == "BUY":
            # Stop hit?
            if sl and ltp <= sl:
                new_status = "HIT_SL"
            # Reversed back below ORH?
            elif orh and ltp < orh * 0.998 and status == "TRIGGERED":
                new_status = "REVERSED"
            # Progress through targets
            elif t2 and ltp >= t2:
                new_status = "AT_T2"
            elif t1 and ltp >= t1:
                new_status = "AT_T1"
            elif ltp >= entry:
                new_status = "TRIGGERED"
            elif ltp >= entry * 0.997:
                new_status = "APPROACHING"
            else:
                new_status = "WATCHING"
        else:  # SHORT
            if sl and ltp >= sl:
                new_status = "HIT_SL"
            elif orl and ltp > orl * 1.002 and status == "TRIGGERED":
                new_status = "REVERSED"
            elif t2 and ltp <= t2:
                new_status = "AT_T2"
            elif t1 and ltp <= t1:
                new_status = "AT_T1"
            elif ltp <= entry:
                new_status = "TRIGGERED"
            elif ltp <= entry * 1.003:
                new_status = "APPROACHING"
            else:
                new_status = "WATCHING"

        if new_status != status:
            try:
                db.update_intraday_signal_status(sym, new_status)
                changes += 1
            except Exception as _ue:
                print(f"⚠ 6-pillar status update failed for {sym}: {_ue}")

    return changes


def quick_refresh(client: "KiteClient | None" = None) -> dict:
    """
    Fast intra-day refresh. Updates LTP + today's stats, re-ranks composite,
    AND recomputes all trade signals against the fresh price.

    Why recompute signals here?
      Signals depend on LTP vs EMA20/EMA50/R1/etc. During the trading day the
      EMA/ATR/RSI values are stable (they're based on prior closes), but LTP
      moves constantly. A stock that was NOT in the pullback zone at yesterday's
      close may drift INTO it intraday → a fresh BUY signal appears. Similarly
      the intraday R1 trigger fires the moment LTP crosses above R1.

    ~10-15 seconds total (OHLC batch fetch + vectorised signal computation).
    """
    t0 = time.time()
    if client is None:
        client = KiteClient()
    metrics = db.load_metrics()
    if metrics.empty:
        return {"error": "No cached metrics. Run Full Rescan first."}

    instruments = [f"NSE:{s}" for s in metrics["tradingsymbol"]]
    quotes = client.get_ohlc_batch(instruments)

    # Update LTP for each row
    updated = 0
    for idx, row in metrics.iterrows():
        key = f"NSE:{row['tradingsymbol']}"
        q = quotes.get(key)
        if not q:
            continue
        new_ltp = q.get("last_price")
        if new_ltp:
            metrics.at[idx, "ltp"] = new_ltp
            # Recompute distance from 50 EMA with fresh price
            if row["ema_50"] and row["ema_50"] > 0:
                metrics.at[idx, "dist_from_50ema_pct"] = (
                    (new_ltp - row["ema_50"]) / row["ema_50"] * 100
                )
            # Recompute distance from 52W high
            if row["high_52w"] and row["high_52w"] > 0:
                metrics.at[idx, "dist_from_52w_high_pct"] = (
                    (row["high_52w"] - new_ltp) / new_ltp * 100
                )
            updated += 1

    # ── Recompute signals against fresh LTP ──────────────────────────────
    # The EMA / ATR / RSI values are based on prior closes so they don't
    # change intraday — only LTP does. Passing a minimal metrics dict (no df)
    # is enough because signal computation uses the stored indicator values.
    sig_cols = [
        "swing_signal", "swing_setup", "swing_entry", "swing_stop",
        "swing_t1", "swing_t2", "swing_rr", "swing_quality", "swing_reason",
        "intraday_signal", "intraday_pivot",
        "intraday_r1", "intraday_r2", "intraday_r3",
        "intraday_s1", "intraday_s2", "intraday_s3",
        "intraday_entry", "intraday_stop", "intraday_t1", "intraday_t2",
        "intraday_confidence", "intraday_gap_flag", "intraday_nifty_gate",
        "intraday_reason",
        "scale_signal", "scale_setup",
        "scale_entry_1", "scale_stop", "scale_trailing_stop",
        "scale_target", "scale_quality", "scale_reason",
    ]
    # Only attempt if signal columns already exist (post first full rescan)
    existing_sig_cols = [c for c in sig_cols if c in metrics.columns]
    if existing_sig_cols:
        from signals import swing_signal as _sw_sig, intraday_signal as _it_sig, scaling_signal as _sc_sig

        # Pivot-level columns are derived from yesterday's OHLCV (H/L/C) and do
        # NOT change during the trading day. With an empty df, intraday_signal()
        # returns all-None for these fields — we must NOT overwrite valid values
        # from the last Full Rescan.
        _PRESERVE_PIVOT = {
            "intraday_pivot",
            "intraday_r1", "intraday_r2", "intraday_r3",
            "intraday_s1", "intraday_s2", "intraday_s3",
            "intraday_entry", "intraday_stop", "intraday_t1", "intraday_t2",
        }

        sigs_updated = 0
        for idx, row in metrics.iterrows():
            m = row.to_dict()
            sw = _sw_sig(pd.DataFrame(), m)   # PULLBACK works; BREAKOUT/NR7 need df → graceful
            it = _it_sig(pd.DataFrame(), m)   # returns None for pivot cols (expected)
            sc = _sc_sig(pd.DataFrame(), m)   # fully metric-driven — works perfectly
            merged = {**sw, **it, **sc}
            for col, val in merged.items():
                if col not in metrics.columns:
                    continue
                # Don't overwrite valid pivot levels with None
                if col in _PRESERVE_PIVOT and val is None:
                    continue
                metrics.at[idx, col] = val
            sigs_updated += 1
    else:
        sigs_updated = 0

    metrics["last_updated"] = _now_ist()
    db.replace_metrics(metrics)

    try:
        db.save_signal_snapshot(metrics, user_id="", nifty_pct_chg=None)
    except Exception:
        pass

    # ── Refresh 6-pillar intraday signal statuses against live LTP ──────────
    _6p_changes = 0
    try:
        ltp_map = {f"NSE:{row['tradingsymbol']}": quotes.get(f"NSE:{row['tradingsymbol']}", {}).get("last_price")
                   for _, row in metrics.iterrows()}
        # Build a plain {symbol: ltp} map
        _sym_ltp = {}
        for _, row in metrics.iterrows():
            _q = quotes.get(f"NSE:{row['tradingsymbol']}")
            if _q:
                _sym_ltp[row["tradingsymbol"]] = _q.get("last_price")
        _6p_changes = refresh_intraday_signal_statuses(_sym_ltp)
    except Exception as _6pe:
        print(f"⚠ 6-pillar status refresh failed: {_6pe}")

    return {
        "stocks_updated": updated,
        "signals_recomputed": sigs_updated,
        "total_in_cache": len(metrics),
        "sixpillar_status_updates": _6p_changes,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def refresh_signals_only(progress_callback=None, user_id: str = "") -> dict:
    """
    Recompute ALL trade signals from OHLCV already stored in the DB.

    No API calls needed — reads historical data from Neon Postgres, runs
    swing / intraday / scaling signal logic, and writes results back.
    Typical runtime: 30–60 s for ~1000 stocks.

    When to use:
      • After a code change to signals.py (new setups, bug fixes)
      • When signal columns are missing/NULL after a failed Full Rescan
      • To refresh signals mid-day without fetching new history

    Loads per-user signal_config overrides (derived from paper trade feedback)
    and applies them to the intraday signal computation.
    """
    from signals import compute_all_signals as _all_sigs

    t0 = time.time()
    metrics = db.load_metrics()
    if metrics.empty:
        return {"error": "No data in DB. Run Full Rescan first."}

    # Load tuned thresholds from paper-trade feedback (falls back to defaults)
    sig_cfg = db.get_signal_config(user_id=user_id)
    _rsi_buy_max  = sig_cfg.get("intraday_rsi_buy_max",  75.0)
    _rsi_sell_min = sig_cfg.get("intraday_rsi_sell_min", 25.0)
    _min_rr       = sig_cfg.get("intraday_min_rr",        1.5)

    total   = len(metrics)
    updated = 0
    errors  = 0

    # Bulk-load ALL OHLCV in a single DB round-trip
    all_tokens = metrics["instrument_token"].dropna().astype(int).tolist()
    ohlcv_map  = db.load_ohlcv_bulk(all_tokens)

    for idx, row in metrics.iterrows():
        sym   = row.get("tradingsymbol", "?")
        token = row.get("instrument_token")
        if not token:
            continue

        if progress_callback:
            progress_callback(idx, total, sym)

        try:
            ohlcv = ohlcv_map.get(int(token), pd.DataFrame())
            if ohlcv.empty or len(ohlcv) < 5:
                continue
            m    = row.to_dict()
            sigs = _all_sigs(ohlcv, m,
                             rsi_buy_max=_rsi_buy_max,
                             rsi_sell_min=_rsi_sell_min,
                             min_rr=_min_rr)
            for col, val in sigs.items():
                if col in metrics.columns:
                    metrics.at[idx, col] = val
            updated += 1
        except Exception:
            errors += 1
            continue

    metrics["last_updated"] = _now_ist()
    db.replace_metrics(metrics)

    try:
        db.save_signal_snapshot(metrics, user_id=user_id, nifty_pct_chg=None)
    except Exception:
        pass

    return {
        "signals_updated": updated,
        "errors":          errors,
        "total":           total,
        "elapsed_sec":     round(time.time() - t0, 1),
        "thresholds_used": {
            "rsi_buy_max":  _rsi_buy_max,
            "rsi_sell_min": _rsi_sell_min,
            "min_rr":       _min_rr,
        },
    }


# ============================================================
# 6-PILLAR INTRADAY SIGNAL SYSTEM
# ============================================================

def select_intraday_candidates(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4.5 — run after compute_metrics_for_universe().

    Selects ALL stocks that clear the ATR% floor — no composite_score rank cap.
    INTRADAY_MAX_CANDIDATES acts as a safety ceiling only (e.g. 300) to prevent
    runaway API usage if the ATR% threshold is misconfigured.  Pre-computes P2
    (daily structure score) so intraday_scan() doesn't re-derive it from DB.
    """
    df = metrics_df.copy()

    # ATR% uses ltp as the price reference (post-market, ltp = yesterday's close)
    df["atr_pct"] = df.apply(
        lambda r: (r["atr_14"] / r["ltp"] * 100)
        if r.get("atr_14") and r.get("ltp") and r["ltp"] > 0
        else None,
        axis=1,
    )

    eligible = df[df["atr_pct"].notna() & (df["atr_pct"] >= config.INTRADAY_MIN_ATR_PCT)].copy()

    # Sort by composite_score so the best stocks appear first in scan output,
    # but do NOT cut to top-N — every stock that clears ATR% gate is included.
    # INTRADAY_MAX_CANDIDATES is a safety ceiling only.
    candidates = (
        eligible
        .sort_values("composite_score", ascending=False)
        .head(config.INTRADAY_MAX_CANDIDATES)          # safety ceiling
        [["tradingsymbol", "instrument_token", "ltp", "atr_14", "atr_pct",
          "ema_20", "ema_200", "composite_score", "rs_vs_nifty_3m",
          "trend_score", "rsi_14"]]
        .copy()
        .reset_index(drop=True)
    )

    def _p2(row) -> int:
        s   = 0
        ltp = row["ltp"] or 0
        e20 = row["ema_20"] or 0
        e200 = row["ema_200"] or 0
        if not ltp:
            return 0
        if ltp > e20:              s += 2
        if ltp > e200:             s += 2
        if e20 > e200:             s += 1
        if ltp < e200:             s -= 3
        return max(-3, s)

    candidates["p2_score_precomputed"] = candidates.apply(_p2, axis=1)
    return candidates


# ── 6-pillar scoring functions ────────────────────────────────────────────────

def _score_p1(adx_val: float, plus_di: float, minus_di: float,
               is_buy: bool) -> tuple[int, bool]:
    """Pillar 1 — Trend Strength (ADX/DMI). Returns (score, adx_gate_passed).
    ADX gate threshold lowered to 15 (was 20) — ADX 15-20 is directional enough intraday.
    """
    score = 0
    if adx_val > 35:
        score += 5
    elif adx_val > 25:
        score += 4
    elif adx_val > 20:
        score += 3
    elif adx_val > 15:
        score += 1
    if is_buy and plus_di > minus_di:
        score += 2
    if not is_buy and minus_di > plus_di:
        score += 2
    return score, adx_val >= 15


def _score_p2(ltp: float, ema_20d: float, ema_200d: float, is_buy: bool) -> int:
    """Pillar 2 — Long-Term Structure (daily EMAs, pre-computed)."""
    s = 0
    if is_buy:
        if ltp > ema_20d:    s += 2
        if ltp > ema_200d:   s += 2
        if ema_20d > ema_200d: s += 1
        if ltp < ema_200d:   s -= 3
    else:
        if ltp < ema_20d:    s += 2
        if ltp < ema_200d:   s += 2
        if ema_20d < ema_200d: s += 1
        if ltp > ema_200d:   s -= 3
    return s


def _score_p3(ema9: float, ema21: float, ema50: float,
               st_bull_5m: bool, st_bull_15m: bool, is_buy: bool) -> int:
    """Pillar 3 — Intraday Trend Direction (5-min EMAs + Supertrend)."""
    s = 0
    if is_buy:
        if st_bull_5m:                        s += 2
        if ema9 > ema21 and ema21 > ema50:    s += 2
        if st_bull_15m:                       s += 1
    else:
        if not st_bull_5m:                    s += 2
        if ema9 < ema21 and ema21 < ema50:    s += 2
        if not st_bull_15m:                   s += 1
    return s


def _score_p4(rsi5: float, macd_hist: float, macd_hist_prev: float,
               macd_line: float, signal_line: float, is_buy: bool) -> int:
    """Pillar 4 — Momentum Acceleration (RSI + MACD histogram)."""
    s = 0
    if is_buy:
        if 50 <= rsi5 <= 65:            s += 2
        if rsi5 > 70:                   s -= 2
        if macd_hist > macd_hist_prev:  s += 2
        if macd_line > signal_line:     s += 1
    else:
        if 35 <= rsi5 <= 50:            s += 2
        if rsi5 < 30:                   s -= 2
        if macd_hist < macd_hist_prev:  s += 2
        if macd_line < signal_line:     s += 1
    return s


def _score_p5(volume: float, avg_vol_20: float,
               obv_series: "pd.Series", is_buy: bool) -> int:
    """Pillar 5 — Volume Flow (OBV + surge)."""
    import numpy as np
    vol_surge  = volume > avg_vol_20 * config.INTRADAY_VOL_MULT
    obv_rising = obv_series.iloc[-1] > obv_series.iloc[-4]  if len(obv_series) >= 4  else False
    obv_new_hi = (obv_series.iloc[-1] > obv_series.iloc[-11:-1].max()
                  if len(obv_series) >= 11 else False)
    s = 0
    if is_buy:
        if vol_surge:        s += 2
        if obv_rising:       s += 2
        if obv_new_hi:       s += 1
        # Penalty reduced to -1 (was -3): flat OBV is neutral, not catastrophic.
        # Active distribution (OBV falling while price rises) is what should hurt.
        if not obv_rising:   s -= 1
    else:
        obv_falling = obv_series.iloc[-1] < obv_series.iloc[-4] if len(obv_series) >= 4 else False
        obv_new_lo  = (obv_series.iloc[-1] < obv_series.iloc[-11:-1].min()
                       if len(obv_series) >= 11 else False)
        if vol_surge:        s += 2
        if obv_falling:      s += 2
        if obv_new_lo:       s += 1
        if not obv_falling:  s -= 1
    return s


def _score_p6(close: float, orh: float, orl: float, vwap: float,
               news_catalyst: bool, is_buy: bool) -> int:
    """Pillar 6 — Structural Trigger (ORB + VWAP)."""
    s = 0
    if is_buy:
        if close > orh:  s += 2
        if close > vwap: s += 2
        if news_catalyst: s += 1
        if close > orh and close < vwap: s -= 3
    else:
        if close < orl:  s += 2
        if close < vwap: s += 2
        if news_catalyst: s += 1
        if close < orl and close > vwap: s -= 3
    return s


def _compute_grade(total_score: int, adx_gate: bool) -> str:
    if not adx_gate:
        return "D"
    if total_score >= config.INTRADAY_GRADE_APLUS: return "A+"
    if total_score >= config.INTRADAY_GRADE_A:     return "A"
    if total_score >= config.INTRADAY_GRADE_B:     return "B"
    if total_score >= config.INTRADAY_GRADE_C:     return "C"
    return "D"


def _compute_supertrend(df: pd.DataFrame, atr_period: int = 10,
                         multiplier: float = 3.0) -> bool:
    """
    Compute Supertrend on the given OHLCV DataFrame.
    Returns True if the most recent candle is in a bullish Supertrend.
    Requires columns: high, low, close.
    """
    import numpy as np
    n = len(df)
    if n < atr_period + 2:
        return True  # not enough data — default neutral/bullish

    df = df.copy().reset_index(drop=True)
    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df[["high", "low", "prev_close"]].apply(
        lambda r: max(
            r["high"] - r["low"],
            abs(r["high"] - (r["prev_close"] or r["low"])),
            abs(r["low"]  - (r["prev_close"] or r["high"])),
        ),
        axis=1,
    )
    df["atr"] = df["tr"].ewm(span=atr_period, adjust=False).mean()

    df["basic_ub"] = (df["high"] + df["low"]) / 2 + multiplier * df["atr"]
    df["basic_lb"] = (df["high"] + df["low"]) / 2 - multiplier * df["atr"]

    final_ub = [0.0] * n
    final_lb = [0.0] * n
    st_bull  = [True] * n

    for i in range(1, n):
        # Final Upper Band
        if df["basic_ub"].iloc[i] < final_ub[i-1] or df["close"].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = df["basic_ub"].iloc[i]
        else:
            final_ub[i] = final_ub[i-1]

        # Final Lower Band
        if df["basic_lb"].iloc[i] > final_lb[i-1] or df["close"].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = df["basic_lb"].iloc[i]
        else:
            final_lb[i] = final_lb[i-1]

        # Direction
        if df["close"].iloc[i] <= final_ub[i]:
            st_bull[i] = False
        elif df["close"].iloc[i] >= final_lb[i]:
            st_bull[i] = True
        else:
            st_bull[i] = st_bull[i-1]

    return st_bull[-1]


# ── Main intraday scan orchestrator ───────────────────────────────────────────

def intraday_scan(client=None, progress_callback=None) -> dict:
    """
    Run 6-pillar intraday signal computation on the pre-selected candidate list.

    Timing : Call after 9:45 AM IST (ORB window closed).
    Runtime: ~2–4 min for 40 stocks (5-min + 15-min candle fetches via Kite).

    Returns a summary dict for UI display.
    """
    import numpy as np

    t0  = time.time()
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(tz=IST)

    # ── Step A: Guard — Kite client ──────────────────────────────────────────
    if client is None:
        return {"error": "Kite client not initialised. Please log in first."}

    # ── Step B: Validate timing ──────────────────────────────────────────────
    orb_close  = now.replace(hour=9,  minute=45, second=0, microsecond=0)
    hard_cutoff = now.replace(hour=11, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)

    if now < orb_close:
        return {
            "error": f"Too early. ORB window closes at 9:45 AM IST. Current time: {now.strftime('%H:%M')}"
        }
    if now > market_close:
        return {"error": "Market closed. Intraday signals are only valid during market hours."}

    late_warning = now > hard_cutoff

    # ── Step C: Load candidates ──────────────────────────────────────────────
    candidates = db.load_intraday_candidates()
    if candidates.empty:
        return {"error": "No candidates found. Run Full Rescan first to build the candidate list."}

    # ── Step D: Fetch Nifty regime context ───────────────────────────────────
    today_date = now.date()
    today_9am  = datetime.combine(today_date, datetime.min.time()).replace(
        hour=9, minute=0, second=0
    )
    nifty_pct_chg = 0.0
    try:
        nifty_candles = client.get_historical(
            instrument_token=config.NIFTY_50_TOKEN,
            from_date=today_9am,
            to_date=now.replace(tzinfo=None),
            interval="5minute",
        )
        if nifty_candles and len(nifty_candles) >= 2:
            nf_open  = float(nifty_candles[0]["open"])
            nf_last  = float(nifty_candles[-1]["close"])
            if nf_open > 0:
                nifty_pct_chg = (nf_last - nf_open) / nf_open * 100
    except Exception as _ne:
        print(f"⚠ Nifty 5-min fetch failed: {_ne}")

    # ── Step E: Scan each candidate ──────────────────────────────────────────
    signals = []
    total_cands = len(candidates)

    for idx, cand in candidates.iterrows():
        sym   = cand["tradingsymbol"]
        token = int(cand["instrument_token"])

        if progress_callback:
            progress_callback(idx, total_cands, sym)
        else:
            print(f"  [{idx+1}/{total_cands}] {sym}")

        try:
            # Fetch 5-min candles (75 bars covers full session + prior context)
            candles_5m = client.get_historical(
                instrument_token=token,
                from_date=today_9am,
                to_date=now.replace(tzinfo=None),
                interval="5minute",
            )
            if not candles_5m or len(candles_5m) < config.INTRADAY_ORB_CANDLES + 3:
                signals.append(_no_signal_row(sym, token, reason="Insufficient 5-min candles", scan_date=today_date))
                continue

            df5 = pd.DataFrame(candles_5m)
            df5["date"] = pd.to_datetime(df5["date"])

            # Fetch 15-min candles for HTF Supertrend only
            candles_15m = client.get_historical(
                instrument_token=token,
                from_date=today_9am,
                to_date=now.replace(tzinfo=None),
                interval="15minute",
            )
            df15 = pd.DataFrame(candles_15m) if candles_15m else pd.DataFrame()
            if not df15.empty:
                df15["date"] = pd.to_datetime(df15["date"])

            # ── ORH / ORL ────────────────────────────────────────────────────
            today_str = str(today_date)
            session_df = df5[df5["date"].dt.date.astype(str) == today_str].copy()
            orb_candles = session_df.head(config.INTRADAY_ORB_CANDLES)

            if len(orb_candles) < config.INTRADAY_ORB_CANDLES:
                signals.append(_no_signal_row(sym, token, reason="ORB window not yet complete", scan_date=today_date))
                continue

            orh = float(orb_candles["high"].max())
            orl = float(orb_candles["low"].min())

            # ── 5-min indicators (computed on full df5 for EWM stability) ────
            df5 = df5.copy().reset_index(drop=True)
            df5["prev_close"] = df5["close"].shift(1)
            df5["tr"] = df5.apply(
                lambda r: max(
                    r["high"] - r["low"],
                    abs(r["high"] - (r["prev_close"] if pd.notna(r["prev_close"]) else r["low"])),
                    abs(r["low"]  - (r["prev_close"] if pd.notna(r["prev_close"]) else r["high"])),
                ), axis=1,
            )
            atr5_series = df5["tr"].ewm(span=14, adjust=False).mean()
            atr5 = float(atr5_series.iloc[-1])

            # RSI(14) on 5-min
            delta_c = df5["close"].diff()
            gain_c  = delta_c.clip(lower=0).ewm(span=14, adjust=False).mean()
            loss_c  = (-delta_c.clip(upper=0)).ewm(span=14, adjust=False).mean()
            rs5     = gain_c / loss_c.replace(0, float("nan"))
            rsi5    = float(100 - (100 / (1 + rs5)).iloc[-1])

            # MACD(12, 26, 9) on 5-min
            ema12       = df5["close"].ewm(span=12, adjust=False).mean()
            ema26       = df5["close"].ewm(span=26, adjust=False).mean()
            macd_line_s = ema12 - ema26
            signal_line_s = macd_line_s.ewm(span=9, adjust=False).mean()
            macd_hist_s   = macd_line_s - signal_line_s
            macd_hist_val  = float(macd_hist_s.iloc[-1])
            macd_hist_prev = float(macd_hist_s.iloc[-2]) if len(macd_hist_s) >= 2 else 0.0
            macd_line_val  = float(macd_line_s.iloc[-1])
            sig_line_val   = float(signal_line_s.iloc[-1])

            # ADX(14) and ±DI on 5-min
            df5["high_diff"] = df5["high"].diff()
            df5["low_diff"]  = df5["low"].shift(1) - df5["low"]
            df5["plus_dm"]   = df5["high_diff"].where(
                (df5["high_diff"] > df5["low_diff"]) & (df5["high_diff"] > 0), 0)
            df5["minus_dm"]  = df5["low_diff"].where(
                (df5["low_diff"] > df5["high_diff"]) & (df5["low_diff"] > 0), 0)
            atr14_sm  = df5["tr"].ewm(span=14, adjust=False).mean()
            plus_di_s = 100 * df5["plus_dm"].ewm(span=14, adjust=False).mean() / atr14_sm.replace(0, float("nan"))
            minus_di_s= 100 * df5["minus_dm"].ewm(span=14, adjust=False).mean() / atr14_sm.replace(0, float("nan"))
            dx_s      = (100 * (plus_di_s - minus_di_s).abs() /
                         (plus_di_s + minus_di_s).replace(0, float("nan")))
            adx_val   = float(dx_s.ewm(span=14, adjust=False).mean().iloc[-1])
            plus_di_val  = float(plus_di_s.iloc[-1])
            minus_di_val = float(minus_di_s.iloc[-1])

            # Session-only indicators (use session_df for VWAP, EMAs, OBV)
            session_df = session_df.copy().reset_index(drop=True)
            session_df["ema9"]  = session_df["close"].ewm(span=9,  adjust=False).mean()
            session_df["ema21"] = session_df["close"].ewm(span=21, adjust=False).mean()
            session_df["ema50"] = session_df["close"].ewm(span=50, adjust=False).mean()

            session_df["tp"]      = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
            session_df["cum_vol"] = session_df["volume"].cumsum()
            session_df["cum_tpv"] = (session_df["tp"] * session_df["volume"]).cumsum()
            session_df["vwap"]    = session_df["cum_tpv"] / session_df["cum_vol"].replace(0, float("nan"))

            session_df["price_diff"] = session_df["close"].diff().fillna(0)
            session_df["obv"] = (np.sign(session_df["price_diff"]) * session_df["volume"]).cumsum()

            avg_vol_20  = float(session_df["volume"].rolling(20, min_periods=1).mean().iloc[-1])
            current_vol = float(session_df["volume"].iloc[-1])

            latest = session_df.iloc[-1]
            close  = float(latest["close"])
            vwap   = float(latest["vwap"]) if pd.notna(latest["vwap"]) else close
            ema9   = float(latest["ema9"])
            ema21  = float(latest["ema21"])
            ema50  = float(latest["ema50"])

            # Supertrend on 5-min and 15-min
            st_bull_5m  = _compute_supertrend(
                df5, config.INTRADAY_ST_ATR_PERIOD, config.INTRADAY_ST_MULTIPLIER)
            st_bull_15m = (_compute_supertrend(
                df15, config.INTRADAY_ST_ATR_PERIOD, config.INTRADAY_ST_MULTIPLIER)
                if not df15.empty else True)

            # ── Gap invalidation ─────────────────────────────────────────────
            prev_close_px = float(cand.get("ltp") or close)
            today_open_px = float(orb_candles["open"].iloc[0])
            atr_pct_val   = (atr5 / close * 100) if close > 0 else 0.0
            gap_pct = abs(today_open_px - prev_close_px) / prev_close_px * 100 if prev_close_px > 0 else 0
            gap_invalidated = gap_pct > (atr_pct_val * 1.5)

            # ── Determine signal direction ───────────────────────────────────
            buy_candidate   = close > orh and close > vwap
            short_candidate = close < orl and close < vwap

            signal_direction = None
            if buy_candidate:
                signal_direction = "BUY"
            elif short_candidate:
                signal_direction = "SHORT"

            # ── 3-candle confirmation: scan full session ──────────────────────
            confirmation_found = False
            entry_price        = None
            c1_idx             = None

            for ci in range(len(session_df) - 2):
                c1 = session_df.iloc[ci]
                c2 = session_df.iloc[ci + 1]
                c3 = session_df.iloc[ci + 2]
                c1_vol = float(c1["volume"])
                if signal_direction == "BUY":
                    if (float(c1["close"]) > orh and
                            c1_vol > avg_vol_20 * config.INTRADAY_VOL_MULT and
                            float(c2["close"]) > orh):
                        confirmation_found = True
                        entry_price = float(c3["open"])
                        c1_idx = ci
                elif signal_direction == "SHORT":
                    if (float(c1["close"]) < orl and
                            c1_vol > avg_vol_20 * config.INTRADAY_VOL_MULT and
                            float(c2["close"]) < orl):
                        confirmation_found = True
                        entry_price = float(c3["open"])
                        c1_idx = ci

            # Use the most recent valid pattern
            # (loop above overwrites — last found = most recent)

            # Staleness check: signal from more than INTRADAY_STALE_CANDLES bars ago
            if confirmation_found and c1_idx is not None:
                bars_ago = len(session_df) - 1 - c1_idx
                if bars_ago > config.INTRADAY_STALE_CANDLES:
                    signals.append(_no_signal_row(
                        sym, token,
                        reason=f"ORB signal stale — breakout occurred {bars_ago * 5} min ago (>{config.INTRADAY_STALE_CANDLES * 5} min threshold)",
                        scan_date=today_date,
                    ))
                    continue

            if not confirmation_found or entry_price is None:
                signals.append(_no_signal_row(
                    sym, token,
                    reason="No valid 3-candle ORB confirmation in this session",
                    scan_date=today_date,
                ))
                continue

            # ── Score all 6 pillars ──────────────────────────────────────────
            is_buy = signal_direction == "BUY"

            p1, adx_gate = _score_p1(adx_val, plus_di_val, minus_di_val, is_buy)
            p2 = int(cand.get("p2_score_precomputed") or 0)
            p3 = _score_p3(ema9, ema21, ema50, st_bull_5m, st_bull_15m, is_buy)
            p4 = _score_p4(rsi5, macd_hist_val, macd_hist_prev, macd_line_val, sig_line_val, is_buy)
            p5 = _score_p5(current_vol, avg_vol_20, session_df["obv"], is_buy)
            p6 = _score_p6(close, orh, orl, vwap, news_catalyst=False, is_buy=is_buy)

            total_score = p1 + p2 + p3 + p4 + p5 + p6
            if gap_invalidated:
                total_score = max(0, total_score - 3)

            grade       = _compute_grade(total_score, adx_gate)
            final_signal = signal_direction if grade != "D" else "NO_SIGNAL"

            # ── Entry / stop / targets ───────────────────────────────────────
            stop_loss = t1 = t2 = t3 = rr_at_t2 = None
            reason = ""
            if final_signal != "NO_SIGNAL":
                atr_pct_val = (atr5 / entry_price * 100) if entry_price > 0 else 0
                if atr_pct_val < config.INTRADAY_MIN_ATR_PCT:
                    final_signal = "NO_SIGNAL"
                    reason = (f"ATR% {atr_pct_val:.2f}% < "
                              f"minimum {config.INTRADAY_MIN_ATR_PCT}%")
                else:
                    if is_buy:
                        stop_loss = max(orl, entry_price - atr5 * config.INTRADAY_ATR_SL_MULT)
                        t1 = entry_price * (1 + config.INTRADAY_T1_PCT)
                        t2 = entry_price * (1 + config.INTRADAY_T2_PCT)
                        t3 = entry_price * (1 + config.INTRADAY_T3_PCT)
                    else:
                        stop_loss = min(orh, entry_price + atr5 * config.INTRADAY_ATR_SL_MULT)
                        t1 = entry_price * (1 - config.INTRADAY_T1_PCT)
                        t2 = entry_price * (1 - config.INTRADAY_T2_PCT)
                        t3 = entry_price * (1 - config.INTRADAY_T3_PCT)

                    stop_dist = abs(entry_price - stop_loss)
                    stop_pct  = stop_dist / entry_price * 100 if entry_price > 0 else 0
                    rr_at_t2  = abs(t2 - entry_price) / stop_dist if stop_dist > 0 else 0

                    if rr_at_t2 < config.INTRADAY_MIN_RR:
                        final_signal = "NO_SIGNAL"
                        reason = f"R:R at T2 {rr_at_t2:.2f} < minimum {config.INTRADAY_MIN_RR}"
                    elif stop_pct > config.INTRADAY_MAX_STOP_PCT:
                        final_signal = "NO_SIGNAL"
                        reason = f"Stop {stop_pct:.2f}% > max {config.INTRADAY_MAX_STOP_PCT}%"

            if final_signal != "NO_SIGNAL":
                breakeven_stop = (entry_price - 1.0) if is_buy else (entry_price + 1.0)
                trail_after_t2 = t1
                valid_until    = now.replace(hour=15, minute=15, second=0,
                                             microsecond=0).isoformat()
            else:
                breakeven_stop = trail_after_t2 = valid_until = None

            signals.append({
                "tradingsymbol":    sym,
                "instrument_token": token,
                "scan_date":        today_date,
                "signal":           final_signal,
                "grade":            grade,
                "total_score":      total_score,
                "p1_adx":           p1,
                "p2_daily":         p2,
                "p3_trend":         p3,
                "p4_momentum":      p4,
                "p5_volume":        p5,
                "p6_trigger":       p6,
                "entry":            entry_price,
                "stop_loss":        stop_loss,
                "t1":               t1,
                "t2":               t2,
                "t3":               t3,
                "t1_shares_pct":    0.40,
                "t2_shares_pct":    0.35,
                "t3_shares_pct":    0.25,
                "breakeven_stop":   breakeven_stop,
                "trail_after_t2":   trail_after_t2,
                "orh":              orh,
                "orl":              orl,
                "atr5":             atr5,
                "atr_pct":          atr_pct_val,
                "rr_at_t2":         rr_at_t2,
                "adx_gate":         adx_gate,
                "gap_invalidated":  gap_invalidated,
                "news_catalyst":    False,
                "reason":           reason,
                "valid_until":      valid_until,
                "signal_generated_at": _now_ist().isoformat(),
            })

            time.sleep(0.4)  # stay within Kite's ~2.5 req/sec limit

        except Exception as _e:
            print(f"  ⚠ {sym}: {_e}")
            signals.append(_no_signal_row(sym, token, reason=str(_e)[:200], scan_date=today_date))
            continue

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        db.replace_intraday_signals(signals_df)

    buy_count   = int((signals_df["signal"] == "BUY").sum())   if not signals_df.empty else 0
    short_count = int((signals_df["signal"] == "SHORT").sum()) if not signals_df.empty else 0
    total_sigs  = buy_count + short_count

    return {
        "signals_found":  total_sigs,
        "buy_signals":    buy_count,
        "short_signals":  short_count,
        "total_scanned":  len(signals),
        "late_warning":   late_warning,
        "nifty_pct_chg":  round(nifty_pct_chg, 2),
        "elapsed_sec":    round(time.time() - t0, 1),
    }


def _no_signal_row(tradingsymbol: str, instrument_token: int,
                   reason: str = "", scan_date=None) -> dict:
    """Return a minimal NO_SIGNAL dict for a stock that couldn't be scored."""
    return {
        "tradingsymbol":    tradingsymbol,
        "instrument_token": instrument_token,
        "scan_date":        scan_date or _now_ist().date(),
        "signal":           "NO_SIGNAL",
        "grade":            "D",
        "total_score":      0,
        "p1_adx": 0, "p2_daily": 0, "p3_trend": 0,
        "p4_momentum": 0, "p5_volume": 0, "p6_trigger": 0,
        "entry": None, "stop_loss": None,
        "t1": None, "t2": None, "t3": None,
        "t1_shares_pct": 0.40, "t2_shares_pct": 0.35, "t3_shares_pct": 0.25,
        "breakeven_stop": None, "trail_after_t2": None,
        "orh": None, "orl": None, "atr5": None, "atr_pct": None,
        "rr_at_t2": None, "adx_gate": False, "gap_invalidated": False,
        "news_catalyst": False, "reason": reason, "valid_until": None,
        "signal_generated_at": _now_ist().isoformat(),
    }
