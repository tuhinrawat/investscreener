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
import time
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd
from tqdm import tqdm

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

    # Try to load from local cache first (refresh if stale > 7 days)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            cached_date = date.fromisoformat(cached.get("date", "2000-01-01"))
            if cached.get("index") == index_name and (date.today() - cached_date).days < 7:
                print(f"Using cached {index_name} list ({cached_date})")
                return set(cached["symbols"])
        except Exception:
            pass

    # Download fresh — NSE requires a prior session hit to set cookies before
    # it will serve the CSV (anti-scraping). We warm up with the homepage first.
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

        # Step 1: warm-up hit to get NSE session cookies
        session.get("https://www.nseindia.com", timeout=10)
        # Step 2: download the index CSV
        r = session.get(url, headers={"Referer": "https://www.nseindia.com"}, timeout=15)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        # NSE CSV has a "Symbol" column (sometimes with trailing spaces)
        symbols = set(df["Symbol"].str.strip().tolist())

        # Cache to disk
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "index": index_name,
            "date": date.today().isoformat(),
            "symbols": sorted(symbols),
        }))
        print(f"✓ Downloaded {index_name}: {len(symbols)} constituents")
        return symbols

    except Exception as e:
        print(f"⚠ Could not fetch {index_name} list ({e}) — using full EQ universe")
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
    df["last_updated"] = datetime.now()

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
    first_error = ""
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
                continue  # nothing to fetch
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
                fail_count += 1
                continue
            for c in candles:
                all_candles.append({
                    "instrument_token": token,
                    "date": c["date"].date() if hasattr(c["date"], "date") else c["date"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                })
            success_count += 1

            # Flush every ~20 stocks (~5,000 candles) so a crash loses at most
            # 20 stocks of progress rather than 200. The incremental logic on
            # restart skips anything already persisted, so this is safe to tune.
            if len(all_candles) > 5_000:
                db.upsert_ohlcv(pd.DataFrame(all_candles))
                all_candles = []

        except Exception as e:
            fail_count += 1
            # Capture the first error for diagnosis — surfaces auth/token issues
            if fail_count == 1:
                first_error = str(e)
            continue

    # Final flush
    if all_candles:
        db.upsert_ohlcv(pd.DataFrame(all_candles))

    print(f"✓ History pulled: {success_count} success, {fail_count} failed")
    if success_count == 0 and fail_count > 0:
        raise RuntimeError(
            f"All {fail_count} historical data fetches failed. "
            f"First error: {first_error}. "
            "This is usually an expired Kite access token — "
            "please re-authenticate Kite from the sidebar."
        )
    return success_count


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

    rows = []
    for row in tqdm(universe_df.itertuples(index=False), total=len(universe_df),
                    desc="Computing indicators"):
        token = int(row.instrument_token)
        symbol = row.tradingsymbol

        ohlcv = db.load_ohlcv(token)
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
        metrics["last_updated"] = datetime.now()
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
        "intraday_r1", "intraday_r2", "intraday_s1", "intraday_s2",
        "intraday_entry", "intraday_stop", "intraday_t1", "intraday_reason",
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
    n_fetched = fetch_historical_for_universe(
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

    # Step 6: Persist
    if not metrics_df.empty:
        db.replace_metrics(metrics_df)
    else:
        print("⚠ compute_metrics_for_universe returned 0 rows — "
              "check liquidity gate thresholds and OHLCV data quality")

    elapsed = time.time() - t0
    return {
        "universe_size":   len(universe),
        "post_pre_filter": len(filtered),
        "ohlcv_rows":      ohlcv_rows,
        "history_fetched": n_fetched,
        "metrics_computed": len(metrics_df),
        "elapsed_sec":     round(elapsed, 1),
    }


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
        "intraday_r1", "intraday_r2", "intraday_s1", "intraday_s2",
        "intraday_entry", "intraday_stop", "intraday_t1", "intraday_reason",
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
            "intraday_pivot", "intraday_r1", "intraday_r2",
            "intraday_s1",    "intraday_s2",
            "intraday_entry", "intraday_stop", "intraday_t1",
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

    metrics["last_updated"] = datetime.now()
    db.replace_metrics(metrics)

    return {
        "stocks_updated": updated,
        "signals_recomputed": sigs_updated,
        "total_in_cache": len(metrics),
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

    for idx, row in metrics.iterrows():
        sym   = row.get("tradingsymbol", "?")
        token = row.get("instrument_token")
        if not token:
            continue

        if progress_callback:
            progress_callback(idx, total, sym)

        try:
            ohlcv = db.load_ohlcv(int(token))
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

    metrics["last_updated"] = datetime.now()
    db.replace_metrics(metrics)

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
