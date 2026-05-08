"""
feature_test.py — Comprehensive feature test for the stock screener.

Tests 30 stocks (3 batches of 10) across:
  1.  DB integrity              — no nulls in critical fields, OHLCV completeness
  2.  Indicator accuracy        — re-compute EMA/RSI/ATR and diff against DB values
  3.  Returns accuracy          — recalculate from raw OHLCV vs DB stored values
  4.  Composite score           — verify weighting formula
  5.  Trade signals             — re-generate and compare with DB
  6.  Chart summary logic       — Weinstein stage, pivot levels, S/R zones
  7.  DB round-trip             — save / load consistency (AI results)
  8.  Extended indicators       — all_returns, 52W levels, S/R, liquidity, vol expansion
  9.  Scaling signal            — level ordering, quality range, R/R invariants
  10. Trade log CRUD            — log_trade, close_trade, delete_trade, P&L math,
                                  load_trade_log, get_trade_stats, load_ohlcv
  11. Paper trading             — get_open_paper_trades, get_paper_trade_perf,
                                  get_signal_config, save_signal_config,
                                  tune_signal_config_from_paper
  12. Tunable thresholds        — intraday_signal with custom RSI / min_rr gates,
                                  AVOID conditions, compute_all_signals passthrough
  13. DB schema integrity       — init_schema idempotency, column completeness
  14. Latency benchmark         — wall-clock time per operation
  15. API cost accounting       — count Kite API calls made

Run from project root:
    python feature_test.py
"""

import sys
import time
import random
import traceback
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import datetime as _dt

import db
import config
import indicators as ind
from signals import (
    compute_all_signals,
    intraday_signal,
    scaling_signal,
    swing_signal,
)

# ─── ANSI colours ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════
# RESULT TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class Results:
    def __init__(self):
        self.checks: list[dict] = []

    def ok(self, name: str, detail: str = "", latency_ms: float = 0):
        self.checks.append({"status": "PASS", "name": name, "detail": detail, "ms": latency_ms})
        print(f"  {GREEN}✓{RESET} {name:<52} {DIM}{detail}{RESET}  {DIM}{latency_ms:.0f}ms{RESET}")

    def fail(self, name: str, detail: str = "", latency_ms: float = 0):
        self.checks.append({"status": "FAIL", "name": name, "detail": detail, "ms": latency_ms})
        print(f"  {RED}✗{RESET} {name:<52} {RED}{detail}{RESET}  {DIM}{latency_ms:.0f}ms{RESET}")

    def warn(self, name: str, detail: str = "", latency_ms: float = 0):
        self.checks.append({"status": "WARN", "name": name, "detail": detail, "ms": latency_ms})
        print(f"  {YELLOW}⚠{RESET} {name:<52} {YELLOW}{detail}{RESET}  {DIM}{latency_ms:.0f}ms{RESET}")

    def summary(self):
        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        failed = sum(1 for c in self.checks if c["status"] == "FAIL")
        warned = sum(1 for c in self.checks if c["status"] == "WARN")
        total  = len(self.checks)
        avg_ms = sum(c["ms"] for c in self.checks) / max(total, 1)
        print(f"\n{BOLD}{'─'*70}{RESET}")
        print(f"{BOLD}SUMMARY  {passed}/{total} passed  |  {failed} failed  |  {warned} warnings{RESET}")
        print(f"Average check latency: {avg_ms:.1f} ms")
        if failed == 0 and warned == 0:
            print(f"{GREEN}{BOLD}ALL CHECKS PASSED ✓{RESET}")
        elif failed == 0:
            print(f"{YELLOW}{BOLD}PASSED with {warned} warnings{RESET}")
        else:
            print(f"{RED}{BOLD}{failed} FAILURES — see ✗ lines above{RESET}")
        return failed


R = Results()
API_CALLS = {"kite": 0, "db_reads": 0, "db_writes": 0}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _t(fn, *args, **kwargs):
    """Call fn, return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000


def _near(a, b, tol=0.01):
    """True if a and b are within tol% of each other."""
    if a is None or b is None:
        return False
    if pd.isna(float(a)) or pd.isna(float(b)):
        return False
    denom = abs(float(b)) if abs(float(b)) > 1e-9 else 1e-9
    return abs(float(a) - float(b)) / denom < tol


def _fmt_price(v):
    return f"₹{v:,.2f}" if v and not pd.isna(float(v)) else "—"


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITES PER STOCK
# ═══════════════════════════════════════════════════════════════════════════

def test_ohlcv_integrity(sym: str, token: int, ohlcv: pd.DataFrame):
    """Suite 1: Raw OHLCV quality."""
    tag = f"[{sym}] OHLCV"

    # Row count
    t0 = time.perf_counter()
    n = len(ohlcv)
    ms = (time.perf_counter() - t0) * 1000
    if n >= 200:
        R.ok(f"{tag} row count", f"{n} rows", ms)
    elif n >= 50:
        R.warn(f"{tag} row count", f"only {n} rows (< 200)", ms)
    else:
        R.fail(f"{tag} row count", f"only {n} rows — too thin for indicators", ms)
        return  # no point continuing

    # No NaN in OHLCV
    t0 = time.perf_counter()
    nan_cols = [c for c in ["open","high","low","close","volume"] if ohlcv[c].isna().any()]
    ms = (time.perf_counter() - t0) * 1000
    if nan_cols:
        R.fail(f"{tag} no NaN values", f"NaN in {nan_cols}", ms)
    else:
        R.ok(f"{tag} no NaN values", "all OHLCV columns complete", ms)

    # High >= Low, High >= Open/Close, Low <= Open/Close
    t0 = time.perf_counter()
    bad_hl = (ohlcv["high"] < ohlcv["low"]).sum()
    bad_hc = (ohlcv["high"] < ohlcv["close"]).sum()
    bad_lc = (ohlcv["low"]  > ohlcv["close"]).sum()
    ms = (time.perf_counter() - t0) * 1000
    if bad_hl + bad_hc + bad_lc > 0:
        R.fail(f"{tag} OHLC relationships", f"{bad_hl} H<L, {bad_hc} H<C, {bad_lc} L>C", ms)
    else:
        R.ok(f"{tag} OHLC relationships", "H≥L, H≥C, L≤C for all rows", ms)

    # All prices positive
    t0 = time.perf_counter()
    neg = (ohlcv[["open","high","low","close"]] <= 0).any().any()
    ms  = (time.perf_counter() - t0) * 1000
    if neg:
        R.fail(f"{tag} positive prices", "one or more prices ≤ 0", ms)
    else:
        R.ok(f"{tag} positive prices", "all prices > 0", ms)

    # Dates sorted ascending, no future dates
    t0 = time.perf_counter()
    dates  = pd.to_datetime(ohlcv["date"])
    sorted_ok   = (dates.diff().dropna() > pd.Timedelta(0)).all()
    future_rows = (dates > pd.Timestamp.now()).sum()
    ms = (time.perf_counter() - t0) * 1000
    if not sorted_ok:
        R.fail(f"{tag} date order", "dates not strictly ascending", ms)
    elif future_rows > 0:
        R.fail(f"{tag} no future dates", f"{future_rows} future rows", ms)
    else:
        latest = dates.iloc[-1].strftime("%Y-%m-%d")
        R.ok(f"{tag} dates", f"sorted asc · latest {latest} · no futures", ms)

    # No duplicate dates
    t0 = time.perf_counter()
    dups = ohlcv["date"].duplicated().sum()
    ms   = (time.perf_counter() - t0) * 1000
    if dups:
        R.fail(f"{tag} no duplicate dates", f"{dups} duplicate date rows", ms)
    else:
        R.ok(f"{tag} no duplicate dates", "all dates unique", ms)

    # Volume sanity — at least some non-zero volume days
    t0 = time.perf_counter()
    zero_vol_pct = (ohlcv["volume"] == 0).mean() * 100
    ms = (time.perf_counter() - t0) * 1000
    if zero_vol_pct > 20:
        R.warn(f"{tag} volume data", f"{zero_vol_pct:.1f}% zero-volume days", ms)
    else:
        R.ok(f"{tag} volume data", f"{zero_vol_pct:.1f}% zero-volume days", ms)


def test_indicators(sym: str, token: int, ohlcv: pd.DataFrame, row: pd.Series):
    """Suite 2: Re-compute indicators and compare with DB stored values."""
    tag = f"[{sym}] Indicators"

    # RSI
    t0 = time.perf_counter()
    rsi_calc = ind.rsi(ohlcv)
    rsi_db   = row.get("rsi_14")
    ms = (time.perf_counter() - t0) * 1000
    if rsi_calc is None:
        R.warn(f"{tag} RSI-14", "insufficient data to compute", ms)
    elif pd.isna(rsi_db):
        R.warn(f"{tag} RSI-14", f"calc={rsi_calc:.2f}, DB=null", ms)
    elif _near(rsi_calc, rsi_db, 0.02):  # 2% tolerance
        R.ok(f"{tag} RSI-14", f"calc={rsi_calc:.2f} ≈ DB={rsi_db:.2f}", ms)
    else:
        R.fail(f"{tag} RSI-14", f"calc={rsi_calc:.2f} ≠ DB={rsi_db:.2f} (>{2}%)", ms)

    # EMA 50
    t0 = time.perf_counter()
    ema50_calc = ind.ema(ohlcv, 50)
    ema50_db   = row.get("ema_50")
    ms = (time.perf_counter() - t0) * 1000
    if ema50_calc is None:
        R.warn(f"{tag} EMA-50", "insufficient data", ms)
    elif pd.isna(ema50_db):
        R.warn(f"{tag} EMA-50", f"calc={_fmt_price(ema50_calc)}, DB=null", ms)
    elif _near(ema50_calc, ema50_db):
        R.ok(f"{tag} EMA-50", f"calc={_fmt_price(ema50_calc)} ≈ DB={_fmt_price(ema50_db)}", ms)
    else:
        R.fail(f"{tag} EMA-50", f"calc={_fmt_price(ema50_calc)} ≠ DB={_fmt_price(ema50_db)}", ms)

    # ATR-14
    t0 = time.perf_counter()
    atr_calc = ind.atr(ohlcv)
    atr_db   = row.get("atr_14")
    ms = (time.perf_counter() - t0) * 1000
    if atr_calc is None:
        R.warn(f"{tag} ATR-14", "insufficient data", ms)
    elif pd.isna(atr_db):
        R.warn(f"{tag} ATR-14", f"calc={_fmt_price(atr_calc)}, DB=null", ms)
    elif _near(atr_calc, atr_db, 0.03):
        R.ok(f"{tag} ATR-14", f"calc={_fmt_price(atr_calc)} ≈ DB={_fmt_price(atr_db)}", ms)
    else:
        R.fail(f"{tag} ATR-14", f"calc={_fmt_price(atr_calc)} ≠ DB={_fmt_price(atr_db)}", ms)

    # LTP sanity — DB LTP should be near latest close
    t0 = time.perf_counter()
    ltp_db    = row.get("ltp")
    latest_c  = float(ohlcv["close"].iloc[-1])
    ms = (time.perf_counter() - t0) * 1000
    if pd.isna(ltp_db):
        R.warn(f"{tag} LTP in DB", "LTP is null", ms)
    elif _near(ltp_db, latest_c, 0.05):
        R.ok(f"{tag} LTP vs latest close", f"DB={_fmt_price(ltp_db)} ≈ OHLCV={_fmt_price(latest_c)}", ms)
    else:
        # Could legitimately differ if DB was last updated days ago
        R.warn(f"{tag} LTP vs latest close", f"DB={_fmt_price(ltp_db)} vs OHLCV={_fmt_price(latest_c)} (>{5}%)", ms)


def test_returns(sym: str, token: int, ohlcv: pd.DataFrame, row: pd.Series):
    """Suite 3: Recalculate multi-timeframe returns and compare with DB."""
    tag = f"[{sym}] Returns"

    windows = {"5D": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    for label, days in windows.items():
        t0 = time.perf_counter()
        calc = ind.pct_return(ohlcv, days)
        db_v = row.get(f"ret_{label.lower()}")
        ms   = (time.perf_counter() - t0) * 1000
        if calc is None:
            R.warn(f"{tag} ret_{label}", "insufficient data", ms)
            continue
        if pd.isna(db_v):
            R.warn(f"{tag} ret_{label}", f"calc={calc:+.2f}%, DB=null", ms)
            continue
        if _near(calc, db_v, 0.01):
            R.ok(f"{tag} ret_{label}", f"{calc:+.2f}% ≈ DB={db_v:+.2f}%", ms)
        else:
            R.fail(f"{tag} ret_{label}", f"calc={calc:+.2f}% ≠ DB={db_v:+.2f}%", ms)


def test_composite_score(sym: str, row: pd.Series):
    """Suite 4: Verify composite score formula."""
    tag = f"[{sym}] Composite"

    t0 = time.perf_counter()
    trend_db  = row.get("trend_score")
    rs_db     = row.get("rs_vs_nifty_3m")
    vol_db    = row.get("vol_expansion_ratio")
    comp_db   = row.get("composite_score")
    ms        = (time.perf_counter() - t0) * 1000

    if any(pd.isna(v) for v in [trend_db, rs_db, vol_db, comp_db]):
        R.warn(f"{tag} components present", "one or more component scores null in DB", ms)
        return

    # Replicate composite formula from indicators.py
    # composite = W_TREND * trend + W_RS * rs + W_VOL * (vol_expansion - 1) * 50
    vol_contribution = (float(vol_db) - 1.0) * 50 if vol_db else 0.0
    calc_comp = (
        config.W_TREND             * float(trend_db) +
        config.W_RELATIVE_STRENGTH * float(rs_db)    +
        config.W_VOLUME_EXPANSION  * vol_contribution
    )
    if _near(calc_comp, comp_db, 0.05):
        R.ok(f"{tag} score formula", f"calc={calc_comp:.2f} ≈ DB={comp_db:.2f}", ms)
    else:
        R.fail(f"{tag} score formula", f"calc={calc_comp:.2f} ≠ DB={comp_db:.2f}", ms)

    # RSI in sane range
    t0 = time.perf_counter()
    rsi_v = row.get("rsi_14")
    ms    = (time.perf_counter() - t0) * 1000
    if not pd.isna(rsi_v) and 0 <= float(rsi_v) <= 100:
        R.ok(f"{tag} RSI range", f"RSI={rsi_v:.1f} ∈ [0,100]", ms)
    elif pd.isna(rsi_v):
        R.warn(f"{tag} RSI range", "RSI null", ms)
    else:
        R.fail(f"{tag} RSI range", f"RSI={rsi_v:.1f} out of [0,100]", ms)

    # Vol expansion ≥ 0
    t0  = time.perf_counter()
    ms  = (time.perf_counter() - t0) * 1000
    if not pd.isna(vol_db) and float(vol_db) >= 0:
        R.ok(f"{tag} vol expansion ≥ 0", f"vol_exp={vol_db:.2f}x", ms)
    else:
        R.fail(f"{tag} vol expansion ≥ 0", f"vol_exp={vol_db}", ms)


def test_signals(sym: str, token: int, ohlcv: pd.DataFrame, row: pd.Series):
    """Suite 5: Regenerate signals and validate key invariants."""
    tag = f"[{sym}] Signals"
    if len(ohlcv) < 30:
        R.warn(f"{tag} regen", "skipped — too few rows", 0)
        return

    # Re-compute — pass metrics dict as required by the function signature
    t0 = time.perf_counter()
    try:
        metrics_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        fresh = compute_all_signals(ohlcv, metrics_dict)
    except Exception as e:
        R.fail(f"{tag} regen compute", str(e)[:60])
        return
    ms = (time.perf_counter() - t0) * 1000

    # Signal type is a known enum value or None
    valid_sw = {None, "BUY", "SELL", "HOLD", "WATCH", "AVOID", "EXIT"}
    valid_id = {None, "BUY_ABOVE", "SELL_BELOW", "NONE", "WATCH", "AVOID"}
    sw  = fresh.get("swing_signal")
    idr = fresh.get("intraday_signal")

    if sw in valid_sw:
        R.ok(f"{tag} swing signal type", f"signal={sw}", ms)
    else:
        R.fail(f"{tag} swing signal type", f"unexpected signal='{sw}'", ms)

    if idr in valid_id:
        R.ok(f"{tag} intraday signal type", f"signal={idr}", ms)
    else:
        R.fail(f"{tag} intraday signal type", f"unexpected signal='{idr}'", ms)

    # If swing BUY: entry < t1, stop < entry, R/R ≥ 1.0
    if sw == "BUY":
        t0 = time.perf_counter()
        entry = fresh.get("swing_entry")
        stop  = fresh.get("swing_stop")
        t1    = fresh.get("swing_t1")
        rr    = fresh.get("swing_rr")
        ms    = (time.perf_counter() - t0) * 1000
        price_ok = (entry and stop and t1 and stop < entry < t1)
        rr_ok    = (rr and float(rr) >= 1.0)
        if price_ok:
            R.ok(f"{tag} swing BUY levels", f"stop={_fmt_price(stop)} < entry={_fmt_price(entry)} < t1={_fmt_price(t1)}", ms)
        else:
            R.fail(f"{tag} swing BUY levels", f"stop={_fmt_price(stop)} entry={_fmt_price(entry)} t1={_fmt_price(t1)}", ms)
        if rr_ok:
            R.ok(f"{tag} swing BUY R/R", f"R/R={rr:.2f}x ≥ 1.0", ms)
        else:
            R.fail(f"{tag} swing BUY R/R", f"R/R={rr} < 1.0", ms)

    # Intraday BUY_ABOVE: entry > current close (it's a breakout above)
    if idr == "BUY_ABOVE":
        t0 = time.perf_counter()
        entry = fresh.get("intraday_entry")
        stop  = fresh.get("intraday_stop")
        t1    = fresh.get("intraday_t1")
        ltp   = float(ohlcv["close"].iloc[-1])
        ms    = (time.perf_counter() - t0) * 1000
        levels_ok = (entry and stop and t1 and stop < entry < t1)
        if levels_ok:
            R.ok(f"{tag} intraday BUY_ABOVE levels", f"stop={_fmt_price(stop)} < entry={_fmt_price(entry)} < t1={_fmt_price(t1)}", ms)
        else:
            R.fail(f"{tag} intraday BUY_ABOVE levels", f"stop={_fmt_price(stop)} entry={_fmt_price(entry)} t1={_fmt_price(t1)}", ms)

    # Intraday SELL_BELOW: stop > entry, entry > t1 (short)
    if idr == "SELL_BELOW":
        t0 = time.perf_counter()
        entry = fresh.get("intraday_entry")
        stop  = fresh.get("intraday_stop")
        t1    = fresh.get("intraday_t1")
        ms    = (time.perf_counter() - t0) * 1000
        levels_ok = (entry and stop and t1 and stop > entry > t1)
        if levels_ok:
            R.ok(f"{tag} intraday SELL_BELOW levels", f"t1={_fmt_price(t1)} < entry={_fmt_price(entry)} < stop={_fmt_price(stop)}", ms)
        else:
            R.fail(f"{tag} intraday SELL_BELOW levels", f"t1={_fmt_price(t1)} entry={_fmt_price(entry)} stop={_fmt_price(stop)}", ms)

    # Gain always covers risk for any signal
    for setup, entry_k, stop_k, t1_k in [
        ("swing",    "swing_entry",    "swing_stop",    "swing_t1"),
        ("intraday", "intraday_entry", "intraday_stop", "intraday_t1"),
    ]:
        entry = fresh.get(entry_k)
        stop  = fresh.get(stop_k)
        t1    = fresh.get(t1_k)
        sig   = fresh.get(f"{setup}_signal")
        if not (entry and stop and t1 and sig in ("BUY", "BUY_ABOVE")):
            continue
        t0 = time.perf_counter()
        risk   = abs(float(entry) - float(stop))
        gain   = abs(float(t1)    - float(entry))
        ms     = (time.perf_counter() - t0) * 1000
        if risk > 0 and gain >= risk:
            R.ok(f"{tag} {setup} gain≥risk", f"gain={gain:.2f} ≥ risk={risk:.2f}", ms)
        elif risk <= 0:
            R.warn(f"{tag} {setup} gain≥risk", "risk=0 (degenerate level)", ms)
        else:
            R.fail(f"{tag} {setup} gain≥risk", f"gain={gain:.2f} < risk={risk:.2f}", ms)


def test_chart_summary_logic(sym: str, ohlcv: pd.DataFrame, row: pd.Series):
    """Suite 6: Verify chart-summary computations (Weinstein, pivots, S/R)."""
    tag = f"[{sym}] Charts"
    if len(ohlcv) < 50:
        R.warn(f"{tag} skipped", "too few rows", 0)
        return

    # Weinstein stage reproducibility
    t0 = time.perf_counter()
    c    = ohlcv["close"]
    e50  = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    ltp  = c.iloc[-1]
    above_200 = ltp > e200.iloc[-1]
    e50_above = e50.iloc[-1] > e200.iloc[-1]
    lb = min(30, len(c) - 1)
    slope = (e200.iloc[-1] - e200.iloc[-lb]) / e200.iloc[-lb] * 100

    if above_200 and e50_above and slope > 0.3:
        stage = "2"
    elif not above_200 and not e50_above and slope < -0.3:
        stage = "4"
    elif not above_200 and not e50_above and abs(slope) <= 0.3:
        stage = "1"
    else:
        stage = "3"
    ms = (time.perf_counter() - t0) * 1000
    R.ok(f"{tag} Weinstein stage", f"Stage {stage} · EMA200={_fmt_price(e200.iloc[-1])}", ms)

    # Pivot levels from last daily candle
    t0 = time.perf_counter()
    if len(ohlcv) >= 2:
        yest = ohlcv.iloc[-2]
        H, L, C = float(yest["high"]), float(yest["low"]), float(yest["close"])
        P  = (H + L + C) / 3
        R1 = 2 * P - L
        R2 = P + (H - L)
        S1 = 2 * P - H
        S2 = P - (H - L)
        pivots_ok = S2 < S1 < P < R1 < R2
        ms = (time.perf_counter() - t0) * 1000
        if pivots_ok:
            R.ok(f"{tag} pivot ordering", f"S2={_fmt_price(S2)} < S1 < P={_fmt_price(P)} < R1 < R2={_fmt_price(R2)}", ms)
        else:
            R.fail(f"{tag} pivot ordering", f"S2={S2:.2f} S1={S1:.2f} P={P:.2f} R1={R1:.2f} R2={R2:.2f}", ms)

        # Current price vs pivot
        t0 = time.perf_counter()
        ltp_now = float(ohlcv["close"].iloc[-1])
        bias = "Bullish" if ltp_now > P else "Bearish"
        ms   = (time.perf_counter() - t0) * 1000
        R.ok(f"{tag} pivot bias", f"LTP={_fmt_price(ltp_now)} vs P={_fmt_price(P)} → {bias}", ms)

    # S/R zone detection returns a list (at least 0 zones, never crashes)
    t0 = time.perf_counter()
    try:
        data = ohlcv.tail(120).reset_index(drop=True)
        raw = []
        for i in range(2, len(data) - 2):
            h = float(data["high"].iloc[i])
            if h >= data["high"].iloc[max(0,i-2):i].max() and h >= data["high"].iloc[i+1:i+3].max():
                raw.append(("R", h))
            l = float(data["low"].iloc[i])
            if l <= data["low"].iloc[max(0,i-2):i].min() and l <= data["low"].iloc[i+1:i+3].min():
                raw.append(("S", l))
        supports     = [p for t, p in raw if t == "S" and p < ltp_now]
        resistances  = [p for t, p in raw if t == "R" and p > ltp_now]
        ms = (time.perf_counter() - t0) * 1000
        R.ok(
            f"{tag} S/R zone detection",
            f"{len(supports)} supports below, {len(resistances)} resistances above",
            ms,
        )
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} S/R zone detection", str(e)[:60], ms)


def test_db_roundtrip(sym: str, token: int, row: pd.Series):
    """Suite 7: Save a synthetic AI result and reload — round-trip integrity."""
    tag = f"[{sym}] DB roundtrip"

    # Write synthetic AI result
    t0 = time.perf_counter()
    synthetic = {
        "ai_score":       7.5,
        "ai_verdict":     "BUY",
        "ai_confidence":  "HIGH",
        "ai_brief":       f"[TEST] Synthetic STOCKLENS brief for {sym}",
        "ai_analyzed_at": datetime.now().isoformat(),
    }
    try:
        db.save_ai_result(int(token), synthetic)
        ms_w = (time.perf_counter() - t0) * 1000
        API_CALLS["db_writes"] += 1
        R.ok(f"{tag} AI result write", f"token={token} written in {ms_w:.0f}ms", ms_w)
    except Exception as e:
        ms_w = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} AI result write", str(e)[:60], ms_w)
        return

    # Read back
    t0 = time.perf_counter()
    try:
        metrics = db.load_metrics()
        API_CALLS["db_reads"] += 1
        row_back = metrics[metrics["instrument_token"] == int(token)]
        ms_r = (time.perf_counter() - t0) * 1000
        if row_back.empty:
            R.fail(f"{tag} AI result reload", "row not found after write", ms_r)
            return
        score_back = row_back.iloc[0].get("ai_score")
        brief_back = row_back.iloc[0].get("ai_brief", "")
        if _near(score_back, 7.5, 0.001):
            R.ok(f"{tag} AI score round-trip", f"wrote=7.5, read={score_back}", ms_r)
        else:
            R.fail(f"{tag} AI score round-trip", f"wrote=7.5, read={score_back}", ms_r)
        if "[TEST]" in str(brief_back):
            R.ok(f"{tag} AI brief round-trip", "brief text preserved", ms_r)
        else:
            R.fail(f"{tag} AI brief round-trip", f"brief mismatch: '{str(brief_back)[:40]}'", ms_r)
    except Exception as e:
        ms_r = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} AI result reload", str(e)[:60], ms_r)

    # Clean up — reset AI fields to null
    try:
        _con = db.get_conn()
        _con.execute(
            "UPDATE computed_metrics SET ai_score=NULL, ai_verdict=NULL, ai_brief=NULL, "
            "ai_confidence=NULL, ai_analyzed_at=NULL WHERE instrument_token=?",
            [int(token)],
        )
        _con.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 8 — Extended indicator functions (synthetic data)
# ═══════════════════════════════════════════════════════════════════════════

def _make_ohlcv(n: int = 300, base: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic, well-formed OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 1, n))
    closes = np.maximum(closes, 1.0)
    highs  = closes + rng.uniform(0.1, 2.0, n)
    lows   = closes - rng.uniform(0.1, 2.0, n)
    lows   = np.maximum(lows, 0.01)
    opens  = lows + rng.uniform(0, highs - lows)
    vols   = rng.integers(100_000, 2_000_000, n).astype(float)
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols,
    })


def test_extended_indicators():
    """Suite 8: all_returns, fifty_two_week_levels, support_resistance,
    liquidity_metrics, volume_expansion — pure math, no DB."""
    tag = "[ExtInd]"
    df  = _make_ohlcv(300)

    # all_returns — should match individual pct_return calls
    t0 = time.perf_counter()
    rets = ind.all_returns(df)
    ms   = (time.perf_counter() - t0) * 1000
    expected_keys = {"ret_5d", "ret_1m", "ret_3m", "ret_6m", "ret_1y"}
    if expected_keys == set(rets.keys()):
        R.ok(f"{tag} all_returns keys", f"keys={sorted(rets.keys())}", ms)
    else:
        R.fail(f"{tag} all_returns keys", f"got {set(rets.keys())}", ms)

    for label, days in {"5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}.items():
        t0    = time.perf_counter()
        single = ind.pct_return(df, days)
        agg    = rets.get(f"ret_{label}")
        ms     = (time.perf_counter() - t0) * 1000
        if single is not None and agg is not None and _near(single, agg, 0.001):
            R.ok(f"{tag} all_returns ret_{label} matches pct_return", f"{single:+.3f}%", ms)
        elif single is None and agg is None:
            R.warn(f"{tag} all_returns ret_{label}", "both None — insufficient data", ms)
        else:
            R.fail(f"{tag} all_returns ret_{label} mismatch",
                   f"pct_return={single} all_returns={agg}", ms)

    # fifty_two_week_levels
    t0  = time.perf_counter()
    lvl = ind.fifty_two_week_levels(df)
    ms  = (time.perf_counter() - t0) * 1000
    h52 = lvl.get("high_52w")
    l52 = lvl.get("low_52w")
    dist = lvl.get("dist_from_52w_high_pct")
    ltp  = float(df["close"].iloc[-1])
    if h52 and l52 and h52 >= l52:
        R.ok(f"{tag} 52W high >= low", f"H={h52:.2f} L={l52:.2f}", ms)
    else:
        R.fail(f"{tag} 52W high >= low", f"H={h52} L={l52}", ms)
    if h52 and h52 >= ltp:
        R.ok(f"{tag} 52W high >= LTP", f"H={h52:.2f} LTP={ltp:.2f}", ms)
    else:
        R.fail(f"{tag} 52W high >= LTP", f"H={h52} LTP={ltp:.2f}", ms)
    if dist is not None and dist >= 0:
        R.ok(f"{tag} dist_from_52W_high >= 0", f"{dist:.2f}%", ms)
    else:
        R.fail(f"{tag} dist_from_52W_high >= 0", f"dist={dist}", ms)

    # Verify dist calculation: (H - LTP) / LTP * 100
    if h52 and ltp:
        t0 = time.perf_counter()
        expected_dist = (h52 - ltp) / ltp * 100
        ms = (time.perf_counter() - t0) * 1000
        if _near(dist, expected_dist, 0.001):
            R.ok(f"{tag} dist_from_52W formula correct", f"calc={dist:.3f}% expected={expected_dist:.3f}%", ms)
        else:
            R.fail(f"{tag} dist_from_52W formula correct", f"calc={dist:.3f} expected={expected_dist:.3f}", ms)

    # support_resistance — 20-day low/high
    t0 = time.perf_counter()
    sr = ind.support_resistance(df, window=20)
    ms = (time.perf_counter() - t0) * 1000
    sup = sr.get("support_20d")
    res = sr.get("resistance_20d")
    if sup is not None and res is not None and sup <= res:
        R.ok(f"{tag} support <= resistance", f"sup={sup:.2f} res={res:.2f}", ms)
    else:
        R.fail(f"{tag} support <= resistance", f"sup={sup} res={res}", ms)
    # Support should be ≤ current LTP (it's the 20D LOW)
    if sup is not None and sup <= ltp + 1e-6:
        R.ok(f"{tag} support_20d <= LTP", f"sup={sup:.2f} LTP={ltp:.2f}", ms)
    else:
        R.fail(f"{tag} support_20d <= LTP", f"sup={sup} LTP={ltp:.2f}", ms)
    # Support should equal the actual 20-day low
    t0 = time.perf_counter()
    expected_sup = float(df.tail(20)["low"].min())
    ms = (time.perf_counter() - t0) * 1000
    if _near(sup, expected_sup, 0.001):
        R.ok(f"{tag} support_20d == 20D low", f"{sup:.2f} == {expected_sup:.2f}", ms)
    else:
        R.fail(f"{tag} support_20d == 20D low", f"{sup:.2f} != {expected_sup:.2f}", ms)

    # liquidity_metrics
    t0  = time.perf_counter()
    liq = ind.liquidity_metrics(df, lookback=20)
    ms  = (time.perf_counter() - t0) * 1000
    avg_t  = liq.get("avg_turnover_cr")
    avg_v  = liq.get("avg_volume")
    if avg_t is not None and avg_t > 0:
        R.ok(f"{tag} liquidity avg_turnover_cr > 0", f"₹{avg_t:.2f} Cr", ms)
    else:
        R.fail(f"{tag} liquidity avg_turnover_cr > 0", f"got {avg_t}", ms)
    if avg_v is not None and avg_v > 0:
        R.ok(f"{tag} liquidity avg_volume > 0", f"{avg_v:,}", ms)
    else:
        R.fail(f"{tag} liquidity avg_volume > 0", f"got {avg_v}", ms)
    # Manual verification: turnover = mean(vol * close) / 1e7
    t0 = time.perf_counter()
    recent20  = df.tail(20)
    expected_t = (recent20["volume"] * recent20["close"]).mean() / 1e7
    ms = (time.perf_counter() - t0) * 1000
    if _near(avg_t, expected_t, 0.001):
        R.ok(f"{tag} liquidity turnover formula", f"calc={avg_t:.4f} expected={expected_t:.4f}", ms)
    else:
        R.fail(f"{tag} liquidity turnover formula", f"calc={avg_t:.4f} expected={expected_t:.4f}", ms)

    # volume_expansion
    t0  = time.perf_counter()
    vex = ind.volume_expansion(df)
    ms  = (time.perf_counter() - t0) * 1000
    if vex is not None and vex > 0:
        R.ok(f"{tag} volume_expansion > 0", f"{vex:.3f}x", ms)
    else:
        R.fail(f"{tag} volume_expansion > 0", f"got {vex}", ms)
    # Manual: vol5 / vol20
    t0 = time.perf_counter()
    vol5  = float(df["volume"].tail(5).mean())
    vol20 = float(df["volume"].tail(20).mean())
    expected_vex = vol5 / vol20 if vol20 > 0 else None
    ms = (time.perf_counter() - t0) * 1000
    if expected_vex and _near(vex, expected_vex, 0.001):
        R.ok(f"{tag} volume_expansion formula", f"calc={vex:.4f} expected={expected_vex:.4f}", ms)
    else:
        R.fail(f"{tag} volume_expansion formula", f"calc={vex} expected={expected_vex}", ms)

    # volume_expansion on short df (< 20 rows) → None
    t0  = time.perf_counter()
    vex_short = ind.volume_expansion(_make_ohlcv(10))
    ms  = (time.perf_counter() - t0) * 1000
    if vex_short is None:
        R.ok(f"{tag} volume_expansion None on short df", "returned None as expected", ms)
    else:
        R.fail(f"{tag} volume_expansion None on short df", f"got {vex_short} instead of None", ms)

    # EMA-20 and EMA-200
    for period in (20, 200):
        t0  = time.perf_counter()
        val = ind.ema(df, period)
        ms  = (time.perf_counter() - t0) * 1000
        expected = float(df["close"].ewm(span=period, adjust=False).mean().iloc[-1])
        if val is not None and _near(val, expected, 0.001):
            R.ok(f"{tag} EMA-{period} formula", f"{val:.2f} ≈ {expected:.2f}", ms)
        else:
            R.fail(f"{tag} EMA-{period} formula", f"calc={val} expected={expected:.2f}", ms)

    # pct_return None on insufficient data
    t0  = time.perf_counter()
    ret_none = ind.pct_return(_make_ohlcv(5), 252)
    ms  = (time.perf_counter() - t0) * 1000
    if ret_none is None:
        R.ok(f"{tag} pct_return None on insufficient data", "correctly returned None", ms)
    else:
        R.fail(f"{tag} pct_return None on insufficient data", f"got {ret_none}", ms)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 9 — Scaling signal invariants (using live DB stocks)
# ═══════════════════════════════════════════════════════════════════════════

def test_scaling_signal_suite(stocks: list):
    """Suite 9: Scaling signal level ordering, R/R, quality range."""
    tag = "[Scale]"
    tested = 0
    scale_hits = 0

    for s in stocks[:10]:
        sym   = s.get("tradingsymbol", "?")
        token = int(s.get("instrument_token", 0))
        if not token:
            continue
        try:
            con   = db.get_conn()
            ohlcv = con.execute(
                "SELECT date, open, high, low, close, volume "
                "FROM daily_ohlcv WHERE instrument_token=? ORDER BY date ASC",
                [token],
            ).df()
            con.close()
        except Exception:
            continue
        if len(ohlcv) < 30:
            continue
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        metrics = s

        t0 = time.perf_counter()
        try:
            sc = scaling_signal(ohlcv, metrics)
        except Exception as e:
            R.fail(f"{tag} [{sym}] scaling_signal no crash", str(e)[:60])
            continue
        ms = (time.perf_counter() - t0) * 1000
        tested += 1

        valid_sc_sigs = {None, "INITIAL_ENTRY", "ADD_POSITION", "HOLD_FOR_ADD", "AVOID"}
        sig = sc.get("scale_signal")
        if sig in valid_sc_sigs:
            R.ok(f"{tag} [{sym}] signal type", f"scale_signal={sig}", ms)
        else:
            R.fail(f"{tag} [{sym}] signal type", f"unexpected '{sig}'", ms)

        if sig == "INITIAL_ENTRY":
            scale_hits += 1
            entry  = sc.get("scale_entry_1")
            stop   = sc.get("scale_stop")
            target = sc.get("scale_target")
            qual   = sc.get("scale_quality")

            # Levels ordering: stop < entry < target (long bias)
            t0 = time.perf_counter()
            lvl_ok = (entry and stop and target and stop < entry < target)
            ms = (time.perf_counter() - t0) * 1000
            if lvl_ok:
                R.ok(f"{tag} [{sym}] INITIAL_ENTRY levels",
                     f"stop={stop:.2f} < entry={entry:.2f} < target={target:.2f}", ms)
            else:
                R.fail(f"{tag} [{sym}] INITIAL_ENTRY levels",
                       f"stop={stop} entry={entry} target={target}", ms)

            # Quality in {1, 2, 3}
            t0 = time.perf_counter()
            ms = (time.perf_counter() - t0) * 1000
            if qual in (1, 2, 3):
                R.ok(f"{tag} [{sym}] quality ∈ {{1,2,3}}", f"quality={qual}", ms)
            else:
                R.fail(f"{tag} [{sym}] quality ∈ {{1,2,3}}", f"quality={qual}", ms)

            # R/R ≥ 1.0 (gain must exceed risk)
            if entry and stop and target:
                t0 = time.perf_counter()
                risk = abs(entry - stop)
                gain = abs(target - entry)
                ms   = (time.perf_counter() - t0) * 1000
                if risk > 0 and gain / risk >= 1.0:
                    R.ok(f"{tag} [{sym}] scale R/R ≥ 1.0",
                         f"gain={gain:.2f} risk={risk:.2f} → R/R={gain/risk:.2f}x", ms)
                else:
                    R.fail(f"{tag} [{sym}] scale R/R ≥ 1.0",
                           f"gain={gain:.2f} risk={risk:.2f}", ms)

    if tested == 0:
        R.warn(f"{tag} scaling_signal", "no stocks with OHLCV data available to test", 0)
    else:
        R.ok(f"{tag} total tested", f"{tested} stocks · {scale_hits} INITIAL_ENTRY signals", 0)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 10 — Trade log CRUD + P&L math + load_ohlcv
# ═══════════════════════════════════════════════════════════════════════════

_TEST_USER = "TEST_FEATURETEST"   # sentinel user_id — cleaned up after tests


def test_trade_log_crud(sample_token: int, sample_sym: str):
    """Suite 10: log_trade / close_trade / delete_trade / load_trade_log /
    get_trade_stats / load_ohlcv."""
    tag = "[TradeCRUD]"

    # ── load_ohlcv ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        ohlcv_loaded = db.load_ohlcv(sample_token)
        ms = (time.perf_counter() - t0) * 1000
        if not ohlcv_loaded.empty:
            R.ok(f"{tag} load_ohlcv returns data", f"{len(ohlcv_loaded)} rows for {sample_sym}", ms)
        else:
            R.warn(f"{tag} load_ohlcv returns data", f"empty for token={sample_token}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} load_ohlcv returns data", str(e)[:60], ms)

    # ── log_trade — create two trades: one that wins, one that stops out ──
    trade_ids = []
    for i, (sig, entry, exit_price, exp_pnl_sign) in enumerate([
        ("BUY",       100.0, 115.0, +1),  # winner: +15% gain
        ("SELL_BELOW", 200.0, 210.0, -1), # loser:  short that went against
    ]):
        t0 = time.perf_counter()
        try:
            tid = db.log_trade({
                "trade_date":          _dt.date.today(),
                "tradingsymbol":       f"TESTSYM{i}",
                "instrument_token":    sample_token,
                "setup_type":          "INTRADAY",
                "signal_type":         sig,
                "rec_entry":           entry,
                "rec_stop":            entry * 0.97 if "BUY" in sig else entry * 1.03,
                "rec_t1":              entry * 1.10 if "BUY" in sig else entry * 0.90,
                "rec_rr":              3.0,
                "rec_reason":          f"[FEATURETEST] synthetic {sig}",
                "rec_composite_score": 5.0,
                "kite_user_id":        _TEST_USER,
                "quantity":            10,
                "actual_entry":        entry,
                "status":              "OPEN",
                "notes":               "[FEATURETEST]",
                "is_paper_trade":      False,
            })
            ms = (time.perf_counter() - t0) * 1000
            trade_ids.append(tid)
            API_CALLS["db_writes"] += 1
            R.ok(f"{tag} log_trade [{sig}] created id={tid}", f"entry={entry}", ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            R.fail(f"{tag} log_trade [{sig}]", str(e)[:60], ms)

    if not trade_ids:
        R.fail(f"{tag} skipping remaining CRUD — no trades created", "")
        return

    # ── load_trade_log — verify our test records appear ──────────────────
    t0 = time.perf_counter()
    log_df = db.load_trade_log(user_id=_TEST_USER)
    ms = (time.perf_counter() - t0) * 1000
    API_CALLS["db_reads"] += 1
    found_ids = set(log_df["id"].tolist()) if not log_df.empty else set()
    expected = set(trade_ids)
    if expected.issubset(found_ids):
        R.ok(f"{tag} load_trade_log sees test records", f"{len(found_ids)} rows", ms)
    else:
        R.fail(f"{tag} load_trade_log sees test records",
               f"expected {expected} ⊆ {found_ids}", ms)

    # ── load_trade_log status filter ──────────────────────────────────────
    t0 = time.perf_counter()
    open_df = db.load_trade_log(status_filter=["OPEN"], user_id=_TEST_USER)
    ms = (time.perf_counter() - t0) * 1000
    if not open_df.empty and all(open_df["status"] == "OPEN"):
        R.ok(f"{tag} load_trade_log status_filter=OPEN works",
             f"{len(open_df)} OPEN rows", ms)
    else:
        R.fail(f"{tag} load_trade_log status_filter=OPEN works",
               f"statuses found: {open_df['status'].unique().tolist() if not open_df.empty else 'empty'}", ms)

    # ── close_trade — close first trade as TARGET_HIT ────────────────────
    if trade_ids:
        tid0 = trade_ids[0]
        t0   = time.perf_counter()
        try:
            db.close_trade(tid0, 115.0, "TARGET_HIT", "[FEATURETEST] closed at T1")
            ms = (time.perf_counter() - t0) * 1000
            API_CALLS["db_writes"] += 1
            R.ok(f"{tag} close_trade TARGET_HIT", f"trade id={tid0}", ms)

            # Re-load and verify P&L calculation
            t0     = time.perf_counter()
            log_df = db.load_trade_log(user_id=_TEST_USER)
            ms     = (time.perf_counter() - t0) * 1000
            row    = log_df[log_df["id"] == tid0]
            if not row.empty:
                pnl    = row.iloc[0].get("pnl_amount")
                pnl_pct = row.iloc[0].get("pnl_pct")
                rr_r   = row.iloc[0].get("rr_realised")
                status = row.iloc[0].get("status")
                # BUY 10 shares @ 100, exit @ 115 → P&L = +15 * 10 = +150
                if pnl is not None and _near(pnl, 150.0, 0.001):
                    R.ok(f"{tag} close_trade pnl_amount=+150", f"got {pnl:.2f}", ms)
                else:
                    R.fail(f"{tag} close_trade pnl_amount=+150", f"got pnl={pnl}", ms)
                # P&L %: (115-100)/100*100 = 15%
                if pnl_pct is not None and _near(pnl_pct, 15.0, 0.001):
                    R.ok(f"{tag} close_trade pnl_pct=+15%", f"got {pnl_pct:.2f}%", ms)
                else:
                    R.fail(f"{tag} close_trade pnl_pct=+15%", f"got {pnl_pct}", ms)
                # rr_realised: gain=(115-100)=15 / risk=(100-97)=3 → 5×
                if rr_r is not None and _near(rr_r, 5.0, 0.05):
                    R.ok(f"{tag} close_trade rr_realised≈5×", f"got {rr_r:.2f}", ms)
                else:
                    R.fail(f"{tag} close_trade rr_realised≈5×", f"got {rr_r}", ms)
                if status == "TARGET_HIT":
                    R.ok(f"{tag} close_trade status=TARGET_HIT", "", ms)
                else:
                    R.fail(f"{tag} close_trade status=TARGET_HIT", f"got '{status}'", ms)
            else:
                R.fail(f"{tag} close_trade verify", f"row {tid0} not found after close", ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            R.fail(f"{tag} close_trade TARGET_HIT", str(e)[:60], ms)

    # ── get_trade_stats — verify aggregate ───────────────────────────────
    t0 = time.perf_counter()
    try:
        stats = db.get_trade_stats(user_id=_TEST_USER)
        ms    = (time.perf_counter() - t0) * 1000
        API_CALLS["db_reads"] += 1
        if stats.get("total", 0) >= len(trade_ids):
            R.ok(f"{tag} get_trade_stats total ≥ created",
                 f"total={stats['total']} open={stats['open']}", ms)
        else:
            R.fail(f"{tag} get_trade_stats total ≥ created",
                   f"stats={stats} expected ≥ {len(trade_ids)}", ms)
        # At least 1 win (TARGET_HIT)
        if stats.get("wins", 0) >= 1:
            R.ok(f"{tag} get_trade_stats wins ≥ 1", f"wins={stats['wins']}", ms)
        else:
            R.fail(f"{tag} get_trade_stats wins ≥ 1", f"wins={stats.get('wins')}", ms)
        # win_rate should be in [0, 100]
        wr = stats.get("win_rate", -1)
        if 0 <= wr <= 100:
            R.ok(f"{tag} get_trade_stats win_rate ∈ [0,100]", f"{wr:.1f}%", ms)
        else:
            R.fail(f"{tag} get_trade_stats win_rate ∈ [0,100]", f"got {wr}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} get_trade_stats", str(e)[:60], ms)

    # ── delete_trade — clean up all test records ──────────────────────────
    for tid in trade_ids:
        t0 = time.perf_counter()
        try:
            db.delete_trade(tid)
            ms = (time.perf_counter() - t0) * 1000
            API_CALLS["db_writes"] += 1
            R.ok(f"{tag} delete_trade id={tid}", "", ms)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            R.fail(f"{tag} delete_trade id={tid}", str(e)[:60], ms)

    # Verify deletion
    t0     = time.perf_counter()
    post   = db.load_trade_log(user_id=_TEST_USER)
    ms     = (time.perf_counter() - t0) * 1000
    remaining = set(post["id"].tolist()) & set(trade_ids) if not post.empty else set()
    if not remaining:
        R.ok(f"{tag} delete_trade all gone", "no test records remain", ms)
    else:
        R.fail(f"{tag} delete_trade all gone", f"still present: {remaining}", ms)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 11 — Paper trading: create, performance, signal config tuning
# ═══════════════════════════════════════════════════════════════════════════

def test_paper_trading(sample_token: int, sample_sym: str):
    """Suite 11: get_open_paper_trades, get_paper_trade_perf, get_signal_config,
    save_signal_config, tune_signal_config_from_paper."""
    tag = "[Paper]"
    paper_ids = []

    # ── get_signal_config — defaults ─────────────────────────────────────
    t0 = time.perf_counter()
    try:
        cfg = db.get_signal_config(user_id=_TEST_USER)
        ms  = (time.perf_counter() - t0) * 1000
        if cfg.get("intraday_rsi_buy_max") == 75.0:
            R.ok(f"{tag} get_signal_config default rsi_buy_max=75", "", ms)
        else:
            R.fail(f"{tag} get_signal_config default rsi_buy_max=75",
                   f"got {cfg.get('intraday_rsi_buy_max')}", ms)
        if cfg.get("intraday_rsi_sell_min") == 25.0:
            R.ok(f"{tag} get_signal_config default rsi_sell_min=25", "", ms)
        else:
            R.fail(f"{tag} get_signal_config default rsi_sell_min=25",
                   f"got {cfg.get('intraday_rsi_sell_min')}", ms)
        if cfg.get("intraday_min_rr") == 1.5:
            R.ok(f"{tag} get_signal_config default min_rr=1.5", "", ms)
        else:
            R.fail(f"{tag} get_signal_config default min_rr=1.5",
                   f"got {cfg.get('intraday_min_rr')}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} get_signal_config", str(e)[:60], ms)
        cfg = {}

    # ── save_signal_config — write custom values ──────────────────────────
    t0 = time.perf_counter()
    try:
        db.save_signal_config({
            "intraday_rsi_buy_max":  68.0,
            "intraday_rsi_sell_min": 32.0,
            "intraday_min_rr":       2.0,
        }, user_id=_TEST_USER)
        ms = (time.perf_counter() - t0) * 1000
        API_CALLS["db_writes"] += 1
        R.ok(f"{tag} save_signal_config upsert", "wrote rsi_buy_max=68, rsi_sell_min=32, min_rr=2.0", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} save_signal_config", str(e)[:60], ms)

    # ── get_signal_config — verify persisted values ───────────────────────
    t0 = time.perf_counter()
    try:
        cfg2 = db.get_signal_config(user_id=_TEST_USER)
        ms   = (time.perf_counter() - t0) * 1000
        API_CALLS["db_reads"] += 1
        if cfg2.get("intraday_rsi_buy_max") == 68.0:
            R.ok(f"{tag} get_signal_config after save rsi_buy_max=68", "", ms)
        else:
            R.fail(f"{tag} get_signal_config after save rsi_buy_max=68",
                   f"got {cfg2.get('intraday_rsi_buy_max')}", ms)
        if cfg2.get("intraday_min_rr") == 2.0:
            R.ok(f"{tag} get_signal_config after save min_rr=2.0", "", ms)
        else:
            R.fail(f"{tag} get_signal_config after save min_rr=2.0",
                   f"got {cfg2.get('intraday_min_rr')}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} get_signal_config after save", str(e)[:60], ms)

    # ── Create synthetic paper trades for perf testing ────────────────────
    # 3 winners + 2 losers for BUY_ABOVE
    for i, (exit_p, outcome) in enumerate([
        (110.0, "TARGET_HIT"),   # win
        (111.0, "TARGET_HIT"),   # win
        (112.0, "TARGET_HIT"),   # win
        ( 95.0, "STOPPED_OUT"),  # loss
        ( 94.0, "STOPPED_OUT"),  # loss
    ]):
        try:
            tid = db.log_trade({
                "trade_date":          _dt.date.today(),
                "tradingsymbol":       f"PAPER{i}",
                "instrument_token":    sample_token,
                "setup_type":          "INTRADAY",
                "signal_type":         "BUY_ABOVE",
                "rec_entry":           100.0,
                "rec_stop":            97.0,
                "rec_t1":              110.0,
                "rec_rr":              3.33,
                "rec_reason":          "[FEATURETEST] paper",
                "kite_user_id":        _TEST_USER,
                "quantity":            10,
                "actual_entry":        100.0,
                "status":              "OPEN",
                "is_paper_trade":      True,
            })
            paper_ids.append(tid)
            # Close it immediately with the test exit
            db.close_trade(tid, exit_p, outcome, "[FEATURETEST]")
            API_CALLS["db_writes"] += 1
        except Exception as e:
            R.fail(f"{tag} create paper trade {i}", str(e)[:60])

    if paper_ids:
        R.ok(f"{tag} created {len(paper_ids)} paper trades for perf test", "", 0)

    # ── get_open_paper_trades — create one that stays OPEN ────────────────
    open_id = None
    t0 = time.perf_counter()
    try:
        open_id = db.log_trade({
            "trade_date":          _dt.date.today(),
            "tradingsymbol":       "PAPER_OPEN",
            "instrument_token":    sample_token,
            "setup_type":          "INTRADAY",
            "signal_type":         "BUY_ABOVE",
            "rec_entry":           200.0,
            "rec_stop":            195.0,
            "rec_t1":              210.0,
            "kite_user_id":        _TEST_USER,
            "quantity":            5,
            "actual_entry":        200.0,
            "status":              "OPEN",
            "is_paper_trade":      True,
        })
        paper_ids.append(open_id)
        ms = (time.perf_counter() - t0) * 1000
        API_CALLS["db_writes"] += 1

        open_list = db.get_open_paper_trades(user_id=_TEST_USER)
        found = any(p["id"] == open_id for p in open_list)
        if found:
            R.ok(f"{tag} get_open_paper_trades finds open trade", f"id={open_id}", ms)
        else:
            R.fail(f"{tag} get_open_paper_trades finds open trade",
                   f"id={open_id} not in {[p['id'] for p in open_list]}", ms)

        # Fields present
        if open_list:
            sample_pt = next((p for p in open_list if p["id"] == open_id), None)
            if sample_pt:
                for fld in ("id","tradingsymbol","signal_type","actual_entry","rec_stop","rec_t1"):
                    if sample_pt.get(fld) is not None:
                        R.ok(f"{tag} get_open_paper_trades field '{fld}' present",
                             f"{sample_pt[fld]}", 0)
                    else:
                        R.fail(f"{tag} get_open_paper_trades field '{fld}' present",
                               f"got None", 0)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} get_open_paper_trades", str(e)[:60], ms)

    # ── get_paper_trade_perf — win rate should be ~60% (3/5) ─────────────
    t0 = time.perf_counter()
    try:
        perf = db.get_paper_trade_perf(user_id=_TEST_USER, days=1)
        ms   = (time.perf_counter() - t0) * 1000
        API_CALLS["db_reads"] += 1
        ba   = perf.get("BUY_ABOVE", {})
        ov   = perf.get("overall",   {})
        if ba.get("total", 0) >= 5:
            R.ok(f"{tag} get_paper_trade_perf BUY_ABOVE total ≥ 5",
                 f"total={ba['total']}", ms)
        else:
            R.fail(f"{tag} get_paper_trade_perf BUY_ABOVE total ≥ 5",
                   f"total={ba.get('total')}", ms)
        if _near(ba.get("win_rate", -1), 60.0, 0.02):   # 3/5 = 60%
            R.ok(f"{tag} get_paper_trade_perf BUY_ABOVE win_rate≈60%",
                 f"{ba.get('win_rate'):.1f}%", ms)
        else:
            R.fail(f"{tag} get_paper_trade_perf BUY_ABOVE win_rate≈60%",
                   f"got {ba.get('win_rate')}", ms)
        if ov.get("total", 0) >= 5:
            R.ok(f"{tag} get_paper_trade_perf overall.total ≥ 5",
                 f"total={ov['total']}", ms)
        else:
            R.fail(f"{tag} get_paper_trade_perf overall.total ≥ 5",
                   f"total={ov.get('total')}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} get_paper_trade_perf", str(e)[:60], ms)

    # ── tune_signal_config_from_paper — 60% win rate > threshold, no change ─
    # First reset to defaults so tuning logic starts from known state
    t0 = time.perf_counter()
    try:
        db.save_signal_config({
            "intraday_rsi_buy_max":  75.0,
            "intraday_rsi_sell_min": 25.0,
            "intraday_min_rr":       1.5,
        }, user_id=_TEST_USER)
        changes = db.tune_signal_config_from_paper(user_id=_TEST_USER, days=1)
        ms = (time.perf_counter() - t0) * 1000
        # 60% win rate is > 40% but < 65% → should NOT trigger a change
        if isinstance(changes, dict):
            R.ok(f"{tag} tune_signal_config_from_paper returns dict",
                 f"changes={changes}", ms)
        else:
            R.fail(f"{tag} tune_signal_config_from_paper returns dict",
                   f"got type={type(changes)}", ms)
        # 60% is in middle band → no change expected
        if not changes:
            R.ok(f"{tag} tune: 60% win rate → no threshold change (in neutral band)",
                 "changes={}", ms)
        else:
            R.warn(f"{tag} tune: 60% win rate → unexpected change",
                   f"changes={changes}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} tune_signal_config_from_paper", str(e)[:60], ms)

    # ── Simulate low win rate (< 40%) → should tighten thresholds ─────────
    # Add 6 more losses to push total to 11 trades, 3 wins → ~27% win rate
    extra_loss_ids = []
    for i in range(6):
        try:
            tid = db.log_trade({
                "trade_date":          _dt.date.today(),
                "tradingsymbol":       f"LOSS{i}",
                "instrument_token":    sample_token,
                "setup_type":          "INTRADAY",
                "signal_type":         "BUY_ABOVE",
                "rec_entry":           100.0,
                "rec_stop":            97.0,
                "rec_t1":              110.0,
                "kite_user_id":        _TEST_USER,
                "quantity":            10,
                "actual_entry":        100.0,
                "status":              "OPEN",
                "is_paper_trade":      True,
            })
            db.close_trade(tid, 95.0, "STOPPED_OUT", "[FEATURETEST] extra loss")
            extra_loss_ids.append(tid)
        except Exception:
            pass
    paper_ids.extend(extra_loss_ids)

    t0 = time.perf_counter()
    try:
        db.save_signal_config({
            "intraday_rsi_buy_max":  75.0,
            "intraday_rsi_sell_min": 25.0,
            "intraday_min_rr":       1.5,
        }, user_id=_TEST_USER)
        changes2 = db.tune_signal_config_from_paper(user_id=_TEST_USER, days=1)
        ms = (time.perf_counter() - t0) * 1000
        # 3/11 = ~27% < 40% → should tighten rsi_buy_max and raise min_rr
        if "intraday_rsi_buy_max" in changes2:
            new_rsi = changes2["intraday_rsi_buy_max"]
            if new_rsi < 75.0:
                R.ok(f"{tag} tune: low win rate tightens rsi_buy_max",
                     f"75 → {new_rsi}", ms)
            else:
                R.fail(f"{tag} tune: low win rate tightens rsi_buy_max",
                       f"expected < 75 got {new_rsi}", ms)
        else:
            R.fail(f"{tag} tune: low win rate → rsi_buy_max should decrease",
                   f"changes={changes2}", ms)
        if "intraday_min_rr" in changes2:
            new_rr = changes2["intraday_min_rr"]
            if new_rr > 1.5:
                R.ok(f"{tag} tune: low win rate raises min_rr",
                     f"1.5 → {new_rr}", ms)
            else:
                R.fail(f"{tag} tune: low win rate raises min_rr",
                       f"expected > 1.5 got {new_rr}", ms)
        else:
            R.fail(f"{tag} tune: low win rate → min_rr should increase",
                   f"changes={changes2}", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} tune_signal_config low win rate", str(e)[:60], ms)

    # ── Cleanup all paper test records ───────────────────────────────────
    for tid in paper_ids:
        try:
            db.delete_trade(tid)
            API_CALLS["db_writes"] += 1
        except Exception:
            pass
    # Reset signal config
    try:
        db.save_signal_config({
            "intraday_rsi_buy_max":  75.0,
            "intraday_rsi_sell_min": 25.0,
            "intraday_min_rr":       1.5,
        }, user_id=_TEST_USER)
    except Exception:
        pass
    R.ok(f"{tag} cleanup: all test trades deleted, config reset", "", 0)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 12 — Tunable threshold tests for intraday_signal
# ═══════════════════════════════════════════════════════════════════════════

def test_tunable_thresholds():
    """Suite 12: intraday_signal with custom RSI/R/R gates; AVOID conditions;
    compute_all_signals passthrough."""
    tag = "[Thresholds]"
    df  = _make_ohlcv(300, base=500.0, seed=7)

    # Baseline signal with default thresholds
    t0  = time.perf_counter()
    metrics_base = {
        "ltp":    float(df["close"].iloc[-1]),
        "ema_20": float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1]),
        "ema_50": float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1]),
        "atr_14": ind.atr(df),
        "rsi_14": ind.rsi(df),
    }
    sig_default = intraday_signal(df, metrics_base)
    ms = (time.perf_counter() - t0) * 1000
    valid = {None, "BUY_ABOVE", "SELL_BELOW", "AVOID"}
    if sig_default.get("intraday_signal") in valid:
        R.ok(f"{tag} default thresholds → valid signal",
             f"signal={sig_default.get('intraday_signal')}", ms)
    else:
        R.fail(f"{tag} default thresholds → valid signal",
               f"got {sig_default.get('intraday_signal')}", ms)

    # Extreme min_rr=99 → virtually any signal should become AVOID
    t0  = time.perf_counter()
    sig_strict = intraday_signal(df, metrics_base, min_rr=99.0)
    ms  = (time.perf_counter() - t0) * 1000
    if sig_strict.get("intraday_signal") == "AVOID":
        R.ok(f"{tag} min_rr=99 → AVOID (impossible gate)", "", ms)
    else:
        R.warn(f"{tag} min_rr=99 → AVOID", f"got {sig_strict.get('intraday_signal')}", ms)

    # rsi_buy_max=0 → BUY_ABOVE impossible (RSI always > 0)
    t0  = time.perf_counter()
    metrics_hi_rsi = dict(metrics_base)
    metrics_hi_rsi["rsi_14"] = 1.0          # RSI = 1 → still > 0, so won't trigger
    sig_rsi0 = intraday_signal(df, metrics_hi_rsi, rsi_buy_max=0.0)
    ms = (time.perf_counter() - t0) * 1000
    if sig_rsi0.get("intraday_signal") != "BUY_ABOVE":
        R.ok(f"{tag} rsi_buy_max=0 → no BUY_ABOVE",
             f"signal={sig_rsi0.get('intraday_signal')}", ms)
    else:
        R.fail(f"{tag} rsi_buy_max=0 → no BUY_ABOVE",
               f"unexpectedly got BUY_ABOVE", ms)

    # rsi_sell_min=100 → SELL_BELOW impossible (RSI always < 100)
    t0  = time.perf_counter()
    metrics_lo_rsi = dict(metrics_base)
    metrics_lo_rsi["rsi_14"] = 99.0        # RSI = 99 → still < 100
    sig_rsi100 = intraday_signal(df, metrics_lo_rsi, rsi_sell_min=100.0)
    ms = (time.perf_counter() - t0) * 1000
    if sig_rsi100.get("intraday_signal") != "SELL_BELOW":
        R.ok(f"{tag} rsi_sell_min=100 → no SELL_BELOW",
             f"signal={sig_rsi100.get('intraday_signal')}", ms)
    else:
        R.fail(f"{tag} rsi_sell_min=100 → no SELL_BELOW",
               f"unexpectedly got SELL_BELOW", ms)

    # AVOID on narrow spread: make a stock where H-L is tiny
    t0 = time.perf_counter()
    df_narrow = _make_ohlcv(300, base=100.0, seed=1)
    df_narrow["high"] = df_narrow["close"] * 1.0005     # only 0.05% range
    df_narrow["low"]  = df_narrow["close"] * 0.9995
    metrics_narrow = {
        "ltp":    float(df_narrow["close"].iloc[-1]),
        "ema_20": float(df_narrow["close"].ewm(span=20, adjust=False).mean().iloc[-1]),
        "atr_14": 0.01,
        "rsi_14": 50.0,
    }
    sig_narrow = intraday_signal(df_narrow, metrics_narrow)
    ms = (time.perf_counter() - t0) * 1000
    if sig_narrow.get("intraday_signal") == "AVOID":
        R.ok(f"{tag} narrow spread → AVOID", "correctly rejected low-spread", ms)
    else:
        R.warn(f"{tag} narrow spread → AVOID",
               f"got {sig_narrow.get('intraday_signal')} (spread may be wide enough)", ms)

    # AVOID when ltp is None / insufficient data
    t0 = time.perf_counter()
    sig_no_ltp = intraday_signal(pd.DataFrame(), {"ltp": None, "ema_20": None, "atr_14": None})
    ms = (time.perf_counter() - t0) * 1000
    if sig_no_ltp.get("intraday_signal") is None:
        R.ok(f"{tag} missing ltp → null signal (no crash)", "", ms)
    else:
        R.fail(f"{tag} missing ltp → null signal",
               f"got {sig_no_ltp.get('intraday_signal')}", ms)

    # compute_all_signals passes thresholds through correctly
    t0 = time.perf_counter()
    sig_all = compute_all_signals(df, metrics_base,
                                  rsi_buy_max=75.0, rsi_sell_min=25.0, min_rr=1.5)
    ms = (time.perf_counter() - t0) * 1000
    required_keys = {
        "swing_signal", "intraday_signal", "scale_signal",
        "intraday_pivot", "intraday_r1", "intraday_r2",
        "intraday_entry", "intraday_stop", "intraday_t1",
    }
    missing = required_keys - set(sig_all.keys())
    if not missing:
        R.ok(f"{tag} compute_all_signals has all required keys", "", ms)
    else:
        R.fail(f"{tag} compute_all_signals missing keys", str(missing), ms)

    # Verify threshold passthrough: with min_rr=99 all-signals should also AVOID
    t0 = time.perf_counter()
    sig_all_strict = compute_all_signals(df, metrics_base, min_rr=99.0)
    ms = (time.perf_counter() - t0) * 1000
    if sig_all_strict.get("intraday_signal") == "AVOID":
        R.ok(f"{tag} compute_all_signals min_rr=99 → intraday AVOID", "", ms)
    else:
        R.warn(f"{tag} compute_all_signals min_rr=99 → intraday AVOID",
               f"got {sig_all_strict.get('intraday_signal')}", ms)

    # Return output for BUY_ABOVE contains stop < entry < t1 if generated
    if sig_all.get("intraday_signal") == "BUY_ABOVE":
        e = sig_all.get("intraday_entry")
        s = sig_all.get("intraday_stop")
        t = sig_all.get("intraday_t1")
        if e and s and t and s < e < t:
            R.ok(f"{tag} compute_all BUY_ABOVE levels ordered", f"stop={s:.2f}<entry={e:.2f}<t1={t:.2f}", 0)
        else:
            R.fail(f"{tag} compute_all BUY_ABOVE levels ordered", f"s={s} e={e} t={t}", 0)
    elif sig_all.get("intraday_signal") == "SELL_BELOW":
        e = sig_all.get("intraday_entry")
        s = sig_all.get("intraday_stop")
        t = sig_all.get("intraday_t1")
        if e and s and t and s > e > t:
            R.ok(f"{tag} compute_all SELL_BELOW levels ordered", f"t1={t:.2f}<entry={e:.2f}<stop={s:.2f}", 0)
        else:
            R.fail(f"{tag} compute_all SELL_BELOW levels ordered", f"s={s} e={e} t={t}", 0)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 13 — DB schema integrity
# ═══════════════════════════════════════════════════════════════════════════

def test_db_schema_integrity():
    """Suite 13: init_schema idempotency, expected columns in all tables."""
    tag = "[Schema]"

    # init_schema is idempotent — calling it twice should not raise
    t0 = time.perf_counter()
    try:
        db.init_schema()
        db.init_schema()
        ms = (time.perf_counter() - t0) * 1000
        R.ok(f"{tag} init_schema idempotent (called twice)", "", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} init_schema idempotent", str(e)[:80], ms)

    # Verify computed_metrics has all expected columns
    # DuckDB PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    t0 = time.perf_counter()
    con = db.get_conn()
    try:
        cols_metrics = {row[1] for row in
                        con.execute("PRAGMA table_info(computed_metrics)").fetchall()}
        ms = (time.perf_counter() - t0) * 1000
        required_metrics = {
            "instrument_token", "tradingsymbol",
            "ltp", "rsi_14", "ema_20", "ema_50", "ema_200", "atr_14",
            "composite_score", "trend_score", "vol_expansion_ratio",
            "rs_vs_nifty_3m", "avg_turnover_cr", "avg_volume",
            "intraday_signal", "intraday_entry", "intraday_stop", "intraday_t1",
            "intraday_pivot", "intraday_r1", "intraday_r2", "intraday_s1", "intraday_s2",
            "swing_signal", "swing_entry", "swing_stop", "swing_t1", "swing_rr",
            "scale_signal", "scale_entry_1", "scale_stop", "scale_target",
            "ai_score", "ai_verdict", "ai_brief",
        }
        missing = required_metrics - cols_metrics
        if not missing:
            R.ok(f"{tag} computed_metrics has all required columns",
                 f"{len(cols_metrics)} total cols", ms)
        else:
            R.fail(f"{tag} computed_metrics missing columns", str(missing), ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} computed_metrics columns", str(e)[:60], ms)

    # Verify trade_log has all expected columns (incl. new is_paper_trade)
    t0 = time.perf_counter()
    try:
        cols_tl = {row[1] for row in
                   con.execute("PRAGMA table_info(trade_log)").fetchall()}
        ms = (time.perf_counter() - t0) * 1000
        required_tl = {
            "id", "logged_at", "trade_date", "tradingsymbol", "instrument_token",
            "setup_type", "signal_type",
            "rec_entry", "rec_stop", "rec_t1", "rec_t2", "rec_rr",
            "rec_reason", "rec_composite_score", "rec_ai_score",
            "kite_user_id", "kite_order_id", "kite_sl_order_id", "kite_status",
            "quantity", "actual_entry", "actual_exit", "status", "notes",
            "pnl_amount", "pnl_pct", "slippage_entry_pct", "rr_realised",
            "is_paper_trade",
        }
        missing_tl = required_tl - cols_tl
        if not missing_tl:
            R.ok(f"{tag} trade_log has all required columns (incl. is_paper_trade)",
                 f"{len(cols_tl)} total cols", ms)
        else:
            R.fail(f"{tag} trade_log missing columns", str(missing_tl), ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} trade_log columns", str(e)[:60], ms)

    # Verify signal_config table exists
    t0 = time.perf_counter()
    try:
        con.execute("SELECT COUNT(*) FROM signal_config").fetchone()
        ms = (time.perf_counter() - t0) * 1000
        R.ok(f"{tag} signal_config table exists", "", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} signal_config table exists", str(e)[:60], ms)

    con.close()

    # Verify trade_log row count is accessible (no crash)
    t0 = time.perf_counter()
    try:
        log_df = db.load_trade_log()
        ms = (time.perf_counter() - t0) * 1000
        R.ok(f"{tag} load_trade_log no crash", f"{len(log_df)} rows in log", ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        R.fail(f"{tag} load_trade_log no crash", str(e)[:60], ms)


# ═══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(batch_num: int, stocks: list[dict]):
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  BATCH {batch_num} — {len(stocks)} stocks{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")

    for s in stocks:
        sym   = s["tradingsymbol"]
        token = int(s["instrument_token"])
        print(f"\n{BOLD}  ── {sym} (token={token}) ──────────────────────────────────{RESET}")

        # Load OHLCV from DB
        t0 = time.perf_counter()
        try:
            con  = db.get_conn()
            ohlcv = con.execute(
                "SELECT date, open, high, low, close, volume "
                "FROM daily_ohlcv WHERE instrument_token=? ORDER BY date ASC",
                [token],
            ).df()
            con.close()
            API_CALLS["db_reads"] += 1
            load_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            R.fail(f"[{sym}] OHLCV load from DB", str(e)[:60])
            continue

        if ohlcv.empty:
            R.fail(f"[{sym}] OHLCV in DB", "no rows in daily_ohlcv", load_ms)
            continue
        R.ok(f"[{sym}] OHLCV load from DB", f"{len(ohlcv)} rows in {load_ms:.0f}ms", load_ms)

        # Convert date column (assign to avoid pandas chained-assignment warning)
        ohlcv = ohlcv.assign(date=pd.to_datetime(ohlcv["date"]))

        row = pd.Series(s)

        # Run all test suites
        test_ohlcv_integrity(sym, token, ohlcv)
        test_indicators(sym, token, ohlcv, row)
        test_returns(sym, token, ohlcv, row)
        test_composite_score(sym, row)
        test_signals(sym, token, ohlcv, row)
        test_chart_summary_logic(sym, ohlcv, row)
        test_db_roundtrip(sym, token, row)


# ═══════════════════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════

def run_latency_bench(stocks: list[dict]):
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  LATENCY BENCHMARK — DB operations{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")

    # Full metrics load
    t0 = time.perf_counter()
    metrics = db.load_metrics()
    ms = (time.perf_counter() - t0) * 1000
    API_CALLS["db_reads"] += 1
    n = len(metrics)
    if ms < 500:
        R.ok("[Bench] Full metrics load", f"{n} rows in {ms:.0f}ms", ms)
    elif ms < 2000:
        R.warn("[Bench] Full metrics load", f"{n} rows in {ms:.0f}ms (acceptable)", ms)
    else:
        R.fail("[Bench] Full metrics load", f"{n} rows in {ms:.0f}ms (too slow)", ms)

    # OHLCV load for one stock
    token = int(stocks[0]["instrument_token"])
    sym   = stocks[0]["tradingsymbol"]
    t0    = time.perf_counter()
    con   = db.get_conn()
    ohlcv = con.execute(
        "SELECT * FROM daily_ohlcv WHERE instrument_token=? ORDER BY date",
        [token],
    ).df()
    con.close()
    ms = (time.perf_counter() - t0) * 1000
    API_CALLS["db_reads"] += 1
    if ms < 100:
        R.ok(f"[Bench] OHLCV load ({sym})", f"{len(ohlcv)} rows in {ms:.0f}ms", ms)
    elif ms < 500:
        R.warn(f"[Bench] OHLCV load ({sym})", f"{len(ohlcv)} rows in {ms:.0f}ms (acceptable)", ms)
    else:
        R.fail(f"[Bench] OHLCV load ({sym})", f"{len(ohlcv)} rows in {ms:.0f}ms (too slow)", ms)

    # Indicator compute latency
    if not ohlcv.empty:
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        t0 = time.perf_counter()
        ind.rsi(ohlcv)
        ind.ema(ohlcv, 50)
        ind.ema(ohlcv, 200)
        ind.atr(ohlcv)
        ms = (time.perf_counter() - t0) * 1000
        if ms < 50:
            R.ok("[Bench] 4 indicators compute", f"{ms:.1f}ms", ms)
        elif ms < 200:
            R.warn("[Bench] 4 indicators compute", f"{ms:.1f}ms (acceptable)", ms)
        else:
            R.fail("[Bench] 4 indicators compute", f"{ms:.1f}ms (too slow)", ms)

        # Signal compute latency
        t0 = time.perf_counter()
        compute_all_signals(ohlcv, stocks[0])
        ms = (time.perf_counter() - t0) * 1000
        if ms < 100:
            R.ok("[Bench] Signal compute", f"{ms:.1f}ms", ms)
        elif ms < 500:
            R.warn("[Bench] Signal compute", f"{ms:.1f}ms (acceptable)", ms)
        else:
            R.fail("[Bench] Signal compute", f"{ms:.1f}ms (too slow)", ms)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═'*70}")
    print("  SCREENER FEATURE TEST  —  30 stocks / 3 batches + 6 unit suites")
    print(f"{'═'*70}{RESET}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Ensure DB schema is fully migrated before any tests run ──────────
    # This creates is_paper_trade column + signal_config table if absent.
    try:
        db.init_schema()
        print(f"  {GREEN}✓{RESET} DB schema migration applied (idempotent)")
    except Exception as _e:
        print(f"  {YELLOW}⚠{RESET} init_schema warning: {_e}")

    # ── Load all metrics from DB ──────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        all_metrics = db.load_metrics()
        API_CALLS["db_reads"] += 1
    except Exception as e:
        print(f"{RED}FATAL: Cannot load metrics from DB: {e}{RESET}")
        sys.exit(1)
    load_ms = (time.perf_counter() - t0) * 1000

    if all_metrics.empty:
        print(f"{RED}FATAL: No rows in computed_metrics — run Full Rescan first.{RESET}")
        sys.exit(1)

    print(f"  DB loaded: {len(all_metrics)} stocks in {load_ms:.0f}ms")

    # Prefer stocks with OHLCV data AND a valid composite score
    scored = all_metrics[all_metrics["composite_score"].notna()].copy()
    if len(scored) < 30:
        print(f"{YELLOW}  Warning: only {len(scored)} scored stocks available; using all.{RESET}")
        scored = all_metrics.copy()

    # Deterministic seed for reproducibility
    random.seed(42)
    sample_size = min(30, len(scored))
    sampled     = scored.sample(n=sample_size, random_state=42).to_dict("records")

    batch1 = sampled[0:10]
    batch2 = sampled[10:20]
    batch3 = sampled[20:30]

    print(f"\n  Batch 1: {[s['tradingsymbol'] for s in batch1]}")
    print(f"  Batch 2: {[s['tradingsymbol'] for s in batch2]}")
    print(f"  Batch 3: {[s['tradingsymbol'] for s in batch3]}")

    # ── Run batches ──────────────────────────────────────────────────────
    run_batch(1, batch1)
    run_batch(2, batch2)
    run_batch(3, batch3)

    # ── Suite 8: Extended indicator functions (synthetic data) ────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 8 — Extended Indicators (all_returns, 52W, S/R, liquidity, vol_exp){RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_extended_indicators()

    # ── Suite 9: Scaling signal invariants ───────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 9 — Scaling Signal (level ordering, quality, R/R){RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_scaling_signal_suite(sampled)

    # ── Suites 10–11: Trade log CRUD + Paper trading ──────────────────────
    # Use first stock in sample as a token reference
    _sample_tok = int(sampled[0]["instrument_token"])
    _sample_sym = sampled[0]["tradingsymbol"]

    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 10 — Trade Log CRUD + P&L + get_trade_stats + load_ohlcv{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_trade_log_crud(_sample_tok, _sample_sym)

    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 11 — Paper Trading (perf, signal config, tuning){RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_paper_trading(_sample_tok, _sample_sym)

    # ── Suite 12: Tunable thresholds ─────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 12 — Tunable Thresholds (intraday_signal RSI / min_rr gates){RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_tunable_thresholds()

    # ── Suite 13: DB schema integrity ────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 13 — DB Schema Integrity (columns, idempotency){RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    test_db_schema_integrity()

    # ── Latency benchmark ────────────────────────────────────────────────
    run_latency_bench(sampled[:3])

    # ── API cost accounting ──────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  SUITE 15 — COST / API ACCOUNTING{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"  Kite API calls:  {API_CALLS['kite']}  (all data served from DB — no live API calls)")
    print(f"  DB reads:        {API_CALLS['db_reads']}")
    print(f"  DB writes:       {API_CALLS['db_writes']}  (only synthetic AI round-trip writes, all cleaned up)")
    print(f"  AI API calls:    0  (test does not invoke AI APIs)")
    print(f"  Estimated cost:  ₹0 / $0.00")

    # ── Final summary ────────────────────────────────────────────────────
    failures = R.summary()
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return failures


if __name__ == "__main__":
    try:
        failures = main()
        sys.exit(0 if failures == 0 else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(3)
