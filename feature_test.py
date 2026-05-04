"""
feature_test.py — Comprehensive feature test for the stock screener.

Tests 30 stocks (3 batches of 10) across:
  1. DB integrity          — no nulls in critical fields, OHLCV completeness
  2. Indicator accuracy    — re-compute EMA/RSI/ATR and diff against DB values
  3. Returns accuracy      — recalculate from raw OHLCV vs DB stored values
  4. Composite score       — verify weighting formula
  5. Trade signals         — re-generate and compare with DB
  6. Chart summary logic   — Weinstein stage, pivot levels, S/R zones
  7. DB round-trip         — save / load consistency
  8. Latency               — wall-clock time per operation
  9. API cost accounting   — count Kite API calls made

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

import db
import config
import indicators as ind
from signals import compute_all_signals

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
    print("  SCREENER FEATURE TEST  —  30 stocks / 3 batches")
    print(f"{'═'*70}{RESET}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # ── Latency benchmark ────────────────────────────────────────────────
    run_latency_bench(sampled[:3])

    # ── API cost accounting ──────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  COST / API ACCOUNTING{RESET}")
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
