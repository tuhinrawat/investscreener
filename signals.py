"""
signals.py — Trade signal engine.

Answers three questions for every stock:
  1. WHEN to BUY      → entry price, stop loss, targets, R/R
  2. WHEN to SELL     → exit trigger with reason
  3. FOR WHICH SETUP  → Intraday, Swing, or Scaling

Three setup types:
  - Swing    : 2–10 day hold, trend-following (EMA stack + volume)
  - Intraday : same-day plan from previous session's candle (pivot levels)
  - Scaling  : multi-week position build on confirmed uptrend stocks

Each setup returns a flat dict ready to merge into the metrics row.
All price levels are rounded to 2 decimal places.
"""
import numpy as np
import pandas as pd


# ─── internal helpers ───────────────────────────────────────────────────────

def _v(x):
    """Return x if finite, else None."""
    if x is None:
        return None
    try:
        return x if np.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def _adx(df: pd.DataFrame, period: int = 14) -> float | None:
    """
    Average Directional Index — measures trend STRENGTH (not direction).
    ADX > 20 = trend present. ADX > 30 = strong trend.
    Direction-blind: use with EMA stack for direction.
    """
    if len(df) < period + 2:
        return None
    high = df["high"].values.astype(float)
    low  = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    tr_arr, pdm_arr, ndm_arr = [], [], []
    for i in range(1, len(high)):
        tr  = max(high[i] - low[i],
                  abs(high[i] - close[i - 1]),
                  abs(low[i]  - close[i - 1]))
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm_arr.append(up   if (up > down and up > 0)   else 0.0)
        ndm_arr.append(down if (down > up and down > 0) else 0.0)
        tr_arr.append(tr)

    alpha = 1 / period
    tr_s  = pd.Series(tr_arr).ewm(alpha=alpha, adjust=False).mean()
    pdi   = pd.Series(pdm_arr).ewm(alpha=alpha, adjust=False).mean() / tr_s * 100
    ndi   = pd.Series(ndm_arr).ewm(alpha=alpha, adjust=False).mean() / tr_s * 100
    denom = (pdi + ndi).replace(0, np.nan)
    dx    = ((pdi - ndi).abs() / denom) * 100
    adx_s = dx.ewm(alpha=alpha, adjust=False).mean()
    val   = adx_s.iloc[-1]
    return float(val) if pd.notna(val) else None


def _nr7(df: pd.DataFrame) -> bool:
    """True if today's candle range is the narrowest in the last 7 days."""
    if len(df) < 7:
        return False
    ranges = (df["high"] - df["low"]).iloc[-7:]
    return bool(ranges.iloc[-1] == ranges.min())


def _vol_drying(df: pd.DataFrame, lookback: int = 3, threshold: float = 0.8,
                vexp_fallback: float | None = None) -> bool:
    """
    True if volume is contracting (pullback on falling volume = healthy).

    Two ways this can be True:
      1. Raw OHLCV: last `lookback` days avg < threshold × 20D avg (primary)
      2. Stored metric fallback: vol_expansion_ratio < threshold (for Quick Scan
         when df is empty — avoids false WATCH signals during intraday refresh)
    """
    if len(df) >= 20:
        avg20  = df["volume"].iloc[-20:].mean()
        recent = df["volume"].iloc[-lookback:].mean()
        return bool(avg20 > 0 and recent < threshold * avg20)
    # Fallback: use stored vol_expansion_ratio from computed_metrics
    if vexp_fallback is not None:
        return bool(vexp_fallback < threshold)
    return False


def _vol_breakout(df: pd.DataFrame, vexp_stored: float | None = None) -> bool:
    """
    True if there is genuine breakout volume.

    Two-level check:
      1. Last candle's individual volume > 1.5× 20D avg (single-bar confirmation)
      2. OR stored vol_expansion_ratio > 1.3 (5D/20D avg from Full Rescan)

    Using both levels means a breakout day is caught even when the 5D avg
    is still catching up (one high-volume candle only moves the 5D avg ~20%).
    """
    if len(df) >= 20:
        avg20 = df["volume"].iloc[-20:].mean()
        last_vol = df["volume"].iloc[-1]
        if avg20 > 0 and last_vol > 1.5 * avg20:
            return True
    if vexp_stored is not None and vexp_stored > 1.3:
        return True
    return False


def _breakout_days_ago(df: pd.DataFrame, lookback: int = 20) -> int | None:
    """
    How many sessions ago did price close above the prior N-day high?
    Returns 1, 2, or 3 (only checks recent 3 candles). None if no breakout.
    """
    if len(df) < lookback + 3:
        return None
    for lag in range(1, 4):
        window_high = df["high"].iloc[-(lookback + lag): -lag].max()
        if df["close"].iloc[-lag] > window_high:
            return lag
    return None


def _pivot_levels(df: pd.DataFrame) -> dict:
    """
    Classic floor-trader pivot points computed from the last completed session.
    Used for intraday planning — these are levels to WATCH on a chart.
    """
    prev = df.iloc[-1]
    H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
    P  = (H + L + C) / 3
    R1 = 2 * P - L
    R2 = P + (H - L)
    R3 = H + 2 * (P - L)
    S1 = 2 * P - H
    S2 = P - (H - L)
    S3 = L - 2 * (H - P)
    return {k: round(v, 2) for k, v in
            dict(pivot=P, r1=R1, r2=R2, r3=R3, s1=S1, s2=S2, s3=S3).items()}


# ─── swing signal ────────────────────────────────────────────────────────────

_SWING_NULL = {
    "swing_signal":  None,
    "swing_setup":   None,
    "swing_entry":   None,
    "swing_stop":    None,
    "swing_t1":      None,
    "swing_t2":      None,
    "swing_rr":      None,
    "swing_quality": None,
    "swing_reason":  None,
}


def swing_signal(df: pd.DataFrame, metrics: dict) -> dict:
    """
    Swing trade signal (2–10 day hold). Detects three buy setups in priority order:
      1. BREAKOUT  — recent 20D high break on above-average volume
      2. PULLBACK  — retracement to 20 EMA in uptrend, volume drying
      3. NR7       — narrowest range in 7 days (pre-breakout coil)

    Also emits SELL signal when holding conditions break down.
    """
    ltp  = _v(metrics.get("ltp"))
    e20  = _v(metrics.get("ema_20"))
    e50  = _v(metrics.get("ema_50"))
    e200 = _v(metrics.get("ema_200"))
    atr  = _v(metrics.get("atr_14"))
    rsi  = _v(metrics.get("rsi_14"))
    vexp = _v(metrics.get("vol_expansion_ratio"))

    # Use explicit None checks — Python's all() treats 0.0 as falsy, which would
    # incorrectly reject RSI=0 (pure downtrend) and block the SELL signal.
    if any(v is None for v in [ltp, e20, e50, atr, rsi]):
        return dict(_SWING_NULL)

    bullish = e20 > e50
    full_stack = bullish and e200 is not None and e50 > e200

    adx_val    = _adx(df)
    trend_ok   = adx_val is not None and adx_val > 18

    # Base quality from market structure
    base_q = 1
    if bullish:    base_q += 1
    if full_stack: base_q += 1
    if trend_ok:   base_q += 1

    def _levels(entry, stop_val):
        stop_val = max(stop_val, entry * 0.85)   # never more than 15% stop
        risk = entry - stop_val
        if risk <= 0:
            return None, None, None, None
        # T1 must be at least 1.5× risk above entry to ensure worthwhile R/R.
        # Use the greater of 2×ATR (standard target) or 1.5× the actual risk.
        t1 = round(max(entry + 2 * atr, entry + 1.5 * risk), 2)
        t2 = round(max(entry + 4 * atr, entry + 3.0 * risk), 2)
        rr = round((t1 - entry) / risk, 2)
        return round(stop_val, 2), t1, t2, rr

    # ── 1. BREAKOUT ───────────────────────────────────────────────────────
    bo = _breakout_days_ago(df, lookback=20)
    has_vol_breakout = _vol_breakout(df, vexp_stored=vexp)
    if (bo is not None and bullish
            and has_vol_breakout
            and 50 <= rsi <= 82):
        entry = round(ltp, 2)
        raw_stop = ltp - 2 * atr
        stop, t1, t2, rr = _levels(entry, raw_stop)
        if stop is None:
            return dict(_SWING_NULL)
        q = min(5, base_q + (1 if vexp > 2.0 else 0) + (1 if bo == 1 else 0))
        return {
            "swing_signal":  "BUY",
            "swing_setup":   "BREAKOUT",
            "swing_entry":   entry,
            "swing_stop":    stop,
            "swing_t1":      t1,
            "swing_t2":      t2,
            "swing_rr":      rr,
            "swing_quality": min(5, q),
            "swing_reason": (
                f"Broke 20D high {bo}d ago on {vexp:.1f}× volume "
                f"(ADX {adx_val:.0f}). "
                f"Buy at market ₹{entry:.2f}. "
                f"Stop 2×ATR below at ₹{stop:.2f}."
            ),
        }

    # ── 2. NR7 contraction (checked before PULLBACK — more specific setup) ──
    # NR7 is a pre-breakout coil: the tightest candle in 7 sessions signals
    # compression before expansion.  Priority over PULLBACK because it's a
    # definitive entry trigger (buy-stop above the candle high), not a zone entry.
    if _nr7(df) and bullish and 43 <= rsi <= 72:
        nr7_high = float(df["high"].iloc[-1])
        entry    = round(nr7_high * 1.003, 2)   # pending buy-stop 0.3% above NR7 high
        raw_stop = float(df["low"].iloc[-1]) - 0.5 * atr
        stop, t1, t2, rr = _levels(entry, raw_stop)
        if stop is None:
            return dict(_SWING_NULL)
        q = min(5, base_q + 1)
        return {
            "swing_signal":  "BUY",
            "swing_setup":   "NR7",
            "swing_entry":   entry,
            "swing_stop":    stop,
            "swing_t1":      t1,
            "swing_t2":      t2,
            "swing_rr":      rr,
            "swing_quality": min(5, q),
            "swing_reason": (
                f"NR7 coil — tightest candle in 7 days (RSI {rsi:.0f}). "
                f"Place buy-stop order at ₹{entry:.2f} (above today's high). "
                f"Stop below candle low at ₹{stop:.2f}."
            ),
        }

    # ── 3. PULLBACK to EMA20 ─────────────────────────────────────────────
    dist20 = (ltp - e20) / e20 * 100 if e20 else None
    # Pass vexp as fallback so Quick Scan (empty df) still detects drying volume
    vol_dry = _vol_drying(df, vexp_fallback=vexp)
    if (bullish
            and dist20 is not None and -3.0 <= dist20 <= 5.0
            and 38 <= rsi <= 65
            and vol_dry):
        entry    = round(ltp, 2)
        raw_stop = e20 - 1.5 * atr
        stop, t1, t2, rr = _levels(entry, raw_stop)
        if stop is None:
            return dict(_SWING_NULL)
        q = min(5, base_q + (1 if abs(dist20) < 1.5 else 0) + (1 if vol_dry else 0))
        return {
            "swing_signal":  "BUY",
            "swing_setup":   "PULLBACK",
            "swing_entry":   entry,
            "swing_stop":    stop,
            "swing_t1":      t1,
            "swing_t2":      t2,
            "swing_rr":      rr,
            "swing_quality": min(5, q),
            "swing_reason": (
                f"{dist20:+.1f}% from EMA20 with volume drying up "
                f"(RSI {rsi:.0f}). "
                f"Pullback entry ₹{entry:.2f}. "
                f"Stop below EMA20 at ₹{stop:.2f}."
            ),
        }

    # ── SELL / EXIT signals ───────────────────────────────────────────────
    sell_reason = None
    if ltp < e20:
        sell_reason = (
            f"Close ₹{ltp:.2f} below EMA20 ₹{e20:.2f} — "
            "uptrend broken. Exit any open long position."
        )
    elif rsi is not None and rsi > 82:
        sell_reason = (
            f"RSI {rsi:.0f} — extremely overbought. "
            "Exit full position, lock in profits."
        )
    elif rsi is not None and rsi > 75:
        sell_reason = (
            f"RSI {rsi:.0f} — overbought zone. "
            "Book partial profits (50%), trail stop on remainder."
        )

    if sell_reason:
        return {
            **_SWING_NULL,
            "swing_signal": "SELL",
            "swing_setup":  "EXIT",
            "swing_reason": sell_reason,
        }

    # ── No actionable setup ───────────────────────────────────────────────
    return {
        **_SWING_NULL,
        "swing_signal": "WATCH",
        "swing_reason": (
            "No setup triggered. "
            "Waiting for pullback to EMA20, volume breakout, or NR7 coil."
        ),
    }


# ─── intraday signal ─────────────────────────────────────────────────────────

_INTRA_NULL = {
    "intraday_signal":     None,
    "intraday_pivot":      None,
    "intraday_r1":         None,
    "intraday_r2":         None,
    "intraday_s1":         None,
    "intraday_s2":         None,
    "intraday_entry":      None,
    "intraday_stop":       None,
    "intraday_t1":         None,
    "intraday_reason":     None,
    "intraday_confidence": None,
}


def compute_intraday_confidence(metrics: dict, rr: float, signal_type: str = "BUY_ABOVE") -> int:
    """
    Score an intraday signal on confidence from 0 to 10.

    Factors and max points:
      R/R ratio             : 0-3 pts  (primary edge driver)
      RSI zone              : 0-2 pts  (momentum positioning)
      Volume expansion      : 0-2 pts  (institutional confirmation)
      Relative strength     : 0-2 pts  (market alignment)
      Composite score       : 0-1 pt   (catch-all trend quality)

    Tiers:
      8-10 = STRONG    → auto-trade, larger capital allocation
      6-7  = MODERATE  → auto-trade, standard allocation
      5    = MARGINAL  → auto-trade only if slots available, smaller allocation
      <5   = LOW       → skip auto-trade (shown in table but not executed)
    """
    is_long = signal_type == "BUY_ABOVE"
    score = 0

    # R/R ratio: 0-3 pts — the single strongest predictor of payout
    if rr >= 3.0:
        score += 3
    elif rr >= 2.0:
        score += 2
    elif rr >= 1.5:
        score += 1

    # RSI zone: 0-2 pts — ideal momentum window differs by direction
    rsi = _v(metrics.get("rsi_14"))
    if rsi is not None:
        if is_long:
            # Sweet spot: building momentum without being overbought
            if 45 <= rsi <= 65:
                score += 2
            elif (35 <= rsi < 45) or (65 < rsi <= 72):
                score += 1
        else:
            # Sweet spot for shorts: weakening but not yet oversold (room to fall)
            if 30 <= rsi <= 55:
                score += 2
            elif (55 < rsi <= 65) or (25 <= rsi < 30):
                score += 1

    # Volume expansion: 0-2 pts — institutional activity confirms breakout
    vexp = _v(metrics.get("vol_expansion_ratio"))
    if vexp is not None:
        if vexp >= 1.5:
            score += 2
        elif vexp >= 1.2:
            score += 1

    # Relative strength vs Nifty: 0-2 pts — trading with market momentum
    rs = _v(metrics.get("rs_vs_nifty_3m"))
    if rs is not None:
        if is_long:
            if rs >= 2.0:
                score += 2
            elif rs >= 0.0:
                score += 1
        else:
            # Negative RS = weak relative to market = good short candidate
            if rs <= -2.0:
                score += 2
            elif rs <= 0.0:
                score += 1

    # Composite score: 0-1 pt — overall trend quality (EMA alignment + RS + volume)
    comp = _v(metrics.get("composite_score"))
    if comp is not None:
        if is_long and comp >= 65:
            score += 1
        elif not is_long and comp <= 40:
            score += 1

    return min(10, score)


def intraday_signal(
    df: pd.DataFrame,
    metrics: dict,
    rsi_buy_max: float = 75.0,
    rsi_sell_min: float = 25.0,
    min_rr: float = 1.5,
) -> dict:
    """
    Day-trading plan derived from the previous session's candle.

    Computes classical pivot levels (P, R1, R2, S1, S2) and generates:
      BUY_ABOVE  : long when price breaks above R1 (trend up, RSI not overbought)
      SELL_BELOW : short when price breaks below S1 (trend down, RSI not oversold)
      AVOID      : structure is ambiguous / risk not worthwhile

    Key rule: ALL intraday trades must be closed by 3:10 PM regardless.

    LONG  entry = 0.1% above R1  | stop = just below R1  | T1 = R2
    SHORT entry = 0.1% below S1  | stop = just above S1  | T1 = S2

    Thresholds (rsi_buy_max, rsi_sell_min, min_rr) are tunable via paper-trade
    feedback — pass values from db.get_signal_config() to adjust the algo.
    """
    ltp = _v(metrics.get("ltp"))
    e20 = _v(metrics.get("ema_20"))
    atr = _v(metrics.get("atr_14"))
    rsi = _v(metrics.get("rsi_14"))

    if any(v is None for v in [ltp, e20, atr]) or len(df) < 5:
        return dict(_INTRA_NULL)

    lvls = _pivot_levels(df)
    P, R1, R2 = lvls["pivot"], lvls["r1"], lvls["r2"]
    S1, S2    = lvls["s1"],    lvls["s2"]

    # Guard: if H-L spread from yesterday is < 0.3% of close, the pivot levels
    # are meaningless (e.g. liquid ETFs, debt funds). AVOID to prevent T1 <= entry.
    prev_hl_range = P * 3 - lvls.get("s1_raw_high", P * 3) if "s1_raw_high" in lvls else None
    _spread_pct = (R2 - S2) / P if P > 0 else 0  # total pivot range as % of pivot
    if _spread_pct < 0.006:          # less than 0.6% total range → no tradeable levels
        return {
            **_INTRA_NULL,
            "intraday_signal": "AVOID",
            "intraday_pivot":  P,
            "intraday_r1":     R1,
            "intraday_r2":     R2,
            "intraday_s1":     S1,
            "intraday_s2":     S2,
            "intraday_reason": "H-L spread too narrow for reliable pivot levels (ETF/fund).",
        }

    # ── LONG: above EMA20, RSI not overbought ──────────────────────────────
    if ltp > e20 and (rsi is None or rsi < rsi_buy_max):
        entry = round(R1 * 1.001, 2)
        # Ensure entry is strictly above R1 (rounding can collapse them for sub-₹10 stocks)
        if entry <= R1:
            entry = round(R1 + max(0.01, R1 * 0.001), 2)

        # Stop = just below R1 (the breakout level).
        # Rationale: once R1 breaks it should act as support. If price falls back
        # below R1 the breakout has failed → exit tight. This gives a 0.3–0.5%
        # risk vs 1–3% gain (R1→R2), producing R/R of 2–6×.
        # The old Pivot-based stop (P – 0.25×ATR) was ≈(P–L) away from entry
        # while the target was only (H–P) away → R/R often < 1.
        stop = round(R1 * 0.997, 2)          # 0.3% below the breakout level

        t1   = R2
        if t1 <= entry:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2,
                "intraday_s1": S1, "intraday_s2": S2,
                "intraday_reason": "R2 target not far enough above R1 — insufficient range for trade.",
            }
        risk = entry - stop
        rr   = round((t1 - entry) / risk, 2) if risk > 0 else 0

        # Enforce minimum R/R gate (tunable via paper-trade feedback)
        if rr < min_rr:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2,
                "intraday_s1": S1, "intraday_s2": S2,
                "intraday_reason": f"R/R {rr:.1f}× < {min_rr:.1f}× minimum — risk not justified.",
            }
        _conf = compute_intraday_confidence(metrics, rr, "BUY_ABOVE")
        return {
            "intraday_signal":     "BUY_ABOVE",
            "intraday_pivot":      P,
            "intraday_r1":         R1,
            "intraday_r2":         R2,
            "intraday_s1":         S1,
            "intraday_s2":         S2,
            "intraday_entry":      entry,
            "intraday_stop":       stop,
            "intraday_t1":         t1,
            "intraday_confidence": _conf,
            "intraday_reason": (
                f"Trend up (above EMA20 ₹{e20:.2f}). "
                f"BUY when price trades above R1 ₹{R1:.2f}. "
                f"Stop just below R1 ₹{stop:.2f}. "
                f"Target R2 ₹{R2:.2f}. R/R {rr:.1f}×. "
                f"Confidence {_conf}/10. Hard exit 3:10 PM."
            ),
        }

    # ── SHORT: below EMA20, RSI not oversold (still room to fall) ──────────
    _short_range_ok = (S1 - S2) / S1 > 0.005 if S1 > 0 else False
    if ltp < e20 and (rsi is None or rsi > rsi_sell_min) and _short_range_ok:
        entry = round(S1 * 0.999, 2)          # 0.1% below S1 = breakdown confirmation
        # Stop = just above S1 (the breakdown level).
        # If price recovers back above S1, the breakdown was false → exit tight.
        stop  = round(S1 * 1.003, 2)          # 0.3% above the breakdown level
        t1    = S2
        risk  = stop - entry
        rr    = round((entry - t1) / risk, 2) if risk > 0 else 0

        # Enforce minimum R/R gate (tunable via paper-trade feedback)
        if rr < min_rr:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2,
                "intraday_s1": S1, "intraday_s2": S2,
                "intraday_reason": f"Short R/R {rr:.1f}× < {min_rr:.1f}× minimum — risk not justified.",
            }
        _conf = compute_intraday_confidence(metrics, rr, "SELL_BELOW")
        return {
            "intraday_signal":     "SELL_BELOW",
            "intraday_pivot":      P,
            "intraday_r1":         R1,
            "intraday_r2":         R2,
            "intraday_s1":         S1,
            "intraday_s2":         S2,
            "intraday_entry":      entry,
            "intraday_stop":       stop,
            "intraday_t1":         t1,
            "intraday_confidence": _conf,
            "intraday_reason": (
                f"Trend down (below EMA20 ₹{e20:.2f}). "
                f"SHORT when price breaks below S1 ₹{S1:.2f}. "
                f"Stop just above S1 ₹{stop:.2f}. "
                f"Target S2 ₹{S2:.2f}. R/R {rr:.1f}×. "
                f"Confidence {_conf}/10. Hard exit 3:10 PM."
            ),
        }

    # ── AVOID: overbought long / oversold short / ambiguous ────────────────
    return {
        **_INTRA_NULL,
        "intraday_signal": "AVOID",
        "intraday_pivot":  P,
        "intraday_r1":     R1,
        "intraday_r2":     R2,
        "intraday_s1":     S1,
        "intraday_s2":     S2,
        "intraday_reason": (
            "Overbought or oversold — risk/reward poor. "
            f"Key levels: Pivot ₹{P:.2f} | R1 ₹{R1:.2f} | S1 ₹{S1:.2f}."
        ),
    }


# ─── scaling signal ──────────────────────────────────────────────────────────

_SCALE_NULL = {
    "scale_signal":        None,
    "scale_setup":         None,
    "scale_entry_1":       None,
    "scale_stop":          None,
    "scale_trailing_stop": None,
    "scale_target":        None,
    "scale_quality":       None,
    "scale_reason":        None,
}


def scaling_signal(df: pd.DataFrame, metrics: dict) -> dict:
    """
    Multi-week position building signal.

    Philosophy: only touch stocks where the market structure is unambiguous
    (full EMA stack, positive RS, proven 6M trend). Then enter at the best
    risk point (EMA50 pullback) and trail up with the 50 EMA as the stop.

    Two states:
      INITIAL_ENTRY : price is in the EMA50 pullback zone → deploy 40% now,
                      plan to add another 30–40% on next base breakout
      HOLD_FOR_ADD  : price is extended above EMA50 → if you're already in,
                      hold and trail; if not, wait for the next pullback
    """
    ltp   = _v(metrics.get("ltp"))
    e20   = _v(metrics.get("ema_20"))
    e50   = _v(metrics.get("ema_50"))
    e200  = _v(metrics.get("ema_200"))
    atr   = _v(metrics.get("atr_14"))
    rsi   = _v(metrics.get("rsi_14"))
    ret6m = _v(metrics.get("ret_6m"))
    rs    = _v(metrics.get("rs_vs_nifty_3m"))

    if any(v is None for v in [ltp, e20, e50, e200, atr, rsi]):
        return dict(_SCALE_NULL)

    full_stack = (e20 > e50 > e200)
    if not full_stack:
        return dict(_SCALE_NULL)

    # Hard gates for scaling — only the strongest stocks
    if rs is not None and rs < 0:
        return dict(_SCALE_NULL)
    if ret6m is not None and ret6m < 8.0:
        return dict(_SCALE_NULL)

    base_q = 3   # full stack is already a quality signal
    if rs  is not None and rs  > 5:  base_q += 1
    if ret6m is not None and ret6m > 20: base_q += 1
    base_q = min(5, base_q)

    dist_e50 = (ltp - e50) / e50 * 100

    # ── INITIAL ENTRY: price at EMA50 pullback zone ───────────────────────
    if 0 <= dist_e50 <= 7 and 43 <= rsi <= 65:
        entry_1  = round(e50 * 1.005, 2)   # 0.5% above EMA50 (confirms it held)
        stop     = round(e50 - 1.5 * atr, 2)
        trailing = round(e50, 2)
        target   = round(entry_1 * 1.18, 2)  # 18% measured-move target
        return {
            "scale_signal":        "INITIAL_ENTRY",
            "scale_setup":         "EMA50_PULLBACK",
            "scale_entry_1":       entry_1,
            "scale_stop":          stop,
            "scale_trailing_stop": trailing,
            "scale_target":        target,
            "scale_quality":       base_q,
            "scale_reason": (
                f"Full EMA stack. Price {dist_e50:.1f}% above EMA50 (pullback zone). "
                f"Deploy 40% position at ₹{entry_1:.2f}. "
                f"Add 30% when price breaks above 20D high. "
                f"Trailing stop: close below EMA50 ₹{trailing:.2f}. "
                f"Final target ₹{target:.2f} (+18%)."
            ),
        }

    # ── HOLD / WAIT FOR ADD: extended above EMA50 ─────────────────────────
    if 7 < dist_e50 <= 20 and rsi <= 75:
        trailing = round(e50, 2)
        target   = round(ltp * 1.10, 2)
        stop     = round(e50 - atr, 2)
        return {
            "scale_signal":        "HOLD_FOR_ADD",
            "scale_setup":         "WAIT_PULLBACK",
            "scale_entry_1":       None,
            "scale_stop":          stop,
            "scale_trailing_stop": trailing,
            "scale_target":        target,
            "scale_quality":       max(2, base_q - 1),
            "scale_reason": (
                f"Strong trend but {dist_e50:.1f}% above EMA50 — too extended for fresh entry. "
                f"If already holding: trail stop below EMA50 ₹{trailing:.2f}. "
                f"If not holding: wait for pullback to EMA50 before adding."
            ),
        }

    return dict(_SCALE_NULL)


# ─── master function ─────────────────────────────────────────────────────────

def compute_all_signals(
    df: pd.DataFrame,
    metrics: dict,
    rsi_buy_max: float = 75.0,
    rsi_sell_min: float = 25.0,
    min_rr: float = 1.5,
) -> dict:
    """
    Runs all three setups and returns a single merged dict.
    Safe to call even if df is empty or metrics are partial.

    Pass rsi_buy_max / rsi_sell_min / min_rr from db.get_signal_config() to
    use paper-trade-tuned thresholds instead of the hard-coded defaults.
    """
    if df is None or df.empty or len(df) < 20:
        return {**_SWING_NULL, **_INTRA_NULL, **_SCALE_NULL}

    df = df.sort_values("date").reset_index(drop=True)
    sw = swing_signal(df, metrics)
    it = intraday_signal(df, metrics,
                         rsi_buy_max=rsi_buy_max,
                         rsi_sell_min=rsi_sell_min,
                         min_rr=min_rr)
    sc = scaling_signal(df, metrics)
    return {**sw, **it, **sc}
