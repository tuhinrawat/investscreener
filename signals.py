"""
signals.py — Trade signal engine.

Answers three questions for every stock:
  1. WHEN to BUY      → entry price, stop loss, targets, R/R
  2. WHEN to SELL     → exit trigger with reason
  3. FOR WHICH SETUP  → Intraday, Swing, Scaling, or Scalping

Four setup types:
  - Swing    : 2–10 day hold, trend-following (EMA stack + volume)
  - Intraday : same-day plan from previous session's candle (pivot levels)
  - Scaling  : multi-week position build on confirmed uptrend stocks
  - Scalping : fast in/out (< 30 min hold) using Opening Range Breakout +
               VWAP alignment + intraday RSI — 3 internal confirmations required

Each setup returns a flat dict ready to merge into the metrics row.
All price levels are rounded to 2 decimal places.

New intraday gates (applied at trigger time):
  - Nifty intraday direction gate: suppresses longs on Nifty < -NIFTY_GATE_PCT,
    suppresses shorts on Nifty > +NIFTY_GATE_PCT
  - Gap detection: flags gap-up entries (today's open far above R1) as risk
  - Partial booking: T1 = R2 (60% exit), T2 = R3 (trailing 40%)
"""
import numpy as np
import pandas as pd
import config as _cfg


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
    # Always use yesterday's complete candle.  If df was fetched during the trading
    # day, the last row is today's partial candle — drop it so pivots are correct.
    _ref = df
    if "date" in df.columns and len(df) >= 2:
        import datetime as _dt2
        _today_str = _dt2.date.today().isoformat()
        _last_date = str(df["date"].iloc[-1])[:10]
        if _last_date == _today_str:
            _ref = df.iloc[:-1]
    elif len(df) >= 2 and hasattr(df.index, "date"):
        import datetime as _dt2
        _today_str = _dt2.date.today().isoformat()
        _last_date = str(df.index[-1])[:10]
        if _last_date == _today_str:
            _ref = df.iloc[:-1]

    prev = _ref.iloc[-1]
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
    "intraday_r3":         None,
    "intraday_s1":         None,
    "intraday_s2":         None,
    "intraday_s3":         None,
    "intraday_entry":      None,
    "intraday_stop":       None,
    "intraday_t1":         None,
    "intraday_t2":         None,
    "intraday_reason":     None,
    "intraday_confidence": None,
    "intraday_gap_flag":   None,
    "intraday_nifty_gate": None,
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
    nifty_pct_change: float = 0.0,
    today_open: float | None = None,
    vwap: float | None = None,
) -> dict:
    """
    Day-trading plan derived from the previous session's candle.

    Computes classical pivot levels (P, R1, R2, R3, S1, S2, S3) and generates:
      BUY_ABOVE  : long when price breaks above R1 (trend up, RSI not overbought)
      SELL_BELOW : short when price breaks below S1 (trend down, RSI not oversold)
      AVOID      : structure is ambiguous / risk not worthwhile

    Key rule: ALL intraday trades must be closed by 3:10 PM regardless.

    LONG  entry = 0.1% above R1  | stop = max(0.3% below R1, 0.5×ATR)  | T1 = R2  | T2 = R3
    SHORT entry = 0.1% below S1  | stop = max(0.3% above S1, 0.5×ATR)  | T1 = S2  | T2 = S3

    Gates applied on top of baseline R/R and RSI checks:
      nifty_pct_change — today's Nifty 50 intraday % change from prev close.
        Long signals are flagged/suppressed when Nifty < -NIFTY_GATE_PCT.
        Short signals are flagged/suppressed when Nifty > +NIFTY_GATE_PCT.
      today_open — today's open price for the stock.  If open is already
        GAP_WARN_PCT% above R1, flag as gap-up risk; if > GAP_SKIP_PCT, AVOID.
      vwap — today's intraday VWAP from 5-min candles.  When
        INTRADAY_REGIME_GATE=True:
          Long: AVOID if stock is below VWAP AND Nifty is bearish.
          Short: AVOID if stock is above VWAP AND Nifty is bullish.

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
    P, R1, R2, R3 = lvls["pivot"], lvls["r1"], lvls["r2"], lvls["r3"]
    S1, S2, S3    = lvls["s1"],    lvls["s2"], lvls["s3"]

    # Guard: if H-L spread from yesterday is < 0.3% of close, the pivot levels
    # are meaningless (e.g. liquid ETFs, debt funds). AVOID to prevent T1 <= entry.
    _spread_pct = (R2 - S2) / P if P > 0 else 0  # total pivot range as % of pivot
    if _spread_pct < 0.006:          # less than 0.6% total range → no tradeable levels
        return {
            **_INTRA_NULL,
            "intraday_signal": "AVOID",
            "intraday_pivot":  P,
            "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
            "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
            "intraday_reason": "H-L spread too narrow for reliable pivot levels (ETF/fund).",
        }

    # ── LONG: above EMA20, RSI not overbought ──────────────────────────────
    if ltp > e20 and (rsi is None or rsi < rsi_buy_max):
        entry = round(R1 * 1.001, 2)
        # Ensure entry is strictly above R1 (rounding can collapse them for sub-₹10 stocks)
        if entry <= R1:
            entry = round(R1 + max(0.01, R1 * 0.001), 2)

        # Stop = max(R1×0.997, R1 − 0.5×ATR).
        _atr_stop = round(R1 - 0.5 * atr, 2) if atr else None
        stop = round(R1 * 0.997, 2)          # 0.3% below R1 (minimum stop)
        if _atr_stop is not None and _atr_stop < stop:
            stop = _atr_stop                 # widen if ATR warrants it

        t1 = R2
        t2 = R3   # partial-booking second target
        if t1 <= entry:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                "intraday_reason": "R2 target not far enough above R1 — insufficient range for trade.",
            }
        risk = entry - stop
        rr   = round((t1 - entry) / risk, 2) if risk > 0 else 0

        # Enforce minimum R/R gate (tunable via paper-trade feedback)
        if rr < min_rr:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                "intraday_reason": f"R/R {rr:.1f}× < {min_rr:.1f}× minimum — risk not justified.",
            }

        # ── Gap-up detection ──────────────────────────────────────────────
        gap_flag = None
        if today_open is not None and R1 > 0:
            gap_pct = (today_open - R1) / R1 * 100
            if gap_pct >= _cfg.GAP_SKIP_PCT:
                return {
                    **_INTRA_NULL,
                    "intraday_signal": "AVOID",
                    "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                    "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                    "intraday_reason": (
                        f"Gap-up too large ({gap_pct:.1f}% above R1) — "
                        "entry already at risk of fading. Skip."
                    ),
                    "intraday_gap_flag": "GAP_SKIP",
                }
            elif gap_pct >= _cfg.GAP_WARN_PCT:
                gap_flag = "GAP_WARN"   # emit signal but warn in the reason

        # ── Nifty + VWAP regime gate ──────────────────────────────────────
        nifty_gate = None
        if nifty_pct_change <= -_cfg.NIFTY_GATE_PCT:
            nifty_gate = "NIFTY_BEARISH"
            # Hard block: if regime gate enabled AND stock is also below VWAP
            if _cfg.INTRADAY_REGIME_GATE and vwap is not None and ltp < vwap:
                return {
                    **_INTRA_NULL,
                    "intraday_signal": "AVOID",
                    "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                    "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                    "intraday_nifty_gate": nifty_gate,
                    "intraday_reason": (
                        f"REGIME BLOCK: Nifty down {abs(nifty_pct_change):.1f}% AND stock "
                        f"below VWAP ₹{vwap:.2f} — long setup has double headwind. AVOID."
                    ),
                }

        _conf = compute_intraday_confidence(metrics, rr, "BUY_ABOVE")

        gap_note   = (f" ⚠️ Gap-up {(today_open - R1) / R1 * 100:.1f}% above R1 — watch for fade."
                      if gap_flag == "GAP_WARN" else "")
        nifty_note = (f" ⚠️ Nifty down {abs(nifty_pct_change):.1f}% today — headwind for longs."
                      if nifty_gate else "")

        return {
            "intraday_signal":     "BUY_ABOVE",
            "intraday_pivot":      P,
            "intraday_r1":         R1,
            "intraday_r2":         R2,
            "intraday_r3":         R3,
            "intraday_s1":         S1,
            "intraday_s2":         S2,
            "intraday_s3":         S3,
            "intraday_entry":      entry,
            "intraday_stop":       stop,
            "intraday_t1":         t1,
            "intraday_t2":         t2,
            "intraday_confidence": _conf,
            "intraday_gap_flag":   gap_flag,
            "intraday_nifty_gate": nifty_gate,
            "intraday_reason": (
                f"Trend up (above EMA20 ₹{e20:.2f}). "
                f"BUY when price trades above R1 ₹{R1:.2f}. "
                f"Stop ₹{stop:.2f} (max 0.3% below R1, 0.5×ATR={0.5*atr:.2f}). "
                f"T1 ₹{t1:.2f} (R2, 60% exit). T2 ₹{t2:.2f} (R3, trail 40%). "
                f"R/R {rr:.1f}×. Confidence {_conf}/10. Hard exit 3:10 PM."
                f"{gap_note}{nifty_note}"
            ),
        }

    # ── SHORT: below EMA20, RSI not oversold (still room to fall) ──────────
    _short_range_ok = (S1 - S2) / S1 > 0.005 if S1 > 0 else False
    if ltp < e20 and (rsi is None or rsi > rsi_sell_min) and _short_range_ok:
        entry = round(S1 * 0.999, 2)          # 0.1% below S1 = breakdown confirmation
        # Stop = max(S1×1.003, S1 + 0.5×ATR) — mirror of the long-side ATR widening.
        _atr_stop_s = round(S1 + 0.5 * atr, 2) if atr else None
        stop = round(S1 * 1.003, 2)           # 0.3% above S1 (minimum stop)
        if _atr_stop_s is not None and _atr_stop_s > stop:
            stop = _atr_stop_s                # widen if ATR warrants it
        t1    = S2
        t2    = S3  # trail target
        risk  = stop - entry
        rr    = round((entry - t1) / risk, 2) if risk > 0 else 0

        # Enforce minimum R/R gate (tunable via paper-trade feedback)
        if rr < min_rr:
            return {
                **_INTRA_NULL,
                "intraday_signal": "AVOID",
                "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                "intraday_reason": f"Short R/R {rr:.1f}× < {min_rr:.1f}× minimum — risk not justified.",
            }

        # ── Gap-down detection for shorts ─────────────────────────────────
        gap_flag = None
        if today_open is not None and S1 > 0:
            gap_pct_s = (S1 - today_open) / S1 * 100   # how far below S1 did we open
            if gap_pct_s >= _cfg.GAP_SKIP_PCT:
                return {
                    **_INTRA_NULL,
                    "intraday_signal": "AVOID",
                    "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                    "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                    "intraday_reason": (
                        f"Gap-down too large ({gap_pct_s:.1f}% below S1) — "
                        "short entry already at risk of reversal. Skip."
                    ),
                    "intraday_gap_flag": "GAP_SKIP",
                }
            elif gap_pct_s >= _cfg.GAP_WARN_PCT:
                gap_flag = "GAP_WARN"

        # ── Nifty + VWAP regime gate for shorts ───────────────────────────
        nifty_gate = None
        if nifty_pct_change >= _cfg.NIFTY_GATE_PCT:
            nifty_gate = "NIFTY_BULLISH"
            # Hard block: if regime gate enabled AND stock is also above VWAP
            if _cfg.INTRADAY_REGIME_GATE and vwap is not None and ltp > vwap:
                return {
                    **_INTRA_NULL,
                    "intraday_signal": "AVOID",
                    "intraday_pivot": P, "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
                    "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
                    "intraday_nifty_gate": nifty_gate,
                    "intraday_reason": (
                        f"REGIME BLOCK: Nifty up {abs(nifty_pct_change):.1f}% AND stock "
                        f"above VWAP ₹{vwap:.2f} — short setup has double headwind. AVOID."
                    ),
                }

        _conf = compute_intraday_confidence(metrics, rr, "SELL_BELOW")

        gap_note   = (f" ⚠️ Gap-down {(S1 - today_open) / S1 * 100:.1f}% below S1 — watch for reversal."
                      if gap_flag == "GAP_WARN" and today_open is not None else "")
        nifty_note = (f" ⚠️ Nifty up {abs(nifty_pct_change):.1f}% today — headwind for shorts."
                      if nifty_gate else "")

        return {
            "intraday_signal":     "SELL_BELOW",
            "intraday_pivot":      P,
            "intraday_r1":         R1,
            "intraday_r2":         R2,
            "intraday_r3":         R3,
            "intraday_s1":         S1,
            "intraday_s2":         S2,
            "intraday_s3":         S3,
            "intraday_entry":      entry,
            "intraday_stop":       stop,
            "intraday_t1":         t1,
            "intraday_t2":         t2,
            "intraday_confidence": _conf,
            "intraday_gap_flag":   gap_flag,
            "intraday_nifty_gate": nifty_gate,
            "intraday_reason": (
                f"Trend down (below EMA20 ₹{e20:.2f}). "
                f"SHORT when price breaks below S1 ₹{S1:.2f}. "
                f"Stop ₹{stop:.2f} (max 0.3% above S1, 0.5×ATR). "
                f"T1 ₹{t1:.2f} (S2, 60% exit). T2 ₹{t2:.2f} (S3, trail 40%). "
                f"R/R {rr:.1f}×. Confidence {_conf}/10. Hard exit 3:10 PM."
                f"{gap_note}{nifty_note}"
            ),
        }

    # ── AVOID: overbought long / oversold short / ambiguous ────────────────
    return {
        **_INTRA_NULL,
        "intraday_signal": "AVOID",
        "intraday_pivot":  P,
        "intraday_r1": R1, "intraday_r2": R2, "intraday_r3": R3,
        "intraday_s1": S1, "intraday_s2": S2, "intraday_s3": S3,
        "intraday_reason": (
            "Overbought or oversold — risk/reward poor. "
            f"Key levels: Pivot ₹{P:.2f} | R1 ₹{R1:.2f} | S1 ₹{S1:.2f}."
        ),
    }


# ─── scalping signal (Opening Range Breakout v2) ─────────────────────────────

_SCALP_NULL = {
    "scalp_signal":        None,
    "scalp_direction":     None,
    "scalp_entry":         None,
    "scalp_stop":          None,
    "scalp_t1":            None,
    "scalp_t2":            None,   # runner target (30% of position)
    "scalp_rr":            None,
    "scalp_confirmations": None,
    "scalp_orb_high":      None,
    "scalp_orb_low":       None,
    "scalp_vwap":          None,
    "scalp_reason":        None,
    "scalp_confidence":    None,
}


def _scalp_confidence(
    n_confirmed: int,
    rr: float,
    atr_pct: float,
    orb_range_vs_atr: float,
    nifty_soft_adverse: bool,
    volume_pace_ok: bool,
    vol_surge_at_breakout: bool,
) -> int:
    """
    Calibrated 0–10 confidence score for ORB scalp signals.
    Weights the factors that actually predict whether a scalp reaches T1.
    """
    score = 0
    # Confirmations: 3/3 better than 2/3
    score += 3 if n_confirmed == 3 else 2
    # R:R quality — primary predictor of payout asymmetry
    if rr >= 3.0:   score += 3
    elif rr >= 2.5: score += 2
    elif rr >= 1.8: score += 1
    # ATR% viability: can the stock move enough in 30 min?
    if atr_pct >= 2.0:   score += 2
    elif atr_pct >= 1.5: score += 1
    # ORB compression ratio: tight coil = coiled spring = higher breakout quality
    if orb_range_vs_atr < 0.4:   score += 2   # very tight coil
    elif orb_range_vs_atr < 0.6: score += 1   # moderate coil
    # Context deductions
    if nifty_soft_adverse:        score -= 1
    if not volume_pace_ok:        score -= 1
    if not vol_surge_at_breakout: score -= 1
    return max(0, min(10, score))


def scalping_signal(
    current_ltp: float,
    orb_high: float | None,
    orb_low: float | None,
    orb_range: float | None,
    vwap_price: float | None,
    rsi_5min: float | None,
    atr: float | None,               # 5-min ATR(7) — NOT daily ATR
    nifty_pct_change: float = 0.0,
    daily_vol_ratio: float | None = None,
    vol_surge_at_breakout: bool = False,
) -> dict:
    """
    Opening Range Breakout (ORB) scalping signal v2.

    Key improvements over v1:
      • Uses 5-min ATR(7) instead of daily ATR for all calculations.
      • Stop uses WIDER of structural (50% ORB retrace) or ATR floor — not tighter.
      • Two-stage targets: T1 = 1× ORB range (70% exit), T2 = 1.5× (30% runner).
      • Both targets capped at 2.5× ATR to prevent unreachable aspirational levels.
      • Narrow ORB is a quality BOOSTER (coiled spring), not a rejection gate.
      • Wide ORB (> 1.5× ATR) is rejected — daily range already exhausted.
      • Regime gate is graduated: hard block at 1.2%, soft penalty at 0.6%.
      • Confidence score is a proper 0–10 scale weighted on R:R, ATR%, compression.
    """
    if orb_high is None or orb_low is None or orb_range is None or orb_range <= 0:
        return dict(_SCALP_NULL)
    if current_ltp is None or current_ltp <= 0:
        return dict(_SCALP_NULL)
    if current_ltp < _cfg.SCALP_MIN_PRICE:
        return dict(_SCALP_NULL)
    if atr is None or atr <= 0:
        return dict(_SCALP_NULL)

    atr_pct         = (atr / current_ltp * 100) if current_ltp > 0 else 0
    orb_range_atr   = orb_range / atr            # compression ratio
    min_conf        = _cfg.SCALP_MIN_CONFIRMATIONS

    # ── ORB range quality gates (INVERTED from v1) ───────────────────────────
    # Wide ORB: stock already used up its daily range → nowhere for target to go
    if orb_range > _cfg.SCALP_ORB_EXHAUSTION_MULT * atr:
        return {
            **_SCALP_NULL,
            "scalp_signal": "WATCH",
            "scalp_orb_high": orb_high,
            "scalp_orb_low": orb_low,
            "scalp_confirmations": 0,
            "scalp_reason": (
                f"ORB_RANGE_EXHAUSTED — range {orb_range:.2f} > "
                f"{_cfg.SCALP_ORB_EXHAUSTION_MULT:.1f}× ATR ({atr:.2f}). "
                "Daily range already consumed. Skip."
            ),
        }
    # Narrow ORB (< 0.5× ATR): coiled spring → do NOT reject, boosts confidence score

    # ── Regime gate — graduated ───────────────────────────────────────────────
    _nifty_adverse_long  = nifty_pct_change <= -_cfg.SCALP_NIFTY_HARD_BLOCK_PCT
    _nifty_adverse_short = nifty_pct_change >=  _cfg.SCALP_NIFTY_HARD_BLOCK_PCT
    _nifty_soft_long     = (nifty_pct_change <= -_cfg.SCALP_NIFTY_SOFT_WARN_PCT
                            and not _nifty_adverse_long)
    _nifty_soft_short    = (nifty_pct_change >=  _cfg.SCALP_NIFTY_SOFT_WARN_PCT
                            and not _nifty_adverse_short)
    _vol_pace_ok         = daily_vol_ratio is None or daily_vol_ratio >= 0.8

    def _build_levels_long():
        entry     = round(orb_high, 2)
        orb_stop  = entry - _cfg.SCALP_STOP_ORB_FRAC * orb_range
        atr_floor = entry - _cfg.SCALP_STOP_ATR_MULT * atr
        pct_floor = entry * (1 - _cfg.SCALP_STOP_FLOOR_PCT)
        # Take WIDER of structural/ATR (most protection), then apply % floor as minimum
        stop      = round(max(min(orb_stop, atr_floor), pct_floor), 2)
        risk      = entry - stop
        orb_t1    = entry + _cfg.SCALP_TARGET_T1_MULT * orb_range
        orb_t2    = entry + _cfg.SCALP_TARGET_T2_MULT * orb_range
        atr_cap   = entry + _cfg.SCALP_ATR_TARGET_CAP * atr
        t1        = round(min(orb_t1, atr_cap), 2)
        t2        = round(min(orb_t2, atr_cap), 2)
        rr        = round((t1 - entry) / risk, 2) if risk > 0 else 0
        return entry, stop, t1, t2, rr

    def _build_levels_short():
        entry     = round(orb_low, 2)
        orb_stop  = entry + _cfg.SCALP_STOP_ORB_FRAC * orb_range
        atr_floor = entry + _cfg.SCALP_STOP_ATR_MULT * atr
        pct_floor = entry * (1 + _cfg.SCALP_STOP_FLOOR_PCT)
        stop      = round(min(max(orb_stop, atr_floor), pct_floor), 2)
        risk      = stop - entry
        orb_t1    = entry - _cfg.SCALP_TARGET_T1_MULT * orb_range
        orb_t2    = entry - _cfg.SCALP_TARGET_T2_MULT * orb_range
        atr_cap   = entry - _cfg.SCALP_ATR_TARGET_CAP * atr
        t1        = round(max(orb_t1, atr_cap), 2)
        t2        = round(max(orb_t2, atr_cap), 2)
        rr        = round((entry - t1) / risk, 2) if risk > 0 else 0
        return entry, stop, t1, t2, rr

    def _check_confirmations(is_long: bool):
        confs, notes = [], []
        confs.append("ORB↑" if is_long else "ORB↓")
        notes.append(f"ORB {'breakout above' if is_long else 'breakdown below'} ₹{orb_high if is_long else orb_low:.2f} ✓")
        if vwap_price is not None:
            ok = current_ltp > vwap_price if is_long else current_ltp < vwap_price
            if ok:
                confs.append("VWAP↑" if is_long else "VWAP↓")
                notes.append(f"{'Above' if is_long else 'Below'} VWAP ₹{vwap_price:.2f} ✓")
            else:
                notes.append(f"{'Below' if is_long else 'Above'} VWAP ₹{vwap_price:.2f} ✗")
        else:
            notes.append("VWAP: n/a")
        if rsi_5min is not None:
            ok = rsi_5min > 55 if is_long else rsi_5min < 45
            thresh = "> 55" if is_long else "< 45"
            if ok:
                confs.append("RSI↑" if is_long else "RSI↓")
                notes.append(f"5-min RSI {rsi_5min:.0f} {thresh} ✓")
            else:
                notes.append(f"5-min RSI {rsi_5min:.0f} {'≤ 55' if is_long else '≥ 45'} ✗")
        else:
            notes.append("5-min RSI: n/a")
        return confs, notes

    # ── LONG scalp ───────────────────────────────────────────────────────────
    if current_ltp > orb_high:
        if _nifty_adverse_long:
            return {
                **_SCALP_NULL,
                "scalp_signal": "BLOCKED",
                "scalp_direction": "LONG",
                "scalp_confirmations": 0,
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_reason": (
                    f"NIFTY_BEARISH_EXTREME — Nifty down {abs(nifty_pct_change):.1f}% "
                    f"(hard block > {_cfg.SCALP_NIFTY_HARD_BLOCK_PCT:.1f}%)"
                ),
            }

        confs, conf_notes = _check_confirmations(is_long=True)
        n_confirmed = len(confs)
        if n_confirmed < min_conf:
            return {
                **_SCALP_NULL,
                "scalp_signal": "WATCH", "scalp_direction": "LONG",
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_vwap": vwap_price, "scalp_confirmations": n_confirmed,
                "scalp_reason": (
                    f"ORB long — only {n_confirmed}/{min_conf} confirmations. "
                    + " | ".join(conf_notes)
                ),
            }

        entry, stop, t1, t2, rr = _build_levels_long()
        if rr < _cfg.SCALP_MIN_RR:
            return {
                **_SCALP_NULL,
                "scalp_signal": "WATCH", "scalp_direction": "LONG",
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_vwap": vwap_price,
                "scalp_reason": f"ORB long — R:R {rr:.1f}× < {_cfg.SCALP_MIN_RR:.1f}× minimum.",
            }

        scalp_conf = _scalp_confidence(
            n_confirmed, rr, atr_pct, orb_range_atr,
            _nifty_soft_long, _vol_pace_ok, vol_surge_at_breakout,
        )
        return {
            "scalp_signal": "LONG", "scalp_direction": "LONG",
            "scalp_entry": entry, "scalp_stop": stop,
            "scalp_t1": t1, "scalp_t2": t2, "scalp_rr": rr,
            "scalp_confirmations": n_confirmed,
            "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
            "scalp_vwap": vwap_price, "scalp_confidence": scalp_conf,
            "scalp_reason": (
                f"ORB LONG. {n_confirmed}/3: " + " | ".join(conf_notes)
                + f". Entry ₹{entry:.2f} | Stop ₹{stop:.2f} | T1 ₹{t1:.2f} (70%) "
                + f"| T2 ₹{t2:.2f} (30% runner). R:R {rr:.1f}× on T1."
                + (f" ⚠️ Nifty soft headwind {abs(nifty_pct_change):.1f}%." if _nifty_soft_long else "")
                + " Hard exit 2:45 PM."
            ),
        }

    # ── SHORT scalp ──────────────────────────────────────────────────────────
    if current_ltp < orb_low:
        if _nifty_adverse_short:
            return {
                **_SCALP_NULL,
                "scalp_signal": "BLOCKED",
                "scalp_direction": "SHORT",
                "scalp_confirmations": 0,
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_reason": (
                    f"NIFTY_BULLISH_EXTREME — Nifty up {nifty_pct_change:.1f}% "
                    f"(hard block > {_cfg.SCALP_NIFTY_HARD_BLOCK_PCT:.1f}%)"
                ),
            }

        confs, conf_notes = _check_confirmations(is_long=False)
        n_confirmed = len(confs)
        if n_confirmed < min_conf:
            return {
                **_SCALP_NULL,
                "scalp_signal": "WATCH", "scalp_direction": "SHORT",
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_vwap": vwap_price, "scalp_confirmations": n_confirmed,
                "scalp_reason": (
                    f"ORB short — only {n_confirmed}/{min_conf} confirmations. "
                    + " | ".join(conf_notes)
                ),
            }

        entry, stop, t1, t2, rr = _build_levels_short()
        if rr < _cfg.SCALP_MIN_RR:
            return {
                **_SCALP_NULL,
                "scalp_signal": "WATCH", "scalp_direction": "SHORT",
                "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
                "scalp_vwap": vwap_price,
                "scalp_reason": f"ORB short — R:R {rr:.1f}× < {_cfg.SCALP_MIN_RR:.1f}× minimum.",
            }

        scalp_conf = _scalp_confidence(
            n_confirmed, rr, atr_pct, orb_range_atr,
            _nifty_soft_short, _vol_pace_ok, vol_surge_at_breakout,
        )
        return {
            "scalp_signal": "SHORT", "scalp_direction": "SHORT",
            "scalp_entry": entry, "scalp_stop": stop,
            "scalp_t1": t1, "scalp_t2": t2, "scalp_rr": rr,
            "scalp_confirmations": n_confirmed,
            "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
            "scalp_vwap": vwap_price, "scalp_confidence": scalp_conf,
            "scalp_reason": (
                f"ORB SHORT. {n_confirmed}/3: " + " | ".join(conf_notes)
                + f". Entry ₹{entry:.2f} | Stop ₹{stop:.2f} | T1 ₹{t1:.2f} (70%) "
                + f"| T2 ₹{t2:.2f} (30% runner). R:R {rr:.1f}× on T1."
                + (f" ⚠️ Nifty soft tailwind for longs {nifty_pct_change:.1f}%." if _nifty_soft_short else "")
                + " Hard exit 2:45 PM."
            ),
        }

    # LTP inside ORB — no signal yet
    return {
        **_SCALP_NULL,
        "scalp_signal": "INSIDE_ORB",
        "scalp_orb_high": orb_high, "scalp_orb_low": orb_low,
        "scalp_vwap": vwap_price,
        "scalp_reason": (
            f"LTP ₹{current_ltp:.2f} inside ORB [{orb_low:.2f}–{orb_high:.2f}]. "
            "Waiting for breakout/breakdown."
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
    nifty_pct_change: float = 0.0,
    today_open: float | None = None,
) -> dict:
    """
    Runs all four setups and returns a single merged dict.
    Safe to call even if df is empty or metrics are partial.

    Pass rsi_buy_max / rsi_sell_min / min_rr from db.get_signal_config() to
    use paper-trade-tuned thresholds instead of the hard-coded defaults.

    nifty_pct_change — today's Nifty intraday % move (from live LTP vs prev close).
    today_open       — today's open price (for gap detection).
    """
    if df is None or df.empty or len(df) < 20:
        return {**_SWING_NULL, **_INTRA_NULL, **_SCALE_NULL, **_SCALP_NULL}

    df = df.sort_values("date").reset_index(drop=True)
    sw = swing_signal(df, metrics)
    it = intraday_signal(df, metrics,
                         rsi_buy_max=rsi_buy_max,
                         rsi_sell_min=rsi_sell_min,
                         min_rr=min_rr,
                         nifty_pct_change=nifty_pct_change,
                         today_open=today_open)
    sc = scaling_signal(df, metrics)
    # Scalping signal is computed live in app.py once ORB + VWAP are available;
    # here we return the null placeholder so the merged dict has consistent keys.
    return {**sw, **it, **sc, **_SCALP_NULL}
