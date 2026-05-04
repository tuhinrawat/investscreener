"""
indicators.py — Pure functions. No I/O, no API calls, just math.

Why pure functions? Because they're trivially testable and you can run
backtests on historical data without touching Kite. Every function here
takes a DataFrame, returns a value or Series.

Mental model: this file is your "math layer" — same philosophy as
HyperTrader. Math decides, narration happens elsewhere.
"""
import numpy as np
import pandas as pd
import config


# ============================================================
# RETURNS — multi-timeframe % change
# ============================================================
def pct_return(df: pd.DataFrame, lookback_days: int) -> float:
    """
    % return over N trading days.
    df must be sorted ascending by date.
    Returns None if insufficient data.
    """
    if len(df) < lookback_days + 1:
        return None
    today_close = df["close"].iloc[-1]
    past_close = df["close"].iloc[-(lookback_days + 1)]
    if past_close <= 0:
        return None
    return float((today_close / past_close - 1) * 100)


def all_returns(df: pd.DataFrame) -> dict:
    """Returns dict of all timeframe returns."""
    return {
        f"ret_{label.lower()}": pct_return(df, days)
        for label, days in config.TREND_WINDOWS.items()
    }


# ============================================================
# RSI — Wilder's smoothing
# ============================================================
def rsi(df: pd.DataFrame, period: int = 14) -> float:
    """
    Standard RSI. Returns latest value.
    Wilder's smoothing (not simple SMA) — that's the canonical version.
    """
    if len(df) < period + 1:
        return None
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing = exponential with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    val = rsi_series.iloc[-1]
    return float(val) if pd.notna(val) else None


# ============================================================
# EMAs
# ============================================================
def ema(df: pd.DataFrame, period: int) -> float:
    """Exponential moving average — latest value."""
    if len(df) < period:
        return None
    val = df["close"].ewm(span=period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


# ============================================================
# ATR — Average True Range (volatility)
# ============================================================
def atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    True Range = max(H-L, |H-prevC|, |L-prevC|)
    ATR = Wilder's smoothing of TR.
    Used for volatility-adjusted stop-loss sizing.
    """
    if len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    val = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


# ============================================================
# 52-WEEK HIGH/LOW + distances
# ============================================================
def fifty_two_week_levels(df: pd.DataFrame) -> dict:
    """
    Uses last 252 trading days. If less data available, uses what we have.
    Returns high, low, and % distance from each.
    """
    window = df.tail(252)
    if window.empty:
        return {"high_52w": None, "low_52w": None,
                "dist_from_52w_high_pct": None}
    high = float(window["high"].max())
    low = float(window["low"].min())
    ltp = float(df["close"].iloc[-1])
    dist_high = (high - ltp) / ltp * 100 if ltp > 0 else None
    return {
        "high_52w": high,
        "low_52w": low,
        "dist_from_52w_high_pct": dist_high,
    }


# ============================================================
# Support/Resistance proxies — 20-day low/high
# ============================================================
def support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Honest disclaimer: these are NOT 'real' S/R levels. They're rolling
    20-day extremes used as proxies for screening. For actual S/R,
    do manual analysis on shortlisted stocks.
    """
    if len(df) < window:
        return {"support_20d": None, "resistance_20d": None}
    recent = df.tail(window)
    return {
        "support_20d": float(recent["low"].min()),
        "resistance_20d": float(recent["high"].max()),
    }


# ============================================================
# LIQUIDITY — daily turnover and avg volume
# ============================================================
def liquidity_metrics(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Turnover = volume × close, averaged over lookback days.
    Why use close not VWAP? VWAP not in daily candles. Close is good enough
    for screening (the order of magnitude is what matters here).
    """
    if len(df) < lookback:
        return {"avg_turnover_cr": None, "avg_volume": None}
    recent = df.tail(lookback)
    avg_vol = recent["volume"].mean()
    avg_turnover = (recent["volume"] * recent["close"]).mean() / 1e7  # crores
    return {
        "avg_turnover_cr": float(avg_turnover),
        "avg_volume": int(avg_vol),
    }


# ============================================================
# VOLUME EXPANSION — short-term vs long-term avg
# ============================================================
def volume_expansion(df: pd.DataFrame) -> float:
    """
    5D avg volume / 20D avg volume.
    > 1.2 = volume expanding (institutional interest building)
    < 0.8 = volume drying up (avoid)
    """
    if len(df) < 20:
        return None
    vol5 = df["volume"].tail(5).mean()
    vol20 = df["volume"].tail(20).mean()
    if vol20 == 0:
        return None
    return float(vol5 / vol20)


# ============================================================
# COMPOSITE SCORE — the funnel ranker
# ============================================================
def trend_score(returns: dict) -> float:
    """
    Weighted average of timeframe returns, normalized.
    Positive return contributes positive score weighted by its bucket.
    """
    weights = config.TREND_WEIGHTS
    score = 0.0
    total_weight = 0.0
    for label, weight in weights.items():
        ret = returns.get(f"ret_{label.lower()}")
        if ret is not None:
            # Cap extreme values to avoid one outlier dominating
            capped = max(-50, min(50, ret))
            score += capped * weight
            total_weight += weight
    if total_weight == 0:
        return None
    return score / total_weight


def composite_score(
    trend_sc: float,
    rs_vs_nifty: float,
    vol_expansion: float,
) -> float:
    """
    The final ranker. Three components, weighted per config.

    Why these three?
      - Trend = is the stock going up across timeframes?
      - RS    = is it going up FASTER than the index? (alpha)
      - Vol   = is the move backed by participation? (conviction)
    """
    if trend_sc is None:
        return None
    score = config.W_TREND * trend_sc
    if rs_vs_nifty is not None:
        score += config.W_RELATIVE_STRENGTH * rs_vs_nifty
    if vol_expansion is not None:
        # Center around 1.0 — a ratio of 1 = no signal
        score += config.W_VOLUME_EXPANSION * (vol_expansion - 1) * 50
    return float(score)


# ============================================================
# AGGREGATE — runs everything for one stock's full history
# ============================================================
def compute_all(df: pd.DataFrame, nifty_3m_return: float = None) -> dict:
    """
    Single function the pipeline calls per stock.
    df = daily OHLCV sorted ascending.
    Returns flat dict ready to insert into computed_metrics.
    """
    if df.empty or len(df) < 30:
        return None

    df = df.sort_values("date").reset_index(drop=True)

    rets = all_returns(df)
    liq = liquidity_metrics(df, config.LIQUIDITY_LOOKBACK_DAYS)
    levels = fifty_two_week_levels(df)
    sr = support_resistance(df)

    rsi_val = rsi(df, config.RSI_PERIOD)
    ema20 = ema(df, config.EMA_FAST)
    ema50 = ema(df, config.EMA_MID)
    ema200 = ema(df, config.EMA_SLOW)
    atr_val = atr(df, config.ATR_PERIOD)
    vol_exp = volume_expansion(df)
    ltp = float(df["close"].iloc[-1])

    # Distance from 50 EMA
    dist_50ema = (ltp - ema50) / ema50 * 100 if ema50 else None

    # Relative strength: stock 3M return - benchmark 3M return
    rs = None
    if rets["ret_3m"] is not None and nifty_3m_return is not None:
        rs = rets["ret_3m"] - nifty_3m_return

    trend_sc = trend_score(rets)
    comp = composite_score(trend_sc, rs, vol_exp)

    return {
        "ltp": ltp,
        **liq,
        **rets,
        "rs_vs_nifty_3m": rs,
        "vol_expansion_ratio": vol_exp,
        "rsi_14": rsi_val,
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "atr_14": atr_val,
        **levels,
        "dist_from_50ema_pct": dist_50ema,
        **sr,
        "trend_score": trend_sc,
        "composite_score": comp,
    }
