"""tests/test_signals.py — Unit tests for signals.py functions."""
import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import signals
import config


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_daily(n: int = 60, trend: str = "up") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    if trend == "up":
        close = 1000 + np.cumsum(np.abs(rng.normal(0, 2, n)))
    elif trend == "down":
        close = 1200 - np.cumsum(np.abs(rng.normal(0, 2, n)))
    else:
        close = 1000 + rng.normal(0, 2, n)
    df = pd.DataFrame({
        "date":   pd.date_range("2026-01-01", periods=n, freq="B"),
        "open":   close * 0.998,
        "high":   close * 1.006,
        "low":    close * 0.994,
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })
    return df


def make_metrics_long(ltp: float = 1050.0) -> dict:
    """Metrics that should produce a BUY_ABOVE signal."""
    return {
        "ltp":        ltp,
        "ema_20":     980.0,    # ltp > ema_20 → uptrend
        "ema_50":     950.0,
        "ema_200":    900.0,
        "atr_14":     15.0,
        "rsi_14":     58.0,     # not overbought
        "avg_volume": 1_000_000,
        "volume":     1_200_000,
    }


def make_metrics_short(ltp: float = 950.0) -> dict:
    """Metrics that should produce a SELL_BELOW signal."""
    return {
        "ltp":        ltp,
        "ema_20":     1020.0,   # ltp < ema_20 → downtrend
        "ema_50":     1040.0,
        "ema_200":    1000.0,
        "atr_14":     15.0,
        "rsi_14":     42.0,     # not oversold
        "avg_volume": 1_000_000,
        "volume":     1_200_000,
    }


# ── Intraday signal — basic ───────────────────────────────────────────────────

def test_intraday_long_fires_in_uptrend():
    df = make_daily(60, "up")
    metrics = make_metrics_long()
    result = signals.intraday_signal(df, metrics, nifty_pct_change=0.0)
    assert result["intraday_signal"] in ("BUY_ABOVE", "AVOID"), result


def test_intraday_short_fires_in_downtrend():
    df = make_daily(60, "down")
    metrics = make_metrics_short()
    result = signals.intraday_signal(df, metrics, nifty_pct_change=0.0)
    assert result["intraday_signal"] in ("SELL_BELOW", "AVOID"), result


def test_intraday_avoid_on_thin_df():
    df = make_daily(3)
    result = signals.intraday_signal(df, make_metrics_long())
    assert result["intraday_signal"] is None or result["intraday_signal"] == "AVOID"


# ── Nifty + VWAP regime gate ──────────────────────────────────────────────────

def make_metrics_long_strong(ltp: float = 1100.0) -> dict:
    """Metrics with a wide H-L spread so R/R passes easily."""
    return {
        "ltp":        ltp,
        "ema_20":     1000.0,   # ltp >> ema_20
        "ema_50":     950.0,
        "ema_200":    900.0,
        "atr_14":     8.0,      # tight ATR → stop stays close to R1
        "rsi_14":     55.0,
        "avg_volume": 1_000_000,
        "volume":     1_200_000,
    }


def make_daily_wide(n: int = 60) -> pd.DataFrame:
    """Daily bars with a wide H-L spread so pivot range is large enough."""
    rng = np.random.default_rng(99)
    base = 1000.0
    close = base + np.cumsum(np.abs(rng.normal(0, 3, n)))
    df = pd.DataFrame({
        "date":   pd.date_range("2026-01-01", periods=n, freq="B"),
        "open":   close * 0.990,
        "high":   close * 1.020,   # 2% high
        "low":    close * 0.980,   # 2% low  → wide pivots
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    })
    return df


def test_long_blocked_by_regime_gate_when_below_vwap():
    """BUY_ABOVE must be AVOID when Nifty bearish AND stock below VWAP."""
    df = make_daily_wide(60)
    metrics = make_metrics_long_strong(ltp=1050.0)
    # Nifty down -1%, stock (ltp=1050) below vwap=1150 → double headwind
    result = signals.intraday_signal(
        df, metrics,
        nifty_pct_change=-1.0,
        vwap=1150.0,
    )
    if config.INTRADAY_REGIME_GATE:
        assert result["intraday_signal"] == "AVOID", (
            f"Expected AVOID under regime gate, got {result['intraday_signal']}: "
            f"{result.get('intraday_reason')}"
        )
        assert "REGIME BLOCK" in (result.get("intraday_reason") or "")


def test_short_blocked_by_regime_gate_when_above_vwap():
    """SELL_BELOW must be AVOID when Nifty bullish AND stock above VWAP."""
    df = make_daily_wide(60)
    # Force downtrend by making ltp below ema_20
    metrics = {
        "ltp":        950.0,
        "ema_20":     1100.0,   # ltp << ema_20
        "ema_50":     1050.0,
        "ema_200":    1000.0,
        "atr_14":     8.0,
        "rsi_14":     42.0,
        "avg_volume": 1_000_000,
        "volume":     1_200_000,
    }
    # Nifty up +1%, stock (ltp=950) above vwap=900 → double headwind for shorts
    result = signals.intraday_signal(
        df, metrics,
        nifty_pct_change=+1.0,
        vwap=900.0,
    )
    if config.INTRADAY_REGIME_GATE:
        assert result["intraday_signal"] == "AVOID", (
            f"Expected AVOID under regime gate, got {result['intraday_signal']}: "
            f"{result.get('intraday_reason')}"
        )
        assert "REGIME BLOCK" in (result.get("intraday_reason") or "")


def test_long_not_blocked_when_above_vwap_despite_bearish_nifty():
    """Long should NOT be hard-blocked if stock is above VWAP even when Nifty is bearish."""
    df = make_daily(60, "up")
    metrics = make_metrics_long(ltp=1100.0)
    result = signals.intraday_signal(
        df, metrics,
        nifty_pct_change=-1.0,
        vwap=1050.0,  # ltp=1100 is ABOVE vwap → no hard block
    )
    # Should not be a regime block (may still be BUY_ABOVE with nifty warning)
    if result.get("intraday_reason"):
        assert "REGIME BLOCK" not in result["intraday_reason"]


# ── Scalp signal ─────────────────────────────────────────────────────────────

def make_5min_with_breakout(orb_high: float = 1050.0,
                             ltp: float = 1058.0,
                             n_orb_candles: int = 8) -> pd.DataFrame:
    times = pd.date_range("2026-05-15 09:15", periods=75, freq="5min")
    rng = np.random.default_rng(3)
    close = np.full(75, ltp)
    close[:n_orb_candles] = orb_high - 5   # ORB candles below orb_high
    close[n_orb_candles] = ltp             # breakout candle
    df = pd.DataFrame({
        "date":   times,
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": rng.integers(500_000, 1_500_000, 75).astype(float),
    })
    df.loc[n_orb_candles, "volume"] = 2_000_000  # breakout candle has 2× volume
    return df


def test_scalp_null_when_atr_zero():
    """scalping_signal returns null scalp when atr=0 (hard guard)."""
    result = signals.scalping_signal(
        current_ltp=1058.0,
        orb_high=1050.0,
        orb_low=1035.0,
        orb_range=15.0,
        vwap_price=1045.0,
        rsi_5min=58.0,
        atr=0.0,                # zero ATR → should return null
        nifty_pct_change=0.0,
    )
    assert result.get("scalp_signal") is None, f"Expected null, got {result}"


def test_scalp_blocked_when_nifty_very_bearish():
    """Long ORB scalp must not fire when Nifty is very bearish."""
    result = signals.scalping_signal(
        current_ltp=1058.0,
        orb_high=1050.0,
        orb_low=1035.0,
        orb_range=15.0,
        vwap_price=1045.0,
        rsi_5min=60.0,
        atr=12.0,
        nifty_pct_change=-2.0,  # well below NIFTY_GATE_PCT=0.6
        daily_vol_ratio=2.0,
    )
    assert result.get("scalp_signal") != "BUY_ORB", (
        f"Long scalp should be blocked on very bearish Nifty, got {result.get('scalp_signal')}"
    )
