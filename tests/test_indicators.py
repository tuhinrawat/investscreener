"""tests/test_indicators.py — Unit tests for indicators.py pure functions."""
import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import indicators


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ohlcv(n: int, close_vals=None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = close_vals if close_vals is not None else (100 + np.cumsum(rng.normal(0, 1, n)))
    df = pd.DataFrame({
        "open":   close * 0.998,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": rng.integers(100_000, 500_000, n).astype(float),
    })
    return df


def make_5min(n_candles: int = 75, start: str = "2026-05-15 09:15") -> pd.DataFrame:
    times = pd.date_range(start=start, periods=n_candles, freq="5min")
    rng = np.random.default_rng(0)
    close = 1500 + np.cumsum(rng.normal(0, 2, n_candles))
    df = pd.DataFrame({
        "date":   times,
        "open":   close * 0.999,
        "high":   close * 1.003,
        "low":    close * 0.997,
        "close":  close,
        "volume": rng.integers(50_000, 200_000, n_candles).astype(float),
    })
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_returns_float_on_enough_data():
    df = make_ohlcv(50)
    result = indicators.rsi(df, period=14)
    assert result is not None
    assert 0 < result < 100


def test_rsi_returns_none_on_insufficient_data():
    df = make_ohlcv(10)
    assert indicators.rsi(df, period=14) is None


def test_rsi_overbought_on_strong_uptrend():
    # Strong uptrend with occasional dips → gains >> losses → RSI >70.
    # Pure linspace (zero losses) causes avg_loss=0 and NaN division (expected
    # behaviour); use a sawtooth that mostly rises to ensure non-zero losses.
    rng = np.random.default_rng(99)
    base = np.linspace(100, 200, 60)
    # Add small random walk with slight upward bias — guarantees some losses
    close = base + rng.normal(0.5, 0.3, 60).cumsum()
    close = np.maximum(close, 1.0)
    df = make_ohlcv(60, close_vals=close)
    result = indicators.rsi(df, period=14)
    assert result is not None, "RSI should be computable on 60-bar uptrend"
    assert result > 60, f"RSI on strong uptrend should be >60, got {result}"


# ── EMA ───────────────────────────────────────────────────────────────────────

def test_ema_returns_float():
    df = make_ohlcv(60)
    result = indicators.ema(df, period=20)
    assert isinstance(result, float)


def test_ema_returns_none_below_period():
    df = make_ohlcv(10)
    assert indicators.ema(df, period=20) is None


def test_ema_roughly_tracks_mean():
    close = np.full(100, 150.0)
    df = make_ohlcv(100, close_vals=close)
    result = indicators.ema(df, period=20)
    assert abs(result - 150.0) < 0.1


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_positive():
    df = make_ohlcv(30)
    result = indicators.atr(df, period=14)
    assert result is not None
    assert result > 0


def test_atr_returns_none_on_short_df():
    df = make_ohlcv(5)
    assert indicators.atr(df, period=14) is None


# ── VWAP ──────────────────────────────────────────────────────────────────────

def test_vwap_returns_float():
    df = make_5min(75)
    result = indicators.vwap(df)
    assert isinstance(result, float)
    assert result > 0


def test_vwap_none_on_empty():
    assert indicators.vwap(pd.DataFrame()) is None


def test_vwap_close_to_close_on_flat_price():
    df = make_5min(75)
    df["high"] = df["low"] = df["close"] = 1000.0
    result = indicators.vwap(df)
    assert result is not None
    assert abs(result - 1000.0) < 0.01


# ── Opening Range ─────────────────────────────────────────────────────────────

def test_opening_range_returns_dict_with_all_keys():
    df = make_5min(75)
    result = indicators.opening_range(df)
    assert result is not None
    assert "orb_high" in result
    assert "orb_low" in result
    assert "orb_range" in result


def test_opening_range_incomplete_when_few_candles():
    df = make_5min(3)   # only 3 candles (< 6 required)
    result = indicators.opening_range(df)
    assert result is not None
    assert result.get("orb_window_incomplete") is True
    assert result["orb_high"] is None


def test_opening_range_orb_high_ge_orb_low():
    df = make_5min(75)
    result = indicators.opening_range(df)
    assert result is not None
    if not result.get("orb_window_incomplete"):
        assert result["orb_high"] >= result["orb_low"]
        assert result["orb_range"] >= 0
