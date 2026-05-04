"""
config.py — Single source of truth for all tunable parameters.

Why centralize? Because when you backtest different weight combinations
you'll change values 50 times. Hardcoding them in screener.py = pain.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """
    Read a secret from (in priority order):
      1. OS environment variable / .env file (local dev)
      2. Streamlit Cloud Secrets (st.secrets — TOML configured in dashboard)
    Returns an empty string if not found anywhere.
    """
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st          # noqa: PLC0415 — lazy import OK here
            val = str(st.secrets.get(key, "") or "")
        except Exception:
            pass
    return val or ""


# ============================================================
# AUTH — credentials from .env (local) or Streamlit Cloud Secrets
# ============================================================
KITE_API_KEY    = _get_secret("KITE_API_KEY")
KITE_API_SECRET = _get_secret("KITE_API_SECRET")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "screener.db"
TOKEN_CACHE = DATA_DIR / "access_token.json"
INSTRUMENTS_CACHE = DATA_DIR / "instruments.csv"

# ============================================================
# UNIVERSE — what stocks even enter the funnel
# ============================================================
EXCHANGE = "NSE"
EXCLUDE_SERIES = ["BE", "BZ", "GB", "IL", "SM"]   # only EQ series — intraday + delivery allowed
INSTRUMENTS_REFRESH_DAYS = 7                       # re-pull instrument master weekly

# NSE index pre-filter: narrows universe before pulling 400-day history.
# "NIFTY500"  → 504 stocks, covers 96% of NSE market cap  (~3.5 min first run)
# "NIFTY200"  → 200 stocks, large + mid cap only            (~1.5 min first run)
# "NIFTY100"  → 100 stocks, large cap only                  (~45 sec first run)
# None        → all ~1800 EQ series stocks                  (~12 min first run)
UNIVERSE_INDEX = "NIFTY500"

# NSE index constituent CSVs (Symbol column used for matching)
NSE_INDEX_URLS = {
    "NIFTY50":   "https://nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100":  "https://nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY200":  "https://nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500":  "https://nseindia.com/content/indices/ind_nifty500list.csv",
}
NSE_INDEX_CACHE = DATA_DIR / "nse_index_symbols.json"  # local cache to avoid re-downloading

# ============================================================
# STAGE 1 — Liquidity gate (kills 75% of universe cheaply)
# ============================================================
MIN_PRICE = 50.0                  # below this = circuit/penny stock risk
MIN_AVG_TURNOVER_CR = 5.0         # ₹5 Cr daily — slippage threshold
MIN_AVG_VOLUME = 100_000          # 1L shares/day
LIQUIDITY_LOOKBACK_DAYS = 20

# ============================================================
# STAGE 2 — Multi-timeframe trend windows (TRADING DAYS, not calendar)
# ============================================================
TREND_WINDOWS = {
    "1Y":  252,
    "6M":  126,
    "3M":  63,
    "1M":  21,
    "5D":  5,
}
HISTORICAL_LOOKBACK_DAYS = 400    # buffer for 252 trading days + holidays

# ============================================================
# STAGE 3 — Composite score weights (SHORT SWING, 1-3 day hold)
# Front-loaded to 5D + 1M because that's where the edge is
# for 1-3 day trades. If you change hold horizon, change these.
# ============================================================
TREND_WEIGHTS = {
    "5D":  30,
    "1M":  30,
    "3M":  20,
    "6M":  12,
    "1Y":  8,
}
# Composite formula weights
W_TREND = 0.50
W_RELATIVE_STRENGTH = 0.30
W_VOLUME_EXPANSION = 0.20

# ============================================================
# STAGE 4 — Technical indicator settings
# ============================================================
RSI_PERIOD = 14
EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
ATR_PERIOD = 14

# Sweet spots for short swing entries
RSI_BULLISH_RANGE = (50, 70)
DISTANCE_FROM_52W_HIGH_MAX = 15.0   # % — within 15% of 52W high = strong
DISTANCE_FROM_50EMA_MAX = 5.0       # % — pullback to 50 EMA = entry zone

# ============================================================
# RATE LIMITING — stay under Kite's caps
# Kite limits: 3 req/sec historical, 1 req/sec quote, 20 req/sec total
# We use 2.5/sec to be safe (network jitter eats margin)
# ============================================================
HIST_REQUESTS_PER_SEC = 2.5
QUOTE_BATCH_SIZE = 250            # OHLC API allows up to 1000, but smaller = faster failures
QUOTE_REQUESTS_PER_SEC = 0.8

# ============================================================
# REFERENCE INDEX — used for relative strength calc
# ============================================================
BENCHMARK_SYMBOL = "NIFTY 50"
BENCHMARK_EXCHANGE = "NSE"

# ============================================================
# NIFTY 50 instrument token (constant) — used for benchmark queries
# ============================================================
NIFTY_50_TOKEN = 256265
