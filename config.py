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
      1. OS environment variable / .env file  (local dev)
      2. Streamlit Cloud Secrets (st.secrets) (cloud deployment via dashboard)
      3. screener_keys.json                   (entered by user in sidebar UI)
    Returns an empty string if not found anywhere.
    """
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st          # noqa: PLC0415 — lazy import OK here
            val = str(st.secrets.get(key, "") or "")
        except Exception:
            pass
    if not val:
        try:
            import json as _json
            _kf = Path(__file__).parent / "screener_keys.json"
            if _kf.exists():
                _stored = _json.loads(_kf.read_text())
                # screener_keys.json uses lowercase keys (kite_api_key / kite_api_secret)
                _map = {"KITE_API_KEY": "kite_api_key", "KITE_API_SECRET": "kite_api_secret"}
                val = str(_stored.get(_map.get(key, key), "") or "")
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

# ============================================================
# PAPER TRADING — virtual capital for signal evaluation
# ============================================================
PAPER_CAPITAL       = 900_000   # ₹9,00,000 virtual capital
PAPER_MAX_POSITIONS = 6         # legacy: kept for reference; capital-based gating now used

# ── Confidence-based capital allocation ──────────────────────────────────────
# Intraday signals are scored 0-10 before auto-triggering.
# Tier thresholds and per-trade capital:
CONFIDENCE_STRONG_MIN   = 8        # 8-10 → STRONG: large allocation
CONFIDENCE_MODERATE_MIN = 6        # 6-7  → MODERATE: standard allocation
CONFIDENCE_MARGINAL_MIN = 5        # 5    → MARGINAL: small allocation, only if capital available
# Below CONFIDENCE_MARGINAL_MIN → LOW: shown in table, NOT auto-traded
PAPER_CAP_STRONG   = 200_000       # ₹2,00,000 per STRONG trade
PAPER_CAP_MODERATE = 150_000       # ₹1,50,000 per MODERATE trade (was the old fixed default)
PAPER_CAP_MARGINAL = 100_000       # ₹1,00,000 per MARGINAL trade
# Total capital hard cap: PAPER_CAPITAL (₹9L) — never exceed regardless of confidence
# Effective max slots: 4 STRONG (₹8L) or 6 MODERATE (₹9L) or 9 MARGINAL (₹9L)
# LTP freshness gate: if last LTP update older than this, pause auto-triggers
LTP_FRESHNESS_SECS = 10

# Daily gain target / trailing cutoff (applies to both paper & real trading)
# Rule:
#   • Below DAILY_TARGET_LOW_PCT  (2%) → always accept new entries
#   • 2% – 5%                         → trailing cutoff = peak_return – DAILY_TRAIL_PCT
#   • At or above DAILY_TARGET_HIGH_PCT (5%) → hard ceiling, block all new entries
# The "peak_return" is the high-water mark of today's realised return.
# Once the live return drops back to (peak – 0.3%) trading is halted for the day.
DAILY_TARGET_LOW_PCT  = 2.0     # activate trailing stop only after this gain
DAILY_TARGET_HIGH_PCT = 5.0     # hard ceiling — never trade past this
DAILY_TRAIL_PCT       = 0.3     # stop trading if return falls this far from peak

# ============================================================
# INTRADAY GATES — market-context filters applied at trigger time
# ============================================================
# Nifty 50 intraday direction gate (applied per signal direction)
#   Long signals are suppressed when Nifty is down more than this %
#   Short signals are suppressed when Nifty is up more than this %
NIFTY_GATE_PCT        = 0.6    # ±0.6% Nifty intraday move triggers direction gate

# Gap-up / gap-down detection
#   If today's open is this far above/below R1/S1, flag as gap entry risk
GAP_WARN_PCT          = 0.8    # 0.8% above R1 = gap-up warning (skip or downgrade confidence)
GAP_SKIP_PCT          = 1.5    # 1.5% above R1 = gap too large, skip signal entirely

# Partial profit booking at T1
#   On hitting T1 (R2 for longs, S2 for shorts), book this fraction of position
PARTIAL_BOOK_RATIO    = 0.60   # 60% exit at T1
# Remaining 40% trails to T2 (R3 / S3) with stop moved to break-even

# ============================================================
# SCALPING — Opening Range Breakout (ORB) strategy
# ============================================================
# Opening range = first N minutes after 9:15 AM IST
SCALP_ORB_MINUTES      = 15    # first 15-min candle(s) define the range

# Scalping targets / stops expressed as multiples of the ORB range width
SCALP_TARGET_MULT      = 1.5   # target  = breakout + 1.5 × ORB_range
SCALP_STOP_MULT        = 0.5   # stop    = breakout – 0.5 × ORB_range (tight)
SCALP_STOP_FLOOR_PCT   = 0.002 # minimum stop = 0.2% below breakout (prevents sub-paise stops)
SCALP_MIN_RR           = 1.8   # minimum R/R to emit a scalp signal

# Scalping requires ≥ this many of the 3 internal confirmations to auto-trade
SCALP_MIN_CONFIRMATIONS = 2    # ORB breakout + VWAP alignment + RSI momentum

# Capital per scalp trade (smaller than intraday — scalping = quick in/out)
SCALP_CAP_PER_TRADE    = 75_000    # ₹75,000 per scalp position
SCALP_MAX_POSITIONS    = 4         # max 4 concurrent scalp trades
SCALP_HARD_EXIT_TIME   = (14, 45)  # (hour, minute) in IST — hard exit all scalps at 2:45 PM

# VWAP band — price must be within this % of VWAP to count as "near VWAP" alignment
VWAP_BAND_PCT          = 0.5   # within 0.5% of VWAP counts as VWAP pull-to trade
