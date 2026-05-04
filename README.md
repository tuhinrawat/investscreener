# NSE Swing Screener

Pre-market dashboard for short-swing (1-3 day) candidate screening on NSE equity.
Pulls data from Kite Connect, computes multi-timeframe momentum + technicals,
ranks via composite score, displays in Streamlit with click-to-chart detail.

## Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Create .env with your Kite Connect creds
cp .env.example .env
# edit .env, paste KITE_API_KEY and KITE_API_SECRET from kite.trade

# 3. First run — will prompt you to login
streamlit run app.py
```

## Daily workflow

**Post-market (after 4:00 PM IST):**
1. Open dashboard → click **Full Rescan** (~3-5 min)
2. Filter results, export shortlist CSV
3. Do manual S/R analysis on top 10 candidates

**Next morning (before 9:15 AM IST):**
1. Open dashboard → click **Quick Refresh** (~10 sec)
2. Check if any shortlisted stocks gapped against you
3. Place orders manually in Kite

## Architecture

```
app.py                    Streamlit UI (refresh buttons, table, chart)
  ↓ calls
data_pipeline.py          Orchestrates fetch + compute
  ↓ uses
kite_client.py            Auth + rate-limited Kite SDK wrapper
indicators.py             Pure math functions (RSI, EMA, returns, score)
db.py                     DuckDB schema + upsert helpers
config.py                 All thresholds and weights
```

## Tunable parameters

All in `config.py`:
- `TREND_WEIGHTS` — change for different hold horizons
- `MIN_AVG_TURNOVER_CR` — liquidity floor
- `W_TREND / W_RELATIVE_STRENGTH / W_VOLUME_EXPANSION` — composite score formula

## Known limits

- **Kite historical API: 3 req/sec.** Full rescan of ~1,500 stocks ≈ 8-10 min.
- **Daily access token expiry:** re-login each day after 6 AM IST
- **Support/resistance levels** are 20-day extremes, not real S/R. Use for screening only.
- **No intraday data** — daily candles only. For intraday signals, use a separate WebSocket-based tool.
