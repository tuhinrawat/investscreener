"""
ai_analyst.py — STOCKLENS AI analysis engine.

Supports OpenAI (gpt-4o) and OpenRouter (perplexity/sonar-pro with web search).
OpenRouter takes preference when both keys are provided.

API keys are persisted locally in screener_keys.json (git-ignored).
"""
import re
import json
import os
import time
from datetime import datetime
from typing import Optional

_KEYS_FILE = os.path.join(os.path.dirname(__file__), "screener_keys.json")

# ── STOCKLENS SYSTEM PROMPT ──────────────────────────────────────────────────
STOCKLENS_SYSTEM = """You are STOCKLENS — a forensic stock analyst for Indian equity markets (NSE/BSE).

Your mandate: When given a stock ticker and timeframe, produce a structured, data-driven, action-oriented investment brief by researching publicly available information across three signal layers: Fundamental, Technical, and Social/Sentiment. You are not a financial advisor — you are an analyst. Your job is to surface signal, assign confidence, and state a directional thesis with clear logic.

## OPERATING PRINCIPLES

1. Evidence-first: Every claim must trace to a source. If you cannot find data, say "data unavailable" — never interpolate or assume.
2. No false precision: Scores are ranges, not guarantees. Say "high confidence" not "95.3% probability".
3. Structured brevity: Each section has a hard word limit. No padding, no generic statements.
4. Action-oriented: Every section ends with an implication — what does this mean for the trader/investor?
5. Conflict surfacing: If fundamental and technical signals conflict, flag it explicitly. Do not paper over contradictions.
6. Recency bias: Prioritize information from the last 90 days. Flag stale data (>6 months old) when you use it.

## ANALYSIS FRAMEWORK — RUN IN THIS SEQUENCE

### LAYER 1: FUNDAMENTAL ANALYSIS (Company Health)
Extract and report:
□ Revenue growth (YoY, QoQ) — flag if growth is decelerating
□ PAT growth — compare to revenue growth (margin trend)
□ Debt-to-Equity ratio — flag if D/E > 1.5
□ Promoter holding % — flag if declining quarter-on-quarter
□ Promoter pledging % — flag if >20% is pledged
□ P/E vs sector average — cheap/fair/expensive verdict
□ Return on Equity (ROE) — flag if <15% for non-financial companies
□ Free Cash Flow trend — positive/negative/improving/deteriorating
□ Recent management guidance, concalls, investor day commentary

FUNDAMENTAL SCORE: [1-10] with 2-line rationale
IMPLICATION: [Bull/Neutral/Bear on fundamentals with 1 reason]

### LAYER 2: TECHNICAL ANALYSIS (Price Action & Structure)
Assess and report:
□ Current price vs 50-DMA and 200-DMA
□ RSI-14 reading
□ Price pattern: breakout/consolidation/downtrend/reversal
□ 52-week range position
□ Volume confirmation
□ Key support and resistance levels
□ Delivery % (institutional accumulation signal)

TECHNICAL SCORE: [1-10] with 2-line rationale
IMPLICATION: [Entry/Wait/Avoid with specific price zones]

### LAYER 3: SOCIAL & SENTIMENT ANALYSIS
Assess and report:
□ Recent news sentiment — cite key headline
□ Analyst consensus: Buy/Hold/Sell + average target price
□ FII holding trend: Increasing/Decreasing/Stable
□ DII holding trend
□ Bulk/block deals
□ Social media buzz — hype risk or genuine accumulation?
□ Any pending regulatory, legal, or ESG risks

SENTIMENT SCORE: [1-10] with 2-line rationale
IMPLICATION: [Institutions accumulating/distributing]

### LAYER 4: MACRO & SECTOR CONTEXT
Assess:
□ Is the sector in favor or out of favor?
□ Macro tailwinds/headwinds for this sector
□ Stock performance vs sector index

MACRO SCORE: [1-10] with 1-line rationale

### LAYER 5: COMPOSITE VERDICT
Weights: Fundamental 35% · Technical 30% · Sentiment 20% · Macro 15%

| Score | Verdict |
|-------|---------|
| 8–10  | STRONG BUY |
| 6.5–7.9 | BUY |
| 5–6.4 | WATCHLIST |
| 3–4.9 | AVOID |
| 0–2.9 | EXIT |

### OUTPUT FORMAT — MANDATORY

═══════════════════════════════════════
STOCKLENS BRIEF: [TICKER] | [DATE] | [TIMEFRAME]
═══════════════════════════════════════

📊 SNAPSHOT
Current Price: ₹XXX | 52W Range: ₹XX–₹XX
Market Cap: ₹X,XXX Cr | Sector: [SECTOR]
P/E: XX.X | ROE: XX% | D/E: X.X

🏗 FUNDAMENTAL [X/10]
[3–4 bullet points]
→ Implication: [1 line]

📈 TECHNICAL [X/10]
[3–4 bullet points]
→ Entry zone: ₹XXX–₹XXX | Stop: ₹XXX | Target: ₹XXX

📣 SENTIMENT [X/10]
[3–4 bullet points]
→ Implication: [1 line]

🌍 MACRO [X/10]
[2 bullet points]

═══════════════════════════════════════
⚡ VERDICT: [STRONG BUY / BUY / WATCHLIST / AVOID / EXIT]
COMPOSITE: [X/10]
CONFIDENCE: [HIGH / MEDIUM / LOW]
HORIZON: [Short 1–4 wks / Medium 1–3 months / Long 3–12 months]
═══════════════════════════════════════

⚠️ KEY RISKS
1. [Risk 1]
2. [Risk 2]
3. [Risk 3]

🔍 WATCH TRIGGERS
- Bull case unlocked if: [specific event/price level]
- Bear case confirmed if: [specific event/price level]
═══════════════════════════════════════

## CONSTRAINTS
- Never fabricate data. If a metric is unavailable, write "N/A — data not found".
- Never give a price prediction without a stop loss.
- Flag any potential conflict between layers.
- Maximum output length: 600 words.
- Do not hedge with generic disclaimers inside the brief.
"""

# ── Key persistence ──────────────────────────────────────────────────────────

def load_keys() -> dict:
    if os.path.exists(_KEYS_FILE):
        try:
            return json.load(open(_KEYS_FILE))
        except Exception:
            pass
    return {"openai_key": "", "openrouter_key": ""}


def save_keys(openai_key: str, openrouter_key: str) -> None:
    with open(_KEYS_FILE, "w") as f:
        json.dump({"openai_key": openai_key, "openrouter_key": openrouter_key}, f)


# ── Client factory ───────────────────────────────────────────────────────────

def get_client(openai_key: str = "", openrouter_key: str = ""):
    """
    Returns (client, provider_str) or (None, None).
    OpenRouter takes priority; both use the openai SDK (same interface).
    """
    from openai import OpenAI
    if openrouter_key and openrouter_key.strip():
        return (
            OpenAI(
                api_key=openrouter_key.strip(),
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title":      "NSE Screener STOCKLENS",
                },
            ),
            "openrouter",
        )
    if openai_key and openai_key.strip():
        return OpenAI(api_key=openai_key.strip()), "openai"
    return None, None


def _model_for(provider: str) -> str:
    """Best model per provider — prioritise web-search capable."""
    if provider == "openrouter":
        return "perplexity/sonar-pro"   # live web search built in
    return "gpt-4o"                     # latest capable OpenAI model


# ── Score / field parser ─────────────────────────────────────────────────────

def _parse(text: str, field: str, cast, default):
    m = re.search(rf'{re.escape(field)}[\s:]+([^\n]+)', text, re.IGNORECASE)
    if not m:
        return default
    raw = m.group(1).strip()
    try:
        if cast is float:
            return float(raw.split("/")[0].strip())
        return cast(raw)
    except Exception:
        return default


# ── Core analysis runner ─────────────────────────────────────────────────────

def run_stocklens(
    symbol:    str,
    stock_row: dict,
    client,
    provider:  str,
    timeframe: str = "Medium (1–3 months)",
) -> dict:
    """
    Run STOCKLENS analysis for one stock.
    Returns dict with: ai_score, ai_verdict, ai_confidence, ai_brief, ai_analyzed_at, error.
    """
    # ── Build context from screener metrics ─────────────────────────────────
    def _fmt(v, fmt="{:.2f}", fb="N/A"):
        if v is None:
            return fb
        try:
            return fmt.format(float(v))
        except Exception:
            return fb

    ctx = "\n".join([
        f"TICKER: {symbol} (NSE)",
        f"Company: {stock_row.get('company_name', symbol)}",
        f"Current Price (LTP): ₹{_fmt(stock_row.get('ltp'), '₹{:,.2f}')}",
        f"52W High: ₹{_fmt(stock_row.get('high_52w'), '{:,.2f}')} | "
        f"52W Low: ₹{_fmt(stock_row.get('low_52w'), '{:,.2f}')}",
        f"Distance from 52W High: {_fmt(stock_row.get('dist_from_52w_high_pct'), '{:.1f}')}%",
        f"RSI-14: {_fmt(stock_row.get('rsi_14'), '{:.1f}')}",
        f"Returns — 5D: {_fmt(stock_row.get('ret_5d'),'{:+.2f}')}% | "
        f"1M: {_fmt(stock_row.get('ret_1m'),'{:+.2f}')}% | "
        f"3M: {_fmt(stock_row.get('ret_3m'),'{:+.2f}')}% | "
        f"6M: {_fmt(stock_row.get('ret_6m'),'{:+.2f}')}% | "
        f"1Y: {_fmt(stock_row.get('ret_1y'),'{:+.2f}')}%",
        f"Avg Daily Turnover: ₹{_fmt(stock_row.get('avg_turnover_cr'),'{:.1f}')} Cr",
        f"Volume Expansion (5D/20D): {_fmt(stock_row.get('vol_expansion_ratio'),'{:.2f}')}×",
        f"Mathematical Composite Score: {_fmt(stock_row.get('composite_score'),'{:.1f}')}",
        f"Trend Score: {_fmt(stock_row.get('trend_score'),'{:.1f}')}",
        f"RS vs Nifty (3M): {_fmt(stock_row.get('rs_vs_nifty_3m'),'{:+.2f}')}%",
        f"Swing Signal: {stock_row.get('swing_signal','N/A')} | "
        f"Intraday Signal: {stock_row.get('intraday_signal','N/A')}",
    ])

    user_msg = f"""Analyse this NSE stock using the full STOCKLENS framework.

SCREENER DATA (use as technical context):
{ctx}

Today's date: {datetime.now().strftime('%d %b %Y')}
Requested timeframe: {timeframe}

Run all 5 layers. Use web search to fetch latest fundamentals, news, FII/DII flows, analyst ratings, and sector outlook.

After your brief, output EXACTLY these three machine-readable lines (no extra text on those lines):
AI_SCORE: X.X/10
AI_VERDICT: STRONG BUY|BUY|WATCHLIST|AVOID|EXIT
AI_CONFIDENCE: HIGH|MEDIUM|LOW"""

    model = _model_for(provider)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STOCKLENS_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1400,
            temperature=0.2,
        )
        brief = resp.choices[0].message.content
    except Exception as exc:
        return {"error": str(exc), "ai_score": None, "ai_verdict": "N/A",
                "ai_confidence": "N/A", "ai_brief": "", "ai_analyzed_at": None}

    ai_score = _parse(brief, "AI_SCORE", float, 5.0)
    ai_score = min(10.0, max(0.0, ai_score))

    return {
        "ai_score":       round(ai_score, 1),
        "ai_verdict":     _parse(brief, "AI_VERDICT",     str, "WATCHLIST").strip().upper(),
        "ai_confidence":  _parse(brief, "AI_CONFIDENCE",  str, "LOW").strip().upper(),
        "ai_brief":       brief,
        "ai_analyzed_at": datetime.now().isoformat(),
        "error":          None,
    }


# ── Batch runner ─────────────────────────────────────────────────────────────

def batch_analyze(
    rows:              list[dict],
    client,
    provider:          str,
    stale_hours:       int  = 24,
    progress_callback       = None,
    min_delay_secs:    float = 1.5,   # stay well under rate limits
) -> list[dict]:
    """
    Analyze a list of stock rows. Skips rows whose ai_analyzed_at is fresher
    than stale_hours. Returns list of result dicts keyed by tradingsymbol.
    """
    results = []
    total   = len(rows)
    for i, row in enumerate(rows):
        sym = row.get("tradingsymbol", "?")

        # Freshness check — skip if already analyzed recently
        analyzed_at = row.get("ai_analyzed_at")
        if analyzed_at:
            try:
                age = (datetime.now() - datetime.fromisoformat(str(analyzed_at))).total_seconds()
                if age < stale_hours * 3600:
                    if progress_callback:
                        progress_callback(i, total, sym, skipped=True)
                    results.append({"tradingsymbol": sym, "skipped": True})
                    continue
            except Exception:
                pass

        if progress_callback:
            progress_callback(i, total, sym, skipped=False)

        result = run_stocklens(sym, row, client, provider)
        result["tradingsymbol"] = sym
        results.append(result)

        # Polite delay between API calls
        if i < total - 1:
            time.sleep(min_delay_secs)

    return results


# ── Verdict badge colours (for UI) ───────────────────────────────────────────

VERDICT_COLOR = {
    "STRONG BUY": "#22c55e",
    "BUY":        "#86efac",
    "WATCHLIST":  "#f59e0b",
    "AVOID":      "#ef4444",
    "EXIT":       "#dc2626",
    "N/A":        "#64748b",
}

VERDICT_EMOJI = {
    "STRONG BUY": "🚀",
    "BUY":        "📈",
    "WATCHLIST":  "👀",
    "AVOID":      "⛔",
    "EXIT":       "🔴",
    "N/A":        "—",
}
