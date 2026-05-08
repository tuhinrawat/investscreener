"""
market_intel.py — India Market Intelligence brief runner.

Reuses the existing OpenRouter (Perplexity/sonar-pro with live web search)
or OpenAI (gpt-4o-search-preview) client from ai_analyst.py.
Runs the full 7-section market intel analysis and parses stock
recommendations into structured dicts for DB storage and UI display.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))

# Model strings
OPENROUTER_MODEL = "perplexity/sonar-pro"      # live web search built in
OPENAI_MODEL     = "gpt-4o-search-preview"     # OpenAI native web search

# Stance labels
BUY        = "BUY"
SHORT      = "SHORT"
AVOID      = "AVOID"
BUY_ON_COND = "BUY_ON_COND"

# Background job executor (1 worker — prevents concurrent duplicate runs)
_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mkt_intel"
)
_active_futures: Dict[str, Future] = {}   # user_id → Future


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ist() -> Tuple[str, str]:
    """Return (date_str, time_str) in IST format."""
    now = datetime.now(_IST)
    return now.strftime("%d %B %Y"), now.strftime("%I:%M %p IST")


def _normalize_symbol(raw: str) -> str:
    """
    Clean up a symbol cell from a markdown table.
    Strips ★ markers, text in parentheses, spaces; uppercases.
    """
    sym = re.sub(r"[★\*]", "", raw).strip()
    sym = re.sub(r"\s*\(.*?\)", "", sym).strip()   # drop "(HDFC Bank)"
    sym = sym.upper().replace(" ", "").replace(".", "-")
    return sym


def _safe(cells: list, idx: int, default: str = "") -> str:
    return cells[idx].strip() if idx < len(cells) else default


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(live_prices: Optional[Dict[str, float]] = None) -> str:
    """
    Construct the full market intel prompt.
    Injects IST timestamp and optional live Kite prices for Section 4 pricing.
    """
    date_str, time_str = _now_ist()

    prices_block = ""
    if live_prices:
        prices_block = (
            f"\n\n## LIVE NSE STOCK PRICES (fetched from Kite at {time_str})\n"
            "These are REAL live prices fetched seconds ago from Zerodha Kite.\n"
            "You MUST use ONLY these values to derive ALL entry triggers, stop losses, "
            "and price levels in Section 4. Do NOT substitute from memory.\n\n"
            "| Symbol | Live Price (₹) |\n|--------|----------------|\n"
        )
        for sym, price in sorted(live_prices.items()):
            prices_block += f"| {sym} | ₹{price:,.2f} |\n"
        prices_block += (
            "\nAny stock in Section 4 that does NOT appear above: "
            "use web search to get current live price before stating any level.\n"
        )

    return f"""Today is {date_str}, {time_str}.{prices_block}

You are an expert India equity market analyst. Execute the following market intelligence skill completely and precisely. Output ALL sections. Do not abbreviate or skip any section.

{_MARKET_INTEL_INSTRUCTIONS}
"""


# ── API runner ────────────────────────────────────────────────────────────────

def run(client, provider: str, prompt: str) -> str:
    """
    Run the market intel prompt using the provided AI client.
    client: openai.OpenAI instance (works for both OpenRouter and OpenAI).
    provider: "openrouter" | "openai"
    """
    model = OPENROUTER_MODEL if provider == "openrouter" else OPENAI_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ── Background job management ─────────────────────────────────────────────────

def start_job(
    user_id: str,
    client,
    provider: str,
    live_prices: Optional[Dict[str, float]] = None,
) -> None:
    """
    Start a market intel job in the background for the given user.
    Call check_job() on subsequent reruns to see if it's done.
    """
    prompt = build_prompt(live_prices=live_prices)

    def _job() -> dict:
        try:
            raw    = run(client, provider, prompt)
            stocks = parse_stocks(raw)
            bias   = parse_bias(raw)
            return {"raw": raw, "stocks": stocks, "bias": bias, "error": None}
        except Exception as exc:
            return {"raw": "", "stocks": [], "bias": {}, "error": str(exc)}

    _active_futures[user_id] = _executor.submit(_job)


def is_running(user_id: str) -> bool:
    """True if a market intel job is currently in progress for this user."""
    f = _active_futures.get(user_id)
    return f is not None and not f.done()


def check_job(user_id: str) -> Optional[dict]:
    """
    Check if the background job has finished.
    Returns the result dict if done, None if still running or not started.
    Clears the future on success.
    """
    f = _active_futures.get(user_id)
    if f is None:
        return None
    if not f.done():
        return None
    _active_futures.pop(user_id, None)
    return f.result()


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_stocks(raw: str) -> List[Dict[str, Any]]:
    """Parse all four Section 4 stance tables from the AI output."""
    results: List[Dict[str, Any]] = []
    results.extend(_extract_table(raw, BUY))
    results.extend(_extract_table(raw, SHORT))
    results.extend(_extract_table(raw, AVOID))
    results.extend(_extract_table(raw, BUY_ON_COND))
    return results


def parse_bias(raw: str) -> dict:
    """Extract Section 5 composite directional bias and confidence."""
    bias       = "NEUTRAL"
    confidence = "MEDIUM"

    m_bias = re.search(r"Overall Market Bias\s*[:\|]\s*([A-Z][A-Z\s]*)", raw, re.IGNORECASE)
    if m_bias:
        # Capture only to end of line (stop at newline / Confidence / bracket)
        raw_bias = m_bias.group(1).split("\n")[0].strip().upper().rstrip(".")
        # Strip trailing CONFIDENCE or other keywords bled in
        raw_bias = re.split(r"\bCONFIDENCE\b|\bREASONING\b", raw_bias)[0].strip()
        bias = raw_bias or bias

    m_conf = re.search(r"\bConfidence\s*[:\|]\s*(HIGH|MEDIUM|LOW)\b", raw, re.IGNORECASE)
    if m_conf:
        confidence = m_conf.group(1).strip().upper()

    return {"bias": bias, "confidence": confidence}


def _extract_table(raw: str, stance: str) -> List[Dict[str, Any]]:
    """Find a stance section in the raw text and parse all table rows from it."""
    # Header patterns that identify each stance section
    _HDR: Dict[str, List[str]] = {
        BUY:        [r"📗[^\n]*BUY[^\n]*Enter", r"\*\*📗 BUY", r"^#+\s*BUY —"],
        SHORT:      [r"📕[^\n]*SHORT",           r"\*\*📕 SHORT", r"^#+\s*SHORT —"],
        AVOID:      [r"📙[^\n]*AVOID",           r"\*\*📙 AVOID", r"^#+\s*AVOID —"],
        BUY_ON_COND:[r"📘[^\n]*BUY ON COND",    r"\*\*📘 BUY ON COND", r"^#+\s*BUY ON CONDITION"],
    }
    # End-of-section markers (next stance or section break)
    _END: Dict[str, str] = {
        BUY:        r"📕|📙|📘|SECTION 5|━━━",
        SHORT:      r"📙|📘|SECTION 5|━━━",
        AVOID:      r"📘|SECTION 5|━━━",
        BUY_ON_COND:r"SECTION 5|━━━|^---",
    }

    section = None
    for pat in _HDR.get(stance, []):
        m = re.search(pat, raw, re.IGNORECASE | re.MULTILINE)
        if m:
            start = m.start()
            end_m = re.search(_END.get(stance, r"SECTION 5"), raw[start + 30:], re.IGNORECASE)
            end   = start + 30 + (end_m.start() if end_m else len(raw))
            section = raw[start:end]
            break

    if not section:
        return []

    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip header and separator rows
        if re.search(r"Stock \(NSE\)|Symbol|-----", line, re.IGNORECASE):
            continue
        cells = [c.strip() for c in line.split("|")[1:]]  # strip leading |
        cells = [c for c in cells if c]                    # drop empties
        if len(cells) < 2:
            continue

        sym = _normalize_symbol(cells[0])
        if len(sym) < 2 or len(sym) > 20:
            continue

        row = _make_row(sym, stance, cells)
        if row:
            rows.append(row)

    return rows


def _make_row(sym: str, stance: str, cells: list) -> Optional[Dict[str, Any]]:
    """Build a standardised stock dict from parsed table cells."""
    base = {
        "tradingsymbol":    sym,
        "stance":           stance,
        "sector":           _safe(cells, 1),
        "fundamental_reason": _safe(cells, 2),
        "entry_trigger":    "",
        "stop_loss":        "",
        "conviction":       "MED",
        "condition_required": "",
        "alert_level":      "",
        "expected_move":    "",
    }
    if stance == BUY:
        base.update({
            "entry_trigger": _safe(cells, 3),
            "stop_loss":     _safe(cells, 4),
            "conviction":    _safe(cells, 5, "MED").upper()[:10],
        })
    elif stance == SHORT:
        base.update({
            "entry_trigger": _safe(cells, 3),
            "stop_loss":     _safe(cells, 4),
            "conviction":    _safe(cells, 5, "MED").upper()[:10],
        })
    elif stance == AVOID:
        base["condition_required"] = _safe(cells, 3)
    elif stance == BUY_ON_COND:
        base.update({
            "condition_required": _safe(cells, 3),
            "alert_level":        _safe(cells, 4),
            "expected_move":      _safe(cells, 5),
        })
    return base


# ── Overlap analysis ──────────────────────────────────────────────────────────

def compute_overlap(
    intel_stocks: List[Dict[str, Any]],
    screener_df,          # pd.DataFrame with tradingsymbol + intraday_signal columns
) -> List[Dict[str, Any]]:
    """
    For each intel stock, check if the same symbol exists in the screener signals.
    Returns enriched intel_stocks with two extra fields:
      overlap_signal   : "BUY_ABOVE" | "SELL_BELOW" | None
      overlap_type     : "SAME_DIR" | "OPPOSITE_DIR" | "AVOID_WARNING" | None
      confidence_delta : +2 / -2 / 0
    """
    if screener_df is None or screener_df.empty or "tradingsymbol" not in screener_df.columns:
        for s in intel_stocks:
            s.setdefault("overlap_signal", None)
            s.setdefault("overlap_type", None)
            s.setdefault("confidence_delta", 0)
        return intel_stocks

    sig_map: Dict[str, str] = {}
    for _, row in screener_df.iterrows():
        sym = str(row.get("tradingsymbol", "")).upper()
        sig = str(row.get("intraday_signal", "") or "")
        if sym and sig:
            sig_map[sym] = sig

    for s in intel_stocks:
        sym   = s["tradingsymbol"].upper()
        sig   = sig_map.get(sym)
        stance = s["stance"]

        if not sig:
            s["overlap_signal"]   = None
            s["overlap_type"]     = None
            s["confidence_delta"] = 0
            continue

        s["overlap_signal"] = sig

        if stance == AVOID:
            s["overlap_type"]     = "AVOID_WARNING"
            s["confidence_delta"] = -2
        elif stance == BUY and sig == "BUY_ABOVE":
            s["overlap_type"]     = "SAME_DIR"
            s["confidence_delta"] = +2
        elif stance == SHORT and sig == "SELL_BELOW":
            s["overlap_type"]     = "SAME_DIR"
            s["confidence_delta"] = +2
        elif stance in (BUY, SHORT):
            s["overlap_type"]     = "OPPOSITE_DIR"
            s["confidence_delta"] = -1
        else:
            s["overlap_type"]     = None
            s["confidence_delta"] = 0

    return intel_stocks


# ── Full market intel prompt (7-section structure) ────────────────────────────

_MARKET_INTEL_INSTRUCTIONS = """\
---
name: india-market-intel
description: >
  Generates a live India market intelligence brief by researching all global macro signals,
  mapping them through the cheat sheet transmission chains, and producing actionable sector
  calls with specific NSE stock symbols.
---

# India Market Intelligence Skill

## What This Skill Does

Researches live global macro data from primary sources, applies the Master Cheat Sheet
transmission framework, and outputs a structured daily intelligence brief with:

1. Live signal readings across all 6 macro dimensions
2. Active trigger identification (which cheat sheet rows are firing today)
3. Full transmission chain walkthrough for each active trigger
4. Sector winners and losers with specific NSE-listed stock symbols
5. Today's directional bias and key levels to watch

---

## Step-by-Step Execution Protocol

### STEP 1 — Research All Live Signals in Parallel

Search for current values of ALL of the following. Use web search for each cluster.
Do not skip any — missing data produces incomplete analysis.

**Cluster A — Global Monetary:**
- US Fed funds rate (current target range)
- Next FOMC meeting date and market expectations (CME FedWatch)
- US 10-year Treasury yield (current %)
- DXY Dollar Index (current level)
- Fed stance language from latest statement (hawkish/neutral/dovish)

**Cluster B — Commodities:**
- Brent crude oil price (current $/barrel)
- WTI crude oil price (current $/barrel)
- Gold price (current $/oz)
- Key commodity events (OPEC decisions, supply disruptions, geopolitical)

**Cluster C — India Domestic:**
- USD/INR rate (current)
- RBI repo rate (current %) and stance
- Latest RBI MPC decision and next meeting date
- Nifty 50 current level / recent trend
- FII net flows — last 5 days (from NSE/BSE provisional data or news)
- India VIX current level
- India CPI inflation (latest)
- India GDP growth (latest quarter)

**Cluster D — China & Geopolitics:**
- China Caixin/NBS Manufacturing PMI (latest reading)
- Any major China economic news
- Active geopolitical conflicts affecting trade routes, oil, commodities
- Any US-China developments

**Cluster E — India-Specific Events:**
- IMD monsoon forecast (if June-September season approaching)
- Any upcoming RBI MPC meeting (within 30 days)
- Any Union Budget announcements or upcoming
- Any major Indian corporate earnings in the week
- Any SEBI regulatory actions
- State elections or general election news

**Cluster F — Sector Signals:**
- Latest Nifty sectoral performance (Bank, IT, Metal, FMCG, Pharma, Realty)
- Any major PLI scheme announcements
- Any sector-specific regulatory or policy news

---

### STEP 2 — Score Each Cheat Sheet Row

After research, map data to each of the 17 cheat sheet triggers. Score each:
- **ACTIVE 🔴** — The trigger condition is clearly firing right now
- **PARTIAL 🟡** — Conditions partially met or trend building
- **INACTIVE ⚪** — Trigger not firing currently
- **EASING 🟢** — Previously active trigger now resolving (potential reversal play)

**The 17 Cheat Sheet Triggers:**

| # | Trigger | Active Condition |
|---|---------|-----------------|
| 1 | US Fed Hikes Rates | Rate hike in last 60 days OR hike signalled at next meeting |
| 2 | US Fed Cuts Rates | Rate cut in last 60 days OR cut probability >60% at next meeting |
| 3 | DXY above 105 | DXY reading above 105 |
| 4 | Crude Oil Rises ($90→$110) | Brent above $95/barrel |
| 5 | Crude Oil Falls ($90→$70) | Brent below $75/barrel |
| 6 | Gold Rises | Gold up >5% in last 30 days AND above $2,500 |
| 7 | China Booms | China PMI >52 AND commodity prices rising |
| 8 | China Slows/Dumps | China PMI <49 OR active dumping news in metals/chemicals |
| 9 | US-China Tensions Rise | New tariffs, sanctions, or geopolitical escalation in 60 days |
| 10 | Good Monsoon | IMD forecast ≥104% of LPA OR actual rainfall above normal |
| 11 | Bad Monsoon | IMD forecast <96% of LPA OR actual deficient rainfall |
| 12 | RBI Hikes Repo Rate | Rate hike in last 60 days OR clearly signalled |
| 13 | RBI Cuts Repo Rate | Rate cut in last 60 days OR clearly signalled |
| 14 | Union Budget High Capex | Budget announced with capex increase >15% YoY |
| 15 | India Election | Within 90 days of election announcement OR active election |
| 16 | India GDP Beats | Latest GDP print >0.3% above consensus |
| 17 | Geopolitical War/Crisis | Active conflict affecting oil, trade routes, or global risk |

---

### STEP 3 — Build the Intelligence Brief

Structure the output as follows. Every section is required. Do not abbreviate.

---

## OUTPUT FORMAT

```
═══════════════════════════════════════════════════════════════
INDIA MARKET INTELLIGENCE BRIEF
Date: [TODAY'S DATE]  |  Prepared: [TIME]  |  Valid for: Current session
═══════════════════════════════════════════════════════════════

SECTION 1 — LIVE SIGNAL DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Signal | Current Value | vs Threshold | Assessment |
|--------|--------------|-------------|------------|
| US Fed Rate | X.XX–X.XX% | Threshold | HOLD/HIKE/CUT |
| DXY | XXX.X | >105 = headwind | ABOVE/BELOW |
| US 10Y Yield | X.XX% | >4.5% = FII pressure | HIGH/NORMAL |
| Brent Crude | $XXX | >$95 = headwind | HOT/NORMAL/COOL |
| Gold | $X,XXX | Rising = risk-off | RISING/FALLING |
| USD/INR | XX.XX | >90 = weak rupee | STRONG/WEAK/RECORD |
| India VIX | XX.X | >20 = volatile | LOW/ELEVATED/HIGH |
| RBI Repo | X.XX% | — | HOLD/CUT/HIKE |
| Nifty 50 | XX,XXX | — | LEVEL + TREND |
| FII Flow (5d) | ₹XX,XXX Cr | — | NET BUY/SELL |

---

SECTION 2 — ACTIVE TRIGGERS TODAY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[List triggers with Status, Evidence, Cheat Sheet Row]

---

SECTION 3 — TRANSMISSION CHAIN ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[For each active trigger, walk through transmission chain with numbered domino steps]

---

SECTION 4 — TRADE STANCE & STOCK-LEVEL TRIGGERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: Before writing ANY price level, fetch live prices from web search for every stock.
Derive ALL entry triggers, stop losses, and levels as offsets from live price.

**📗 BUY — Enter Now (with technical confirmation)**

| Stock (NSE) | Sector | Fundamental Reason | Technical Entry Trigger | Stop Loss | Conviction |
|-------------|--------|--------------------|------------------------|-----------|------------|
[Minimum 5 stocks, maximum 10. Nifty 50 stocks first.]

**📕 SHORT — Active Short Setup**

| Stock (NSE) | Sector | Fundamental Reason | Technical Breakdown Trigger | Stop Loss | Conviction |
|-------------|--------|--------------------|----------------------------|-----------|------------|
[Maximum 5 shorts. Only when BOTH fundamental deterioration AND technical breakdown confirm.]

**📙 AVOID — Environment Unfavourable (No Entry, No Short)**

| Stock (NSE) | Sector | Why Avoid | What Would Change This Call |
|-------------|--------|-----------|-----------------------------|
[5–8 stocks with structural headwinds but no trade edge.]

**📘 BUY ON CONDITION — Set Alert, Don't Enter Yet**

| Stock (NSE) | Sector | Fundamental Setup | Condition Required | Alert Level | Expected Move |
|-------------|--------|-------------------|--------------------|-------------|---------------|
[Minimum 4 entries. Conditions must be specific enough to set an alert.]

---

SECTION 5 — COMPOSITE DIRECTIONAL BIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Market Bias: [BULLISH / MILDLY BULLISH / NEUTRAL / MILDLY BEARISH / BEARISH]
Confidence: [HIGH / MEDIUM / LOW]

Reasoning: [2-3 sentences]
Positive forces: [List]
Negative forces: [List]

---

SECTION 6 — USER ACTION CENTRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ MARKET LEVELS
| Index/Asset | Current | Key Support | Key Resistance | If Breaks Support | If Breaks Resistance |
[Nifty 50, BankNifty, USD/INR, Brent Crude, India VIX]

▶ ALERTS TO SET RIGHT NOW
🔔 Alert 1: [Asset] crosses [Level] → [Action]
[Minimum 4 alerts]

▶ CALENDAR — Events This Week
| Date | Time (IST) | Event | Expected Impact | Your Prep |

▶ TODAY'S TRADE ACTIONS
1. □ [Most urgent — specific stock, entry level, size]
...up to 7 numbered actions

▶ RISK RULES TODAY
Position Size: [% of normal]
Overnight Hold Rule: [Hold/Reduce/No overnight]
Stop Loss Discipline: [Tight/Normal/Wide]
Maximum Simultaneous Positions: [Number]

▶ WHAT WOULD CHANGE THE WHOLE PICTURE
Bullish flip: [Event + Nifty target + stocks to buy]
Bearish flip: [Event + Nifty target + stocks to short/exit]

---

SECTION 7 — DATA SOURCES & CONFIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[List data sources with date/time accessed. Flag stale (>24h) with ⚠️]
Overall data freshness: [X/18 signals confirmed live today]
```

---

### STEP 4 — Quality Checks Before Delivering

1. Section 4 has all four stances populated (BUY, SHORT, AVOID, BUY ON CONDITION)
2. Every stock in Section 4 has BOTH Fundamental Reason AND Technical Trigger
3. Section 6 Action Centre is self-contained
4. Numbers are cited, not estimated
5. Competing triggers for same stock → AVOID
6. BUY ON CONDITION has specific alert levels (not vague "when oil falls")
"""
