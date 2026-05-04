"""
charts.py — Stock Analysis Console
4 purpose-built Plotly chart views for the stock detail panel.

Each function takes (df: DataFrame of OHLCV, stock_row: Series of computed metrics)
and returns a Plotly Figure ready for st.plotly_chart().

Chart 1: chart_trend_canvas   — 1-year view, Weinstein stage, golden/death cross
Chart 2: chart_momentum_lab   — 3-month view, Bollinger Bands, RSI divergence, MACD
Chart 3: chart_trade_setup    — 20-day view, pivot levels, signal overlays, ATR bands
Chart 4: chart_market_structure — 6-month view, auto S/R zones, Volume Profile
"""

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── colour palette (dark theme) ────────────────────────────────────────────
C = {
    "green":        "#22c55e",
    "green_muted":  "#86efac",
    "red":          "#ef4444",
    "red_muted":    "#fca5a5",
    "blue":         "#3b82f6",
    "purple":       "#8b5cf6",
    "amber":        "#f59e0b",
    "teal":         "#14b8a6",
    "white":        "#f1f5f9",
    "slate":        "#64748b",
    "bg":           "#0f172a",
    "panel":        "#1e293b",
    "grid":         "#1e293b",
    "lime":         "#84cc16",
    "pink":         "#ec4899",
}

_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["bg"],
    # Legend sits below the chart — avoids overlapping subplot titles
    margin=dict(l=10, r=10, t=48, b=64),
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.06,
        xanchor="center", x=0.5,
        font=dict(size=11),
        bgcolor="rgba(15,23,42,0.0)",  # transparent so it doesn't box over the chart
        itemsizing="constant",
        tracegroupgap=4,
    ),
    font=dict(color="#e2e8f0"),
    # TradingView-style: drag = pan, scroll wheel = zoom
    dragmode="pan",
    # Crosshair spike lines across all axes
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(15,23,42,0.92)",
        bordercolor="#334155",
        font=dict(color="#e2e8f0", size=12),
        align="left",
    ),
    modebar=dict(
        bgcolor="rgba(15,23,42,0.8)",
        color="#94a3b8",
        activecolor="#3b82f6",
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _bar_seconds(df: pd.DataFrame) -> float:
    """Return the typical bar duration in seconds. Used to detect intraday data."""
    if len(df) < 2:
        return 86400  # assume daily
    dates = pd.to_datetime(df["date"].values)
    diffs = [(dates[i+1] - dates[i]).total_seconds() for i in range(min(5, len(dates)-1))
             if (dates[i+1] - dates[i]).total_seconds() > 0]
    return min(diffs) if diffs else 86400

def _is_intraday_df(df: pd.DataFrame) -> bool:
    """True when the DataFrame contains sub-daily (intraday) candles."""
    return _bar_seconds(df) < 3600  # less than 1-hour bars

def _vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP anchored to the start of each calendar day."""
    df2 = df.copy().reset_index(drop=True)
    df2["_dt"]  = pd.to_datetime(df2["date"])
    df2["_day"] = df2["_dt"].dt.date
    df2["_tp"]  = (df2["high"] + df2["low"] + df2["close"]) / 3
    df2["_tv"]  = df2["_tp"] * df2["volume"]
    df2["_cum_tv"] = df2.groupby("_day")["_tv"].cumsum()
    df2["_cum_v"]  = df2.groupby("_day")["volume"].cumsum()
    vwap = df2["_cum_tv"] / df2["_cum_v"].replace(0, float("nan"))
    return vwap.ffill()

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Returns (upper, mid, lower) Bollinger Band series."""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) series."""
    ema_fast    = _ema(close, fast)
    ema_slow    = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _weinstein_stage(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Classify Weinstein stage from daily OHLCV.
    Returns (stage_number, stage_label, background_color).

    Stage 2 (Advancing): price above EMA200, EMA50 > EMA200, EMA200 rising
    Stage 4 (Declining): price below EMA200, EMA50 < EMA200, EMA200 falling
    Stage 1 (Basing):    EMA200 flat (±2%), price oscillating around it
    Stage 3 (Topping):   EMA50 crossing below EMA200 near recent highs
    """
    if len(df) < 50:
        return "?", "INSUFFICIENT DATA", "rgba(100,116,139,0.08)"

    c    = df["close"]
    e50  = _ema(c, 50)
    e200 = _ema(c, 200)
    ltp  = c.iloc[-1]

    above_200      = ltp > e200.iloc[-1]
    e50_above_200  = e50.iloc[-1] > e200.iloc[-1]

    lookback = min(30, len(df) - 1)
    e200_slope_pct = (e200.iloc[-1] - e200.iloc[-lookback]) / e200.iloc[-lookback] * 100

    if above_200 and e50_above_200 and e200_slope_pct > 0.3:
        return "2", "STAGE 2 — ADVANCING", "rgba(34,197,94,0.07)"
    elif not above_200 and not e50_above_200 and e200_slope_pct < -0.3:
        return "4", "STAGE 4 — DECLINING", "rgba(239,68,68,0.07)"
    elif not above_200 and not e50_above_200 and abs(e200_slope_pct) <= 0.3:
        return "1", "STAGE 1 — BASING", "rgba(100,116,139,0.07)"
    else:
        return "3", "STAGE 3 — DISTRIBUTION", "rgba(245,158,11,0.07)"


def _find_sr_zones(df: pd.DataFrame, lookback: int = 120, tolerance: float = 0.015):
    """
    Auto-detect support/resistance zones by clustering swing highs/lows.
    A swing high: candle whose high > 2 neighbours on both sides.
    A swing low:  candle whose low  < 2 neighbours on both sides.
    Zones within `tolerance` (%) of each other are merged.
    Returns list of {center, lo, hi, touches, type} dicts sorted by price.
    """
    data = df.tail(lookback).reset_index(drop=True)
    if len(data) < 6:
        return []

    raw: list[dict] = []
    for i in range(2, len(data) - 2):
        h = float(data["high"].iloc[i])
        if (h >= data["high"].iloc[i - 1] and h >= data["high"].iloc[i - 2] and
                h >= data["high"].iloc[i + 1] and h >= data["high"].iloc[i + 2]):
            raw.append({"price": h, "type": "R"})

        l = float(data["low"].iloc[i])
        if (l <= data["low"].iloc[i - 1] and l <= data["low"].iloc[i - 2] and
                l <= data["low"].iloc[i + 1] and l <= data["low"].iloc[i + 2]):
            raw.append({"price": l, "type": "S"})

    if not raw:
        return []

    # Cluster
    raw_sorted = sorted(raw, key=lambda x: x["price"])
    zones: list[dict] = []
    for item in raw_sorted:
        matched = False
        for z in zones:
            if abs(item["price"] - z["center"]) / z["center"] < tolerance:
                z["prices"].append(item["price"])
                z["center"] = float(np.mean(z["prices"]))
                z["touches"] += 1
                z["r_count"] += item["type"] == "R"
                z["s_count"] += item["type"] == "S"
                matched = True
                break
        if not matched:
            zones.append({
                "center":  item["price"],
                "prices":  [item["price"]],
                "touches": 1,
                "type":    item["type"],
                "r_count": int(item["type"] == "R"),
                "s_count": int(item["type"] == "S"),
            })

    result = []
    for z in zones:
        if z["touches"] < 2:
            continue
        ztype = "R" if z["r_count"] >= z["s_count"] else "S"
        c     = z["center"]
        result.append({
            "center":  round(c, 2),
            "lo":      round(c * (1 - 0.0075), 2),
            "hi":      round(c * (1 + 0.0075), 2),
            "touches": z["touches"],
            "type":    ztype,
        })

    return sorted(result, key=lambda x: x["center"])


def _volume_profile(df: pd.DataFrame, bins: int = 40):
    """
    Price-by-Volume histogram.
    For each price bucket, sum the volume of all candles whose range overlaps it.
    Returns sorted list of {lo, hi, center, volume} dicts.
    """
    lo_all = float(df["low"].min())
    hi_all = float(df["high"].max())
    if hi_all <= lo_all:
        return []
    bin_size = (hi_all - lo_all) / bins
    buckets  = []
    for i in range(bins):
        bl = lo_all + i * bin_size
        bh = lo_all + (i + 1) * bin_size
        mask = (df["low"] <= bh) & (df["high"] >= bl)
        vol  = int(df.loc[mask, "volume"].sum())
        buckets.append({"lo": bl, "hi": bh, "center": (bl + bh) / 2, "volume": vol})
    return buckets


def _detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20):
    """
    Simple divergence detector over the last `lookback` bars.
    Returns list of dicts: {type, price_idx, rsi_idx, label, color}

    Bearish divergence: price makes higher high, RSI makes lower high.
    Bullish divergence: price makes lower low,  RSI makes higher low.
    """
    if len(close) < lookback + 5:
        return []

    c   = close.iloc[-lookback:].reset_index(drop=True)
    r   = rsi.iloc[-lookback:].reset_index(drop=True)
    div = []

    # Find two most recent swing highs in price
    highs = []
    for i in range(1, len(c) - 1):
        if c.iloc[i] > c.iloc[i - 1] and c.iloc[i] >= c.iloc[i + 1]:
            highs.append(i)
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if c.iloc[i2] > c.iloc[i1] and r.iloc[i2] < r.iloc[i1]:
            div.append({
                "type": "BEARISH", "idx1": i1, "idx2": i2,
                "label": "Bearish Divergence",
                "color": C["red"],
            })

    # Find two most recent swing lows in price
    lows = []
    for i in range(1, len(c) - 1):
        if c.iloc[i] < c.iloc[i - 1] and c.iloc[i] <= c.iloc[i + 1]:
            lows.append(i)
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if c.iloc[i2] < c.iloc[i1] and r.iloc[i2] > r.iloc[i1]:
            div.append({
                "type": "BULLISH", "idx1": i1, "idx2": i2,
                "label": "Bullish Divergence",
                "color": C["green"],
            })

    return div


def _safe(val, default=None):
    """Return val if it's a finite number, else default."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _apply_zoom(fig: go.Figure, x_range=None) -> None:
    """
    Apply timeframe window, enable free zoom on both axes, and add crosshair spikes.
    Drag = pan (TradingView style). Scroll wheel = zoom (set via scrollZoom config).
    x_range: optional (start, end) date tuple for the visible window.
    """
    if x_range is not None:
        start, end = str(x_range[0])[:10], str(x_range[1])[:10]
        fig.update_xaxes(range=[start, end])
    # Free drag-pan + scroll-zoom on both axes
    fig.update_xaxes(
        fixedrange=False,
        showspikes=True, spikemode="across+toaxis",
        spikesnap="cursor", spikecolor="#64748b",
        spikethickness=1, spikedash="dot",
    )
    fig.update_yaxes(
        fixedrange=False,
        showspikes=True, spikemode="across+toaxis",
        spikesnap="cursor", spikecolor="#64748b",
        spikethickness=1, spikedash="dot",
    )


def _candle_colors(df: pd.DataFrame) -> list[str]:
    return [C["green"] if c >= o else C["red"]
            for o, c in zip(df["open"], df["close"])]


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — TREND CANVAS  (1 year, Weinstein stage)
# ═══════════════════════════════════════════════════════════════════════════

def chart_trend_canvas(df: pd.DataFrame, stock_row, x_range=None, candle_label: str = "1D") -> go.Figure:
    """
    Trend Canvas — adapts automatically to daily vs intraday candles.

    Daily mode  : EMA 50/200, Weinstein stage shading, Golden/Death Cross,
                  52W High/Low, volume + 20D avg volume.
    Intraday mode: EMA 9/21, VWAP (day-anchored), session separator lines,
                   volume + session-avg volume. No Weinstein / no 52W lines
                   (those concepts don't apply to sub-daily bars).
    """
    sym      = stock_row.get("tradingsymbol", "")
    intraday = _is_intraday_df(df)
    tail_n   = len(df) if intraday else 252
    data     = df.sort_values("date").tail(tail_n).copy().reset_index(drop=True)

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    c   = data["close"]
    vol = data["volume"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.02,
        subplot_titles=(f"{sym} — Trend Canvas [{candle_label}]", "Volume"),
    )

    # ── Background shading (daily only — Weinstein stages) ───────────────
    if not intraday:
        stage_num, stage_label, stage_bg = _weinstein_stage(data)
        fig.add_hrect(y0=0, y1=1, yref="paper", fillcolor=stage_bg,
                      layer="below", line_width=0)

    # ── Candlesticks ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=data["date"],
        open=data["open"], high=data["high"],
        low=data["low"],   close=data["close"],
        name="OHLC", showlegend=False,
        increasing_line_color=C["green"],
        decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"],
        decreasing_fillcolor=C["red"],
        line=dict(width=1),
        hovertext=[
            f"O: ₹{o:,.2f}  H: ₹{h:,.2f}  L: ₹{l:,.2f}  C: ₹{c2:,.2f}"
            for o, h, l, c2 in zip(data["open"], data["high"], data["low"], data["close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    if intraday:
        # ── INTRADAY mode: EMA 9 / 21 + VWAP ────────────────────────────
        e9  = _ema(c, 9)
        e21 = _ema(c, 21)
        fig.add_trace(go.Scatter(
            x=data["date"], y=e9,
            name="EMA 9", line=dict(color=C["blue"], width=1.4),
            hovertemplate="EMA 9: ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data["date"], y=e21,
            name="EMA 21", line=dict(color=C["purple"], width=1.6),
            hovertemplate="EMA 21: ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        try:
            vwap_series = _vwap(data)
            fig.add_trace(go.Scatter(
                x=data["date"], y=vwap_series,
                name="VWAP", line=dict(color=C["amber"], width=1.4, dash="dot"),
                hovertemplate="VWAP: ₹%{y:,.2f}<extra></extra>",
            ), row=1, col=1)
        except Exception:
            pass

        # Session separator lines (vertical dashed lines at each new trading day)
        data["_dt"] = pd.to_datetime(data["date"])
        data["_day"] = data["_dt"].dt.date
        day_starts = data[data["_day"] != data["_day"].shift(1)]["date"].tolist()
        for ds in day_starts[1:]:   # skip the very first
            fig.add_vline(x=str(ds)[:19], line_dash="dash",
                          line_color="rgba(100,116,139,0.35)", line_width=1)

        # Volume bars + session avg
        sess_avg = data.groupby("_day")["volume"].transform("mean")
        vol_col  = [C["green"] if cl >= op else C["red"]
                    for cl, op in zip(data["close"], data["open"])]
        fig.add_trace(go.Bar(
            x=data["date"], y=vol, marker_color=vol_col,
            name="Volume", showlegend=False, opacity=0.7,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data["date"], y=sess_avg,
            name="Session Avg Vol", line=dict(color=C["amber"], width=1.2, dash="dot"),
            hovertemplate="Session Avg: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        # Intraday label
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.98,
            text=f"<b>{candle_label} · Intraday</b>",
            showarrow=False, align="right",
            font=dict(size=11, color=C["teal"]),
            bgcolor="rgba(15,23,42,0.75)",
            bordercolor=C["teal"], borderwidth=1, borderpad=5,
        )

    else:
        # ── DAILY mode: EMA 50/200 + 52W + Weinstein + Golden/Death cross ─
        e50  = _ema(c, 50)
        e200 = _ema(c, 200) if len(data) >= 50 else pd.Series([np.nan] * len(data))
        vol_avg20 = vol.rolling(20).mean()

        fig.add_trace(go.Scatter(
            x=data["date"], y=e50,
            name="EMA 50", line=dict(color=C["blue"], width=1.5),
            hovertemplate="EMA 50: ₹%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        if len(data) >= 50:
            fig.add_trace(go.Scatter(
                x=data["date"], y=e200,
                name="EMA 200", line=dict(color=C["purple"], width=1.8),
                hovertemplate="EMA 200: ₹%{y:,.2f}<extra></extra>",
            ), row=1, col=1)

        # 52W High / Low
        h52 = _safe(stock_row.get("high_52w"))
        l52 = _safe(stock_row.get("low_52w"))
        if h52:
            fig.add_hline(y=h52, line_dash="dot", line_color=C["green"],
                          line_width=1, annotation_text=f"52W H ₹{h52:,.0f}",
                          annotation_font=dict(color=C["green"], size=10),
                          annotation_bgcolor="rgba(0,0,0,0)", row=1, col=1)
        if l52:
            fig.add_hline(y=l52, line_dash="dot", line_color=C["red"],
                          line_width=1, annotation_text=f"52W L ₹{l52:,.0f}",
                          annotation_font=dict(color=C["red"], size=10),
                          annotation_bgcolor="rgba(0,0,0,0)", row=1, col=1)

        # Golden / Death cross markers
        crosses = []
        if len(data) >= 52:
            e50v, e200v = e50.values, e200.values
            for i in range(1, len(data)):
                if np.isnan(e50v[i]) or np.isnan(e200v[i]):
                    continue
                if e50v[i-1] < e200v[i-1] and e50v[i] >= e200v[i]:
                    crosses.append(("GOLDEN", i, data["date"].iloc[i], float(c.iloc[i])))
                elif e50v[i-1] > e200v[i-1] and e50v[i] <= e200v[i]:
                    crosses.append(("DEATH", i, data["date"].iloc[i], float(c.iloc[i])))
        for cross_type, idx, dt, price in crosses[-3:]:
            color  = C["amber"] if cross_type == "GOLDEN" else C["pink"]
            symbol = "triangle-up" if cross_type == "GOLDEN" else "triangle-down"
            hover  = (
                "<b>Golden Cross</b><br>EMA 50 crossed <b>above</b> EMA 200<br>"
                "📈 Bullish signal — institutional buy interest<br>"
                "Action: Look for swing BUY setups while price stays above EMA 50"
            ) if cross_type == "GOLDEN" else (
                "<b>Death Cross</b><br>EMA 50 crossed <b>below</b> EMA 200<br>"
                "📉 Bearish signal — downtrend accelerating<br>"
                "Action: Avoid new longs; consider shorts if RSI is not oversold"
            )
            fig.add_trace(go.Scatter(
                x=[dt], y=[price * (0.97 if cross_type == "DEATH" else 1.03)],
                mode="markers+text",
                marker=dict(symbol=symbol, size=14, color=color),
                text=["Golden ✕" if cross_type == "GOLDEN" else "Death ✕"],
                textposition="top center", textfont=dict(size=9, color=color),
                name=("🟡 Golden Cross" if cross_type == "GOLDEN" else "💀 Death Cross"),
                showlegend=False,
                hovertemplate=hover + "<extra></extra>",
            ), row=1, col=1)

        # Volume bars + 20D avg line
        vcols = _candle_colors(data)
        fig.add_trace(go.Bar(
            x=data["date"], y=vol,
            marker_color=vcols, name="Volume",
            showlegend=False, opacity=0.7,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data["date"], y=vol_avg20,
            name="Avg Vol 20D", line=dict(color=C["amber"], width=1.2, dash="dot"),
            hovertemplate="Avg Vol 20D: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        # Stage label annotation
        stage_colors = {"2": C["green"], "4": C["red"], "1": C["slate"], "3": C["amber"], "?": C["slate"]}
        scol = stage_colors.get(stage_num, C["slate"])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.98,
            text=f"<b>{stage_label}</b>",
            showarrow=False, align="right",
            font=dict(size=12, color=scol),
            bgcolor="rgba(15,23,42,0.75)",
            bordercolor=scol, borderwidth=1, borderpad=5,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=620,
        xaxis_rangeslider_visible=False,
        xaxis2=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="Max"),
                ],
                bgcolor=C["panel"], activecolor=C["blue"],
                bordercolor=C["slate"], font=dict(color=C["white"], size=11),
            ),
        ) if not intraday else {},
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1,
                     gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     gridcolor=C["grid"], showticklabels=False)
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)
    _apply_zoom(fig, x_range)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — MOMENTUM LAB  (3 months, Bollinger + RSI divergence + MACD)
# ═══════════════════════════════════════════════════════════════════════════

def chart_momentum_lab(df: pd.DataFrame, stock_row, x_range=None, candle_label: str = "1D") -> go.Figure:
    """
    3-month (60-day) momentum analysis:
    - Price candles + EMA 20 + Bollinger Bands with OB/OS shaded fills
    - Band-walk annotation when price clings to upper band
    - RSI-14 with zone fills + divergence markers
    - MACD (12,26,9) with acceleration-encoded histogram + crossover signals
    """
    sym  = stock_row.get("tradingsymbol", "")
    data = df.sort_values("date").tail(80).copy().reset_index(drop=True)

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    c    = data["close"]
    e20  = _ema(c, 20)
    bb_upper, bb_mid, bb_lower = _bollinger(c, 20, 2.0)
    rsi  = _rsi(c, 14)
    macd_line, sig_line, hist = _macd(c, 12, 26, 9)
    divergences = _detect_rsi_divergence(c, rsi, lookback=min(40, len(data)))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.025,
        subplot_titles=(
            f"{sym} — Momentum Lab [{candle_label}]",
            "RSI-14  [overbought ▲70 | oversold ▼30]",
            "MACD (12,26,9)  [histogram = momentum acceleration]",
        ),
    )

    # ── Panel A: Price + Bollinger ────────────────────────────────────────
    # Bollinger fill — hidden from legend (it's a visual band, not a signal)
    fig.add_trace(go.Scatter(
        x=pd.concat([data["date"], data["date"].iloc[::-1]]),
        y=pd.concat([bb_upper, bb_lower.iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.07)",
        line=dict(width=0),
        name="BB Band", showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # Candlesticks — OHLC hover
    fig.add_trace(go.Candlestick(
        x=data["date"],
        open=data["open"], high=data["high"],
        low=data["low"],   close=data["close"],
        name="OHLC", showlegend=False,
        increasing_line_color=C["green"],  decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"],   decreasing_fillcolor=C["red"],
        line=dict(width=1),
        hovertext=[
            f"O: ₹{o:,.2f}  H: ₹{h:,.2f}  L: ₹{l:,.2f}  C: ₹{c:,.2f}"
            for o, h, l, c in zip(data["open"], data["high"], data["low"], data["close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    # EMA 20
    fig.add_trace(go.Scatter(
        x=data["date"], y=e20,
        name="EMA 20", line=dict(color=C["blue"], width=1.5),
        hovertemplate="EMA 20: ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # BB upper and lower lines
    fig.add_trace(go.Scatter(
        x=data["date"], y=bb_upper,
        name="BB Upper",
        line=dict(color="rgba(239,68,68,0.5)", width=1, dash="dot"),
        showlegend=True,
        hovertemplate="BB Upper: ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data["date"], y=bb_lower,
        name="BB Lower",
        line=dict(color="rgba(34,197,94,0.5)", width=1, dash="dot"),
        showlegend=True,
        hovertemplate="BB Lower: ₹%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Band-walk detection: ≥5 consecutive closes above upper BB
    if len(c) >= 10:
        consec = 0
        for i in range(len(c) - 1, max(len(c) - 20, 0), -1):
            if not (pd.isna(bb_upper.iloc[i]) or pd.isna(c.iloc[i])):
                if c.iloc[i] >= bb_upper.iloc[i]:
                    consec += 1
                else:
                    break
        if consec >= 5:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.98,
                text=f"<b>BAND WALK — Strong Uptrend ({consec}d above upper BB)</b>",
                showarrow=False, font=dict(color=C["amber"], size=11),
                bgcolor="rgba(15,23,42,0.8)", bordercolor=C["amber"],
                borderwidth=1, borderpad=4,
            )

    # RSI divergence markers on price panel
    offset = max(0, len(data) - 40)
    for div in divergences:
        i2 = div["idx2"] + offset
        if i2 < len(data):
            price_at_div = float(data["close"].iloc[i2])
            fig.add_annotation(
                x=data["date"].iloc[i2],
                y=price_at_div * (1.015 if div["type"] == "BEARISH" else 0.985),
                text=f"<b>{div['label']}</b>",
                showarrow=True, arrowhead=2,
                arrowcolor=div["color"], font=dict(color=div["color"], size=9),
                ax=0, ay=-30 if div["type"] == "BEARISH" else 30,
                row=1, col=1,
            )

    # ── Panel B: RSI ──────────────────────────────────────────────────────
    # OB/OS zone fills
    fig.add_hrect(y0=70, y1=100, row=2, col=1,
                  fillcolor="rgba(239,68,68,0.12)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  row=2, col=1,
                  fillcolor="rgba(34,197,94,0.12)", line_width=0)

    fig.add_trace(go.Scatter(
        x=data["date"], y=rsi,
        name="RSI-14", line=dict(color=C["purple"], width=1.8),
        showlegend=True,
        hovertemplate="RSI-14: %{y:.1f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=C["red"],   line_width=0.8, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=C["slate"], line_width=0.6, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=C["green"], line_width=0.8, row=2, col=1)

    # RSI divergence marker on RSI panel
    for div in divergences:
        i2 = div["idx2"] + offset
        if i2 < len(data):
            rsi_at_div = float(rsi.iloc[i2]) if not pd.isna(rsi.iloc[i2]) else 50
            fig.add_annotation(
                x=data["date"].iloc[i2], y=rsi_at_div,
                text="▲" if div["type"] == "BULLISH" else "▼",
                showarrow=False, font=dict(color=div["color"], size=14),
                row=2, col=1,
            )

    # ── Panel C: MACD ────────────────────────────────────────────────────
    # Acceleration-encoded histogram colours
    hist_colors = []
    for i in range(len(hist)):
        h_val  = hist.iloc[i]
        h_prev = hist.iloc[i - 1] if i > 0 else h_val
        if np.isnan(h_val):
            hist_colors.append(C["slate"])
        elif h_val >= 0:
            hist_colors.append(C["green"] if h_val >= h_prev else C["green_muted"])
        else:
            hist_colors.append(C["red"] if h_val <= h_prev else C["red_muted"])

    fig.add_trace(go.Bar(
        x=data["date"], y=hist,
        marker_color=hist_colors, name="MACD Hist",
        showlegend=True, opacity=0.85,
        hovertemplate="Histogram: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data["date"], y=macd_line,
        name="MACD", line=dict(color=C["blue"], width=1.5),
        hovertemplate="MACD: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data["date"], y=sig_line,
        name="Signal", line=dict(color=C["amber"], width=1.2),
        hovertemplate="Signal: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color=C["slate"], line_width=0.6, row=3, col=1)

    # MACD crossover markers (last 3)
    crossovers = []
    m = macd_line.values
    s = sig_line.values
    for i in range(1, len(m)):
        if np.isnan(m[i]) or np.isnan(s[i]):
            continue
        if m[i - 1] < s[i - 1] and m[i] >= s[i]:
            crossovers.append(("BUY ✕", i, C["green"], "triangle-up"))
        elif m[i - 1] > s[i - 1] and m[i] <= s[i]:
            crossovers.append(("SELL ✕", i, C["red"], "triangle-down"))

    for label, idx, color, sym_marker in crossovers[-3:]:
        fig.add_trace(go.Scatter(
            x=[data["date"].iloc[idx]],
            y=[float(macd_line.iloc[idx])],
            mode="markers+text",
            marker=dict(symbol=sym_marker, size=10, color=color),
            text=[label], textposition="top center",
            textfont=dict(size=8, color=color),
            showlegend=False,
        ), row=3, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=640,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price (₹)",     row=1, col=1,
                     gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_yaxes(title_text="RSI",            row=2, col=1,
                     range=[0, 100], gridcolor=C["grid"])
    fig.update_yaxes(title_text="MACD",           row=3, col=1,
                     gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)
    _apply_zoom(fig, x_range)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — TRADE SETUP  (20 days, pivot levels, signal overlays, ATR bands)
# ═══════════════════════════════════════════════════════════════════════════

def chart_trade_setup(df: pd.DataFrame, stock_row, x_range=None, display_days: int = 20, candle_label: str = "1D") -> go.Figure:
    """
    20-day 'cockpit view' for trade execution.
    Y-axis is clamped to the 20-day candle range (+12% padding) so candles
    always fill the chart. Levels outside the visible window are still drawn
    as clipped lines at the boundary — their annotation labels sit inside a
    dedicated legend table below the chart rather than cluttering the y-axis.
    Annotations alternate left/right to prevent label overlap.
    """
    sym  = stock_row.get("tradingsymbol", "")
    data = df.sort_values("date").tail(30).copy().reset_index(drop=True)

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    display  = data.tail(max(display_days, 5)).reset_index(drop=True)
    vol_avg  = float(data["volume"].mean())
    ltp      = _safe(stock_row.get("ltp")) or float(display["close"].iloc[-1])
    atr      = _safe(stock_row.get("atr_14"))

    # ── Compute y-axis range from 20-day candles (not from signal levels) ──
    raw_lo = float(display["low"].min())
    raw_hi = float(display["high"].max())
    pad    = (raw_hi - raw_lo) * 0.12
    y_lo   = raw_lo - pad
    y_hi   = raw_hi + pad

    # Collect all signal price levels; clip the range to include any levels
    # that are within 20% of the current price (relevant to today's trade).
    # Levels further than 20% are intentionally excluded — the clipped line
    # edge + the legend table below still communicate them.
    def _clip_extend(val):
        if val is None:
            return
        nonlocal y_lo, y_hi
        if abs(val - ltp) / ltp <= 0.20:
            y_lo = min(y_lo, val * 0.99)
            y_hi = max(y_hi, val * 1.01)

    for key in ["intraday_r2", "intraday_r1", "intraday_pivot",
                "intraday_s1", "intraday_s2",
                "swing_entry", "swing_stop", "swing_t1",
                "intraday_entry", "intraday_stop", "intraday_t1"]:
        _clip_extend(_safe(stock_row.get(key)))

    # Helper: draw a line and choose left vs right annotation to avoid pile-up
    _ann_side_toggle = [True]   # True = right, False = left (mutable toggle)

    def _hline(fig, y, color, dash, label, width=1.2, bold=False):
        side = "right" if _ann_side_toggle[0] else "left"
        _ann_side_toggle[0] = not _ann_side_toggle[0]
        prefix = "  " if side == "right" else ""
        suffix = "  " if side == "left" else ""
        txt    = f"{prefix}<b>{label}</b>{suffix}" if bold else f"{prefix}{label}{suffix}"
        fig.add_hline(
            y=y, line_dash=dash, line_color=color, line_width=width,
            annotation_text=txt,
            annotation_position=f"top {side}",
            annotation_font=dict(color=color, size=9),
            annotation_bgcolor="rgba(15,23,42,0.7)",
            row=1, col=1,
        )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.02,
        subplot_titles=(
            f"{sym} — Trade Setup [{candle_label}]",
            "Volume  [orange = high-volume candle > 1.5× avg]",
        ),
    )

    # ── High-volume candle detection ──────────────────────────────────────
    high_vol_mask = display["volume"] > vol_avg * 1.5
    vcols = [C["amber"] if hv else (C["green"] if c >= o else C["red"])
             for hv, o, c in zip(high_vol_mask, display["open"], display["close"])]

    # Candlesticks — OHLC hover
    fig.add_trace(go.Candlestick(
        x=display["date"],
        open=display["open"], high=display["high"],
        low=display["low"],   close=display["close"],
        name="OHLC", showlegend=False,
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"],  decreasing_fillcolor=C["red"],
        line=dict(width=1),
        hovertext=[
            f"O: ₹{o:,.2f}  H: ₹{h:,.2f}  L: ₹{l:,.2f}  C: ₹{c:,.2f}"
            for o, h, l, c in zip(display["open"], display["high"],
                                   display["low"],  display["close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    # High-volume markers
    hv_dates = display.loc[high_vol_mask, "date"]
    hv_lows  = display.loc[high_vol_mask, "low"] * 0.998
    if not hv_dates.empty:
        fig.add_trace(go.Scatter(
            x=hv_dates, y=hv_lows,
            mode="markers",
            marker=dict(symbol="triangle-up", color=C["amber"], size=8),
            name="High Vol", showlegend=True,
        ), row=1, col=1)

    # ── ATR ±1 band ───────────────────────────────────────────────────────
    if atr and ltp:
        fig.add_hrect(
            y0=ltp - atr, y1=ltp + atr,
            fillcolor="rgba(59,130,246,0.07)",
            line=dict(color="rgba(59,130,246,0.30)", width=1, dash="dot"),
            annotation_text=f"ATR ±₹{atr:.2f}",
            annotation_position="top left",
            annotation_font=dict(color=C["blue"], size=9),
            annotation_bgcolor="rgba(15,23,42,0.7)",
            row=1, col=1,
        )

    # ── Pivot levels (alternating left/right) ─────────────────────────────
    pivot_map = [
        ("intraday_r2",    "R2",  C["green"],  "dash"),
        ("intraday_r1",    "R1",  C["lime"],   "dash"),
        ("intraday_pivot", "P",   C["white"],  "dot"),
        ("intraday_s1",    "S1",  C["amber"],  "dash"),
        ("intraday_s2",    "S2",  C["red"],    "dash"),
    ]
    for db_col, label, color, dash in pivot_map:
        val = _safe(stock_row.get(db_col))
        if val:
            _hline(fig, val, color, dash, f"{label} ₹{val:,.2f}")

    # ── Swing signal overlays ─────────────────────────────────────────────
    sw_sig   = stock_row.get("swing_signal")
    sw_entry = _safe(stock_row.get("swing_entry"))
    sw_stop  = _safe(stock_row.get("swing_stop"))
    sw_t1    = _safe(stock_row.get("swing_t1"))
    sw_t2    = _safe(stock_row.get("swing_t2"))
    sw_setup = stock_row.get("swing_setup") or ""

    if sw_sig == "BUY" and sw_entry:
        _hline(fig, sw_entry, C["green"], "solid",
               f"Swing Entry ₹{sw_entry:,.2f} ({sw_setup})", width=1.8, bold=True)
        if sw_stop:
            _hline(fig, sw_stop, C["red"], "dash",
                   f"Swing Stop ₹{sw_stop:,.2f}")
            fig.add_hrect(y0=sw_stop, y1=sw_entry,
                          fillcolor="rgba(239,68,68,0.07)", line_width=0, row=1, col=1)
        if sw_t1:
            _hline(fig, sw_t1, C["teal"], "dot", f"T1 ₹{sw_t1:,.2f}")
        if sw_t2:
            _hline(fig, sw_t2, "rgba(20,184,166,0.55)", "dot", f"T2 ₹{sw_t2:,.2f}")

    # ── Intraday signal overlays ──────────────────────────────────────────
    intra_sig   = stock_row.get("intraday_signal")
    intra_entry = _safe(stock_row.get("intraday_entry"))
    intra_stop  = _safe(stock_row.get("intraday_stop"))
    intra_t1    = _safe(stock_row.get("intraday_t1"))

    if intra_sig == "BUY_ABOVE" and intra_entry:
        _hline(fig, intra_entry, C["green"], "solid",
               f"▶ BUY ABOVE ₹{intra_entry:,.2f}", width=2, bold=True)
        if intra_stop:
            _hline(fig, intra_stop, C["red"], "dash",
                   f"Stop ₹{intra_stop:,.2f}")
        if intra_t1:
            _hline(fig, intra_t1, C["lime"], "dot",
                   f"Target ₹{intra_t1:,.2f}")

    elif intra_sig == "SELL_BELOW" and intra_entry:
        _hline(fig, intra_entry, C["red"], "solid",
               f"▼ SELL BELOW ₹{intra_entry:,.2f}", width=2, bold=True)
        if intra_stop:
            _hline(fig, intra_stop, C["amber"], "dash",
                   f"Cover Stop ₹{intra_stop:,.2f}")
        if intra_t1:
            _hline(fig, intra_t1, C["lime"], "dot",
                   f"Target ₹{intra_t1:,.2f}")

    # ── Volume bars ───────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=display["date"], y=display["volume"],
        marker_color=vcols, name="Volume",
        showlegend=False, opacity=0.8,
    ), row=2, col=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        height=680,
        xaxis_rangeslider_visible=False,
    )
    # Clamp y-axis to the computed range — this keeps candles full-height
    fig.update_yaxes(
        title_text="Price (₹)", row=1, col=1,
        range=[y_lo, y_hi],
        gridcolor=C["grid"], zerolinecolor=C["grid"],
    )
    fig.update_yaxes(title_text="Vol", row=2, col=1,
                     gridcolor=C["grid"], showticklabels=False)
    fig.update_xaxes(gridcolor=C["grid"], showgrid=False)
    _apply_zoom(fig, x_range)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — MARKET STRUCTURE  (6 months, auto S/R zones + Volume Profile)
# ═══════════════════════════════════════════════════════════════════════════

def chart_market_structure(df: pd.DataFrame, stock_row, x_range=None, candle_label: str = "1D") -> go.Figure:
    """
    6-month market structure with:
    - Auto-detected S/R zones (swing high/low clustering, opacity = touch count)
    - Volume Profile (Price-by-Volume histogram) on right axis
    - Point of Control (POC) — price where most volume has traded
    """
    sym  = stock_row.get("tradingsymbol", "")
    data = df.sort_values("date").tail(126).copy().reset_index(drop=True)

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    sr_zones    = _find_sr_zones(data, lookback=126, tolerance=0.015)
    vol_profile = _volume_profile(data, bins=50)
    ltp         = _safe(stock_row.get("ltp")) or float(data["close"].iloc[-1])

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.80, 0.20],
        shared_yaxes=True,
        horizontal_spacing=0.01,
        subplot_titles=(
            f"{sym} — Market Structure [{candle_label}]",
            "Volume Profile",
        ),
    )

    # ── Candlesticks — OHLC hover ─────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=data["date"],
        open=data["open"], high=data["high"],
        low=data["low"],   close=data["close"],
        name="OHLC", showlegend=False,
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        increasing_fillcolor=C["green"],  decreasing_fillcolor=C["red"],
        line=dict(width=1),
        hovertext=[
            f"O: ₹{o:,.2f}  H: ₹{h:,.2f}  L: ₹{l:,.2f}  C: ₹{c:,.2f}"
            for o, h, l, c in zip(data["open"], data["high"],
                                   data["low"],  data["close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    # ── S/R Zone bands ────────────────────────────────────────────────────
    max_touches = max((z["touches"] for z in sr_zones), default=1)
    ltp_now = ltp

    for zone in sr_zones:
        is_support  = zone["type"] == "S"
        base_color  = "34,197,94" if is_support else "239,68,68"
        opacity     = 0.06 + 0.14 * (zone["touches"] / max_touches)
        border_op   = 0.20 + 0.30 * (zone["touches"] / max_touches)
        touch_label = f"{'Support' if is_support else 'Resistance'} ×{zone['touches']}"

        fig.add_hrect(
            y0=zone["lo"], y1=zone["hi"],
            fillcolor=f"rgba({base_color},{opacity:.2f})",
            line=dict(color=f"rgba({base_color},{border_op:.2f})", width=0.8),
            annotation_text=f"  {touch_label}  ₹{zone['center']:,.0f}",
            annotation_font=dict(
                color=C["green"] if is_support else C["red"],
                size=8,
            ),
            annotation_bgcolor="rgba(0,0,0,0)",
            row=1, col=1,
        )

    # ── Current price line ────────────────────────────────────────────────
    fig.add_hline(
        y=ltp_now, line_dash="solid", line_color=C["amber"], line_width=1.5,
        annotation_text=f"  LTP ₹{ltp_now:,.2f}",
        annotation_font=dict(color=C["amber"], size=10),
        row=1, col=1,
    )

    # ── Volume Profile (horizontal bars, right panel) ─────────────────────
    if vol_profile:
        max_vol = max(b["volume"] for b in vol_profile)
        poc     = max(vol_profile, key=lambda b: b["volume"])
        poc_price = poc["center"]

        vp_colors = []
        for b in vol_profile:
            if abs(b["center"] - poc_price) / poc_price < 0.005:
                vp_colors.append(C["amber"])   # POC highlighted in amber
            elif b["center"] < ltp_now:
                vp_colors.append("rgba(34,197,94,0.55)")
            else:
                vp_colors.append("rgba(239,68,68,0.55)")

        fig.add_trace(go.Bar(
            x=[b["volume"] for b in vol_profile],
            y=[b["center"] for b in vol_profile],
            orientation="h",
            marker_color=vp_colors,
            name="Vol Profile",
            showlegend=True,
            width=[(b["hi"] - b["lo"]) * 0.9 for b in vol_profile],
        ), row=1, col=2)

        # POC annotation
        fig.add_annotation(
            xref="x2", yref="y",
            x=max_vol * 0.85, y=poc_price,
            text=f"<b>POC ₹{poc_price:,.0f}</b>",
            showarrow=True, arrowhead=2, arrowcolor=C["amber"],
            font=dict(color=C["amber"], size=10),
            ax=40, ay=0,
        )

        # POC horizontal line on main chart
        fig.add_hline(
            y=poc_price, line_dash="dot", line_color=C["amber"], line_width=1,
            annotation_text=f"  POC ₹{poc_price:,.0f}",
            annotation_font=dict(color=C["amber"], size=9),
            row=1, col=1,
        )

    # ── Zone proximity note ───────────────────────────────────────────────
    near_zones = sorted(sr_zones, key=lambda z: abs(z["center"] - ltp_now))[:2]
    if near_zones:
        msgs = []
        for z in near_zones:
            dist_pct = (z["center"] - ltp_now) / ltp_now * 100
            direction = "above" if dist_pct > 0 else "below"
            ztype = "Resistance" if z["type"] == "R" else "Support"
            msgs.append(f"{ztype} ₹{z['center']:,.0f} ({abs(dist_pct):.1f}% {direction})")
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98,
            text="  " + "  |  ".join(msgs),
            showarrow=False, align="left",
            font=dict(size=10, color=C["white"]),
            bgcolor="rgba(15,23,42,0.80)",
            bordercolor=C["slate"], borderwidth=1, borderpad=5,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=580,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1,
                     gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_yaxes(row=1, col=2, gridcolor=C["grid"])
    fig.update_xaxes(title_text="", row=1, col=1,
                     gridcolor=C["grid"], showgrid=False)
    fig.update_xaxes(title_text="Volume", row=1, col=2,
                     showticklabels=False, gridcolor=C["grid"])
    _apply_zoom(fig, x_range)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS CONTEXT BAR
# Returns a dict of labels for rendering as HTML pills in app.py
# ═══════════════════════════════════════════════════════════════════════════

def build_context_bar(df: pd.DataFrame, stock_row) -> list[dict]:
    """
    Returns a list of {label, value, color} dicts for the analysis summary bar.
    Rendered in app.py as styled HTML pills.
    """
    pills = []

    # Stage
    data = df.sort_values("date").tail(252).copy().reset_index(drop=True)
    stage_num, stage_label, _ = _weinstein_stage(data)
    stage_colors = {"2": C["green"], "4": C["red"], "1": C["slate"], "3": C["amber"], "?": C["slate"]}
    pills.append({
        "label": "Trend Phase",
        "value": stage_label,
        "color": stage_colors.get(stage_num, C["slate"]),
    })

    # RSI state
    rsi_val = _safe(stock_row.get("rsi_14"))
    if rsi_val is not None:
        if rsi_val >= 70:
            rsi_label, rsi_color = f"RSI {rsi_val:.0f} — Overbought", C["red"]
        elif rsi_val <= 30:
            rsi_label, rsi_color = f"RSI {rsi_val:.0f} — Oversold",   C["green"]
        elif 50 <= rsi_val < 70:
            rsi_label, rsi_color = f"RSI {rsi_val:.0f} — Bullish",    C["green_muted"]
        else:
            rsi_label, rsi_color = f"RSI {rsi_val:.0f} — Bearish",    C["red_muted"]
        pills.append({"label": "Momentum", "value": rsi_label, "color": rsi_color})

    # Swing signal
    sw_sig   = stock_row.get("swing_signal")
    sw_setup = stock_row.get("swing_setup") or ""
    sw_entry = _safe(stock_row.get("swing_entry"))
    sw_rr    = _safe(stock_row.get("swing_rr"))
    if sw_sig == "BUY" and sw_entry:
        rr_txt = f"  R/R {sw_rr:.1f}×" if sw_rr else ""
        pills.append({
            "label": "Swing",
            "value": f"{sw_setup} — Entry ₹{sw_entry:,.2f}{rr_txt}",
            "color": C["green"],
        })
    elif sw_sig == "SELL":
        pills.append({"label": "Swing", "value": "EXIT signal active", "color": C["red"]})
    elif sw_sig == "WATCH":
        pills.append({"label": "Swing", "value": "Watching — no setup", "color": C["slate"]})

    # Intraday signal
    intra_sig   = stock_row.get("intraday_signal")
    intra_entry = _safe(stock_row.get("intraday_entry"))
    if intra_sig == "BUY_ABOVE" and intra_entry:
        pills.append({
            "label": "Intraday",
            "value": f"BUY ABOVE ₹{intra_entry:,.2f}",
            "color": C["green"],
        })
    elif intra_sig == "SELL_BELOW" and intra_entry:
        pills.append({
            "label": "Intraday",
            "value": f"SELL BELOW ₹{intra_entry:,.2f}",
            "color": C["red"],
        })
    elif intra_sig == "AVOID":
        pills.append({"label": "Intraday", "value": "AVOID", "color": C["slate"]})

    # Scaling signal
    scale_sig = stock_row.get("scale_signal")
    scale_e1  = _safe(stock_row.get("scale_entry_1"))
    if scale_sig == "SCALE_IN" and scale_e1:
        pills.append({
            "label": "Scaling",
            "value": f"ADD at ₹{scale_e1:,.2f}",
            "color": C["teal"],
        })

    return pills
