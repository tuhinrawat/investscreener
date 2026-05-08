"""
db.py — DuckDB schema and helpers.

Why DuckDB and not SQLite?
  - DuckDB is columnar — analytical queries (window functions, aggregates
    over 252-day series) are 10-100x faster
  - Native pandas integration via .df()
  - Single-file like SQLite, no server
  - Trade-off: single-writer (fine for us, screener is single-user)

Tables:
  - instruments:      changes weekly, ~2K rows
  - daily_ohlcv:      changes daily, ~500K rows after 252 days × 2K stocks
  - computed_metrics: derived, recomputed on full rescan, ~2K rows
  - trade_log:        user trade journal — recommendation vs actual outcome
"""
import duckdb
import pandas as pd
from pathlib import Path
import config


def get_conn():
    """Returns a fresh connection. DuckDB is single-process; reopen each time."""
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DB_PATH))


def _migrate_trade_log_columns(con):
    """
    Idempotent migration — adds columns introduced after the initial schema.
    Safe to call on both new and existing databases.
    """
    extra_cols = [
        ("kite_user_id",     "VARCHAR"),   # per-user isolation
        ("kite_order_id",    "VARCHAR"),
        ("kite_sl_order_id", "VARCHAR"),
        ("kite_status",      "VARCHAR"),
        ("is_paper_trade",   "BOOLEAN"),   # True = virtual paper trade, False/NULL = real
    ]
    for col, dtype in extra_cols:
        try:
            con.execute(
                f"ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS {col} {dtype}"
            )
        except Exception:
            pass


def _migrate_signal_columns(con):
    """
    Adds trade-signal columns to computed_metrics for DBs that pre-date this
    feature. DuckDB supports ADD COLUMN IF NOT EXISTS so this is idempotent.
    """
    new_cols = [
        ("swing_signal",        "VARCHAR"),
        ("swing_setup",         "VARCHAR"),
        ("swing_entry",         "DOUBLE"),
        ("swing_stop",          "DOUBLE"),
        ("swing_t1",            "DOUBLE"),
        ("swing_t2",            "DOUBLE"),
        ("swing_rr",            "DOUBLE"),
        ("swing_quality",       "INTEGER"),
        ("swing_reason",        "VARCHAR"),
        ("intraday_signal",     "VARCHAR"),
        ("intraday_pivot",      "DOUBLE"),
        ("intraday_r1",         "DOUBLE"),
        ("intraday_r2",         "DOUBLE"),
        ("intraday_s1",         "DOUBLE"),
        ("intraday_s2",         "DOUBLE"),
        ("intraday_entry",      "DOUBLE"),
        ("intraday_stop",       "DOUBLE"),
        ("intraday_t1",         "DOUBLE"),
        ("intraday_reason",     "VARCHAR"),
        ("scale_signal",        "VARCHAR"),
        ("scale_setup",         "VARCHAR"),
        ("scale_entry_1",       "DOUBLE"),
        ("scale_stop",          "DOUBLE"),
        ("scale_trailing_stop", "DOUBLE"),
        ("scale_target",        "DOUBLE"),
        ("scale_quality",       "INTEGER"),
        ("scale_reason",        "VARCHAR"),
        # AI analysis columns
        ("ai_score",            "DOUBLE"),
        ("ai_verdict",          "VARCHAR"),
        ("ai_confidence",       "VARCHAR"),
        ("ai_brief",            "TEXT"),
        ("ai_analyzed_at",      "TIMESTAMP"),
    ]
    for col, dtype in new_cols:
        try:
            con.execute(
                f"ALTER TABLE computed_metrics ADD COLUMN IF NOT EXISTS {col} {dtype}"
            )
        except Exception:
            pass   # column already exists or table doesn't exist yet — both fine


def init_schema():
    """Creates tables if not exist. Idempotent — safe to call every run."""
    con = get_conn()
    con.execute("""
        CREATE TABLE IF NOT EXISTS instruments (
            instrument_token  BIGINT PRIMARY KEY,
            tradingsymbol     VARCHAR,
            name              VARCHAR,
            exchange          VARCHAR,
            segment           VARCHAR,
            instrument_type   VARCHAR,
            tick_size         DOUBLE,
            lot_size          INTEGER,
            last_updated      TIMESTAMP
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_ohlcv (
            instrument_token  BIGINT,
            date              DATE,
            open              DOUBLE,
            high              DOUBLE,
            low               DOUBLE,
            close             DOUBLE,
            volume            BIGINT,
            PRIMARY KEY (instrument_token, date)
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS computed_metrics (
            instrument_token        BIGINT PRIMARY KEY,
            tradingsymbol           VARCHAR,
            ltp                     DOUBLE,
            -- Liquidity
            avg_turnover_cr         DOUBLE,
            avg_volume              BIGINT,
            -- Trends (% returns)
            ret_5d                  DOUBLE,
            ret_1m                  DOUBLE,
            ret_3m                  DOUBLE,
            ret_6m                  DOUBLE,
            ret_1y                  DOUBLE,
            -- Relative strength vs Nifty (3M)
            rs_vs_nifty_3m          DOUBLE,
            -- Volume expansion
            vol_expansion_ratio     DOUBLE,
            -- Technicals
            rsi_14                  DOUBLE,
            ema_20                  DOUBLE,
            ema_50                  DOUBLE,
            ema_200                 DOUBLE,
            atr_14                  DOUBLE,
            high_52w                DOUBLE,
            low_52w                 DOUBLE,
            dist_from_52w_high_pct  DOUBLE,
            dist_from_50ema_pct     DOUBLE,
            support_20d             DOUBLE,
            resistance_20d          DOUBLE,
            -- Composite
            trend_score             DOUBLE,
            composite_score         DOUBLE,
            -- Swing signals
            swing_signal            VARCHAR,
            swing_setup             VARCHAR,
            swing_entry             DOUBLE,
            swing_stop              DOUBLE,
            swing_t1                DOUBLE,
            swing_t2                DOUBLE,
            swing_rr                DOUBLE,
            swing_quality           INTEGER,
            swing_reason            VARCHAR,
            -- Intraday signals (pivot-based day plan)
            intraday_signal         VARCHAR,
            intraday_pivot          DOUBLE,
            intraday_r1             DOUBLE,
            intraday_r2             DOUBLE,
            intraday_s1             DOUBLE,
            intraday_s2             DOUBLE,
            intraday_entry          DOUBLE,
            intraday_stop           DOUBLE,
            intraday_t1             DOUBLE,
            intraday_reason         VARCHAR,
            -- Scaling signals (multi-week position build)
            scale_signal            VARCHAR,
            scale_setup             VARCHAR,
            scale_entry_1           DOUBLE,
            scale_stop              DOUBLE,
            scale_trailing_stop     DOUBLE,
            scale_target            DOUBLE,
            scale_quality           INTEGER,
            scale_reason            VARCHAR,
            -- AI Analysis (STOCKLENS)
            ai_score                DOUBLE,
            ai_verdict              VARCHAR,
            ai_confidence           VARCHAR,
            ai_brief                TEXT,
            ai_analyzed_at          TIMESTAMP,
            -- Meta
            last_updated            TIMESTAMP
        );
    """)
    _migrate_signal_columns(con)
    # trade_log — must always be created (safe: IF NOT EXISTS)
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS trade_log_id_seq START 1;
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id                  BIGINT DEFAULT nextval('trade_log_id_seq') PRIMARY KEY,
            logged_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trade_date          DATE,
            tradingsymbol       VARCHAR,
            instrument_token    BIGINT,
            setup_type          VARCHAR,   -- SWING / INTRADAY / SCALING
            signal_type         VARCHAR,   -- BUY / SELL / BUY_ABOVE / SELL_BELOW

            -- ── ORIGINAL RECOMMENDATION snapshot ──────────────────────────
            rec_entry           DOUBLE,
            rec_stop            DOUBLE,
            rec_t1              DOUBLE,
            rec_t2              DOUBLE,
            rec_rr              DOUBLE,
            rec_reason          VARCHAR,
            rec_composite_score DOUBLE,
            rec_ai_score        DOUBLE,

            -- ── WHO placed this trade (per-user isolation) ───────────────
            kite_user_id        VARCHAR,   -- Kite user_id (e.g. "ZY1234")

            -- ── KITE ORDER IDs (set when order placed via Kite API) ───────
            kite_order_id       VARCHAR,   -- entry order id returned by Kite
            kite_sl_order_id    VARCHAR,   -- stop-loss order id (optional)
            kite_status         VARCHAR,   -- last known Kite order status

            -- ── ACTUAL TRADE (what user did / Kite filled) ────────────────
            quantity            INTEGER,
            actual_entry        DOUBLE,
            actual_exit         DOUBLE,    -- NULL = still open
            status              VARCHAR,   -- OPEN / CLOSED / TARGET_HIT / STOPPED_OUT / CANCELLED / REJECTED
            notes               TEXT,

            -- ── CALCULATED OUTCOMES (populated on close) ─────────────────
            pnl_amount          DOUBLE,    -- (exit-entry)*qty, sign-aware for shorts
            pnl_pct             DOUBLE,    -- (exit-entry)/entry*100
            slippage_entry_pct  DOUBLE,    -- (actual_entry-rec_entry)/rec_entry*100
            rr_realised         DOUBLE     -- actual_gain/actual_risk
        );
    """)
    _migrate_trade_log_columns(con)
    # signal_config — persists algo-tuning overrides derived from paper trade feedback
    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_config (
            config_key   VARCHAR NOT NULL,
            kite_user_id VARCHAR NOT NULL DEFAULT '',
            value        DOUBLE  NOT NULL,
            updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (config_key, kite_user_id)
        );
    """)
    con.close()


def _compute_outcomes(quantity: int, actual_entry: float, actual_exit: float,
                      rec_entry: float, rec_stop: float, signal_type: str) -> dict:
    """Compute pnl_amount, pnl_pct, slippage_entry_pct, rr_realised from trade data."""
    direction = -1 if signal_type in ("SELL", "SELL_BELOW") else 1
    pnl_amount = None
    pnl_pct    = None
    rr_realised = None
    if actual_exit is not None and actual_entry:
        pnl_amount  = direction * (actual_exit - actual_entry) * (quantity or 1)
        pnl_pct     = direction * (actual_exit - actual_entry) / actual_entry * 100
        risk        = abs(actual_entry - rec_stop) if rec_stop else None
        actual_gain = direction * (actual_exit - actual_entry)
        if risk and risk > 0:
            rr_realised = actual_gain / risk
    slippage = None
    if rec_entry and actual_entry:
        slippage = (actual_entry - rec_entry) / rec_entry * 100
    return {
        "pnl_amount":         pnl_amount,
        "pnl_pct":            pnl_pct,
        "slippage_entry_pct": slippage,
        "rr_realised":        rr_realised,
    }


def upsert_instruments(df: pd.DataFrame):
    """Replaces full instruments table (it's small and we re-pull weekly)."""
    con = get_conn()
    con.execute("DELETE FROM instruments;")
    con.register("incoming", df)
    con.execute("INSERT INTO instruments SELECT * FROM incoming;")
    con.close()


def upsert_ohlcv(df: pd.DataFrame):
    """Append-or-replace daily candles. Uses DuckDB's INSERT OR REPLACE."""
    if df.empty:
        return
    con = get_conn()
    con.register("incoming", df)
    con.execute("""
        INSERT OR REPLACE INTO daily_ohlcv
        SELECT instrument_token, date, open, high, low, close, volume
        FROM incoming;
    """)
    con.close()


def replace_metrics(df: pd.DataFrame):
    """Full table replace — metrics are always recomputed end-to-end."""
    con = get_conn()
    # Fetch the actual column names defined in the DB table so we can match
    # by name, not by position.  This is critical when signal columns were added
    # via ALTER TABLE migration (appended at the end) while the DataFrame may
    # have them in a different order — positional SELECT * would map VARCHAR
    # values into TIMESTAMP columns and trigger a ConversionError.
    table_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'computed_metrics'"
        ).fetchall()
    }
    # Only keep DataFrame columns that exist in the table (drop extras silently)
    keep = [c for c in df.columns if c in table_cols]
    df_clean = df[keep]
    con.execute("DELETE FROM computed_metrics;")
    con.register("incoming", df_clean)
    # BY NAME matches columns by name, not position — safe across schema versions
    con.execute("INSERT INTO computed_metrics BY NAME SELECT * FROM incoming;")
    con.close()


def load_metrics() -> pd.DataFrame:
    """Returns current screener output, joined with instrument names."""
    con = get_conn()
    df = con.execute("""
        SELECT m.*, i.name AS company_name
        FROM computed_metrics m
        LEFT JOIN instruments i USING (instrument_token)
        ORDER BY composite_score DESC NULLS LAST
    """).df()
    con.close()
    return df


def load_ohlcv(instrument_token: int) -> pd.DataFrame:
    """For chart panel — returns all cached daily candles for one stock."""
    con = get_conn()
    df = con.execute("""
        SELECT date, open, high, low, close, volume
        FROM daily_ohlcv
        WHERE instrument_token = ?
        ORDER BY date
    """, [instrument_token]).df()
    con.close()
    return df


def save_ai_result(instrument_token: int, result: dict):
    """Persist STOCKLENS AI analysis for one stock."""
    con = get_conn()
    con.execute("""
        UPDATE computed_metrics
        SET ai_score       = ?,
            ai_verdict     = ?,
            ai_confidence  = ?,
            ai_brief       = ?,
            ai_analyzed_at = ?
        WHERE instrument_token = ?
    """, [
        result.get("ai_score"),
        result.get("ai_verdict"),
        result.get("ai_confidence"),
        result.get("ai_brief"),
        result.get("ai_analyzed_at"),
        instrument_token,
    ])
    con.close()


def get_instruments_age_days() -> int:
    """Returns days since instruments table was last refreshed (-1 if empty)."""
    con = get_conn()
    result = con.execute("""
        SELECT EXTRACT(DAY FROM (CURRENT_TIMESTAMP - MAX(last_updated)))
        FROM instruments
    """).fetchone()
    con.close()
    if result is None or result[0] is None:
        return -1
    return int(result[0])


def get_last_metrics_update() -> str:
    """Returns ISO timestamp of last full rescan, or 'never'."""
    con = get_conn()
    result = con.execute("SELECT MAX(last_updated) FROM computed_metrics").fetchone()
    con.close()
    if result is None or result[0] is None:
        return "never"
    return result[0].strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════════
# TRADE LOG — journal of recommendations vs actual outcomes
# ═══════════════════════════════════════════════════════════════════════════

def log_trade(trade: dict) -> int:
    """
    Insert a new trade log entry. Returns the new row id.

    trade dict keys (all optional unless marked *required*):
      *tradingsymbol, instrument_token, setup_type, signal_type,
      trade_date, rec_entry, rec_stop, rec_t1, rec_t2, rec_rr, rec_reason,
      rec_composite_score, rec_ai_score,
      quantity, actual_entry, actual_exit, status, notes,
      kite_order_id, kite_sl_order_id, kite_status.
    """
    outcomes = _compute_outcomes(
        trade.get("quantity") or 1,
        trade.get("actual_entry"),
        trade.get("actual_exit"),
        trade.get("rec_entry"),
        trade.get("rec_stop"),
        trade.get("signal_type", ""),
    )
    con = get_conn()
    con.execute("""
        INSERT INTO trade_log (
            trade_date, tradingsymbol, instrument_token,
            setup_type, signal_type,
            rec_entry, rec_stop, rec_t1, rec_t2, rec_rr, rec_reason,
            rec_composite_score, rec_ai_score,
            kite_user_id, kite_order_id, kite_sl_order_id, kite_status,
            quantity, actual_entry, actual_exit, status, notes,
            pnl_amount, pnl_pct, slippage_entry_pct, rr_realised,
            is_paper_trade
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        trade.get("trade_date"),
        trade.get("tradingsymbol"),
        trade.get("instrument_token"),
        trade.get("setup_type"),
        trade.get("signal_type"),
        trade.get("rec_entry"),
        trade.get("rec_stop"),
        trade.get("rec_t1"),
        trade.get("rec_t2"),
        trade.get("rec_rr"),
        trade.get("rec_reason"),
        trade.get("rec_composite_score"),
        trade.get("rec_ai_score"),
        trade.get("kite_user_id"),
        trade.get("kite_order_id"),
        trade.get("kite_sl_order_id"),
        trade.get("kite_status"),
        trade.get("quantity"),
        trade.get("actual_entry"),
        trade.get("actual_exit"),
        trade.get("status", "OPEN"),
        trade.get("notes"),
        outcomes["pnl_amount"],
        outcomes["pnl_pct"],
        outcomes["slippage_entry_pct"],
        outcomes["rr_realised"],
        bool(trade.get("is_paper_trade", False)),
    ])
    row_id = con.execute("SELECT max(id) FROM trade_log").fetchone()[0]
    con.close()
    return row_id


def close_trade(trade_id: int, actual_exit: float, status: str, notes: str = None):
    """Update an OPEN trade with exit price and final status, recompute outcomes."""
    con = get_conn()
    row = con.execute(
        "SELECT quantity, actual_entry, rec_entry, rec_stop, signal_type, notes "
        "FROM trade_log WHERE id = ?", [trade_id]
    ).fetchone()
    if not row:
        con.close()
        return
    qty, ae, re, rs, sig, old_notes = row
    outcomes = _compute_outcomes(qty or 1, ae, actual_exit, re, rs, sig or "")
    merged_notes = "\n".join(filter(None, [old_notes, notes]))
    con.execute("""
        UPDATE trade_log
        SET actual_exit         = ?,
            status              = ?,
            notes               = ?,
            pnl_amount          = ?,
            pnl_pct             = ?,
            slippage_entry_pct  = ?,
            rr_realised         = ?
        WHERE id = ?
    """, [
        actual_exit,
        status,
        merged_notes,
        outcomes["pnl_amount"],
        outcomes["pnl_pct"],
        outcomes["slippage_entry_pct"],
        outcomes["rr_realised"],
        trade_id,
    ])
    con.close()


def delete_trade(trade_id: int):
    """Remove a trade log entry (e.g. accidental log)."""
    con = get_conn()
    con.execute("DELETE FROM trade_log WHERE id = ?", [trade_id])
    con.close()


def load_trade_log(status_filter: list = None, user_id: str = "") -> pd.DataFrame:
    """Load trade log entries for a specific user, newest first."""
    con = get_conn()
    _uid_clause  = "AND kite_user_id = ?" if user_id else ""
    _uid_params  = [user_id] if user_id else []
    if status_filter:
        placeholders = ",".join(["?"] * len(status_filter))
        where = f"WHERE status IN ({placeholders}) {_uid_clause}"
        df = con.execute(
            f"SELECT * FROM trade_log {where} ORDER BY logged_at DESC",
            status_filter + _uid_params,
        ).df()
    else:
        where = f"WHERE 1=1 {_uid_clause}" if user_id else ""
        df = con.execute(
            f"SELECT * FROM trade_log {where} ORDER BY logged_at DESC",
            _uid_params,
        ).df()
    con.close()
    return df


def sync_from_kite_orders(orders: list, user_id: str = "") -> int:
    """
    Given today's Kite order list (from client.get_orders()), update matching
    OPEN trade_log entries that have a kite_order_id.

    Rules:
      - If Kite status is COMPLETE: update actual_entry to filled average_price,
        keep our status as OPEN (still needs an exit to be "closed").
      - If Kite status is REJECTED or CANCELLED: set status = that value in DB.

    Returns the number of trade_log rows updated.
    """
    if not orders:
        return 0
    orders_map = {str(o.get("order_id", "")): o for o in orders}
    con = get_conn()
    _uid_clause = "AND kite_user_id = ?" if user_id else ""
    _uid_param  = [user_id] if user_id else []
    open_rows = con.execute(
        "SELECT id, kite_order_id, actual_entry, quantity, signal_type, rec_entry, rec_stop "
        f"FROM trade_log WHERE status = 'OPEN' AND kite_order_id IS NOT NULL {_uid_clause}",
        _uid_param,
    ).fetchall()
    updated = 0
    for row in open_rows:
        tid, kid, ae, qty, sig, re, rs = row
        if not kid:
            continue
        k = orders_map.get(str(kid))
        if not k:
            continue
        kstat = k.get("status", "")
        filled_price = k.get("average_price") or ae
        if kstat == "COMPLETE":
            # Update kite_status + actual_entry (fill price); keep trade OPEN
            # until user records an exit price
            con.execute(
                "UPDATE trade_log SET kite_status=?, actual_entry=? WHERE id=?",
                [kstat, float(filled_price) if filled_price else ae, tid],
            )
            updated += 1
        elif kstat in ("REJECTED", "CANCELLED"):
            con.execute(
                "UPDATE trade_log SET status=?, kite_status=? WHERE id=?",
                [kstat, kstat, tid],
            )
            updated += 1
    con.close()
    return updated


def get_trade_stats(user_id: str = "", is_paper: bool | None = None) -> dict:
    """
    Aggregate stats for the Activity Log summary header, scoped to user.

    Parameters
    ----------
    user_id  : Kite user ID filter ('' = all users).
    is_paper : True → paper trades only | False → real trades only | None → all.
    """
    con = get_conn()
    clauses: list[str] = []
    params: list = []
    if user_id:
        clauses.append("kite_user_id = ?")
        params.append(user_id)
    if is_paper is True:
        clauses.append("is_paper_trade = TRUE")
    elif is_paper is False:
        clauses.append("(is_paper_trade = FALSE OR is_paper_trade IS NULL)")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    row = con.execute(f"""
        SELECT
            COUNT(*)                                             AS total,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END)   AS open_count,
            SUM(CASE WHEN pnl_amount > 0   THEN 1 ELSE 0 END)  AS wins,
            SUM(CASE WHEN pnl_amount <= 0 AND status != 'OPEN' THEN 1 ELSE 0 END) AS losses,
            COALESCE(SUM(pnl_amount), 0)                        AS total_pnl,
            AVG(CASE WHEN rr_realised IS NOT NULL THEN rr_realised END) AS avg_rr,
            MAX(pnl_amount)                                      AS best_trade,
            MIN(pnl_amount)                                      AS worst_trade
        FROM trade_log {where}
    """, params).fetchone()
    con.close()
    if row is None:
        return {}
    total, open_c, wins, losses, tot_pnl, avg_rr, best, worst = row
    closed = (total or 0) - (open_c or 0)
    win_rate = (wins / closed * 100) if closed > 0 else 0.0
    return {
        "total":      total or 0,
        "open":       open_c or 0,
        "closed":     closed,
        "wins":       wins or 0,
        "losses":     losses or 0,
        "win_rate":   win_rate,
        "total_pnl":  tot_pnl or 0.0,
        "avg_rr":     avg_rr,
        "best_trade": best,
        "worst_trade": worst,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PAPER TRADING — virtual trade tracking and signal tuning
# ═══════════════════════════════════════════════════════════════════════════

def get_open_paper_trades(user_id: str = "", trade_date=None) -> list:
    """
    Return open paper trades for a given date (defaults to today) scoped to user.
    Each row → dict with id, tradingsymbol, signal_type, actual_entry, rec_stop, rec_t1.
    """
    import datetime as _dt
    d = trade_date or _dt.date.today()
    con = get_conn()
    _uid_clause = "AND kite_user_id = ?" if user_id else ""
    _params = [str(d)] + ([user_id] if user_id else [])
    rows = con.execute(
        f"""SELECT id, tradingsymbol, signal_type, actual_entry, rec_stop, rec_t1
            FROM trade_log
            WHERE is_paper_trade = TRUE
              AND status = 'OPEN'
              AND trade_date = ?
              {_uid_clause}""",
        _params,
    ).fetchall()
    con.close()
    return [
        {
            "id":            r[0],
            "tradingsymbol": r[1],
            "signal_type":   r[2],
            "actual_entry":  r[3],
            "rec_stop":      r[4],
            "rec_t1":        r[5],
        }
        for r in rows
    ]


def get_today_closed_pnl(user_id: str = "", is_paper: bool | None = None) -> float:
    """
    Returns the total realised P&L for trades closed TODAY.

    "Closed today" means the trade has a non-OPEN status AND was either:
      • opened today (trade_date = today), OR
      • logged/updated today (logged_at date = today)
    We use GREATEST(trade_date, DATE(logged_at)) so intraday trades opened and
    closed on the same day are always captured, even if the exit timestamp
    rounds to a different date in edge cases.

    Parameters
    ----------
    user_id  : Kite user ID filter ('' = all users).
    is_paper : True → paper trades only | False → real trades only | None → both.
    """
    _ensure_connection()
    import datetime as _dt
    _today = _dt.date.today().isoformat()
    clauses: list[str] = [
        "status IN ('CLOSED', 'TARGET_HIT', 'STOPPED_OUT')",
        "pnl_amount IS NOT NULL",
        "(trade_date = ? OR CAST(logged_at AS DATE) = ?)",
    ]
    params: list = [_today, _today]
    if user_id:
        clauses.append("kite_user_id = ?")
        params.append(user_id)
    if is_paper is True:
        clauses.append("is_paper_trade = TRUE")
    elif is_paper is False:
        clauses.append("(is_paper_trade = FALSE OR is_paper_trade IS NULL)")
    where = " AND ".join(clauses)
    try:
        row = _conn.execute(
            f"SELECT COALESCE(SUM(pnl_amount), 0.0) FROM trade_log WHERE {where}",
            params,
        ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def get_paper_trade_perf(user_id: str = "", days: int = 30) -> dict:
    """
    Aggregate paper trade performance over the last `days` calendar days,
    broken down by signal_type (BUY_ABOVE / SELL_BELOW).

    Returns:
      {
        "BUY_ABOVE":  {"total": int, "wins": int, "losses": int, "win_rate": float,
                       "avg_rr": float, "avg_pnl_pct": float},
        "SELL_BELOW": {...},
        "overall":    {"total": int, "win_rate": float, "avg_rr": float},
      }
    """
    con = get_conn()
    _uid_clause = "AND kite_user_id = ?" if user_id else ""
    _params = [days] + ([user_id] if user_id else [])
    rows = con.execute(
        f"""SELECT signal_type,
                   COUNT(*)                                              AS total,
                   SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END)     AS wins,
                   SUM(CASE WHEN pnl_amount <= 0 AND pnl_amount IS NOT NULL
                            THEN 1 ELSE 0 END)                          AS losses,
                   AVG(CASE WHEN rr_realised IS NOT NULL
                            THEN rr_realised END)                       AS avg_rr,
                   AVG(pnl_pct)                                         AS avg_pnl_pct
            FROM trade_log
            WHERE is_paper_trade = TRUE
              AND status IN ('TARGET_HIT','STOPPED_OUT','CLOSED')
              AND trade_date >= CURRENT_DATE - INTERVAL (?) DAY
              {_uid_clause}
            GROUP BY signal_type""",
        _params,
    ).fetchall()
    con.close()

    result = {}
    overall_total = overall_wins = 0
    overall_rr_sum = overall_rr_n = 0
    for r in rows:
        sig, total, wins, losses, avg_rr, avg_pnl_pct = r
        closed = total or 0
        win_rate = (wins / closed * 100) if closed > 0 else 0.0
        result[sig] = {
            "total":       total or 0,
            "wins":        wins  or 0,
            "losses":      losses or 0,
            "win_rate":    round(win_rate, 1),
            "avg_rr":      round(avg_rr,   2) if avg_rr else None,
            "avg_pnl_pct": round(avg_pnl_pct, 2) if avg_pnl_pct else None,
        }
        overall_total += total or 0
        overall_wins  += wins  or 0
        if avg_rr:
            overall_rr_sum += avg_rr * (total or 0)
            overall_rr_n   += total or 0

    result["overall"] = {
        "total":    overall_total,
        "win_rate": round(overall_wins / overall_total * 100, 1) if overall_total else 0.0,
        "avg_rr":   round(overall_rr_sum / overall_rr_n, 2) if overall_rr_n else None,
    }
    return result


# ─── Signal config (algo-tuning overrides) ───────────────────────────────────

_SIGNAL_CONFIG_DEFAULTS = {
    "intraday_rsi_buy_max":  75.0,
    "intraday_rsi_sell_min": 25.0,
    "intraday_min_rr":        1.5,
}


def get_signal_config(user_id: str = "") -> dict:
    """
    Load per-user signal tuning overrides from signal_config table.
    Falls back to defaults from _SIGNAL_CONFIG_DEFAULTS for missing keys.
    """
    con = get_conn()
    uid = user_id or ""
    rows = con.execute(
        "SELECT config_key, value FROM signal_config WHERE kite_user_id = ?",
        [uid],
    ).fetchall()
    con.close()
    cfg = dict(_SIGNAL_CONFIG_DEFAULTS)
    for key, val in rows:
        cfg[key] = val
    return cfg


def save_signal_config(cfg: dict, user_id: str = "") -> None:
    """
    Upsert signal tuning overrides into signal_config table.
    Only keys present in cfg are written; others are left untouched.
    """
    uid = user_id or ""
    con = get_conn()
    for key, val in cfg.items():
        con.execute(
            """INSERT INTO signal_config (config_key, kite_user_id, value, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT (config_key, kite_user_id)
               DO UPDATE SET value = excluded.value,
                             updated_at = excluded.updated_at""",
            [key, uid, float(val)],
        )
    con.close()


def tune_signal_config_from_paper(user_id: str = "", days: int = 30) -> dict:
    """
    Derive updated signal thresholds from recent paper trade performance.

    Tuning rules (conservative — only moves in one direction per call):
      BUY_ABOVE win rate < 40% → tighten RSI max by 5 (floor 55)
                                  raise min R/R by 0.5 (cap 3.0)
      BUY_ABOVE win rate > 65% → relax RSI max by 3 (cap 75)
                                  lower min R/R by 0.25 (floor 1.5)
      SELL_BELOW win rate < 40% → tighten RSI min by 5 (cap 45)
      SELL_BELOW win rate > 65% → relax RSI min by 3 (floor 25)

    Returns a dict of {config_key: new_value, ...} that was changed,
    or an empty dict if no adjustment was made.
    """
    perf = get_paper_trade_perf(user_id=user_id, days=days)
    current = get_signal_config(user_id=user_id)
    changes: dict = {}

    long_data  = perf.get("BUY_ABOVE",  {})
    short_data = perf.get("SELL_BELOW", {})

    # Only tune when we have at least 5 closed trades to avoid noise
    if long_data.get("total", 0) >= 5:
        wr = long_data["win_rate"]
        cur_rsi = current["intraday_rsi_buy_max"]
        cur_rr  = current["intraday_min_rr"]
        if wr < 40.0:
            new_rsi = max(55.0, cur_rsi - 5.0)
            new_rr  = min(3.0,  cur_rr  + 0.5)
            if new_rsi != cur_rsi: changes["intraday_rsi_buy_max"] = new_rsi
            if new_rr  != cur_rr:  changes["intraday_min_rr"]      = new_rr
        elif wr > 65.0:
            new_rsi = min(75.0, cur_rsi + 3.0)
            new_rr  = max(1.5,  cur_rr  - 0.25)
            if new_rsi != cur_rsi: changes["intraday_rsi_buy_max"] = new_rsi
            if new_rr  != cur_rr:  changes["intraday_min_rr"]      = new_rr

    if short_data.get("total", 0) >= 5:
        wr = short_data["win_rate"]
        cur_rsi = current["intraday_rsi_sell_min"]
        if wr < 40.0:
            new_rsi = min(45.0, cur_rsi + 5.0)
            if new_rsi != cur_rsi: changes["intraday_rsi_sell_min"] = new_rsi
        elif wr > 65.0:
            new_rsi = max(25.0, cur_rsi - 3.0)
            if new_rsi != cur_rsi: changes["intraday_rsi_sell_min"] = new_rsi

    if changes:
        merged = {**current, **changes}
        save_signal_config(merged, user_id=user_id)

    return changes
