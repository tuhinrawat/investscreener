"""
db.py — DuckDB schema and helpers.

Why DuckDB and not SQLite?
  - DuckDB is columnar — analytical queries (window functions, aggregates
    over 252-day series) are 10-100x faster
  - Native pandas integration via .df()
  - Single-file like SQLite, no server
  - Trade-off: single-writer (fine for us, screener is single-user)

Why three tables?
  - instruments: changes weekly, ~2K rows
  - daily_ohlcv: changes daily, ~500K rows after 252 days × 2K stocks
  - computed_metrics: derived, recomputed on full rescan, ~2K rows
"""
import duckdb
import pandas as pd
from pathlib import Path
import config


def get_conn():
    """Returns a fresh connection. DuckDB is single-process; reopen each time."""
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(config.DB_PATH))


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
    con.close()


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
