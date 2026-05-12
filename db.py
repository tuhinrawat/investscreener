"""
db.py — PostgreSQL (Neon) schema and helpers.

Previously used DuckDB (local file). Migrated to Neon serverless Postgres so
that trade history, signals and activity log survive Streamlit Cloud re-deploys
and support multiple concurrent users.

Connection:
  DATABASE_URL env var (set in .env locally, Streamlit secrets in cloud).
  Falls back to a local SQLite-style warning if not set.

Tables:
  instruments      - weekly refresh, ~2K rows
  daily_ohlcv      - daily candles, ~500K rows
  computed_metrics - derived screener output, ~2K rows
  trade_log        - paper + real trade journal
  signal_config    - per-user algo-tuning overrides
  market_intel_log - AI market intel run log
  market_intel_stocks - stocks returned by market intel
  _db_meta         - one-time migration flags
"""

import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
import pandas as pd
from datetime import datetime as _dt_cls, timezone as _tz, timedelta as _td
import config

_IST = _tz(_td(hours=5, minutes=30))


def _now_ist() -> _dt_cls:
    """Current datetime in IST, timezone-naive (safe for TIMESTAMP columns)."""
    return _dt_cls.now(_IST).replace(tzinfo=None)


# ── Connection pool ───────────────────────────────────────────────────────────
# ThreadedConnectionPool is safe for Streamlit's multi-threaded execution model.
# Neon's built-in PgBouncer absorbs rapid connect/disconnect cycles.

_DATABASE_URL: str = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("NEON_DATABASE_URL")
    or ""
)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool, _DATABASE_URL
    if not _DATABASE_URL:
        # 1. Try Streamlit secrets (cloud deployment)
        try:
            import streamlit as _st_inner
            _u = _st_inner.secrets.get("DATABASE_URL", "")
            if _u:
                _DATABASE_URL = _u
                os.environ["DATABASE_URL"] = _u
        except Exception:
            pass
    if not _DATABASE_URL:
        # 2. Try .env with explicit path (local dev, CWD-independent)
        try:
            from dotenv import load_dotenv
            from pathlib import Path as _PL
            load_dotenv(dotenv_path=_PL(__file__).parent / ".env", override=False)
            _DATABASE_URL = os.environ.get("DATABASE_URL", "")
        except ImportError:
            pass
    if not _DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to .env or Streamlit secrets."
        )
    if _pool is None or _pool.closed:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=20, dsn=_DATABASE_URL
        )
    return _pool


def get_conn() -> psycopg2.extensions.connection:
    """
    Borrow a connection from the pool.

    Uses conn.closed (synchronous psycopg2 attribute — no network round-trip)
    as a fast pre-check. Only falls back to a full SELECT 1 ping when the
    connection appears open but may be stale (Neon idle timeout). Rebuilds the
    entire pool on dead-connection detection.
    """
    global _pool
    pool = _get_pool()
    conn = pool.getconn()
    if conn.closed:
        # Definitely dead — rebuild pool immediately
        try:
            pool.closeall()
        except Exception:
            pass
        _pool = None
        return _get_pool().getconn()
    try:
        # Fast transaction-state reset — no extra round-trip if already clean
        conn.rollback()
    except Exception:
        # rollback failed → connection is dead despite conn.closed == 0
        try:
            pool.closeall()
        except Exception:
            pass
        _pool = None
        conn = _get_pool().getconn()
    return conn


def release_conn(conn) -> None:
    """Return a connection to the pool (preferred over conn.close())."""
    try:
        # Roll back any open transaction so the connection is clean for reuse
        try:
            conn.rollback()
        except Exception:
            pass
        _get_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def _df_from_cursor(cur) -> pd.DataFrame:
    """Build a DataFrame from an executed psycopg2 cursor."""
    if cur.description is None:
        return pd.DataFrame()
    cols = [d.name for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


# ── Schema ────────────────────────────────────────────────────────────────────

_schema_initialized: bool = False  # module-level guard — init runs once per process

def init_schema():
    """Creates all tables if they don't exist. Idempotent — safe every run.

    Protected by a module-level flag so it only executes once per Python process.
    Repeated calls from Streamlit reruns are instant no-ops.
    """
    global _schema_initialized
    if _schema_initialized:
        return
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_token  BIGINT PRIMARY KEY,
                tradingsymbol     VARCHAR,
                name              VARCHAR,
                exchange          VARCHAR,
                segment           VARCHAR,
                instrument_type   VARCHAR,
                tick_size         DOUBLE PRECISION,
                lot_size          INTEGER,
                last_updated      TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_ohlcv (
                instrument_token  BIGINT,
                date              DATE,
                open              DOUBLE PRECISION,
                high              DOUBLE PRECISION,
                low               DOUBLE PRECISION,
                close             DOUBLE PRECISION,
                volume            BIGINT,
                PRIMARY KEY (instrument_token, date)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS computed_metrics (
                instrument_token        BIGINT PRIMARY KEY,
                tradingsymbol           VARCHAR,
                ltp                     DOUBLE PRECISION,
                avg_turnover_cr         DOUBLE PRECISION,
                avg_volume              BIGINT,
                ret_5d                  DOUBLE PRECISION,
                ret_1m                  DOUBLE PRECISION,
                ret_3m                  DOUBLE PRECISION,
                ret_6m                  DOUBLE PRECISION,
                ret_1y                  DOUBLE PRECISION,
                rs_vs_nifty_3m          DOUBLE PRECISION,
                vol_expansion_ratio     DOUBLE PRECISION,
                rsi_14                  DOUBLE PRECISION,
                ema_20                  DOUBLE PRECISION,
                ema_50                  DOUBLE PRECISION,
                ema_200                 DOUBLE PRECISION,
                atr_14                  DOUBLE PRECISION,
                high_52w                DOUBLE PRECISION,
                low_52w                 DOUBLE PRECISION,
                dist_from_52w_high_pct  DOUBLE PRECISION,
                dist_from_50ema_pct     DOUBLE PRECISION,
                support_20d             DOUBLE PRECISION,
                resistance_20d          DOUBLE PRECISION,
                trend_score             DOUBLE PRECISION,
                composite_score         DOUBLE PRECISION,
                -- Swing
                swing_signal            VARCHAR,
                swing_setup             VARCHAR,
                swing_entry             DOUBLE PRECISION,
                swing_stop              DOUBLE PRECISION,
                swing_t1                DOUBLE PRECISION,
                swing_t2                DOUBLE PRECISION,
                swing_rr                DOUBLE PRECISION,
                swing_quality           INTEGER,
                swing_reason            VARCHAR,
                -- Intraday
                intraday_signal         VARCHAR,
                intraday_pivot          DOUBLE PRECISION,
                intraday_r1             DOUBLE PRECISION,
                intraday_r2             DOUBLE PRECISION,
                intraday_r3             DOUBLE PRECISION,
                intraday_s1             DOUBLE PRECISION,
                intraday_s2             DOUBLE PRECISION,
                intraday_s3             DOUBLE PRECISION,
                intraday_entry          DOUBLE PRECISION,
                intraday_stop           DOUBLE PRECISION,
                intraday_t1             DOUBLE PRECISION,
                intraday_t2             DOUBLE PRECISION,
                intraday_reason         VARCHAR,
                intraday_confidence     INTEGER,
                intraday_gap_flag       VARCHAR,
                intraday_nifty_gate     VARCHAR,
                -- Scaling
                scale_signal            VARCHAR,
                scale_setup             VARCHAR,
                scale_entry_1           DOUBLE PRECISION,
                scale_stop              DOUBLE PRECISION,
                scale_trailing_stop     DOUBLE PRECISION,
                scale_target            DOUBLE PRECISION,
                scale_quality           INTEGER,
                scale_reason            VARCHAR,
                -- AI analysis
                ai_score                DOUBLE PRECISION,
                ai_verdict              VARCHAR,
                ai_confidence           VARCHAR,
                ai_brief                TEXT,
                ai_analyzed_at          TIMESTAMP,
                last_updated            TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id                  BIGSERIAL PRIMARY KEY,
                logged_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trade_date          DATE,
                tradingsymbol       VARCHAR,
                instrument_token    BIGINT,
                setup_type          VARCHAR,
                signal_type         VARCHAR,
                rec_entry           DOUBLE PRECISION,
                rec_stop            DOUBLE PRECISION,
                rec_t1              DOUBLE PRECISION,
                rec_t2              DOUBLE PRECISION,
                rec_rr              DOUBLE PRECISION,
                rec_reason          VARCHAR,
                rec_composite_score DOUBLE PRECISION,
                rec_ai_score        DOUBLE PRECISION,
                kite_user_id        VARCHAR,
                kite_order_id       VARCHAR,
                kite_sl_order_id    VARCHAR,
                kite_target_order_id VARCHAR,
                kite_status         VARCHAR,
                quantity            INTEGER,
                actual_entry        DOUBLE PRECISION,
                actual_exit         DOUBLE PRECISION,
                status              VARCHAR,
                notes               TEXT,
                pnl_amount          DOUBLE PRECISION,
                pnl_pct             DOUBLE PRECISION,
                slippage_entry_pct  DOUBLE PRECISION,
                rr_realised         DOUBLE PRECISION,
                is_paper_trade      BOOLEAN,
                intraday_confidence INTEGER
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_config (
                config_key   VARCHAR     NOT NULL,
                kite_user_id VARCHAR     NOT NULL DEFAULT '',
                value        DOUBLE PRECISION NOT NULL,
                updated_at   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (config_key, kite_user_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS _db_meta (
                key   VARCHAR PRIMARY KEY,
                value VARCHAR
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_intel_log (
                id                  BIGSERIAL PRIMARY KEY,
                kite_user_id        VARCHAR DEFAULT '',
                created_at          TIMESTAMP,
                raw_output          TEXT,
                overall_bias        VARCHAR,
                overall_confidence  VARCHAR
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_intel_stocks (
                id                   BIGSERIAL PRIMARY KEY,
                intel_id             BIGINT,
                kite_user_id         VARCHAR DEFAULT '',
                created_at           TIMESTAMP,
                tradingsymbol        VARCHAR,
                stance               VARCHAR,
                sector               VARCHAR,
                fundamental_reason   TEXT,
                entry_trigger        TEXT,
                stop_loss            VARCHAR,
                conviction           VARCHAR,
                condition_required   TEXT,
                alert_level          VARCHAR,
                expected_move        VARCHAR
            );
        """)

        # ── Users and sessions ─────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id                      BIGSERIAL PRIMARY KEY,
                username                VARCHAR(100) UNIQUE NOT NULL,
                password_hash           VARCHAR NOT NULL,
                kite_api_key            VARCHAR DEFAULT '',
                kite_api_secret         VARCHAR DEFAULT '',
                kite_user_id            VARCHAR DEFAULT '',
                kite_access_token       VARCHAR DEFAULT '',
                kite_token_updated_at   TIMESTAMP,
                openrouter_key          VARCHAR DEFAULT '',
                openai_key              VARCHAR DEFAULT '',
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at           TIMESTAMP
            );
        """)
        # Migrate existing tables that may not have the AI key columns yet
        for _col, _dflt in [("openrouter_key", "''"), ("openai_key", "''")] :
            cur.execute("""
                DO $$ BEGIN
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS openrouter_key VARCHAR DEFAULT '';
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS openai_key      VARCHAR DEFAULT '';
                EXCEPTION WHEN others THEN NULL;
                END $$;
            """)
            break  # single idempotent block is enough
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id              BIGSERIAL PRIMARY KEY,
                user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_token   VARCHAR(128) UNIQUE NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_capital (
                user_id         BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                paper_balance   DOUBLE PRECISION NOT NULL DEFAULT 900000,
                last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ── Idempotent column additions for older DBs ──────────────────────────
        _add_column_if_missing(cur, "computed_metrics", "intraday_r3",       "DOUBLE PRECISION")
        _add_column_if_missing(cur, "computed_metrics", "intraday_s3",       "DOUBLE PRECISION")
        _add_column_if_missing(cur, "computed_metrics", "intraday_t2",       "DOUBLE PRECISION")
        _add_column_if_missing(cur, "computed_metrics", "intraday_gap_flag", "VARCHAR")
        _add_column_if_missing(cur, "computed_metrics",  "intraday_nifty_gate",    "VARCHAR")
        _add_column_if_missing(cur, "trade_log",         "intraday_confidence",    "INTEGER")
        _add_column_if_missing(cur, "trade_log",         "rec_t2",                 "DOUBLE PRECISION")
        _add_column_if_missing(cur, "trade_log",         "kite_target_order_id",   "VARCHAR")

        # One-time migration: UTC → IST shift on logged_at
        cur.execute("SELECT value FROM _db_meta WHERE key = 'logged_at_utc_to_ist_done'")
        if not cur.fetchone():
            cur.execute("""
                UPDATE trade_log
                SET logged_at = logged_at + INTERVAL '5 hours 30 minutes'
                WHERE logged_at IS NOT NULL
            """)
            cur.execute(
                "INSERT INTO _db_meta (key, value) VALUES ('logged_at_utc_to_ist_done', '1') "
                "ON CONFLICT (key) DO NOTHING"
            )

        # ── Performance indexes (idempotent) ──────────────────────────────────
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_log_user_status
                ON trade_log (kite_user_id, status, trade_date DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_log_paper_date
                ON trade_log (is_paper_trade, status, trade_date)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_token_date
                ON daily_ohlcv (instrument_token, date DESC)
        """)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)
    _schema_initialized = True


def _add_column_if_missing(cur, table: str, col: str, dtype: str) -> None:
    """Postgres-compatible idempotent column add."""
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        [table, col],
    )
    if not cur.fetchone():
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")


# ── Pure math helpers ─────────────────────────────────────────────────────────

def compute_trade_charges(
    entry: float,
    exit_price: float,
    qty: int,
    setup_type: str = "INTRADAY",
    exchange: str = "NSE",
) -> dict:
    """
    Zerodha/exchange statutory charges for one completed equity trade.
    Rates: https://zerodha.com/charges/ (May 2026).
    """
    zero = {k: 0.0 for k in ["brokerage", "stt", "txn_charges", "sebi", "gst", "stamp", "dp", "total"]}
    if not (entry > 0 and exit_price > 0 and qty > 0):
        return zero

    buy_val  = float(entry)      * int(qty)
    sell_val = float(exit_price) * int(qty)
    turnover = buy_val + sell_val

    is_intraday = str(setup_type).upper() in ("INTRADAY", "SCALP")

    if is_intraday:
        brokerage = min(20.0, 0.0003 * buy_val) + min(20.0, 0.0003 * sell_val)
        stt   = 0.00025 * sell_val
        stamp = 0.00003 * buy_val
        dp    = 0.0
    else:
        brokerage = 0.0
        stt   = 0.001   * turnover
        stamp = 0.00015 * buy_val
        dp    = 15.34

    txn_rate    = 0.0000307 if exchange.upper() == "NSE" else 0.0000375
    txn_charges = txn_rate * turnover
    sebi        = 0.000001 * turnover
    gst         = 0.18 * (brokerage + txn_charges + sebi)
    total       = brokerage + stt + txn_charges + sebi + gst + stamp + dp

    return {
        "brokerage":   round(brokerage,   2),
        "stt":         round(stt,         2),
        "txn_charges": round(txn_charges, 2),
        "sebi":        round(sebi,        4),
        "gst":         round(gst,         2),
        "stamp":       round(stamp,       2),
        "dp":          round(dp,          2),
        "total":       round(total,       2),
    }


def _compute_outcomes(quantity, actual_entry, actual_exit,
                      rec_entry, rec_stop, signal_type) -> dict:
    direction = -1 if signal_type in ("SELL", "SELL_BELOW", "SELL_ORB") else 1
    pnl_amount = pnl_pct = rr_realised = slippage = None
    if actual_exit is not None and actual_entry:
        pnl_amount  = direction * (actual_exit - actual_entry) * (quantity or 1)
        pnl_pct     = direction * (actual_exit - actual_entry) / actual_entry * 100
        risk        = abs(actual_entry - rec_stop) if rec_stop else None
        actual_gain = direction * (actual_exit - actual_entry)
        if risk and risk > 0:
            rr_realised = actual_gain / risk
    if rec_entry and actual_entry:
        # For LONG: paying more than rec = bad → positive slippage
        # For SHORT: selling below rec = bad → positive slippage (flip sign)
        raw_slip = (actual_entry - rec_entry) / rec_entry * 100
        slippage = raw_slip if direction == 1 else -raw_slip
    return {
        "pnl_amount":         pnl_amount,
        "pnl_pct":            pnl_pct,
        "slippage_entry_pct": slippage,
        "rr_realised":        rr_realised,
    }


# ── Instrument / OHLCV / Metrics ──────────────────────────────────────────────

def upsert_instruments(df: pd.DataFrame):
    """Replace full instruments table (small, weekly refresh)."""
    if df.empty:
        return
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM instruments")
        cols = ["instrument_token", "tradingsymbol", "name", "exchange",
                "segment", "instrument_type", "tick_size", "lot_size", "last_updated"]
        rows = [tuple(row[c] for c in cols) for _, row in df[cols].iterrows()]
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO instruments (instrument_token, tradingsymbol, name, exchange, "
            "segment, instrument_type, tick_size, lot_size, last_updated) VALUES %s",
            rows,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


_OHLCV_SQL = """
    INSERT INTO daily_ohlcv (instrument_token, date, open, high, low, close, volume)
    VALUES %s
    ON CONFLICT (instrument_token, date) DO UPDATE SET
        open   = EXCLUDED.open,
        high   = EXCLUDED.high,
        low    = EXCLUDED.low,
        close  = EXCLUDED.close,
        volume = EXCLUDED.volume
"""
_UPSERT_CHUNK = 2000   # rows per commit — keeps each connection window short
_UPSERT_RETRIES = 3    # retry on SSL drop with a fresh connection


def upsert_ohlcv(df: pd.DataFrame):
    """
    Append-or-replace daily candles.

    Chunked into _UPSERT_CHUNK rows per commit so a long full-scan never
    holds a single Neon connection open long enough for the SSL idle-timeout
    to fire.  Each chunk is retried up to _UPSERT_RETRIES times with a fresh
    connection on OperationalError / InterfaceError (SSL drop).
    """
    if df.empty:
        return
    cols  = ["instrument_token", "date", "open", "high", "low", "close", "volume"]
    avail = [c for c in cols if c in df.columns]
    rows  = [tuple(row[c] for c in avail) for _, row in df[avail].iterrows()]

    for chunk_start in range(0, len(rows), _UPSERT_CHUNK):
        chunk = rows[chunk_start: chunk_start + _UPSERT_CHUNK]
        last_exc = None
        for attempt in range(_UPSERT_RETRIES):
            conn = get_conn()          # always fresh from pool (pool rebuilds if dead)
            cur  = conn.cursor()
            try:
                psycopg2.extras.execute_values(cur, _OHLCV_SQL, chunk, page_size=500)
                conn.commit()
                last_exc = None
                break                  # success — move to next chunk
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                # SSL connection dropped — don't bother rolling back a dead conn
                last_exc = exc
                try:
                    conn.close()
                except Exception:
                    pass
                # Force pool rebuild so next get_conn() opens a fresh SSL session
                global _pool
                try:
                    if _pool:
                        _pool.closeall()
                except Exception:
                    pass
                _pool = None
            except Exception as exc:
                last_exc = exc
                try:
                    conn.rollback()
                except Exception:
                    pass
                break                  # non-retriable error — propagate below
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
                try:
                    release_conn(conn)
                except Exception:
                    pass
        if last_exc is not None:
            raise last_exc


def replace_metrics(df: pd.DataFrame):
    """Full table replace — metrics are always recomputed end-to-end."""
    if df.empty:
        return
    conn = get_conn()
    cur  = conn.cursor()
    try:
        # Get actual table columns from Postgres catalog
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'computed_metrics' ORDER BY ordinal_position"
        )
        table_cols = [r[0] for r in cur.fetchall()]
        keep = [c for c in df.columns if c in table_cols]
        df_clean = df[keep].copy()

        cur.execute("DELETE FROM computed_metrics")

        if df_clean.empty:
            conn.commit()
            return

        col_list  = ", ".join(keep)
        placeholders = ", ".join(["%s"] * len(keep))
        rows = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in df_clean.itertuples(index=False, name=None)
        ]
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO computed_metrics ({col_list}) VALUES %s",
            rows,
            page_size=500,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def load_metrics() -> pd.DataFrame:
    """Returns current screener output joined with instrument names."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT m.*, i.name AS company_name
            FROM computed_metrics m
            LEFT JOIN instruments i USING (instrument_token)
            ORDER BY composite_score DESC NULLS LAST
        """)
        return _df_from_cursor(cur)
    finally:
        cur.close()
        release_conn(conn)


def load_ohlcv(instrument_token: int) -> pd.DataFrame:
    """All cached daily candles for one stock (for chart panel)."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT date, open, high, low, close, volume FROM daily_ohlcv "
            "WHERE instrument_token = %s ORDER BY date",
            [instrument_token],
        )
        return _df_from_cursor(cur)
    finally:
        cur.close()
        release_conn(conn)


def load_ohlcv_bulk(instrument_tokens: list) -> dict:
    """Load OHLCV for ALL tokens in a single query. Returns {token: DataFrame}.

    Replaces N individual load_ohlcv() calls in compute_metrics / refresh_signals
    — one round-trip instead of N, dramatically faster on Neon serverless.
    """
    if not instrument_tokens:
        return {}
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT instrument_token, date, open, high, low, close, volume "
            "FROM daily_ohlcv WHERE instrument_token = ANY(%s) ORDER BY instrument_token, date",
            [list(instrument_tokens)],
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        release_conn(conn)

    result: dict = {}
    for row in rows:
        tok = row[0]
        if tok not in result:
            result[tok] = []
        result[tok].append(row[1:])  # (date, open, high, low, close, volume)

    return {
        tok: pd.DataFrame(vals, columns=["date", "open", "high", "low", "close", "volume"])
        for tok, vals in result.items()
    }


def save_ai_result(instrument_token: int, result: dict):
    """Persist STOCKLENS AI analysis for one stock."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            UPDATE computed_metrics
            SET ai_score      = %s,
                ai_verdict    = %s,
                ai_confidence = %s,
                ai_brief      = %s,
                ai_analyzed_at= %s
            WHERE instrument_token = %s
        """, [
            result.get("ai_score"),
            result.get("ai_verdict"),
            result.get("ai_confidence"),
            result.get("ai_brief"),
            result.get("ai_analyzed_at"),
            instrument_token,
        ])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_instruments_age_days() -> int:
    """Days since instruments table was last refreshed (-1 if empty)."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT EXTRACT(DAY FROM (NOW() - MAX(last_updated))) FROM instruments"
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return -1
        return int(row[0])
    finally:
        cur.close()
        release_conn(conn)


def get_last_metrics_update() -> str:
    """ISO timestamp of last full rescan, or 'never'."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT MAX(last_updated) FROM computed_metrics")
        row = cur.fetchone()
        if row is None or row[0] is None:
            return "never"
        return row[0].strftime("%Y-%m-%d %H:%M:%S")
    finally:
        cur.close()
        release_conn(conn)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LOG
# ═══════════════════════════════════════════════════════════════════════════════

def log_trade(trade: dict) -> int:
    """
    Insert a new trade log entry. Returns the new row id.
    Accepts all keys from the trade dict; unknown keys are ignored.
    """
    outcomes = _compute_outcomes(
        trade.get("quantity") or 1,
        trade.get("actual_entry"),
        trade.get("actual_exit"),
        trade.get("rec_entry"),
        trade.get("rec_stop"),
        trade.get("signal_type", ""),
    )
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO trade_log (
                trade_date, tradingsymbol, instrument_token,
                setup_type, signal_type,
                rec_entry, rec_stop, rec_t1, rec_t2, rec_rr, rec_reason,
                rec_composite_score, rec_ai_score,
                kite_user_id, kite_order_id, kite_sl_order_id, kite_target_order_id,
                kite_status,
                quantity, actual_entry, actual_exit, status, notes,
                pnl_amount, pnl_pct, slippage_entry_pct, rr_realised,
                is_paper_trade, intraday_confidence, logged_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            ) RETURNING id
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
            trade.get("kite_target_order_id"),
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
            trade.get("intraday_confidence"),
            trade.get("logged_at") or _now_ist(),
        ])
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def close_trade(trade_id: int, actual_exit: float, status: str, notes: str = None):
    """Update an OPEN trade with exit price and final status, recompute P&L.

    For paper trades the realised P&L is immediately reflected in user_capital.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT quantity, actual_entry, rec_entry, rec_stop, signal_type, notes, "
            "       is_paper_trade, kite_user_id "
            "FROM trade_log WHERE id = %s",
            [trade_id],
        )
        row = cur.fetchone()
        if not row:
            return
        qty, ae, re, rs, sig, old_notes, is_paper, kite_uid = row
        outcomes = _compute_outcomes(qty or 1, ae, actual_exit, re, rs, sig or "")
        merged_notes = "\n".join(filter(None, [old_notes, notes]))
        cur.execute("""
            UPDATE trade_log
            SET actual_exit        = %s,
                status             = %s,
                notes              = %s,
                pnl_amount         = %s,
                pnl_pct            = %s,
                slippage_entry_pct = %s,
                rr_realised        = %s
            WHERE id = %s
        """, [
            actual_exit, status, merged_notes,
            outcomes["pnl_amount"], outcomes["pnl_pct"],
            outcomes["slippage_entry_pct"], outcomes["rr_realised"],
            trade_id,
        ])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)

    # ── Cumulative capital: adjust balance for paper trades only ──────────
    if is_paper and kite_uid and outcomes.get("pnl_amount") is not None:
        try:
            adjust_user_capital(kite_uid, outcomes["pnl_amount"])
        except Exception:
            pass  # never let a capital-update failure block a trade close


def get_nifty50_tokens() -> dict:
    """
    Return {instrument_token: tradingsymbol} for Nifty 50 constituent stocks.
    Queries the instruments table using the hardcoded symbol list from config.
    Returns an empty dict if the instruments table is empty or hasn't been loaded.
    """
    from config import NIFTY50_SYMBOLS  # noqa: PLC0415
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT instrument_token, tradingsymbol FROM instruments "
                "WHERE exchange = 'NSE' AND tradingsymbol = ANY(%s)",
                [NIFTY50_SYMBOLS],
            )
            return {int(row[0]): str(row[1]) for row in cur.fetchall()}
    except Exception:
        return {}
    finally:
        put_conn(conn)


def get_user_capital(user_id) -> float:
    """Return the user's current paper balance.  Falls back to PAPER_CAPITAL if no row yet."""
    from config import PAPER_CAPITAL  # local import to avoid circular
    if not user_id:
        return float(PAPER_CAPITAL)
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT paper_balance FROM user_capital WHERE user_id = %s",
            [user_id],
        )
        row = cur.fetchone()
        return float(row[0]) if row else float(PAPER_CAPITAL)
    finally:
        cur.close()
        release_conn(conn)


def seed_user_capital_if_missing(user_id) -> None:
    """Insert a default ₹9L capital row for an existing user who predates this feature."""
    from config import PAPER_CAPITAL
    if not user_id:
        return
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO user_capital (user_id, paper_balance)
               VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING""",
            [user_id, PAPER_CAPITAL],
        )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        release_conn(conn)


def adjust_user_capital(user_id, delta: float) -> float:
    """Add *delta* (positive = profit, negative = loss) to the user's paper balance.

    Uses INSERT … ON CONFLICT (upsert) so it works even if the row doesn't exist yet
    (seeds from PAPER_CAPITAL + delta in that case).
    Returns the new balance.
    """
    from config import PAPER_CAPITAL
    if not user_id:
        return float(PAPER_CAPITAL)
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO user_capital (user_id, paper_balance, last_updated_at)
                VALUES (%s, %s + %s, NOW())
            ON CONFLICT (user_id) DO UPDATE
                SET paper_balance   = user_capital.paper_balance + EXCLUDED.paper_balance - %s,
                    last_updated_at = NOW()
            RETURNING paper_balance
            """,
            [user_id, PAPER_CAPITAL, delta, PAPER_CAPITAL],
        )
        new_bal = cur.fetchone()[0]
        conn.commit()
        return float(new_bal)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def note_partial_t1(trade_id: int, t1_price: float, note: str):
    """Append a note for a partial T1 booking WITHOUT closing the trade."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT notes FROM trade_log WHERE id = %s", [trade_id])
        row = cur.fetchone()
        old_notes = (row[0] or "") if row else ""
        merged = "\n".join(filter(None, [old_notes, note]))
        cur.execute("UPDATE trade_log SET notes = %s WHERE id = %s", [merged, trade_id])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def delete_trade(trade_id: int):
    """Remove a trade log entry."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM trade_log WHERE id = %s", [trade_id])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def load_trade_log(status_filter: list = None, user_id: str = "") -> pd.DataFrame:
    """Load trade log entries for a user, newest first.

    If user_id is empty (Kite not yet authenticated), returns an empty DataFrame
    rather than leaking all users' trades.
    """
    if not user_id:
        return pd.DataFrame()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses, params = ["kite_user_id = %s"], [user_id]
        if status_filter:
            placeholders = ",".join(["%s"] * len(status_filter))
            clauses.append(f"status IN ({placeholders})")
            params.extend(status_filter)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(f"SELECT * FROM trade_log {where} ORDER BY logged_at DESC", params)
        return _df_from_cursor(cur)
    finally:
        cur.close()
        release_conn(conn)


def sync_from_kite_orders(orders: list, user_id: str = "") -> int:
    """Update OPEN trade_log entries from Kite order status.

    Requires a non-empty user_id to prevent cross-user order leakage.
    """
    if not orders or not user_id:
        return 0
    orders_map = {str(o.get("order_id", "")): o for o in orders}
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = ["status = 'OPEN'", "kite_order_id IS NOT NULL", "kite_user_id = %s"]
        params  = [user_id]
        cur.execute(
            "SELECT id, kite_order_id, actual_entry, quantity, signal_type, rec_entry, rec_stop "
            "FROM trade_log WHERE " + " AND ".join(clauses),
            params,
        )
        open_rows = cur.fetchall()
        updated = 0
        for tid, kid, ae, qty, sig, re, rs in open_rows:
            if not kid:
                continue
            k = orders_map.get(str(kid))
            if not k:
                continue
            kstat       = k.get("status", "")
            filled_price = k.get("average_price") or ae
            if kstat == "COMPLETE":
                cur.execute(
                    "UPDATE trade_log SET kite_status=%s, actual_entry=%s WHERE id=%s",
                    [kstat, float(filled_price) if filled_price else ae, tid],
                )
                updated += 1
            elif kstat in ("REJECTED", "CANCELLED"):
                cur.execute(
                    "UPDATE trade_log SET status=%s, kite_status=%s WHERE id=%s",
                    [kstat, kstat, tid],
                )
                updated += 1
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_trade_stats(user_id: str = "", is_paper: bool | None = None) -> dict:
    """Aggregate stats for Activity Log header, scoped to user.

    Returns zeroed stats if user_id is empty to prevent cross-user data leaks.
    """
    if not user_id:
        return {"total": 0, "open": 0, "closed": 0, "wins": 0, "losses": 0,
                "total_pnl": 0.0, "avg_rr": None, "best_trade": None, "win_rate": 0.0}
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses, params = ["kite_user_id = %s"], [user_id]
        if is_paper is True:
            clauses.append("is_paper_trade = TRUE")
        elif is_paper is False:
            clauses.append("(is_paper_trade = FALSE OR is_paper_trade IS NULL)")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(f"""
            SELECT
                COUNT(*)                                               AS total,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END)     AS open_count,
                SUM(CASE WHEN pnl_amount > 0   THEN 1 ELSE 0 END)    AS wins,
                SUM(CASE WHEN pnl_amount <= 0 AND status != 'OPEN'
                         THEN 1 ELSE 0 END)                           AS losses,
                COALESCE(SUM(pnl_amount), 0)                         AS total_pnl,
                AVG(CASE WHEN rr_realised IS NOT NULL THEN rr_realised END) AS avg_rr,
                MAX(pnl_amount)                                        AS best_trade,
                MIN(pnl_amount)                                        AS worst_trade
            FROM trade_log {where}
        """, params)
        row = cur.fetchone()
        if row is None:
            return {}
        total, open_c, wins, losses, tot_pnl, avg_rr, best, worst = row
        total    = int(total  or 0)
        open_c   = int(open_c or 0)
        wins     = int(wins   or 0)
        losses   = int(losses or 0)
        closed   = total - open_c
        win_rate = (wins / closed * 100) if closed > 0 else 0.0
        return {
            "total":      total,
            "open":       open_c,
            "closed":     closed,
            "wins":       wins or 0,
            "losses":     losses or 0,
            "win_rate":   win_rate,
            "total_pnl":  tot_pnl or 0.0,
            "avg_rr":     avg_rr,
            "best_trade": best,
            "worst_trade": worst,
        }
    finally:
        cur.close()
        release_conn(conn)


def get_total_charges(user_id: str = "", is_paper: bool | None = None) -> float:
    """Sum of Zerodha statutory charges for all CLOSED trades.

    Returns 0.0 if user_id is empty to prevent cross-user data leaks.
    """
    if not user_id:
        return 0.0
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = [
            "kite_user_id = %s",
            "status != 'OPEN'",
            "actual_entry IS NOT NULL",
            "actual_exit IS NOT NULL",
            "quantity IS NOT NULL",
        ]
        params = [user_id]
        if is_paper is True:
            clauses.append("is_paper_trade = TRUE")
        elif is_paper is False:
            clauses.append("(is_paper_trade = FALSE OR is_paper_trade IS NULL)")
        cur.execute(
            "SELECT actual_entry, actual_exit, quantity, setup_type, signal_type FROM trade_log "
            "WHERE " + " AND ".join(clauses),
            params,
        )
        rows  = cur.fetchall()
        total = 0.0
        for entry, exit_p, qty, stype, sig_type in rows:
            try:
                # ORB signals are always intraday regardless of stored setup_type
                effective_stype = "SCALP" if str(sig_type or "").upper() in ("BUY_ORB", "SELL_ORB") \
                                  else str(stype or "INTRADAY")
                total += compute_trade_charges(
                    float(entry), float(exit_p), int(qty), effective_stype
                ).get("total", 0.0)
            except Exception:
                pass
        return round(total, 2)
    finally:
        cur.close()
        release_conn(conn)


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER TRADING helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_open_paper_trades(user_id: str = "", trade_date=None) -> list:
    """Open paper trades for a given date (defaults to today IST), scoped to user."""
    import datetime as _dt
    _IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    d = trade_date or _dt.datetime.now(_IST).date()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = ["is_paper_trade = TRUE", "status = 'OPEN'", "trade_date = %s"]
        params  = [str(d)]
        if user_id:
            clauses.append("kite_user_id = %s")
            params.append(user_id)
        cur.execute(
            "SELECT id, tradingsymbol, signal_type, actual_entry, rec_stop, rec_t1 "
            "FROM trade_log WHERE " + " AND ".join(clauses),
            params,
        )
        return [
            {"id": r[0], "tradingsymbol": r[1], "signal_type": r[2],
             "actual_entry": r[3], "rec_stop": r[4], "rec_t1": r[5]}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        release_conn(conn)


def get_all_today_paper_trades(user_id: str = "", trade_date=None) -> list:
    """
    ALL paper trades for today IST (any status). Used on startup to rebuild
    paper_triggered / scalp_triggered so closed trades can't re-fire after
    a page refresh.
    """
    import datetime as _dt
    _IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    d = trade_date or _dt.datetime.now(_IST).date()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = ["is_paper_trade = TRUE", "trade_date = %s"]
        params  = [str(d)]
        if user_id:
            clauses.append("kite_user_id = %s")
            params.append(user_id)
        cur.execute(
            "SELECT id, tradingsymbol, signal_type, actual_entry, rec_stop, rec_t1, "
            "       status, setup_type, rec_t2, intraday_confidence, is_paper_trade "
            "FROM trade_log WHERE " + " AND ".join(clauses) + " ORDER BY id",
            params,
        )
        return [
            {
                "id":                 r[0],
                "tradingsymbol":      r[1],
                "signal_type":        r[2],
                "actual_entry":       r[3],
                "rec_stop":           r[4],
                "rec_t1":             r[5],
                "status":             r[6],
                "setup_type":         r[7] or "INTRADAY",
                "rec_t2":             r[8],
                "intraday_confidence": r[9] or 0,
                "is_paper_trade":     r[10],
            }
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        release_conn(conn)


def get_today_closed_pnl(user_id: str = "", is_paper: bool | None = None) -> float:
    """Total realised P&L for trades closed today.

    Returns 0 if user_id is empty to prevent cross-user data leaks.
    """
    if not user_id:
        return 0.0
    import datetime as _dt
    _IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    _today = _dt.datetime.now(_IST).date().isoformat()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = [
            "kite_user_id = %s",
            "status IN ('CLOSED', 'TARGET_HIT', 'STOPPED_OUT')",
            "pnl_amount IS NOT NULL",
            "(trade_date = %s OR logged_at::DATE = %s)",
        ]
        params: list = [user_id, _today, _today]
        if is_paper is True:
            clauses.append("is_paper_trade = TRUE")
        elif is_paper is False:
            clauses.append("(is_paper_trade = FALSE OR is_paper_trade IS NULL)")
        cur.execute(
            "SELECT COALESCE(SUM(pnl_amount), 0.0) FROM trade_log WHERE "
            + " AND ".join(clauses),
            params,
        )
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0
    finally:
        cur.close()
        release_conn(conn)


def get_paper_trade_perf(user_id: str = "", days: int = 30) -> dict:
    """
    Aggregate paper trade performance over the last `days` calendar days,
    broken down by signal_type.
    """
    import datetime as _dt
    cutoff = (_dt.date.today() - _td(days=days)).isoformat()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = [
            "is_paper_trade = TRUE",
            "status IN ('TARGET_HIT','STOPPED_OUT','CLOSED')",
            "trade_date >= %s",
        ]
        params: list = [cutoff]
        if user_id:
            clauses.append("kite_user_id = %s")
            params.append(user_id)
        cur.execute(
            "SELECT signal_type, "
            "COUNT(*) AS total, "
            "SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN pnl_amount <= 0 AND pnl_amount IS NOT NULL THEN 1 ELSE 0 END) AS losses, "
            "AVG(CASE WHEN rr_realised IS NOT NULL THEN rr_realised END) AS avg_rr, "
            "AVG(pnl_pct) AS avg_pnl_pct "
            "FROM trade_log WHERE " + " AND ".join(clauses) + " GROUP BY signal_type",
            params,
        )
        rows   = cur.fetchall()
        result = {}
        overall_total = overall_wins = 0
        overall_rr_sum = overall_rr_n = 0
        for sig, total, wins, losses, avg_rr, avg_pnl_pct in rows:
            closed   = total or 0
            win_rate = (wins / closed * 100) if closed > 0 else 0.0
            result[sig] = {
                "total":       total  or 0,
                "wins":        wins   or 0,
                "losses":      losses or 0,
                "win_rate":    round(win_rate, 1),
                "avg_rr":      round(avg_rr, 2)      if avg_rr      else None,
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
    finally:
        cur.close()
        release_conn(conn)


def get_archived_paper_trades_for_analysis(user_id: str = "") -> pd.DataFrame:
    """All closed paper trades from days BEFORE today IST for algorithm re-tuning."""
    import datetime as _dt
    _IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    today = str(_dt.datetime.now(_IST).date())
    conn = get_conn()
    cur  = conn.cursor()
    try:
        clauses = [
            "is_paper_trade = TRUE",
            "status IN ('TARGET_HIT', 'STOPPED_OUT', 'CLOSED')",
            "trade_date < %s",
        ]
        params: list = [today]
        if user_id:
            clauses.append("kite_user_id = %s")
            params.append(user_id)
        cur.execute(
            "SELECT id, tradingsymbol, signal_type, trade_date, logged_at, "
            "       actual_entry, actual_exit, rec_entry, rec_stop, rec_t1, "
            "       pnl_amount, pnl_pct, rr_realised, intraday_confidence, "
            "       status, quantity "
            "FROM trade_log WHERE " + " AND ".join(clauses)
            + " ORDER BY trade_date DESC, id DESC",
            params,
        )
        return _df_from_cursor(cur)
    finally:
        cur.close()
        release_conn(conn)


def get_open_real_trades(user_id: str = "") -> list:
    """All real (non-paper) OPEN trades for today IST — used by 3:20 PM Kite sync."""
    import datetime as _dt_mod
    _IST_r = _dt_mod.timezone(_dt_mod.timedelta(hours=5, minutes=30))
    today = str(_dt_mod.datetime.now(_IST_r).date())
    conn  = get_conn()
    cur   = conn.cursor()
    try:
        clauses = ["is_paper_trade = FALSE", "status = 'OPEN'", "trade_date = %s"]
        params  = [today]
        if user_id:
            clauses.append("kite_user_id = %s")
            params.append(user_id)
        cur.execute(
            "SELECT id, tradingsymbol, signal_type, actual_entry, rec_stop, "
            "kite_sl_order_id, kite_target_order_id, quantity "
            "FROM trade_log WHERE " + " AND ".join(clauses) + " ORDER BY id",
            params,
        )
        return [
            {"id": r[0], "tradingsymbol": r[1], "signal_type": r[2],
             "actual_entry": r[3], "rec_stop": r[4],
             "kite_sl_order_id": r[5], "kite_target_order_id": r[6],
             "quantity": r[7]}
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        release_conn(conn)


# ─── Signal config ────────────────────────────────────────────────────────────

_SIGNAL_CONFIG_DEFAULTS = {
    "intraday_rsi_buy_max":  75.0,
    "intraday_rsi_sell_min": 25.0,
    "intraday_min_rr":        1.5,
}


def get_signal_config(user_id: str = "") -> dict:
    """Per-user signal tuning overrides, with defaults for missing keys."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT config_key, value FROM signal_config WHERE kite_user_id = %s",
            [user_id or ""],
        )
        cfg = dict(_SIGNAL_CONFIG_DEFAULTS)
        for key, val in cur.fetchall():
            cfg[key] = val
        return cfg
    finally:
        cur.close()
        release_conn(conn)


def save_signal_config(cfg: dict, user_id: str = "") -> None:
    """Upsert signal tuning overrides."""
    uid  = user_id or ""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        for key, val in cfg.items():
            cur.execute(
                """INSERT INTO signal_config (config_key, kite_user_id, value, updated_at)
                   VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                   ON CONFLICT (config_key, kite_user_id)
                   DO UPDATE SET value = EXCLUDED.value,
                                 updated_at = EXCLUDED.updated_at""",
                [key, uid, float(val)],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def tune_signal_config_from_paper(user_id: str = "", days: int = 30) -> dict:
    """Derive updated signal thresholds from recent paper trade performance."""
    perf    = get_paper_trade_perf(user_id=user_id, days=days)
    current = get_signal_config(user_id=user_id)
    changes: dict = {}

    long_data  = perf.get("BUY_ABOVE",  {})
    short_data = perf.get("SELL_BELOW", {})

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
        wr      = short_data["win_rate"]
        cur_rsi = current["intraday_rsi_sell_min"]
        if wr < 40.0:
            new_rsi = min(45.0, cur_rsi + 5.0)
            if new_rsi != cur_rsi: changes["intraday_rsi_sell_min"] = new_rsi
        elif wr > 65.0:
            new_rsi = max(25.0, cur_rsi - 3.0)
            if new_rsi != cur_rsi: changes["intraday_rsi_sell_min"] = new_rsi

    if changes:
        save_signal_config({**current, **changes}, user_id=user_id)
    return changes


# ─── Market Intel ─────────────────────────────────────────────────────────────

def save_market_intel(
    user_id: str,
    raw_output: str,
    bias: str,
    confidence: str,
    stocks: list,
) -> int:
    """Save a market intel run (log row + N stock rows). Returns intel_id."""
    uid  = user_id or ""
    now  = _now_ist()
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM market_intel_stocks WHERE kite_user_id = %s", [uid]
        )
        cur.execute(
            "INSERT INTO market_intel_log "
            "(kite_user_id, created_at, raw_output, overall_bias, overall_confidence) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            [uid, now, raw_output[:50_000], bias, confidence],
        )
        intel_id = cur.fetchone()[0]

        for s in stocks:
            cur.execute(
                "INSERT INTO market_intel_stocks "
                "(intel_id, kite_user_id, created_at, tradingsymbol, stance, sector, "
                " fundamental_reason, entry_trigger, stop_loss, conviction, "
                " condition_required, alert_level, expected_move) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    intel_id, uid, now,
                    s.get("tradingsymbol", ""),
                    s.get("stance", ""),
                    s.get("sector", "")[:200],
                    s.get("fundamental_reason", "")[:500],
                    s.get("entry_trigger", "")[:500],
                    s.get("stop_loss", "")[:100],
                    s.get("conviction", "")[:20],
                    s.get("condition_required", "")[:500],
                    s.get("alert_level", "")[:200],
                    s.get("expected_move", "")[:200],
                ],
            )
        conn.commit()
        return intel_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_latest_market_intel(user_id: str = "") -> dict:
    """Most recent market intel log row for this user."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id, created_at, overall_bias, overall_confidence, raw_output "
            "FROM market_intel_log WHERE kite_user_id = %s "
            "ORDER BY created_at DESC LIMIT 1",
            [user_id or ""],
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "id":         row[0],
            "created_at": row[1],
            "bias":       row[2] or "NEUTRAL",
            "confidence": row[3] or "MEDIUM",
            "raw_output": row[4] or "",
        }
    finally:
        cur.close()
        release_conn(conn)


def get_market_intel_stocks(user_id: str = "") -> list:
    """All stock rows from the latest market intel run for this user."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """SELECT mis.tradingsymbol, mis.stance, mis.sector,
                      mis.fundamental_reason, mis.entry_trigger, mis.stop_loss,
                      mis.conviction, mis.condition_required,
                      mis.alert_level, mis.expected_move, mis.created_at
               FROM market_intel_stocks mis
               INNER JOIN market_intel_log mil ON mis.intel_id = mil.id
               WHERE mis.kite_user_id = %s
                 AND mil.id = (
                     SELECT id FROM market_intel_log
                     WHERE kite_user_id = %s
                     ORDER BY created_at DESC LIMIT 1
                 )
               ORDER BY mis.stance, mis.id""",
            [user_id or "", user_id or ""],
        )
        return [
            {
                "tradingsymbol":      r[0],
                "stance":             r[1],
                "sector":             r[2],
                "fundamental_reason": r[3],
                "entry_trigger":      r[4],
                "stop_loss":          r[5],
                "conviction":         r[6],
                "condition_required": r[7],
                "alert_level":        r[8],
                "expected_move":      r[9],
                "created_at":         r[10],
            }
            for r in cur.fetchall()
        ]
    finally:
        cur.close()
        release_conn(conn)


# ═══════════════════════════════════════════════════════════════════════════════
# USER AUTH — accounts, sessions, Kite credential storage
# ═══════════════════════════════════════════════════════════════════════════════

def create_user(username: str, password_hash: str,
                kite_api_key: str = "", kite_api_secret: str = "") -> int:
    """
    Create a new user. Returns the new user id.
    Raises psycopg2.errors.UniqueViolation if username already exists.
    """
    from config import PAPER_CAPITAL
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO users (username, password_hash, kite_api_key, kite_api_secret)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            [username.strip().lower(), password_hash, kite_api_key, kite_api_secret],
        )
        uid = cur.fetchone()[0]
        # Seed the paper trading balance for the new user
        cur.execute(
            """INSERT INTO user_capital (user_id, paper_balance)
               VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING""",
            [uid, PAPER_CAPITAL],
        )
        conn.commit()
        return uid
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_user_by_username(username: str) -> dict | None:
    """Return the full users row as a dict, or None if not found."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id, username, password_hash, kite_api_key, kite_api_secret, "
            "       kite_user_id, kite_access_token, kite_token_updated_at, "
            "       created_at, last_login_at, "
            "       COALESCE(openrouter_key,''), COALESCE(openai_key,'') "
            "FROM users WHERE username = %s",
            [username.strip().lower()],
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id":                    row[0],
            "username":              row[1],
            "password_hash":         row[2],
            "kite_api_key":          row[3] or "",
            "kite_api_secret":       row[4] or "",
            "kite_user_id":          row[5] or "",
            "kite_access_token":     row[6] or "",
            "kite_token_updated_at": row[7],
            "created_at":            row[8],
            "last_login_at":         row[9],
            "openrouter_key":        row[10] or "",
            "openai_key":            row[11] or "",
        }
    finally:
        cur.close()
        release_conn(conn)


def create_session(user_id: int, session_token: str) -> None:
    """Insert a new session row. Token is generated by auth.py."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO user_sessions (user_id, session_token) VALUES (%s, %s) "
            "ON CONFLICT (session_token) DO NOTHING",
            [user_id, session_token],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_user_by_session(session_token: str) -> dict | None:
    """
    Validate a session token and return the matching users row.
    Also bumps last_seen_at on the session.
    Returns None if token not found.
    """
    if not session_token:
        return None
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """SELECT u.id, u.username, u.password_hash, u.kite_api_key,
                      u.kite_api_secret, u.kite_user_id, u.kite_access_token,
                      u.kite_token_updated_at, u.created_at, u.last_login_at,
                      COALESCE(u.openrouter_key,''), COALESCE(u.openai_key,'')
               FROM user_sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.session_token = %s""",
            [session_token],
        )
        row = cur.fetchone()
        if not row:
            return None
        # Bump last_seen_at (best-effort, non-fatal)
        try:
            cur.execute(
                "UPDATE user_sessions SET last_seen_at = %s WHERE session_token = %s",
                [_now_ist(), session_token],
            )
        except Exception:
            pass
        conn.commit()
        return {
            "id":                    row[0],
            "username":              row[1],
            "password_hash":         row[2],
            "kite_api_key":          row[3] or "",
            "kite_api_secret":       row[4] or "",
            "kite_user_id":          row[5] or "",
            "kite_access_token":     row[6] or "",
            "kite_token_updated_at": row[7],
            "created_at":            row[8],
            "last_login_at":         row[9],
            "openrouter_key":        row[10] or "",
            "openai_key":            row[11] or "",
        }
    finally:
        cur.close()
        release_conn(conn)


def delete_session(session_token: str) -> None:
    """Logout — remove the session token."""
    if not session_token:
        return
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM user_sessions WHERE session_token = %s", [session_token])
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        release_conn(conn)


def update_last_login(user_id: int) -> None:
    """Stamp last_login_at whenever a user successfully logs in."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE users SET last_login_at = %s WHERE id = %s",
            [_now_ist(), user_id],
        )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        release_conn(conn)


def update_kite_credentials(user_id: int, kite_api_key: str, kite_api_secret: str) -> None:
    """Update stored Kite API key and secret for a user."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE users SET kite_api_key = %s, kite_api_secret = %s WHERE id = %s",
            [kite_api_key, kite_api_secret, user_id],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def update_kite_auth(user_id: int, kite_user_id: str, access_token: str) -> None:
    """
    Store Kite access token after a successful OAuth exchange.
    Also records kite_user_id (e.g. "ZY1234") which links to all trade data.
    """
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            """UPDATE users
               SET kite_user_id = %s,
                   kite_access_token = %s,
                   kite_token_updated_at = %s
               WHERE id = %s""",
            [kite_user_id, access_token, _now_ist(), user_id],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_conn(conn)


def get_ai_keys(user_id: int) -> dict:
    """Return stored OpenRouter and OpenAI keys for the user."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT openrouter_key, openai_key FROM users WHERE id = %s",
                [user_id],
            )
            row = cur.fetchone()
            if row:
                return {"openrouter_key": row[0] or "", "openai_key": row[1] or ""}
            return {"openrouter_key": "", "openai_key": ""}
    except Exception:
        return {"openrouter_key": "", "openai_key": ""}
    finally:
        release_conn(conn)


def update_ai_keys(user_id: int, openrouter_key: str, openai_key: str) -> None:
    """Persist AI API keys in the users table (stored securely in Neon PostgreSQL)."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "UPDATE users SET openrouter_key = %s, openai_key = %s WHERE id = %s",
            [openrouter_key.strip(), openai_key.strip(), user_id],
        )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        release_conn(conn)
