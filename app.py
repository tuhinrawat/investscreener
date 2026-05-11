"""
app.py — Streamlit dashboard.

Three sections:
  1. Sidebar — refresh buttons + filters
  2. Main table — sortable, filterable, with composite score ranking
  3. Detail panel — when you click a row, shows candlestick + indicators

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import charts as _charts
from datetime import datetime, timezone, timedelta, date as date_type

import os as _os

# ── Database URL bootstrap ────────────────────────────────────────────────────
# Load DATABASE_URL from .env (local dev) or Streamlit secrets (cloud) so that
# db.get_conn() can build the connection pool before any table function runs.
# Priority: already in env > Streamlit secrets > .env file.
if not _os.environ.get("DATABASE_URL"):
    try:
        # Streamlit Cloud: secrets are in st.secrets
        import streamlit as _st_boot
        _db_url = _st_boot.secrets.get("DATABASE_URL", "")
        if _db_url:
            _os.environ["DATABASE_URL"] = _db_url
    except Exception:
        pass
if not _os.environ.get("DATABASE_URL"):
    try:
        from dotenv import load_dotenv as _lde
        _lde(override=False)
    except ImportError:
        pass

import config
import db
import data_pipeline
import ai_analyst as _ai
import market_intel as _mi
from kite_client import KiteClient

# Background-tab-safe autorefresh: uses a Web Worker (background thread) that
# browsers cannot throttle, unlike plain setInterval used by fragment run_every.
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
    def _st_autorefresh(interval=2000, limit=None, key="ar", debounce=False):  # noqa
        return 0

# True when running inside a Streamlit Cloud container.
# Keys must NEVER be written to shared disk in that environment.
_ON_CLOUD: bool = _os.environ.get("HOME", "").rstrip("/").endswith("appuser")


# ============================================================
# LIVE QUOTE FRAGMENT — auto-refreshes every 10 s independently
# of the rest of the page (Streamlit 1.37+ fragment feature).
# Only invoked when the user enables the Live toggle.
# ============================================================
@st.fragment(run_every="10s")
def _live_quote_fragment(symbol: str):
    """Fetches and displays a real-time OHLC quote for one symbol."""
    # Don't poll the API outside market hours
    from datetime import datetime, timezone, timedelta
    _ist = timezone(timedelta(hours=5, minutes=30))
    _now = datetime.now(_ist)
    _mkt_open = (
        _now.weekday() < 5
        and _now.replace(hour=9,  minute=15, second=0, microsecond=0) <= _now
        <= _now.replace(hour=15, minute=30, second=0, microsecond=0)
    )
    if not _mkt_open:
        st.caption("⏸ Market closed — live quote paused.")
        return

    try:
        client = st.session_state.get("kite_client") or KiteClient(
            api_key=st.session_state.get("kite_api_key", ""),
            api_secret=st.session_state.get("kite_api_secret", ""),
            access_token=st.session_state.get("kite_access_token", ""),
        )
        quotes  = client.get_ohlc_batch([f"NSE:{symbol}"])
        q       = quotes.get(f"NSE:{symbol}")
    except Exception as err:
        st.warning(f"Live data unavailable: {err}")
        return

    if not q:
        st.caption("No live quote returned.")
        return

    ltp        = q.get("last_price", 0)
    ohlc_today = q.get("ohlc", {})
    prev_close = ohlc_today.get("close", 0)
    day_chg    = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
    day_abs    = ltp - prev_close
    badge_col  = "#22c55e" if day_chg >= 0 else "#ef4444"
    arrow      = "▲" if day_chg >= 0 else "▼"

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                    padding:14px 22px;margin-bottom:8px;display:flex;
                    align-items:center;gap:24px;flex-wrap:wrap">
          <span style="font-size:1.7rem;font-weight:700;color:#f1f5f9;
                       letter-spacing:-0.5px">{symbol}</span>
          <span style="font-size:1.5rem;font-weight:700;color:#f8fafc">
            ₹{ltp:,.2f}</span>
          <span style="font-size:1.05rem;font-weight:600;color:{badge_col}">
            {arrow} {day_chg:+.2f}%&nbsp;&nbsp;({day_abs:+.2f})</span>
          <span style="font-size:0.88rem;color:#64748b">
            O&nbsp;<b style="color:#94a3b8">₹{ohlc_today.get('open',0):,.2f}</b>
            &nbsp;&nbsp;H&nbsp;<b style="color:#94a3b8">₹{ohlc_today.get('high',0):,.2f}</b>
            &nbsp;&nbsp;L&nbsp;<b style="color:#94a3b8">₹{ohlc_today.get('low',0):,.2f}</b>
            &nbsp;&nbsp;Prev&nbsp;<b style="color:#94a3b8">₹{prev_close:,.2f}</b>
          </span>
          <span style="margin-left:auto;font-size:0.78rem;color:#475569">
            ⏱ {datetime.now().strftime('%H:%M:%S')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Store latest LTP in session so the chart can use it without re-fetching
    st.session_state[f"live_ltp_{symbol}"]       = ltp
    st.session_state[f"live_ohlc_{symbol}"]      = ohlc_today
    st.session_state[f"live_prev_close_{symbol}"] = prev_close


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NSE Swing Screener",
    page_icon="📊",
    layout="wide",
)

# ── Browser localStorage (per-user, per-domain) ──────────────────────
# ls_get / ls_set / ls_delete persist values in the user's own browser.
# No external package needed — just a tiny vanilla-JS Streamlit component.
from ls_store import ls_get as _ls_get, ls_set as _ls_set, ls_delete as _ls_del
import auth as _auth

# ============================================================
# USER AUTH GATE  — username / password + Kite OAuth
#
# Flow:
#   1. Check localStorage for "screener_session" token (persists forever)
#   2. Validate token against user_sessions DB table → auto-login
#   3. If no valid token → show Login / Signup page
#   4. After login: load Kite credentials from users table
#   5. If Kite token is fresh (issued today) → KiteClient is ready
#   6. If Kite token is expired → show "Re-authenticate with Kite" button
#
# Session state keys set by this block:
#   app_user          — full users row dict (set = logged in)
#   kite_api_key      — from users table
#   kite_api_secret   — from users table
#   kite_access_token — from users table (if fresh)
#   kite_authenticated — bool
#   kite_user_id      — Kite profile user_id (e.g. "ZY1234")
#   kite_user_name    — Kite display name
#   kite_client       — KiteClient instance (if authenticated)
# ============================================================

# ── Capture Kite OAuth request_token before anything else ─────────────
# Zerodha redirects back with ?request_token=XXX on a fresh page load.
# Save it immediately so it survives the re-renders needed for the DB
# session lookup and localStorage component to fire.
if "request_token" in st.query_params:
    st.session_state["_pending_rt"] = st.query_params["request_token"]
    st.query_params.clear()


# ── Helper: seed session_state Kite keys from a users-table row ───────
def _load_kite_from_user(user: dict) -> None:
    """Populate Kite session_state from a validated users row."""
    st.session_state["kite_api_key"]    = user.get("kite_api_key",    "")
    st.session_state["kite_api_secret"] = user.get("kite_api_secret", "")
    st.session_state["kite_user_id"]    = user.get("kite_user_id",    "")
    token = user.get("kite_access_token", "")
    fresh = _auth.is_kite_token_fresh(user.get("kite_token_updated_at"))
    if token and fresh:
        st.session_state["kite_access_token"]  = token
        st.session_state["kite_authenticated"] = True
        try:
            _kc_boot = KiteClient(
                api_key    = user["kite_api_key"],
                api_secret = user["kite_api_secret"],
            )
            _kc_boot.set_access_token(token)
            st.session_state["kite_client"] = _kc_boot
        except Exception:
            pass
    else:
        st.session_state["kite_access_token"]  = ""
        st.session_state["kite_authenticated"] = False
        st.session_state.pop("kite_client", None)
    st.session_state.setdefault("kite_user_name", "")
    st.session_state.setdefault("kite_access_date", "")


# ── Step 1: try auto-login from localStorage session token ────────────
if "app_user" not in st.session_state:
    _stored_token = _ls_get("screener_session")
    if _stored_token:
        _restored_user = db.get_user_by_session(_stored_token)
        if _restored_user:
            st.session_state["app_user"]          = _restored_user
            st.session_state["_session_token"]    = _stored_token
            _load_kite_from_user(_restored_user)


# ── Step 2: if still not logged in → show Login / Signup page ─────────
def _show_auth_page() -> None:
    """Full-screen login / signup. Calls st.stop() so the main app doesn't render."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        .auth-card {
            background:#0f172a;border:1px solid #1e293b;border-radius:16px;
            padding:40px 44px;max-width:480px;margin:60px auto 0;
        }
        .auth-title {font-size:1.6rem;font-weight:800;color:#f1f5f9;margin-bottom:4px;}
        .auth-sub   {font-size:0.9rem;color:#64748b;margin-bottom:28px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, _col, _ = st.columns([1, 2, 1])
    with _col:
        st.markdown(
            '<div style="text-align:center;margin-bottom:8px;font-size:2.4rem;">📊</div>'
            '<div style="text-align:center;font-size:1.5rem;font-weight:800;color:#f1f5f9'
            ';margin-bottom:2px;">NSE Screener</div>'
            '<div style="text-align:center;color:#64748b;font-size:0.85rem;margin-bottom:28px;">'
            'Intraday · Swing · Scalping · Market Intel</div>',
            unsafe_allow_html=True,
        )

        _auth_tab, _signup_tab = st.tabs(["🔑 Login", "✨ Sign Up"])

        # ── LOGIN ──────────────────────────────────────────────────────────
        with _auth_tab:
            _login_user = st.text_input("Username", key="login_username",
                                        placeholder="your username")
            _login_pass = st.text_input("Password", type="password",
                                        key="login_password",
                                        placeholder="your password")
            if st.button("Login", type="primary", use_container_width=True,
                         key="login_btn"):
                if not (_login_user.strip() and _login_pass.strip()):
                    st.error("Enter username and password.")
                else:
                    _u = db.get_user_by_username(_login_user.strip())
                    if not _u or not _auth.verify_password(_login_pass, _u["password_hash"]):
                        st.error("Incorrect username or password.")
                    else:
                        _tok = _auth.new_session_token()
                        db.create_session(_u["id"], _tok)
                        db.update_last_login(_u["id"])
                        st.session_state["app_user"]       = _u
                        st.session_state["_session_token"] = _tok
                        _ls_set("screener_session", _tok, expires_days=36500)
                        _load_kite_from_user(_u)
                        st.rerun()

        # ── SIGN UP ────────────────────────────────────────────────────────
        with _signup_tab:
            _su_user = st.text_input("Choose a username", key="su_username",
                                     placeholder="e.g. tuhin")
            _su_pass = st.text_input("Choose a password", type="password",
                                     key="su_password",
                                     placeholder="min 6 characters")
            _su_pass2 = st.text_input("Confirm password", type="password",
                                      key="su_password2",
                                      placeholder="repeat password")
            st.markdown("##### Kite Connect credentials")
            st.caption(
                "Get these from [developers.kite.trade](https://developers.kite.trade) "
                "→ My Apps. Set redirect URL to your app URL."
            )
            _su_key = st.text_input("Kite API Key", key="su_kite_key",
                                    type="password", placeholder="Kite API Key")
            _su_sec = st.text_input("Kite API Secret", key="su_kite_secret",
                                    type="password", placeholder="Kite API Secret")

            if st.button("Create Account", type="primary",
                         use_container_width=True, key="signup_btn"):
                _errs = []
                if not _su_user.strip():      _errs.append("Username is required.")
                if len(_su_pass) < 6:         _errs.append("Password must be at least 6 characters.")
                if _su_pass != _su_pass2:     _errs.append("Passwords do not match.")
                if not _su_key.strip():       _errs.append("Kite API Key is required.")
                if not _su_sec.strip():       _errs.append("Kite API Secret is required.")
                if _errs:
                    for _e in _errs:
                        st.error(_e)
                else:
                    try:
                        _ph  = _auth.hash_password(_su_pass)
                        _uid = db.create_user(
                            _su_user.strip(), _ph,
                            _su_key.strip(), _su_sec.strip(),
                        )
                        _tok = _auth.new_session_token()
                        db.create_session(_uid, _tok)
                        _new_u = db.get_user_by_username(_su_user.strip())
                        st.session_state["app_user"]       = _new_u
                        st.session_state["_session_token"] = _tok
                        _ls_set("screener_session", _tok, expires_days=36500)
                        _load_kite_from_user(_new_u)
                        st.success(f"Welcome, {_su_user.strip()}! Redirecting…")
                        st.rerun()
                    except Exception as _se:
                        if "unique" in str(_se).lower():
                            st.error("That username is already taken. Choose another.")
                        else:
                            st.error(f"Sign-up failed: {_se}")

    st.stop()


if "app_user" not in st.session_state:
    _show_auth_page()

# ── From here: user is authenticated ──────────────────────────────────
_app_user    = st.session_state["app_user"]
_cur_user_id = st.session_state.get("kite_user_id", "") or _app_user.get("kite_user_id", "")

# ── Kite OAuth callback: exchange pending request_token ───────────────
_pending_rt    = st.session_state.get("_pending_rt", "")
_ss_api_key    = st.session_state.get("kite_api_key",    "") or _app_user.get("kite_api_key",    "")
_ss_api_secret = st.session_state.get("kite_api_secret", "") or _app_user.get("kite_api_secret", "")

if _pending_rt and _ss_api_key and _ss_api_secret:
    with st.spinner("Completing Zerodha authentication…"):
        try:
            _client    = KiteClient(api_key=_ss_api_key, api_secret=_ss_api_secret)
            _acc_token = _client.complete_auth(_pending_rt)
            _profile   = _client.get_profile()
            _kite_uid  = _profile.get("user_id",   "")
            _kite_name = _profile.get("user_name", "")
            _today_str = date_type.today().isoformat()
            # Persist token to DB (tied to user account)
            db.update_kite_auth(_app_user["id"], _kite_uid, _acc_token)
            # Update in-memory user dict
            _app_user["kite_user_id"]          = _kite_uid
            _app_user["kite_access_token"]     = _acc_token
            st.session_state["app_user"]       = _app_user
            st.session_state["kite_access_token"]  = _acc_token
            st.session_state["kite_access_date"]   = _today_str
            st.session_state["kite_user_id"]       = _kite_uid
            st.session_state["kite_user_name"]     = _kite_name
            st.session_state["kite_authenticated"] = True
            st.session_state["kite_client"]        = _client
            st.session_state.pop("_pending_rt", None)
            st.rerun()
        except Exception as _auth_err:
            st.session_state.pop("_pending_rt", None)
            st.error(f"Kite authentication failed: {_auth_err}")
            st.stop()

# ── Step 2 — restore session token if we have one for today ───────────
if not st.session_state.get("kite_authenticated"):
    _ss_token = st.session_state.get("kite_access_token", "")
    _ss_date  = st.session_state.get("kite_access_date",  "")
    if _ss_token and _ss_date == date_type.today().isoformat():
        # Token is from today — restore the KiteClient
        try:
            _rc = KiteClient(api_key=_ss_api_key, api_secret=_ss_api_secret,
                             access_token=_ss_token)
            if _rc.authenticated:
                _profile = _rc.get_profile()
                st.session_state["kite_user_id"]   = _profile.get("user_id",   st.session_state.get("kite_user_id", ""))
                st.session_state["kite_user_name"] = _profile.get("user_name", st.session_state.get("kite_user_name", ""))
                st.session_state["kite_authenticated"] = True
                st.session_state["kite_client"]    = _rc
        except Exception:
            pass
    elif not _ON_CLOUD:
        # Local dev only: try disk cache (single-user, no sharing risk)
        try:
            _rc = KiteClient(api_key=_ss_api_key, api_secret=_ss_api_secret)
            if _rc.authenticated:
                _profile = _rc.get_profile()
                st.session_state["kite_user_id"]   = _profile.get("user_id",  "")
                st.session_state["kite_user_name"] = _profile.get("user_name","")
                st.session_state["kite_authenticated"] = True
                st.session_state["kite_client"]    = _rc
        except Exception:
            pass

# ── Step 3 — show login page if still not authenticated ───────────────
if not st.session_state.get("kite_authenticated"):
    _login_client = KiteClient(api_key=_ss_api_key, api_secret=_ss_api_secret)
    _login_url = _login_client.get_login_url()

    # Global CSS injected at column-0 (no leading spaces) so Markdown
    # does NOT treat it as an indented code block (4-space rule).
    st.markdown(
"""<style>
[data-testid="stSidebar"]{display:none}
[data-testid="stHeader"]{background:transparent}
.block-container{padding:2rem 1rem 1rem 1rem !important;max-width:500px !important;margin:0 auto}
</style>""",
        unsafe_allow_html=True,
    )

    # Visual login card — uses st.html() which bypasses Markdown entirely
    st.html("""
<div style="text-align:center;padding:24px 0 8px 0">
  <div style="width:52px;height:52px;background:linear-gradient(135deg,#3b82f6,#6366f1);
              border-radius:14px;display:inline-flex;align-items:center;
              justify-content:center;font-size:26px;margin-bottom:12px">📊</div>
  <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.4px">
    NSE Swing Screener</div>
  <div style="font-size:0.78rem;color:#3b82f6;font-weight:600;
              letter-spacing:0.08em;text-transform:uppercase;margin-top:2px">
    Powered by Zerodha Kite</div>
</div>

<div style="text-align:center;margin:20px 0 24px 0">
  <div style="font-size:1.6rem;font-weight:800;color:#f8fafc;line-height:1.2;margin-bottom:8px">
    Trade with an edge.</div>
  <div style="font-size:0.9rem;color:#64748b;line-height:1.5">
    AI-powered swing signals, live portfolio tracking<br>and direct order execution — all in one place.</div>
</div>

<div style="display:flex;gap:7px;justify-content:center;flex-wrap:wrap;margin-bottom:24px">
  <span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
               color:#93c5fd;font-size:0.73rem;padding:4px 11px;border-radius:20px">📈 Swing &amp; Intraday</span>
  <span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
               color:#93c5fd;font-size:0.73rem;padding:4px 11px;border-radius:20px">🤖 AI Analysis</span>
  <span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
               color:#93c5fd;font-size:0.73rem;padding:4px 11px;border-radius:20px">⚡ Live Orders</span>
  <span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
               color:#93c5fd;font-size:0.73rem;padding:4px 11px;border-radius:20px">📒 Trade Log</span>
</div>

<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
            border-radius:16px;padding:20px 24px;margin-bottom:14px;text-align:center">
  <div style="font-size:0.72rem;font-weight:600;color:#475569;
              letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px">
    Connect your Zerodha account</div>
  <div style="font-size:0.83rem;color:#64748b">Click the button below to log in via Zerodha.</div>
</div>

<div style="background:rgba(251,191,36,0.05);border:1px solid rgba(251,191,36,0.18);
            border-radius:12px;padding:14px 18px;font-size:0.8rem;
            color:#94a3b8;line-height:1.65;margin-bottom:8px">
  <strong style="color:#fbbf24">One-time setup &middot;</strong>
  In your <a href="https://developers.kite.trade" target="_blank"
             style="color:#60a5fa;text-decoration:none">Kite Developer Console</a>,
  set the app <strong style="color:#94a3b8">Redirect URL</strong> to:<br><br>
  <code style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);
               padding:2px 8px;border-radius:6px;font-size:0.76rem;color:#e2e8f0">
    https://tuhinrawat-investscreener-app-miqshh.streamlit.app/</code>
  <span style="color:#475569;font-size:0.74rem"> — Streamlit Cloud</span><br>
  <code style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);
               padding:2px 8px;border-radius:6px;font-size:0.76rem;color:#e2e8f0">
    http://127.0.0.1:8501</code>
  <span style="color:#475569;font-size:0.74rem"> — Local dev</span>
</div>
<div style="text-align:center;font-size:0.71rem;color:#334155;padding:6px 0 2px 0">
  Market data &amp; orders via Zerodha Kite Connect API.
  Credentials are never shared between users.
</div>
""")

    _, _btn_col, _ = st.columns([1, 2, 1])
    with _btn_col:
        st.link_button(
            "🔑  Login with Zerodha",
            _login_url,
            use_container_width=True,
            type="primary",
        )

    st.stop()


# ============================================================
# AUTHENTICATED — initialize DB and continue
# ============================================================
db.init_schema()

# Ensure kite_client is set for the authenticated session
if "kite_client" not in st.session_state or not st.session_state["kite_client"]:
    _kc = KiteClient(
        api_key=st.session_state.get("kite_api_key",    ""),
        api_secret=st.session_state.get("kite_api_secret",  ""),
        access_token=st.session_state.get("kite_access_token", ""),
    )
    if _kc.authenticated:
        st.session_state["kite_client"] = _kc

# Convenience alias — current Kite user id for per-user DB filtering
_cur_user_id: str = st.session_state.get("kite_user_id", "")

# ── Paper trade session state ────────────────────────────────────────────────
# paper_triggered: {(date_str, sym): trade_id} — prevents re-triggering same signal
# paper_open:      {trade_id: {sym, stop, t1, signal_type, entry, cap}} — exit monitoring; cap = capital allocated (tier-based)
import datetime as _dt
_today_str = _dt.date.today().isoformat()

if "paper_triggered" not in st.session_state:
    st.session_state["paper_triggered"] = {}
if "paper_open" not in st.session_state:
    st.session_state["paper_open"] = {}

# trading_mode: "paper" | "real" | "off"
# real_triggered: {(date_str, sym): trade_id} — prevents re-entering same signal for real
if "trading_mode" not in st.session_state:
    st.session_state["trading_mode"] = "paper"
if "real_triggered" not in st.session_state:
    st.session_state["real_triggered"] = {}

# _entry_confirm_since: {sym: datetime} — tracks when LTP first crossed the entry
# level.  Auto-trigger fires only after the price stays above entry for
# ENTRY_CONFIRM_SECS seconds (false-breakout filter).
if "_entry_confirm_since" not in st.session_state:
    st.session_state["_entry_confirm_since"] = {}
_ENTRY_CONFIRM_SECS = 30   # require 30 s sustained above entry before firing

# ── Scalp trade tracking ──────────────────────────────────────────────────────
# scalp_triggered: {(date_str, sym, direction): trade_id} — prevents re-entering
# scalp_open:      {trade_id: {sym, stop, t1, signal_type, entry, cap, setup_type}}
#   setup_type = "SCALP" lets the 2:45 PM hard-exit know which trades to close early
if "scalp_triggered" not in st.session_state:
    st.session_state["scalp_triggered"] = {}
if "scalp_open" not in st.session_state:
    st.session_state["scalp_open"] = {}
_SCALP_ENTRY_CONFIRM_SECS = 10  # scalps need faster execution: 10 s (vs 30 s intraday)

# ── Market Intel state ────────────────────────────────────────────────────────
# _intel_job_status : "idle" | "running" | "done" | "error: <msg>"
# _intel_result     : {"raw": str, "stocks": list, "bias": dict}
# _intel_applied    : True once user clicked Apply in the dialog
# _intel_stocks_cache : copy of last applied stocks list (for intraday overlay)
if "_intel_job_status" not in st.session_state:
    st.session_state["_intel_job_status"] = "idle"
if "_intel_stocks_cache" not in st.session_state:
    # Restore from DB on first load
    try:
        _cached_uid = st.session_state.get("kite_user_id", "")
        _cached_stks = db.get_market_intel_stocks(user_id=_cached_uid)
        st.session_state["_intel_stocks_cache"] = _cached_stks
        if _cached_stks:
            st.session_state["_intel_applied"] = True
    except Exception:
        st.session_state["_intel_stocks_cache"] = []

# ── Daily gain gate — reset every new trading day ─────────────────────────────
# paper_day_hwm_pct : high-water mark of today's realised return % (paper trades)
# paper_day_blocked : True once the trailing cutoff is triggered
# real_day_hwm_pct  : same but for real trades
# real_day_blocked  : True once real-trade trailing cutoff is triggered
if st.session_state.get("_day_gate_date") != _today_str:
    st.session_state["paper_day_hwm_pct"]  = 0.0
    st.session_state["paper_day_blocked"]  = False
    st.session_state["real_day_hwm_pct"]   = 0.0
    st.session_state["real_day_blocked"]   = False
    st.session_state["_day_gate_date"]     = _today_str
    # Reset intraday Nifty tracking on new day
    st.session_state.pop("_nifty_intraday_pct", None)
    st.session_state.pop("_nifty_live_ltp",     None)
    st.session_state.pop("_nifty_prev_close",   None)
    st.session_state.pop("_scalp_candle_cache", None)

# Initialise Nifty prev-close from base_df once computed_metrics are loaded
if "_nifty_prev_close" not in st.session_state:
    try:
        _bdf = st.session_state.get("_signals_base_df", pd.DataFrame())
        if not _bdf.empty and "ema_20" in _bdf.columns:
            # Use the most recent Nifty 50 LTP from stored metrics as a proxy for
            # yesterday's close — a rough but reliable bootstrap for the first fetch.
            # The real Nifty prev close is the last daily close from the benchmarks data.
            # We'll rely on the first live LTP fetch to set a more accurate reference;
            # for now, zero = "not yet set" so gate is neutral until first fetch.
            pass
    except Exception:
        pass

# On first load (or after page refresh), re-sync paper_triggered + paper_open
# from the DB so we never double-create paper trades.
# Run init_schema() first to ensure is_paper_trade column + signal_config table
# exist on databases that pre-date this feature (idempotent — safe every run).
if st.session_state.get("_paper_sync_date") != _today_str:
    try:
        db.init_schema()
    except Exception:
        pass
    try:
        # Load ALL of today's paper trades (open + closed) so that symbols whose
        # trades already closed cannot re-trigger after a page refresh.
        _all_today_trades = db.get_all_today_paper_trades(user_id=_cur_user_id)
        for _pt in _all_today_trades:
            _k = (_today_str, _pt["tradingsymbol"])
            # Always mark symbol as triggered (blocks re-entry for any status)
            st.session_state["paper_triggered"][_k] = _pt["id"]
            # Only add OPEN trades to the exit-monitoring dict
            if _pt["status"] == "OPEN":
                _sig_t = _pt.get("signal_type", "")
                _setup  = _pt.get("setup_type", "INTRADAY")
                _mon_dict = (
                    st.session_state["scalp_open"]
                    if _setup == "SCALP"
                    else st.session_state["paper_open"]
                )
                _mon_dict[_pt["id"]] = {
                    "sym":         _pt["tradingsymbol"],
                    "stop":        float(_pt["rec_stop"] or 0),
                    "t1":          float(_pt["rec_t1"] or 0),
                    "t2":          float(_pt.get("rec_t2") or 0),
                    "signal_type": _sig_t,
                    "entry":       float(_pt["actual_entry"] or 0),
                    "setup_type":  _setup,
                    "partial_booked": False,
                }
                # Also mark scalp as triggered so it doesn't re-enter
                if _setup == "SCALP":
                    _sk = (_today_str, _pt["tradingsymbol"], _sig_t)
                    st.session_state["scalp_triggered"][_sk] = _pt["id"]
    except Exception:
        pass
    st.session_state["_paper_sync_date"] = _today_str


# ============================================================
# SIDEBAR — Controls
# ============================================================
st.sidebar.title("⚙️ Controls")

# ── User profile + logout ──────────────────────────────────────────────
with st.sidebar.expander(f"👤 {_app_user.get('username','').upper()}", expanded=False):
    st.caption(f"**Account:** {_app_user.get('username','')}")
    if _app_user.get("kite_user_id"):
        st.caption(f"**Kite ID:** {_app_user['kite_user_id']}")
    st.caption(
        f"**Kite:** {'✅ Connected' if st.session_state.get('kite_authenticated') else '⚠️ Token expired'}"
    )
    if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout_btn"):
        _tok_to_del = st.session_state.get("_session_token", "")
        if _tok_to_del:
            db.delete_session(_tok_to_del)
        _ls_del("screener_session")
        for _k in ["app_user", "_session_token", "kite_authenticated", "kite_client",
                   "kite_access_token", "kite_user_id", "kite_user_name"]:
            st.session_state.pop(_k, None)
        st.rerun()
st.sidebar.markdown("---")

# --- Refresh status
last_update = db.get_last_metrics_update()
st.sidebar.caption(f"**Last metrics update:** {last_update}")

st.sidebar.markdown("---")
st.sidebar.subheader("Refresh")
st.sidebar.caption(
    "**Quick Scan** → updates live prices + refreshes all trade signals against today's LTP.  \n"
    "**Full Rescan** → rebuilds everything from 400-day history (run after 3:30 PM)."
)

# --- Quick refresh button
if st.sidebar.button("⚡ Quick Scan (~15s)", use_container_width=True,
                     help="Pulls live LTP + recomputes trade signals. If AI key is set, "
                          "auto-analyses signal stocks not reviewed in 7 days (max 20)."):
    with st.spinner("Pulling fresh quotes + recomputing signals..."):
        try:
            result = data_pipeline.quick_refresh(client=st.session_state.get("kite_client"))
            if "error" in result:
                st.sidebar.error(result["error"])
            else:
                sigs = result.get("signals_recomputed", 0)
                st.sidebar.success(
                    f"✓ {result['stocks_updated']} prices updated, "
                    f"{sigs} signals refreshed in {result['elapsed_sec']}s"
                )
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

    # ── Auto-AI: analyse signal stocks that haven't been reviewed in 7 days ──
    _qscan_ai_client   = st.session_state.get("ai_client")
    _qscan_ai_provider = st.session_state.get("ai_provider")
    if _qscan_ai_client:
        _all_metrics = db.load_metrics()
        # Only stocks with an active trade signal
        _signal_mask = (
            _all_metrics["swing_signal"].isin(["BUY", "SELL"]) |
            _all_metrics["intraday_signal"].isin(["BUY_ABOVE", "SELL_BELOW"]) |
            (_all_metrics.get("scale_signal", pd.Series(dtype=str)) == "INITIAL_ENTRY")
        ) if "swing_signal" in _all_metrics.columns else pd.Series(False, index=_all_metrics.index)
        _signal_stocks = _all_metrics[_signal_mask].copy()

        if not _signal_stocks.empty:
            # Prioritise by composite score; cap at 20 stocks per run
            _signal_stocks = _signal_stocks.sort_values(
                "composite_score", ascending=False
            ).head(20)
            _n_signal = len(_signal_stocks)

            _ai_sidebar_prog = st.sidebar.progress(0, text=f"🤖 AI: analysing {_n_signal} signal stocks…")
            _ai_qs_state = {"done": 0, "skipped": 0}

            def _ai_qs_cb(i, total, sym, skipped=False):
                _ai_qs_state["done"] += 1
                if skipped:
                    _ai_qs_state["skipped"] += 1
                pct = min(1.0, _ai_qs_state["done"] / max(total, 1))
                lbl = "⏭" if skipped else "🔍"
                _ai_sidebar_prog.progress(pct, text=f"{lbl} AI: {sym} ({_ai_qs_state['done']}/{total})")

            _qs_results = _ai.batch_analyze(
                _signal_stocks.to_dict("records"),
                _qscan_ai_client, _qscan_ai_provider,
                stale_hours=168,        # 7 days — fundamentals don't change daily
                progress_callback=_ai_qs_cb,
            )
            _qs_new = 0
            for _res in _qs_results:
                if _res.get("skipped") or _res.get("error"):
                    continue
                _sym = _res.get("tradingsymbol")
                _tok_rows = _all_metrics[_all_metrics["tradingsymbol"] == _sym]
                if not _tok_rows.empty:
                    db.save_ai_result(int(_tok_rows.iloc[0]["instrument_token"]), _res)
                    _qs_new += 1

            _qs_skip = _ai_qs_state["skipped"]
            _ai_sidebar_prog.progress(
                1.0,
                text=f"✅ AI done — {_qs_new} new, {_qs_skip} cached (7d) of {_n_signal} signal stocks"
            )

# --- Refresh Signals button (no API calls — reads existing DB OHLCV)
if st.sidebar.button("📡 Refresh Signals (~30s)", use_container_width=True,
                     help="Recomputes all swing / intraday / scaling signals from existing "
                          "historical data in the DB. No API calls — run this if signal "
                          "columns are empty after a schema change or failed Full Rescan."):
    _sig_bar = st.sidebar.progress(0)
    _sig_status = st.sidebar.empty()

    def _sig_progress(idx, total, sym):
        _sig_bar.progress((idx + 1) / total)
        _sig_status.caption(f"{idx+1}/{total}: {sym}")

    try:
        # Tune thresholds first, then apply them during signal refresh
        _pre_tune = db.tune_signal_config_from_paper(user_id=_cur_user_id, days=30)
        _sig_result = data_pipeline.refresh_signals_only(
            progress_callback=_sig_progress,
            user_id=_cur_user_id,
        )
        _sig_bar.progress(1.0)
        _sig_status.caption("Done")
        if "error" in _sig_result:
            st.sidebar.error(_sig_result["error"])
        else:
            _thresholds = _sig_result.get("thresholds_used", {})
            _thresh_str = (
                f"RSI buy≤{_thresholds.get('rsi_buy_max', 75):.0f} · "
                f"RSI sell≥{_thresholds.get('rsi_sell_min', 25):.0f} · "
                f"Min R/R {_thresholds.get('min_rr', 1.5):.1f}×"
            ) if _thresholds else ""
            _tune_note = ""
            if _pre_tune:
                _tune_note = " · 🧠 " + ", ".join(
                    f"{k.replace('intraday_', '').replace('_', ' ')}→{v}"
                    for k, v in _pre_tune.items()
                )
            st.sidebar.success(
                f"✓ {_sig_result['signals_updated']} signals refreshed "
                f"({_sig_result['errors']} errors) in {_sig_result['elapsed_sec']}s\n"
                f"{_thresh_str}{_tune_note}"
            )
            st.rerun()
    except Exception as _e:
        st.sidebar.error(f"Failed: {_e}")

# --- Full rescan button
if st.sidebar.button("🔄 Full Rescan (~3-5 min)", use_container_width=True,
                     help="Re-pulls 400 days of history for all NSE EQ stocks. "
                          "Run once daily post-market."):
    progress_bar = st.sidebar.progress(0)
    status = st.sidebar.empty()

    def update_progress(idx, total, symbol):
        progress_bar.progress((idx + 1) / total)
        status.caption(f"{idx+1}/{total}: {symbol}")

    try:
        result = data_pipeline.full_rescan(
            progress_callback=update_progress,
            client=st.session_state.get("kite_client"),
        )
        progress_bar.progress(1.0)
        status.caption("Done")
        st.sidebar.success(
            f"✓ {result['metrics_computed']} stocks scored "
            f"in {result['elapsed_sec']}s"
        )
        st.sidebar.json(result)

        # ── Algo feedback: tune thresholds from paper trade performance ───
        try:
            _tuned = db.tune_signal_config_from_paper(user_id=_cur_user_id, days=30)
            if _tuned:
                _tune_lines = "\n".join(
                    f"• {k.replace('intraday_', '').replace('_', ' ')}: {v}"
                    for k, v in _tuned.items()
                )
                st.sidebar.info(
                    f"🧠 **Signal thresholds auto-tuned** from paper trade results:\n{_tune_lines}\n\n"
                    f"Run **Refresh Signals** to apply to the current universe.",
                    icon="🧠",
                )
        except Exception:
            pass
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())

st.sidebar.markdown("---")

# ─── Kite connection status + key management ────────────────
st.sidebar.subheader("🔑 Zerodha Kite Connect")
_kc_live     = st.session_state.get("kite_client")
_ss_uid      = st.session_state.get("kite_user_id",   "")
_ss_uname    = st.session_state.get("kite_user_name", "")
if _kc_live and _kc_live.authenticated:
    _id_label = f" · {_ss_uname or _ss_uid}" if (_ss_uname or _ss_uid) else ""
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:#22c55e;padding:2px 0 6px 0;">'
        f'🟢 <b>Kite connected{_id_label}</b> — intraday candles &amp; live prices active'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div style="font-size:12px;color:#f59e0b;padding:2px 0 4px 0;">'
        '🟡 <b>Kite not connected</b> — intraday candles unavailable'
        '</div>',
        unsafe_allow_html=True,
    )
    _reauth_client = KiteClient(
        api_key=st.session_state.get("kite_api_key",    ""),
        api_secret=st.session_state.get("kite_api_secret", ""),
    )
    _reauth_url = _reauth_client.get_login_url()
    if _reauth_url:
        st.sidebar.link_button(
            "🔑 Authenticate Kite",
            _reauth_url,
            use_container_width=True,
        )

# ── Update / rotate API keys (per-session; also persisted for local dev) ─
with st.sidebar.expander("🔄 Update API Keys", expanded=False):
    _upd_k = st.text_input(
        "API Key",
        value=st.session_state.get("kite_api_key", ""),
        type="password",
        key="upd_kite_key",
        help="Kite Developer Console → My Apps → API Key",
    )
    _upd_s = st.text_input(
        "API Secret",
        value=st.session_state.get("kite_api_secret", ""),
        type="password",
        key="upd_kite_secret",
        help="Kite Developer Console → My Apps → API Secret",
    )
    if st.button("💾 Save Keys", key="upd_kite_save", use_container_width=True):
        if _upd_k.strip() and _upd_s.strip():
            st.session_state["kite_api_key"]    = _upd_k.strip()
            st.session_state["kite_api_secret"] = _upd_s.strip()
            # Persist to users table (single source of truth)
            try:
                db.update_kite_credentials(
                    _app_user["id"], _upd_k.strip(), _upd_s.strip()
                )
                _app_user["kite_api_key"]    = _upd_k.strip()
                _app_user["kite_api_secret"] = _upd_s.strip()
                st.session_state["app_user"] = _app_user
            except Exception:
                pass
            for _k in ("kite_authenticated", "kite_client", "kite_access_token",
                       "kite_access_date", "kite_user_id", "kite_user_name"):
                st.session_state.pop(_k, None)
            st.success("Keys updated — please re-authenticate Kite above.")
            st.rerun()
        else:
            st.error("Both fields required.")

# ── Sign out / forget this browser ─────────────────────────────────
if _ON_CLOUD and st.sidebar.button("🚪 Sign out & forget keys",
                                   use_container_width=True,
                                   help="Clears your API keys and token from this browser"):
    for _lk in ("kite_api_key", "kite_api_secret", "kite_access_token", "kite_access_date"):
        _ls_del(_lk)
    for _sk in ("kite_authenticated", "kite_client", "kite_access_token",
                "kite_access_date", "kite_api_key", "kite_api_secret",
                "kite_user_id", "kite_user_name", "kite_ss_initialized"):
        st.session_state.pop(_sk, None)
    st.rerun()

st.sidebar.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# AI ANALYSIS SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.subheader("🤖 AI Analysis (STOCKLENS)")
st.sidebar.caption(
    "Connect an AI API to enrich screener results with fundamental, "
    "sentiment, and macro analysis. OpenRouter (Perplexity) uses live web search."
)

# Load persisted keys
_ai_keys = _ai.load_keys()

_or_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=_ai_keys.get("openrouter_key", ""),
    type="password",
    help="Preferred. Get a free key at openrouter.ai · Uses Perplexity/sonar-pro with live web search.",
    key="sidebar_or_key",
)
_oa_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=_ai_keys.get("openai_key", ""),
    type="password",
    help="Fallback when OpenRouter key is not set. Uses gpt-4o.",
    key="sidebar_oa_key",
)

# Save keys whenever they change
if _or_key != _ai_keys.get("openrouter_key", "") or _oa_key != _ai_keys.get("openai_key", ""):
    _ai.save_keys(_oa_key, _or_key)

# Show connection status
_ai_client, _ai_provider = _ai.get_client(_oa_key, _or_key)
if _ai_client:
    _provider_label = "OpenRouter (Perplexity/sonar-pro)" if _ai_provider == "openrouter" else "OpenAI (gpt-4o)"
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:#22c55e;padding:2px 0 6px 0;">'
        f'🟢 AI connected · {_provider_label}</div>',
        unsafe_allow_html=True,
    )
    # Store client in session state so the rest of the app can use it
    st.session_state["ai_client"]   = _ai_client
    st.session_state["ai_provider"] = _ai_provider
else:
    st.sidebar.markdown(
        '<div style="font-size:12px;color:#64748b;padding:2px 0 6px 0;">'
        '⚪ AI not connected — add a key above to enable STOCKLENS scoring</div>',
        unsafe_allow_html=True,
    )
    st.session_state.pop("ai_client",   None)
    st.session_state.pop("ai_provider", None)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")


# ============================================================
# LOAD DATA
# ============================================================
df = db.load_metrics()
# Keep df in session_state so the live-signal fragment always sees the latest
# base data (entry/stop/pivot levels) even across fragment-only reruns.
st.session_state["_signals_base_df"] = df

# ── Nifty prev-close bootstrap ────────────────────────────────────────────
# Extract Nifty 50's previous close from the benchmark data stored in DB so
# the intraday direction gate can compute today's % change on the first LTP fetch.
# We look for a row with tradingsymbol = "NIFTY 50" in the metrics DF; if missing,
# we'll use the first authenticated LTP fetch to set a rough reference.
if "_nifty_prev_close" not in st.session_state and not df.empty:
    _nf_row = df[df["tradingsymbol"].astype(str).str.upper() == "NIFTY 50"] if "tradingsymbol" in df.columns else pd.DataFrame()
    if not _nf_row.empty and "ltp" in _nf_row.columns:
        _nf_prev_val = float(_nf_row["ltp"].iloc[0] or 0)
        if _nf_prev_val > 0:
            st.session_state["_nifty_prev_close"] = _nf_prev_val

if df.empty:
    st.warning("⚠ No data yet. Click 'Full Rescan' in the sidebar to bootstrap.")
    st.stop()


# ============================================================
# SIDEBAR FILTERS (continued, now that we have data)
# ============================================================
min_score, max_score = float(df["composite_score"].min() or 0), float(df["composite_score"].max() or 100)
score_threshold = st.sidebar.slider(
    "Min composite score", min_score, max_score, min_score,
    help="Higher = stronger momentum + RS + volume"
)

rsi_min, rsi_max = st.sidebar.slider(
    "RSI(14) range", 0, 100, (40, 75),
    help="Sweet spot for short swing: 50-70 (trending up, not overbought)"
)

max_dist_52w_high = st.sidebar.slider(
    "Max % below 52W high", 0, 100, 25,
    help="Lower = closer to 52W high = stronger stock"
)

min_5d_return = st.sidebar.slider(
    "Min 5D return %", -20, 20, -5,
    help="Negative values allowed for pullback setups"
)

min_turnover = st.sidebar.number_input(
    "Min avg turnover (₹ Cr)", value=config.MIN_AVG_TURNOVER_CR,
    min_value=1.0, step=1.0,
    help="Stocks below this daily turnover are excluded. ₹5 Cr is the slippage threshold for retail traders."
)

min_avg_volume = st.sidebar.number_input(
    "Min avg daily volume (shares)", value=int(config.MIN_AVG_VOLUME),
    min_value=10_000, step=10_000,
    help="Low-volume stocks have wide bid-ask spreads and are hard to exit quickly."
)

vol_expansion_min = st.sidebar.slider(
    "Min volume expansion (5D/20D)", 0.0, 3.0, 0.0, step=0.1,
    help=(
        "Ratio of 5-day avg volume ÷ 20-day avg volume. "
        "> 1.2 = volume building (institutional buying). "
        "1.0 = neutral. < 0.8 = volume drying up (avoid)."
    )
)

# Trend alignment toggle
require_all_positive = st.sidebar.checkbox(
    "Require ALL timeframes positive", value=False,
    help="1Y, 6M, 3M, 1M, 5D all > 0%"
)


# ============================================================
# APPLY FILTERS
# ============================================================
filtered = df.copy()
filtered = filtered[filtered["composite_score"].fillna(-999) >= score_threshold]
filtered = filtered[filtered["rsi_14"].between(rsi_min, rsi_max, inclusive="both") |
                    filtered["rsi_14"].isna()]
filtered = filtered[filtered["dist_from_52w_high_pct"].fillna(999) <= max_dist_52w_high]
filtered = filtered[filtered["ret_5d"].fillna(-999) >= min_5d_return]
filtered = filtered[filtered["avg_turnover_cr"].fillna(0) >= min_turnover]
filtered = filtered[filtered["avg_volume"].fillna(0) >= min_avg_volume]
if vol_expansion_min > 0:
    filtered = filtered[filtered["vol_expansion_ratio"].fillna(0) >= vol_expansion_min]

if require_all_positive:
    for col in ["ret_1y", "ret_6m", "ret_3m", "ret_1m", "ret_5d"]:
        filtered = filtered[filtered[col].fillna(-999) > 0]


# ============================================================
# TAB LAYOUT — Screener | Trade Signals | Activity Log
# ============================================================
tab_screener, tab_signals, tab_activity = st.tabs([
    "📋 Screener", "🎯 Trade Signals", "📒 Activity Log"
])

# ─── helper used by both tabs ───────────────────────────────
def _isna(v) -> bool:
    """True for Python None, float nan, pd.NA, np.nan — anything falsy-for-numbers."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _stars(q) -> str:
    """Convert integer 1-5 to filled/empty star string."""
    if _isna(q):
        return "—"
    try:
        q = int(q)
    except (TypeError, ValueError):
        return "—"
    return "★" * q + "☆" * (5 - q)


def _fmt(v, fmt="₹{:,.2f}", fallback="—"):
    """Format a number safely, return fallback on None/NaN/pd.NA."""
    if _isna(v):
        return fallback
    try:
        return fmt.format(float(v))
    except Exception:
        return fallback


def _fv(v):
    """Return float or None, safe for NaN."""
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except Exception:
        return None


def _render_order_panel(signal_df: pd.DataFrame, setup_type: str, form_key: str):
    """
    Unified order panel shown below each signal table.

    When Kite is connected (st.session_state["kite_client"] exists):
      → Shows "🚀 Place Order via Kite" which places the order on the exchange
        AND auto-logs it to the Activity Log with the Kite order ID.
      → Optionally places a companion stop-loss order.

    When Kite is not connected:
      → Falls back to a manual "📝 Log Trade" form (records only in local DB).

    signal_df must include: tradingsymbol, rec_entry, rec_stop, rec_t1, rec_t2,
    rec_rr, rec_reason, instrument_token, composite_score, ai_score, signal_type.
    setup_type: 'SWING' | 'INTRADAY' | 'SCALING'
    """
    if signal_df.empty:
        return

    _kc = st.session_state.get("kite_client")
    kite_connected = _kc is not None and getattr(_kc, "authenticated", False)

    # ── Real-trade daily gate ─────────────────────────────────────────────────
    # Mirrors the paper-trade trailing-stop logic using today's realised real P&L.
    _uid_op = st.session_state.get("kite_user_id", "")
    try:
        _real_pnl_today = db.get_today_closed_pnl(user_id=_uid_op, is_paper=False)
    except Exception:
        _real_pnl_today = 0.0
    _rcap  = config.PAPER_CAPITAL   # use same base; user can adjust in config
    _r_ret = (_real_pnl_today / _rcap * 100) if _rcap else 0.0
    _r_hwm = max(st.session_state.get("real_day_hwm_pct", 0.0), _r_ret)
    st.session_state["real_day_hwm_pct"] = _r_hwm
    _r_low   = config.DAILY_TARGET_LOW_PCT
    _r_high  = config.DAILY_TARGET_HIGH_PCT
    _r_trail = config.DAILY_TRAIL_PCT
    _r_cutoff = (_r_hwm - _r_trail) if _r_hwm >= _r_low else None
    _real_blocked = (
        (_r_cutoff is not None and _r_ret <= _r_cutoff) or _r_ret >= _r_high
    )
    st.session_state["real_day_blocked"] = _real_blocked

    expander_label = (
        "🚀 Place Order via Kite" if kite_connected else "📝 Log a trade from this list"
    )
    if _real_blocked:
        expander_label += "  🚫 Daily gate closed"

    with st.expander(expander_label, expanded=False):
        if _real_blocked:
            _r_reason = (
                f"Realised gain **{_r_ret:.2f}%** hit ceiling **{_r_high:.0f}%**"
                if _r_ret >= _r_high
                else f"Realised gain **{_r_ret:.2f}%** dropped to cutoff **{_r_cutoff:.2f}%** "
                     f"(peak **{_r_hwm:.2f}%** − {_r_trail:.1f}% trail)"
            )
            st.warning(
                f"**Daily trading gate closed.** No new real orders for the rest of today.  \n{_r_reason}",
                icon="🚫",
            )
            return
        sym_col, _ = st.columns([2, 5])
        sym_options = sorted(signal_df["tradingsymbol"].dropna().unique().tolist())
        selected_sym = sym_col.selectbox(
            "Select symbol", sym_options, key=f"{form_key}_sym"
        )

        row = signal_df[signal_df["tradingsymbol"] == selected_sym]
        if row.empty:
            return
        row = row.iloc[0]

        r_entry   = _fv(row.get("rec_entry")  or row.get("swing_entry") or row.get("intraday_entry") or row.get("scale_entry_1"))
        r_stop    = _fv(row.get("rec_stop")   or row.get("swing_stop")  or row.get("intraday_stop")  or row.get("scale_stop"))
        r_t1      = _fv(row.get("rec_t1")     or row.get("swing_t1")    or row.get("intraday_t1")    or row.get("scale_target"))
        r_t2      = _fv(row.get("rec_t2")     or row.get("swing_t2"))
        r_rr      = _fv(row.get("rec_rr")     or row.get("swing_rr"))
        r_reason  = str(row.get("rec_reason") or row.get("swing_reason") or row.get("intraday_reason") or row.get("scale_reason") or "")
        r_token   = int(row.get("instrument_token") or 0)
        r_cscore  = _fv(row.get("composite_score"))
        r_ai      = _fv(row.get("ai_score"))
        r_sigtype = str(row.get("signal_type") or row.get("swing_signal") or row.get("intraday_signal") or row.get("scale_signal") or "BUY")

        cols_rec = st.columns(5)
        cols_rec[0].metric("Rec Entry", _fmt(r_entry))
        cols_rec[1].metric("Rec Stop",  _fmt(r_stop))
        cols_rec[2].metric("T1",        _fmt(r_t1))
        cols_rec[3].metric("T2",        _fmt(r_t2) if r_t2 else "—")
        cols_rec[4].metric("R/R",       f"{r_rr:.1f}×" if r_rr else "—")

        # ── Determine smart defaults ────────────────────────────────────
        is_sell   = r_sigtype in ("SELL", "SELL_BELOW")
        txn_type  = "SELL" if is_sell else "BUY"
        # Intraday signals use SL-M (stop-market); others use LIMIT
        def_otype = "SL-M" if r_sigtype in ("BUY_ABOVE", "SELL_BELOW") else "LIMIT"
        def_prod  = "MIS" if setup_type == "INTRADAY" else "CNC"

        if kite_connected:
            # ── KITE ORDER FORM ─────────────────────────────────────────
            st.markdown(
                '<div style="font-size:12px;color:#22c55e;margin-bottom:8px;">'
                '⚡ Kite connected — order will be placed directly on NSE and '
                'auto-logged to your Activity Log.'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.form(key=f"{form_key}_kite_{selected_sym}"):
                c1, c2, c3, c4 = st.columns(4)
                qty        = c1.number_input("Qty (shares)", min_value=1, value=1, step=1)
                order_type = c2.selectbox("Order type", ["LIMIT", "MARKET", "SL-M", "SL"],
                                          index=["LIMIT", "MARKET", "SL-M", "SL"].index(def_otype))
                product    = c3.selectbox("Product", ["CNC", "MIS", "NRML"],
                                          index=["CNC", "MIS", "NRML"].index(def_prod))
                txn_disp   = c4.selectbox("Transaction", ["BUY", "SELL"],
                                          index=["BUY", "SELL"].index(txn_type))

                c5, c6 = st.columns(2)
                entry_price   = c5.number_input(
                    "Entry price ₹ (LIMIT / SL)",
                    min_value=0.0, value=float(r_entry or 0), step=0.05, format="%.2f",
                )
                trigger_price = c6.number_input(
                    "Trigger price ₹ (SL / SL-M)",
                    min_value=0.0, value=float(r_entry or 0), step=0.05, format="%.2f",
                )

                place_sl = st.checkbox(
                    f"Also place stop-loss order at ₹{r_stop or 0:,.2f}",
                    value=bool(r_stop),
                )
                sl_price = st.number_input(
                    "Stop-loss trigger ₹",
                    min_value=0.0, value=float(r_stop or 0), step=0.05, format="%.2f",
                    disabled=not place_sl,
                )
                notes_k = st.text_input("Notes (optional)", placeholder="e.g. R/R 2× trigger on EMA bounce")

                btn_col, warn_col = st.columns([2, 3])
                submitted_kite = btn_col.form_submit_button(
                    f"🚀 Place {txn_disp} Order on Kite", type="primary", use_container_width=True
                )
                warn_col.markdown(
                    '<div style="font-size:11px;color:#f59e0b;padding-top:8px;">'
                    '⚠ Orders are real and go to the exchange. Double-check qty and price.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if submitted_kite:
                try:
                    ep  = float(entry_price)   if entry_price   else None
                    tp  = float(trigger_price) if trigger_price else None
                    # Place entry order
                    order_id = _kc.place_order(
                        tradingsymbol   = selected_sym,
                        qty             = int(qty),
                        transaction_type= txn_disp,
                        order_type      = order_type,
                        product         = product,
                        price           = ep  if order_type in ("LIMIT", "SL") else None,
                        trigger_price   = tp  if order_type in ("SL", "SL-M")  else None,
                        tag             = f"scr_{setup_type[:3].lower()}",
                    )

                    # Optionally place companion SL order
                    sl_order_id = None
                    if place_sl and sl_price and sl_price > 0:
                        sl_txn = "BUY" if txn_disp == "SELL" else "SELL"
                        try:
                            sl_order_id = _kc.place_order(
                                tradingsymbol    = selected_sym,
                                qty              = int(qty),
                                transaction_type = sl_txn,
                                order_type       = "SL-M",
                                product          = product,
                                trigger_price    = float(sl_price),
                                tag              = f"sl_{setup_type[:3].lower()}",
                            )
                        except Exception as _sl_err:
                            st.warning(f"Stop-loss order failed: {_sl_err}. Main order still placed.")

                    # Auto-log to DB (tagged with current user)
                    trade_dict = {
                        "trade_date":          pd.Timestamp.today().date(),
                        "tradingsymbol":       selected_sym,
                        "instrument_token":    r_token,
                        "setup_type":          setup_type,
                        "signal_type":         r_sigtype,
                        "rec_entry":           r_entry,
                        "rec_stop":            r_stop,
                        "rec_t1":              r_t1,
                        "rec_t2":              r_t2,
                        "rec_rr":              r_rr,
                        "rec_reason":          r_reason[:200],
                        "rec_composite_score": r_cscore,
                        "rec_ai_score":        r_ai,
                        "kite_user_id":        _cur_user_id,
                        "kite_order_id":       order_id,
                        "kite_sl_order_id":    sl_order_id,
                        "kite_status":         "OPEN",
                        "quantity":            int(qty),
                        "actual_entry":        ep or r_entry,
                        "status":              "OPEN",
                        "notes":               notes_k or None,
                    }
                    new_id = db.log_trade(trade_dict)
                    sl_note = f" + SL order {sl_order_id}" if sl_order_id else ""
                    st.success(
                        f"✅ **Order placed!** Kite order ID: `{order_id}`{sl_note}  "
                        f"— logged to Activity Log (id={new_id}). "
                        f"Check **📒 Activity Log** tab to track status."
                    )
                except Exception as _e:
                    st.error(f"Order failed: {_e}")

        else:
            # ── MANUAL LOG FALLBACK ─────────────────────────────────────
            st.markdown(
                '<div style="font-size:12px;color:#94a3b8;margin-bottom:8px;">'
                'Record your actual entry/exit against the recommendation. '
                'Connect Kite to place orders directly from here.'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.form(key=f"{form_key}_manual_{selected_sym}"):
                c1, c2, c3, c4 = st.columns(4)
                trade_date   = c1.date_input("Trade date", value=pd.Timestamp.today().date())
                quantity     = c2.number_input("Qty (shares)", min_value=1, value=1, step=1)
                actual_entry = c3.number_input("Actual entry ₹", min_value=0.01,
                                               value=float(r_entry or 0) or 0.01, step=0.05, format="%.2f")
                status_m     = c4.selectbox("Status", ["OPEN", "CLOSED", "TARGET_HIT", "STOPPED_OUT", "CANCELLED"])

                c5, c6 = st.columns(2)
                actual_exit = c5.number_input("Actual exit ₹ (0 = still open)",
                                              min_value=0.0, value=0.0, step=0.05, format="%.2f")
                notes_m     = c6.text_input("Notes (optional)", placeholder="e.g. gapped up at open")

                submitted_manual = st.form_submit_button("💾 Save Trade Log", type="primary", use_container_width=True)

            if submitted_manual:
                trade_dict = {
                    "trade_date":          trade_date,
                    "tradingsymbol":       selected_sym,
                    "instrument_token":    r_token,
                    "setup_type":          setup_type,
                    "signal_type":         r_sigtype,
                    "rec_entry":           r_entry,
                    "rec_stop":            r_stop,
                    "rec_t1":              r_t1,
                    "rec_t2":              r_t2,
                    "rec_rr":              r_rr,
                    "rec_reason":          r_reason[:200],
                    "rec_composite_score": r_cscore,
                    "rec_ai_score":        r_ai,
                    "kite_user_id":        _cur_user_id,
                    "quantity":            quantity,
                    "actual_entry":        float(actual_entry) if actual_entry else None,
                    "actual_exit":         float(actual_exit)  if actual_exit  else None,
                    "status":              status_m,
                    "notes":               notes_m or None,
                }
                try:
                    new_id = db.log_trade(trade_dict)
                    if actual_exit and actual_entry:
                        direction = -1 if r_sigtype in ("SELL", "SELL_BELOW") else 1
                        pnl     = direction * (float(actual_exit) - float(actual_entry)) * quantity
                        pnl_pct = direction * (float(actual_exit) - float(actual_entry)) / float(actual_entry) * 100
                        st.success(f"✅ Logged (id={new_id}) — P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    else:
                        st.success(f"✅ Trade logged as {status_m} (id={new_id})")
                except Exception as _e:
                    st.error(f"Failed to save: {_e}")


# ============================================================
# MARKET INTEL DIALOG — defined here (before any tab) so it
# is available from both the Screener tab AND the auto-trigger.
# ============================================================
@st.dialog("🧠 India Market Intelligence Brief", width="large")
def _show_market_intel_dialog(uid: str) -> None:
    """Display the market intel brief, parsed stock tables, and Apply button."""
    result = st.session_state.get("_intel_result", {})
    if not result:
        st.warning("No Market Intel results in session. Re-run from the Screener tab.")
        return

    raw    = result.get("raw", "")
    stocks = result.get("stocks", [])
    bias_d = result.get("bias", {})

    bias       = bias_d.get("bias", "NEUTRAL")
    confidence = bias_d.get("confidence", "MEDIUM")

    _bias_colors = {
        "BULLISH":         "#22c55e",
        "MILDLY BULLISH":  "#86efac",
        "NEUTRAL":         "#f59e0b",
        "MILDLY BEARISH":  "#f87171",
        "BEARISH":         "#ef4444",
    }
    _bias_color = _bias_colors.get(bias, "#f59e0b")

    st.markdown(
        f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
        f'padding:10px 20px;margin-bottom:10px;display:flex;gap:20px;align-items:center">'
        f'<span style="font-size:0.78rem;color:#64748b;font-weight:600;text-transform:uppercase">🧠 Market Intel</span>'
        f'<span style="font-size:1rem;color:{_bias_color};font-weight:700">{bias}</span>'
        f'<span style="font-size:0.8rem;color:#94a3b8">Confidence: <b>{confidence}</b></span>'
        f'<span style="font-size:0.75rem;color:#64748b">'
        f'{len([s for s in stocks if s["stance"]=="BUY"])} BUY · '
        f'{len([s for s in stocks if s["stance"]=="SHORT"])} SHORT · '
        f'{len([s for s in stocks if s["stance"]=="AVOID"])} AVOID · '
        f'{len([s for s in stocks if s["stance"]=="BUY_ON_COND"])} BUY ON COND'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _base_df = st.session_state.get("_signals_base_df", None)
    stocks   = _mi.compute_overlap(stocks, _base_df)

    _buy_stk  = [s for s in stocks if s["stance"] == _mi.BUY]
    _shrt_stk = [s for s in stocks if s["stance"] == _mi.SHORT]
    _avd_stk  = [s for s in stocks if s["stance"] == _mi.AVOID]
    _cnd_stk  = [s for s in stocks if s["stance"] == _mi.BUY_ON_COND]

    if _buy_stk:
        st.markdown("#### 📗 BUY — Enter Now")
        _buy_rows = []
        for s in _buy_stk:
            _ovlp = s.get("overlap_type")
            _badge = (
                "🔥 Both++" if _ovlp == "SAME_DIR"    else
                "⚠️ Conflict" if _ovlp == "OPPOSITE_DIR" else
                "🧠 Intel only"
            )
            _buy_rows.append({
                "Symbol":         s["tradingsymbol"],
                "Sector":         s["sector"][:30],
                "Source":         _badge,
                "Why Buy":        s["fundamental_reason"][:120],
                "Entry Trigger":  s["entry_trigger"][:100],
                "Stop Loss":      s["stop_loss"][:50],
                "Conviction":     s["conviction"],
            })
        st.dataframe(pd.DataFrame(_buy_rows), hide_index=True, use_container_width=True)

    if _shrt_stk:
        st.markdown("#### 📕 SHORT — Active Short Setup")
        _short_rows = []
        for s in _shrt_stk:
            _ovlp = s.get("overlap_type")
            _badge = (
                "🔥 Both++" if _ovlp == "SAME_DIR" else
                "⚠️ Conflict" if _ovlp == "OPPOSITE_DIR" else
                "🧠 Intel only"
            )
            _short_rows.append({
                "Symbol":            s["tradingsymbol"],
                "Sector":            s["sector"][:30],
                "Source":            _badge,
                "Why Short":         s["fundamental_reason"][:120],
                "Breakdown Trigger": s["entry_trigger"][:100],
                "Stop Loss":         s["stop_loss"][:50],
                "Conviction":        s["conviction"],
            })
        st.dataframe(pd.DataFrame(_short_rows), hide_index=True, use_container_width=True)

    if _avd_stk:
        st.markdown("#### 📙 AVOID — Stay Out")
        _avd_rows = []
        for s in _avd_stk:
            _ovlp  = s.get("overlap_type")
            _badge = "⚠️ In Signals!" if _ovlp == "AVOID_WARNING" else "—"
            _avd_rows.append({
                "Symbol":             s["tradingsymbol"],
                "Sector":             s["sector"][:30],
                "In Signals":         _badge,
                "Why Avoid":          s["fundamental_reason"][:150],
                "What Changes This":  s["condition_required"][:100],
            })
        st.dataframe(pd.DataFrame(_avd_rows), hide_index=True, use_container_width=True)

    if _cnd_stk:
        st.markdown("#### 📘 BUY ON CONDITION — Set Alert")
        _cnd_rows = []
        for s in _cnd_stk:
            _cnd_rows.append({
                "Symbol":        s["tradingsymbol"],
                "Sector":        s["sector"][:30],
                "Setup":         s["fundamental_reason"][:100],
                "Condition":     s["condition_required"][:120],
                "Alert Level":   s["alert_level"][:50],
                "Expected Move": s["expected_move"][:50],
            })
        st.dataframe(pd.DataFrame(_cnd_rows), hide_index=True, use_container_width=True)

    if not stocks:
        st.info(
            "No stocks were parsed from the brief. The AI may have used a different table format. "
            "Check the full brief below.",
            icon="ℹ️",
        )

    with st.expander("📄 View Full Market Intelligence Brief", expanded=False):
        st.markdown(raw or "_No raw output available._")

    st.markdown("---")
    _ap1, _ap2 = st.columns([2, 1])

    if stocks:
        if _ap1.button(
            "✅ Apply to Intraday Signals",
            type="primary",
            use_container_width=True,
            help="Saves Market Intel stocks to DB and shows them in the Intraday Plan tab",
        ):
            db.save_market_intel(
                user_id=uid,
                raw_output=raw,
                bias=bias,
                confidence=confidence,
                stocks=stocks,
            )
            st.session_state["_intel_applied"]      = True
            st.session_state["_intel_stocks_cache"] = stocks
            st.success(
                f"✅ Applied {len(stocks)} Market Intel signals! "
                "Switch to **🎯 Trade Signals → Intraday Plan** to see them.",
                icon="🧠",
            )

    if _ap2.button("❌ Close", use_container_width=True, key="intel_dialog_close"):
        st.rerun()


# ── Market Intel background poller ───────────────────────────────────────────
# Defined here (before tab_screener renders) so both tabs can call it.
# Runs every 5 s ONLY while a job is in flight; instant no-op when idle.
@st.fragment(run_every=5)
def _intel_poller():
    """Light fragment: check if background market intel job finished."""
    if st.session_state.get("_intel_job_status") != "running":
        return
    _uid_poll = st.session_state.get("kite_user_id", "")
    result = _mi.check_job(_uid_poll)
    if result is not None:
        if result.get("error"):
            st.session_state["_intel_job_status"] = f"error: {result['error']}"
            st.toast(f"❌ Market Intel failed: {result['error'][:80]}", icon="❌")
        else:
            st.session_state["_intel_job_status"]  = "done"
            st.session_state["_intel_result"]      = result
            st.session_state["_intel_open_dialog"] = True   # triggers auto-show
            n = len(result.get("stocks", []))
            st.toast(f"🧠 Market Intel complete — {n} stocks identified!", icon="✅")
        # Full-page rerun so the dialog auto-trigger fires regardless of active tab
        st.rerun(scope="app")


# ── Auto-open Market Intel dialog when job completes (any tab) ────────────────
# Runs on every full-page rerun. When _intel_poller sets _intel_open_dialog,
# this pops the flag and opens the dialog before tabs are rendered.
if st.session_state.pop("_intel_open_dialog", False):
    _show_market_intel_dialog(uid=st.session_state.get("kite_user_id", ""))


# ─── SCREENER TAB ───────────────────────────────────────────
with tab_screener:
    _hc1, _hc2, _hc3, _hc4 = st.columns(4)
    _hc1.metric("Universe",    len(df))
    _hc2.metric("Filtered",    len(filtered))
    _hc3.metric("Pct retained", f"{len(filtered)/max(len(df),1)*100:.1f}%")
    _hc4.metric("Last update", last_update.split()[1] if last_update != "never" else "-")

    _intel_poller()   # background job watcher (no-op when idle)

    # ── MARKET INTEL BANNER ────────────────────────────────────────────────
    _mi_ai_client   = st.session_state.get("ai_client")
    _mi_ai_provider = st.session_state.get("ai_provider", "")
    _mi_uid         = st.session_state.get("kite_user_id", "")

    _mi_col1, _mi_col2, _mi_col3 = st.columns([2, 2, 3])
    with _mi_col1:
        if _mi_ai_client:
            if st.button(
                "🧠 Run Market Intel",
                type="primary",
                use_container_width=True,
                help=(
                    "Runs a live India market intelligence brief using Perplexity/sonar-pro "
                    "(web search enabled). Analyses global macro, active triggers, and produces "
                    "BUY / SHORT / AVOID / BUY ON CONDITION stock recommendations."
                ),
                key="btn_run_market_intel",
            ):
                # Gather Kite live prices for Section 4 pricing
                _mi_live_prices: dict = {}
                _kc_mi = st.session_state.get("kite_client")
                if _kc_mi and getattr(_kc_mi, "authenticated", False):
                    try:
                        _nifty50 = [
                            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
                            "BHARTIARTL", "KOTAKBANK", "SBIN", "LT", "BAJFINANCE",
                            "HINDUNILVR", "ITC", "MARUTI", "M&M", "AXISBANK",
                            "SUNPHARMA", "TATAMOTORS", "NTPC", "ONGC", "ADANIPORTS",
                            "WIPRO", "HCLTECH", "BAJAJ-AUTO", "ULTRACEMCO", "TITAN",
                            "POWERGRID", "TECHM", "GRASIM", "TATASTEEL", "JSWSTEEL",
                        ]
                        _lt_resp = _kc_mi.get_ltp(
                            [f"NSE:{s}" for s in _nifty50]
                        )
                        for _sym, _v in _lt_resp.items():
                            _clean = _sym.replace("NSE:", "")
                            _mi_live_prices[_clean] = _v.get("last_price", 0)
                    except Exception:
                        pass

                _mi.start_job(
                    user_id=_mi_uid,
                    client=_mi_ai_client,
                    provider=_mi_ai_provider,
                    live_prices=_mi_live_prices or None,
                )
                st.session_state["_intel_job_status"] = "running"
                st.session_state.pop("_intel_result", None)
                st.toast("🧠 Market Intel started — results in ~2-4 min", icon="🔄")
        else:
            st.button(
                "🧠 Run Market Intel",
                disabled=True,
                use_container_width=True,
                help="Add an OpenRouter or OpenAI key in the sidebar (🤖 AI Analysis) to enable Market Intel",
                key="btn_run_market_intel_disabled",
            )

    with _mi_col2:
        # Check if background job completed on this rerun
        _mi_job_done = _mi.check_job(_mi_uid)
        if _mi_job_done is not None:
            if _mi_job_done.get("error"):
                st.session_state["_intel_job_status"] = f"error: {_mi_job_done['error']}"
            else:
                st.session_state["_intel_job_status"] = "done"
                st.session_state["_intel_result"]     = _mi_job_done

        _mi_status = st.session_state.get("_intel_job_status", "idle")

        if _mi_status == "running":
            st.info("🔄 Running (~2-4 min)… switch tabs freely", icon="🔄")
        elif _mi_status == "done":
            _mi_result_peek = st.session_state.get("_intel_result", {})
            _n_parsed = len(_mi_result_peek.get("stocks", []))
            if st.button(
                f"📊 View Intel Results ({_n_parsed} stocks)",
                use_container_width=True,
                key="btn_view_intel",
            ):
                _show_market_intel_dialog(uid=_mi_uid)
        elif _mi_status.startswith("error"):
            st.error(f"Market Intel failed: {_mi_status[6:]}", icon="❌")

    with _mi_col3:
        # Show applied intel status + latest run timestamp
        _mi_latest = None
        try:
            _mi_latest = db.get_latest_market_intel(user_id=_mi_uid)
        except Exception:
            pass
        if _mi_latest and st.session_state.get("_intel_applied"):
            _stks = db.get_market_intel_stocks(user_id=_mi_uid)
            _mi_bias = _mi_latest.get("bias", "NEUTRAL")
            _mi_ts   = _mi_latest.get("created_at")
            _ts_str  = _mi_ts.strftime("%d/%m %H:%M") if _mi_ts else "—"
            st.markdown(
                f'<div style="font-size:0.75rem;color:#64748b;padding-top:6px;">'
                f'🧠 Intel applied · <b style="color:#f8fafc">{len(_stks)} stocks</b> · '
                f'Bias: <b style="color:#f59e0b">{_mi_bias}</b> · Run at {_ts_str}</div>',
                unsafe_allow_html=True,
            )
        elif not _mi_ai_client:
            st.markdown(
                '<div style="font-size:0.75rem;color:#64748b;padding-top:6px;">'
                '⚪ Add OpenRouter or OpenAI key in sidebar to enable Market Intel</div>',
                unsafe_allow_html=True,
            )

    st.subheader(f"Screener Results — {len(filtered)} candidates")

    # ── AI status line (no bulk-run button — analysis is per-stock or auto on signals) ──
    _ai_ready = "ai_client" in st.session_state
    _ai_analyzed_in_view = (
        filtered["ai_score"].notna().sum()
        if "ai_score" in filtered.columns else 0
    )
    _ai_analyzed_total = (
        df["ai_score"].notna().sum()
        if "ai_score" in df.columns else 0
    )
    if _ai_ready:
        st.markdown(
            f'<div style="font-size:12px;color:#22c55e;margin-bottom:4px;">'
            f'🤖 AI connected · {_ai_analyzed_in_view} of {len(filtered)} stocks in this view have STOCKLENS scores'
            f' ({_ai_analyzed_total} total in DB) · '
            f'Click any stock below to analyse it, or run Quick Scan to auto-analyse signal stocks.'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif _ai_analyzed_total > 0:
        st.markdown(
            f'<div style="font-size:12px;color:#94a3b8;margin-bottom:4px;">'
            f'🤖 {_ai_analyzed_in_view} of {len(filtered)} filtered stocks have cached AI scores '
            f'({_ai_analyzed_total} total in DB) · Add an API key in sidebar to run more.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Activity filter info badge
    _high_vol = (filtered["vol_expansion_ratio"].fillna(0) >= 1.2).sum() if "vol_expansion_ratio" in filtered.columns else 0
    _liquid   = (filtered["avg_turnover_cr"].fillna(0) >= 10).sum() if "avg_turnover_cr" in filtered.columns else 0
    st.markdown(
        f'<div style="margin:0 0 8px 0; font-size:12px; color:#94a3b8;">'
        f'🔍 Activity filters active: '
        f'<b>Min turnover ₹{min_turnover:.0f} Cr</b> · '
        f'<b>Min volume {min_avg_volume:,} shares/day</b>'
        + (f' · <b>Vol expansion ≥ {vol_expansion_min:.1f}×</b>' if vol_expansion_min > 0 else '') +
        f' &nbsp;|&nbsp; '
        f'<span style="color:#22c55e;">{_high_vol} stocks with volume surging (≥1.2×)</span> · '
        f'<span style="color:#3b82f6;">{_liquid} stocks with ≥₹10 Cr daily turnover</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Blended score = 70% composite + 30% AI — only for stocks with real AI scores ──
    # Check against the FULL dataset, not just filtered — filters must not hide AI columns
    _has_ai_globally = (
        "ai_score" in df.columns
        and df["ai_score"].notna().any()
    )
    _has_ai = (
        "ai_score" in filtered.columns
        and filtered["ai_score"].notna().any()
    )
    if _has_ai:
        _cs_min   = filtered["composite_score"].min()
        _cs_max   = filtered["composite_score"].max()
        _cs_range = max(_cs_max - _cs_min, 1e-6)
        _norm_cs  = (filtered["composite_score"] - _cs_min) / _cs_range * 10
        # Blended score is NaN ("—") for stocks without an AI score
        _blended  = (0.70 * _norm_cs + 0.30 * filtered["ai_score"]).round(1)
        filtered  = filtered.assign(blended_score=_blended)
    else:
        filtered  = filtered.assign(blended_score=float("nan"))

    # Show AI columns whenever ANY stock in the full DB has been analysed —
    # filters must not make columns disappear just because filtered stocks lack AI scores.
    _ai_display_cols = (
        ["blended_score", "ai_score", "ai_verdict"]
        if _has_ai_globally else []
    )

    display_cols = (
        ["tradingsymbol", "company_name", "ltp"]
        + _ai_display_cols
        + [
            "composite_score", "trend_score", "rs_vs_nifty_3m",
            "ret_1y", "ret_6m", "ret_3m", "ret_1m", "ret_5d",
            "rsi_14", "vol_expansion_ratio",
            "dist_from_52w_high_pct", "dist_from_50ema_pct",
            "avg_turnover_cr",
            "support_20d", "resistance_20d",
        ]
    )
    display_cols = [c for c in display_cols if c in filtered.columns]

    # Pretty number formatting
    formatters = {
        "ltp": "₹{:,.2f}",
        "blended_score":   "{:.1f}",
        "composite_score": "{:.1f}",
        "ai_score":        "{:.1f}",
        "trend_score": "{:.1f}",
        "rs_vs_nifty_3m": "{:+.2f}%",
        "ret_5d": "{:+.2f}%",
        "ret_1m": "{:+.2f}%",
        "ret_3m": "{:+.2f}%",
        "ret_6m": "{:+.2f}%",
        "ret_1y": "{:+.2f}%",
        "rsi_14": "{:.1f}",
        "vol_expansion_ratio": "{:.2f}x",
        "dist_from_52w_high_pct": "{:.1f}%",
        "dist_from_50ema_pct": "{:+.1f}%",
        "avg_turnover_cr": "₹{:.1f} Cr",
        "support_20d": "₹{:,.2f}",
        "resistance_20d": "₹{:,.2f}",
    }

    # Render with conditional coloring on returns
    def color_returns(val):
        if pd.isna(val):
            return ""
        color = "#22c55e" if val > 0 else "#ef4444" if val < 0 else ""
        return f"color: {color}; font-weight: 600"

    def _color_verdict(val):
        return f"color: {_ai.VERDICT_COLOR.get(str(val).upper(), '#94a3b8')}; font-weight: 700"

    def _color_ai_score(val):
        try:
            v = float(val)
            if v >= 8:   return "color: #22c55e; font-weight: 700"
            if v >= 6.5: return "color: #86efac; font-weight: 600"
            if v >= 5:   return "color: #f59e0b; font-weight: 600"
            return "color: #ef4444; font-weight: 600"
        except Exception:
            return ""

    _style_cols  = ["ret_1y", "ret_6m", "ret_3m", "ret_1m", "ret_5d", "rs_vs_nifty_3m"]
    _verdict_col = [c for c in ["ai_verdict"] if c in display_cols]
    _ai_score_col = [c for c in ["ai_score", "blended_score"] if c in display_cols]

    styled = (
        filtered[display_cols]
        .style
        .format(formatters, na_rep="—")
        .map(color_returns, subset=_style_cols)
    )
    if _verdict_col:
        styled = styled.map(_color_verdict, subset=_verdict_col)
    if _ai_score_col:
        styled = styled.map(_color_ai_score, subset=_ai_score_col)

    _screener_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter (e.g. HDFC, RELIANCE)…",
        key="screener_sym_search", label_visibility="collapsed",
    )
    if _screener_sym_q:
        _mask = filtered["tradingsymbol"].str.contains(
            _screener_sym_q.strip(), case=False, na=False, regex=False
        )
        filtered = filtered[_mask]
        styled = (
            filtered[display_cols]
            .style
            .format(formatters, na_rep="—")
            .map(color_returns, subset=_style_cols)
        )
        if _verdict_col:
            styled = styled.map(_color_verdict, subset=_verdict_col)
        if _ai_score_col:
            styled = styled.map(_color_ai_score, subset=_ai_score_col)

    _W = config.TREND_WEIGHTS
    _cc = st.column_config
    st.dataframe(
    styled,
    use_container_width=True,
    height=500,
    hide_index=True,
    column_config={
        "tradingsymbol": _cc.TextColumn(
            "Symbol",
            help="NSE ticker symbol. Click the stock name in the detail panel below to see its full chart.",
        ),
        "company_name": _cc.TextColumn(
            "Company",
            help="Full registered company name.",
        ),
        "ltp": _cc.TextColumn(
            "LTP (₹)",
            help=(
                "Last Traded Price — the closing price from the most recent trading session stored in the DB. "
                "Enable 📡 Live mode in the detail panel for the real-time quote."
            ),
        ),
        "composite_score": _cc.TextColumn(
            "Score",
            help=(
                f"Composite Score = {int(config.W_TREND*100)}% Trend + {int(config.W_RELATIVE_STRENGTH*100)}% RS vs Nifty + {int(config.W_VOLUME_EXPANSION*100)}% Volume expansion.\n\n"
                "Higher is better. Ranks all stocks in the filtered universe so the strongest momentum setups float to the top.\n\n"
                "Use it to prioritise which charts to review first — not as a buy signal on its own."
            ),
        ),
        "trend_score": _cc.TextColumn(
            "Trend",
            help=(
                f"Weighted average of 5 timeframe returns:\n"
                f"  5D × {_W['5D']}%  |  1M × {_W['1M']}%  |  3M × {_W['3M']}%  |  6M × {_W['6M']}%  |  1Y × {_W['1Y']}%\n\n"
                "Each return is capped at ±50% before weighting to prevent one outlier dominating.\n\n"
                "Positive = trending up across multiple timeframes (alignment). Negative = downtrend. "
                "Front-loaded to 5D + 1M because short-swing edge is in recent momentum."
            ),
        ),
        "rs_vs_nifty_3m": _cc.TextColumn(
            "RS vs Nifty",
            help=(
                "Relative Strength vs Nifty 50 over 3 months.\n\n"
                "Formula: stock 3M return − Nifty 3M return.\n\n"
                "Positive = stock is outperforming the index (generating alpha). "
                "Negative = underperforming. "
                "Key criterion: you want to be in stocks that are rising faster than the market, "
                "especially on pullbacks. A stock with strong RS tends to recover faster and go higher."
            ),
        ),
        "ret_1y": _cc.TextColumn(
            "1Y Return",
            help=(
                "1-year % return: (today's close / close 252 trading days ago − 1) × 100.\n\n"
                "Big-picture trend health. A stock with a strong 1Y return is in a long-term uptrend — "
                "the best environment for short swing trades is to trade with the larger trend, not against it."
            ),
        ),
        "ret_6m": _cc.TextColumn(
            "6M Return",
            help=(
                "6-month % return: (today's close / close 126 trading days ago − 1) × 100.\n\n"
                "Medium-term trend. Confirms whether the stock is in a sustained move or just a short spike. "
                "Strong 6M + strong 1M = trend continuity."
            ),
        ),
        "ret_3m": _cc.TextColumn(
            "3M Return",
            help=(
                "3-month % return: (today's close / close 63 trading days ago − 1) × 100.\n\n"
                "Also used as the base for the RS vs Nifty calculation. "
                "Positive 3M in a negative-market environment is a strong alpha signal."
            ),
        ),
        "ret_1m": _cc.TextColumn(
            "1M Return",
            help=(
                "1-month % return: (today's close / close 21 trading days ago − 1) × 100.\n\n"
                "Recent momentum. A 1M pullback into a rising 3M/6M trend is a classic swing entry setup. "
                "Strong 1M alone without 3M/6M support is often just noise."
            ),
        ),
        "ret_5d": _cc.TextColumn(
            "5D Return",
            help=(
                "5-day % return: (today's close / close 5 trading days ago − 1) × 100.\n\n"
                "Most recent short-term momentum. Useful for timing entries on stocks already on the watchlist. "
                "A small negative 5D (mild pullback) inside a strong 1M/3M trend is an ideal entry zone."
            ),
        ),
        "rsi_14": _cc.TextColumn(
            "RSI(14)",
            help=(
                "Relative Strength Index over 14 periods using Wilder's smoothing.\n\n"
                "Scale: 0–100.\n"
                "  < 30  = oversold (possible reversal, but can stay low in downtrends)\n"
                "  30–50 = recovering / weak\n"
                "  50–70 = bullish momentum ← sweet spot for swing entries\n"
                "  > 70  = overbought (avoid chasing; wait for pullback to 50–60)\n\n"
                "RSI alone is not a signal — use it with trend and volume confirmation."
            ),
        ),
        "vol_expansion_ratio": _cc.TextColumn(
            "Vol Expansion",
            help=(
                "5-day average volume ÷ 20-day average volume.\n\n"
                "  > 1.5x = strong volume surge — institutional participation, high conviction move\n"
                "  1.2–1.5x = moderate expansion — worth noting\n"
                "  0.8–1.2x = neutral — no volume signal\n"
                "  < 0.8x = volume drying up — avoid breakouts on low volume\n\n"
                "Rule of thumb: a price breakout on 2x+ volume is far more reliable than the same move on 0.5x volume."
            ),
        ),
        "dist_from_52w_high_pct": _cc.TextColumn(
            "From 52W High",
            help=(
                "% below the 52-week high: (52W_high − LTP) / LTP × 100.\n\n"
                "  0–5%  = at or near 52W high — strong stock, potential breakout zone\n"
                "  5–15% = in the zone of strength — good for swing entries\n"
                "  > 30% = far from highs — avoid unless there's a strong reversal thesis\n\n"
                "Stocks near their 52W high tend to continue higher (momentum effect). "
                "Stocks far below their highs need a strong reason to reverse."
            ),
        ),
        "dist_from_50ema_pct": _cc.TextColumn(
            "vs 50 EMA",
            help=(
                "% distance of LTP from the 50-day Exponential Moving Average.\n\n"
                "  Positive = price is above 50 EMA (bullish structure)\n"
                "  Negative = price is below 50 EMA (bearish / in pullback)\n\n"
                "  −5% to +5% = healthy pullback entry zone — price testing 50 EMA\n"
                "  > +15%    = extended, risk of mean reversion — wait for pullback\n"
                "  < −10%    = broken down — avoid\n\n"
                "The 50 EMA is widely watched by institutional traders, making it a self-fulfilling support level."
            ),
        ),
        "blended_score": _cc.TextColumn(
            "⚡ Blended",
            help=(
                "Blended score (0–10) = 70% mathematical composite + 30% AI (STOCKLENS) score.\n"
                "Only shown for stocks that have an AI analysis — '—' means not yet analysed.\n\n"
                "Click any stock to run STOCKLENS individually, or hit Quick Scan to "
                "auto-analyse signal stocks. Sort by this column for the highest-confidence ideas."
            ),
        ),
        "ai_score": _cc.TextColumn(
            "🤖 AI Score",
            help=(
                "STOCKLENS AI score out of 10 — combines fundamental health, technical structure, "
                "news sentiment, FII/DII flows, and macro context.\n\n"
                "  8–10  = STRONG BUY  · High conviction\n"
                "  6.5–8 = BUY         · Good risk/reward\n"
                "  5–6.5 = WATCHLIST   · Wait for trigger\n"
                "  3–5   = AVOID       · Risk > reward\n"
                "  0–3   = EXIT        · Structural deterioration\n\n"
                "Click a stock below to run STOCKLENS on it, or use Quick Scan to "
                "auto-analyse all signal stocks. Results cached for 7 days."
            ),
        ),
        "ai_verdict": _cc.TextColumn(
            "🤖 Verdict",
            help="STOCKLENS verdict: STRONG BUY / BUY / WATCHLIST / AVOID / EXIT. "
                 "Click a stock row and expand 'STOCKLENS Brief' below for the full analysis. "
                 "Shown only for stocks that have been analysed.",
        ),
        "avg_turnover_cr": _cc.TextColumn(
            "Avg Turnover",
            help=(
                "Average daily turnover in ₹ Crores over the last 20 trading days.\n"
                "Formula: mean(daily_volume × close_price) / 1,00,00,000.\n\n"
                "  ≥ ₹50 Cr  = highly liquid — institutional-grade, tight spreads\n"
                "  ₹10–50 Cr = liquid enough for most retail swing traders\n"
                "  ₹5–10 Cr  = borderline — slippage risk on large orders\n"
                "  < ₹5 Cr   = excluded by the screener's liquidity gate\n\n"
                "Low liquidity = wide bid-ask spread + slippage + hard to exit in a hurry."
            ),
        ),
        "support_20d": _cc.TextColumn(
            "Support (20D)",
            help=(
                "Lowest price (intraday low) over the last 20 trading days.\n\n"
                "Used as a rough near-term support proxy for stop-loss placement. "
                "A close below this level often signals a short-term trend break.\n\n"
                "⚠ Note: this is a screening proxy, not a hand-drawn S/R level. "
                "Always confirm on the chart before using as an actual stop."
            ),
        ),
        "resistance_20d": _cc.TextColumn(
            "Resistance (20D)",
            help=(
                "Highest price (intraday high) over the last 20 trading days.\n\n"
                "Used as a rough near-term resistance proxy for target setting. "
                "A close above this level can signal a breakout.\n\n"
                "⚠ Note: same caveat as Support — confirm on the chart."
            ),
        ),
    },
)

    # Export button
    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export shortlist to CSV", csv,
        file_name=f"screener_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    # ============================================================
    # DETAIL PANEL — click-to-inspect any stock
    # ============================================================
    st.markdown("---")
    st.subheader("🔍 Stock Detail")

    sel_col, live_col = st.columns([4, 1])
    with sel_col:
        selected = st.selectbox(
            "Select a stock to inspect",
            filtered["tradingsymbol"].tolist(),
            index=0 if not filtered.empty else None,
            label_visibility="collapsed",
        )
    with live_col:
        live_mode = st.toggle("📡 Live", value=False,
                              help="Fetch today's real-time quote from Kite")

    if selected:
        stock_row = filtered[filtered["tradingsymbol"] == selected].iloc[0]
        token = int(stock_row["instrument_token"])
        ohlcv = db.load_ohlcv(token)

        # ============================================================
        # LIVE QUOTE — fragment auto-refreshes every 10 s independently
        # ============================================================
        if live_mode:
            _live_quote_fragment(selected)

        # ============================================================
        # METRIC TILES
        # ============================================================
        if ohlcv.empty:
            st.warning("No historical data cached for this stock.")
        else:
            # Use live LTP from session state if available, else fall back to cached
            display_ltp = st.session_state.get(f"live_ltp_{selected}") or stock_row["ltp"]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("LTP", f"₹{display_ltp:,.2f}")
            m2.metric("Composite",
                      f"{stock_row['composite_score']:.1f}" if pd.notna(stock_row["composite_score"]) else "—")
            m3.metric("RSI(14)",
                      f"{stock_row['rsi_14']:.1f}" if pd.notna(stock_row["rsi_14"]) else "—")
            m4.metric("From 52W High",
                      f"-{stock_row['dist_from_52w_high_pct']:.1f}%"
                      if pd.notna(stock_row["dist_from_52w_high_pct"]) else "—")
            m5.metric("Avg Turnover",
                      f"₹{stock_row['avg_turnover_cr']:.1f} Cr"
                      if pd.notna(stock_row["avg_turnover_cr"]) else "—")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("5D",  f"{stock_row['ret_5d']:+.2f}%"  if pd.notna(stock_row["ret_5d"])  else "—")
            m2.metric("1M",  f"{stock_row['ret_1m']:+.2f}%"  if pd.notna(stock_row["ret_1m"])  else "—")
            m3.metric("3M",  f"{stock_row['ret_3m']:+.2f}%"  if pd.notna(stock_row["ret_3m"])  else "—")
            m4.metric("6M",  f"{stock_row['ret_6m']:+.2f}%"  if pd.notna(stock_row["ret_6m"])  else "—")
            m5.metric("1Y",  f"{stock_row['ret_1y']:+.2f}%"  if pd.notna(stock_row["ret_1y"])  else "—")

            # ============================================================
            # CHART DATA — inject live candle when in live mode
            # ============================================================
            chart_df = ohlcv.sort_values("date").tail(400).copy()

            if live_mode and f"live_ltp_{selected}" in st.session_state:
                IST = timezone(timedelta(hours=5, minutes=30))
                now_ist = datetime.now(IST)
                is_market_hours = (
                    now_ist.weekday() < 5
                    and now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
                    <= now_ist
                    <= now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
                )
                today = date_type.today()
                latest_cached = chart_df["date"].max()
                if hasattr(latest_cached, "date"):
                    latest_cached = latest_cached.date()

                if latest_cached < today or is_market_hours:
                    ltp_live   = st.session_state[f"live_ltp_{selected}"]
                    ohlc_today = st.session_state.get(f"live_ohlc_{selected}", {})
                    today_row  = pd.DataFrame([{
                        "date": today, "open": ohlc_today.get("open", ltp_live),
                        "high": ohlc_today.get("high", ltp_live),
                        "low":  ohlc_today.get("low",  ltp_live),
                        "close": ltp_live, "volume": 0,
                    }])
                    chart_df = pd.concat(
                        [chart_df[chart_df["date"] != today], today_row],
                        ignore_index=True,
                    )

            # ============================================================
            # ANALYSIS CONTEXT BAR — stage + signal pills
            # ============================================================
            context_pills = _charts.build_context_bar(chart_df, stock_row)  # always daily
            if context_pills:
                pill_html_parts = []
                for p in context_pills:
                    pill_html_parts.append(
                        f'<span style="'
                        f'display:inline-block;'
                        f'padding:3px 10px;'
                        f'margin:2px 4px;'
                        f'border-radius:12px;'
                        f'border:1px solid {p["color"]};'
                        f'color:{p["color"]};'
                        f'font-size:12px;'
                        f'background:rgba(15,23,42,0.7);'
                        f'">'
                        f'<span style="opacity:0.6;font-size:10px;">{p["label"]}&nbsp;&nbsp;</span>'
                        f'<b>{p["value"]}</b>'
                        f'</span>'
                    )
                st.markdown(
                    '<div style="margin:6px 0 10px 0;">' +
                    "".join(pill_html_parts) +
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ============================================================
            # AI SYNTHESIS CARD — combines AI brief + all mathematical signals
            # ============================================================
            _syn_ai_brief   = stock_row.get("ai_brief")
            _syn_ai_score   = stock_row.get("ai_score")
            _syn_ai_verdict = str(stock_row.get("ai_verdict") or "")
            _syn_ai_conf    = str(stock_row.get("ai_confidence") or "")
            _syn_ai_ts      = stock_row.get("ai_analyzed_at")

            # Build mathematical signal summary from stored metrics
            _syn_sw_sig   = str(stock_row.get("swing_signal") or "")
            _syn_sw_entry = stock_row.get("swing_entry")
            _syn_sw_stop  = stock_row.get("swing_stop")
            _syn_sw_t1    = stock_row.get("swing_t1")
            _syn_sw_t2    = stock_row.get("swing_t2")
            _syn_sw_rr    = stock_row.get("swing_rr")
            _syn_id_sig   = str(stock_row.get("intraday_signal") or "")
            _syn_id_entry = stock_row.get("intraday_entry")
            _syn_id_stop  = stock_row.get("intraday_stop")
            _syn_id_t1    = stock_row.get("intraday_t1")
            _syn_sc_sig   = str(stock_row.get("scale_signal") or "")
            _syn_rsi      = stock_row.get("rsi_14")
            _syn_ltp      = stock_row.get("ltp") or float(chart_df["close"].iloc[-1])
            _syn_ret_1m   = stock_row.get("ret_1m")
            _syn_ret_3m   = stock_row.get("ret_3m")
            _syn_comp     = stock_row.get("composite_score")
            _syn_rs       = stock_row.get("rs_vs_nifty_3m")
            _syn_vol_exp  = stock_row.get("vol_expansion_ratio")
            _syn_dist52   = stock_row.get("dist_from_52w_high_pct")

            # Only show when AI brief is available (otherwise no synthesis possible)
            if _syn_ai_brief and str(_syn_ai_brief).strip():
                # Compute quick Weinstein stage from OHLCV for synthesis
                _syn_c    = chart_df["close"]
                _syn_e50  = _syn_c.ewm(span=50,  adjust=False).mean()
                _syn_e200 = _syn_c.ewm(span=200, adjust=False).mean()
                _syn_ltp_f = float(_syn_c.iloc[-1])
                if len(_syn_c) >= 50:
                    _above_200 = _syn_ltp_f > _syn_e200.iloc[-1]
                    _e50_above = _syn_e50.iloc[-1] > _syn_e200.iloc[-1]
                    _lb = min(30, len(_syn_c) - 1)
                    _slope = (_syn_e200.iloc[-1] - _syn_e200.iloc[-_lb]) / _syn_e200.iloc[-_lb] * 100
                    if _above_200 and _e50_above and _slope > 0.3:
                        _syn_stage = "Stage 2 Advancing"
                    elif not _above_200 and not _e50_above and _slope < -0.3:
                        _syn_stage = "Stage 4 Declining"
                    elif not _above_200 and not _e50_above and abs(_slope) <= 0.3:
                        _syn_stage = "Stage 1 Basing"
                    else:
                        _syn_stage = "Stage 3 Distribution"
                else:
                    _syn_stage = "—"

                # MACD state
                if len(_syn_c) >= 26:
                    _em12 = _syn_c.ewm(span=12, adjust=False).mean()
                    _em26 = _syn_c.ewm(span=26, adjust=False).mean()
                    _mcd  = _em12 - _em26
                    _mcs  = _mcd.ewm(span=9, adjust=False).mean()
                    _hist = _mcd - _mcs
                    _syn_macd = ("positive accelerating" if _hist.iloc[-1] > 0 and _hist.iloc[-1] > _hist.iloc[-2]
                                 else "positive slowing"   if _hist.iloc[-1] > 0
                                 else "negative improving" if _hist.iloc[-1] > _hist.iloc[-2]
                                 else "negative falling")
                else:
                    _syn_macd = "—"

                # Pivot bias
                if len(chart_df) >= 2:
                    _yc = chart_df.iloc[-2]
                    _P  = (float(_yc["high"]) + float(_yc["low"]) + float(_yc["close"])) / 3
                    _syn_piv_bias = "above pivot (bullish)" if _syn_ltp_f > _P else "below pivot (bearish)"
                else:
                    _syn_piv_bias = "—"

                # Assemble math context string for synthesis
                _math_ctx_parts = [
                    f"Weinstein: {_syn_stage}",
                    f"RSI-14: {_syn_rsi:.1f}" if _syn_rsi and not pd.isna(_syn_rsi) else "RSI: —",
                    f"MACD histogram: {_syn_macd}",
                    f"Pivot: {_syn_piv_bias}",
                    f"Composite score: {_syn_comp:.1f}" if _syn_comp and not pd.isna(_syn_comp) else "",
                    f"RS vs Nifty (3M): {_syn_rs:+.1f}%" if _syn_rs and not pd.isna(_syn_rs) else "",
                    f"Vol expansion: {_syn_vol_exp:.2f}×" if _syn_vol_exp and not pd.isna(_syn_vol_exp) else "",
                    f"Dist from 52W High: {_syn_dist52:+.1f}%" if _syn_dist52 and not pd.isna(_syn_dist52) else "",
                    f"1M return: {_syn_ret_1m:+.1f}%" if _syn_ret_1m and not pd.isna(_syn_ret_1m) else "",
                    f"3M return: {_syn_ret_3m:+.1f}%" if _syn_ret_3m and not pd.isna(_syn_ret_3m) else "",
                ]
                if _syn_sw_sig == "BUY":
                    _math_ctx_parts.append(
                        f"Swing BUY: entry ₹{_syn_sw_entry:.2f} / stop ₹{_syn_sw_stop:.2f} / T1 ₹{_syn_sw_t1:.2f}"
                        + (f" / T2 ₹{_syn_sw_t2:.2f}" if _syn_sw_t2 and not pd.isna(_syn_sw_t2) else "")
                        + (f" R/R {_syn_sw_rr:.1f}×" if _syn_sw_rr and not pd.isna(_syn_sw_rr) else "")
                    )
                elif _syn_sw_sig == "SELL":
                    _math_ctx_parts.append(f"Swing SELL signal active: entry ₹{_syn_sw_entry:.2f}")
                if _syn_id_sig == "BUY_ABOVE":
                    _math_ctx_parts.append(f"Intraday BUY ABOVE ₹{_syn_id_entry:.2f} / stop ₹{_syn_id_stop:.2f} / T1 ₹{_syn_id_t1:.2f}")
                elif _syn_id_sig == "SELL_BELOW":
                    _math_ctx_parts.append(f"Intraday SELL BELOW ₹{_syn_id_entry:.2f} / stop ₹{_syn_id_stop:.2f} / T1 ₹{_syn_id_t1:.2f}")
                if _syn_sc_sig == "INITIAL_ENTRY":
                    _math_ctx_parts.append("Scaling signal: INITIAL_ENTRY — start building position")

                _math_ctx = " | ".join(p for p in _math_ctx_parts if p)

                # AI brief snippet (first 600 chars for the synthesis prompt)
                _ai_brief_snip = str(_syn_ai_brief)[:600]

                # Call AI to synthesize — only if client available
                _syn_client   = st.session_state.get("ai_client")
                _syn_provider = st.session_state.get("ai_provider")

                _syn_cache_key = f"synthesis_{selected}_{_syn_ai_ts}"
                if _syn_cache_key not in st.session_state:
                    if _syn_client:
                        try:
                            import ai_analyst as _ai_mod
                            _syn_model = _ai_mod._model_for(_syn_provider)
                            _syn_resp  = _syn_client.chat.completions.create(
                                model=_syn_model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a senior equity analyst. Given quantitative technical signals "
                                            "AND a fundamental/sentiment AI research brief for the same stock, "
                                            "synthesize a single, concise (max 5 bullet points) actionable recommendation. "
                                            "Format: start with a one-line VERDICT in ALL CAPS (STRONG BUY / BUY / HOLD / REDUCE / SELL), "
                                            "then 3-5 bullets each ≤15 words covering: trend alignment, momentum, key levels, risk, and timing. "
                                            "Be direct. No disclaimers. No padding."
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Stock: {selected}\n"
                                            f"Quantitative signals: {_math_ctx}\n"
                                            f"AI research brief: {_ai_brief_snip}"
                                        ),
                                    },
                                ],
                                max_tokens=250,
                                temperature=0.3,
                            )
                            st.session_state[_syn_cache_key] = _syn_resp.choices[0].message.content.strip()
                        except Exception as _syn_e:
                            st.session_state[_syn_cache_key] = None
                    else:
                        # No client — build a rule-based synthesis from math signals alone
                        st.session_state[_syn_cache_key] = "__math_only__"

                _syn_text = st.session_state.get(_syn_cache_key)

                # Render synthesis card
                _vd_col_map = {
                    "STRONG BUY": "#22c55e", "BUY": "#86efac", "HOLD": "#f59e0b",
                    "REDUCE": "#f97316", "SELL": "#ef4444", "AVOID": "#ef4444",
                    "WATCHLIST": "#f59e0b",
                }
                _syn_hdr_col = _vd_col_map.get(_syn_ai_verdict.upper(), "#3b82f6")

                if _syn_text and _syn_text != "__math_only__":
                    # Render AI-synthesised card
                    _syn_lines = [l.strip() for l in _syn_text.strip().split("\n") if l.strip()]
                    _verdict_line = _syn_lines[0] if _syn_lines else ""
                    _bullet_lines = _syn_lines[1:]
                    # Choose border colour based on first word of verdict
                    _first_word = _verdict_line.split()[0].rstrip(":").upper() if _verdict_line else ""
                    _card_col = _vd_col_map.get(_first_word, "#3b82f6")
                    _bullets_html = "".join(
                        f'<li style="margin:4px 0;color:#e2e8f0;font-size:13px;">{bl.lstrip("-•▸● ").lstrip("*")}</li>'
                        for bl in _bullet_lines
                    )
                    _ts_str = ""
                    if _syn_ai_ts:
                        try:
                            _ts_str = pd.to_datetime(_syn_ai_ts).strftime("AI analysed %b %d, %H:%M")
                        except Exception:
                            pass
                    _conf_badge = (
                        f'<span style="background:rgba(34,197,94,0.15);border:1px solid #22c55e;'
                        f'color:#22c55e;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:8px;">'
                        f'{_syn_ai_conf}</span>'
                        if _syn_ai_conf else ""
                    )
                    st.markdown(
                        f'<div style="background:rgba(15,23,42,0.95);border:1px solid {_card_col};'
                        f'border-radius:10px;padding:14px 18px;margin:8px 0 12px 0;">'
                        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;">'
                        f'🤖 STOCKLENS × QUANT SYNTHESIS {_conf_badge}'
                        f'<span style="float:right;font-size:10px;color:#475569;">{_ts_str}</span>'
                        f'</div>'
                        f'<div style="font-size:15px;font-weight:700;color:{_card_col};margin-bottom:8px;">'
                        f'{_verdict_line}'
                        f'</div>'
                        f'<ul style="margin:0;padding-left:18px;">{_bullets_html}</ul>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                elif _syn_text == "__math_only__" or (_syn_ai_brief and not _syn_client):
                    # Math-only synthesis — derive verdict from available signals
                    _mo_signals = []
                    if _syn_sw_sig == "BUY":
                        _mo_signals.append(f"🟢 Swing BUY @ ₹{_syn_sw_entry:.2f} · R/R {_syn_sw_rr:.1f}×")
                    elif _syn_sw_sig == "SELL":
                        _mo_signals.append(f"🔴 Swing SELL signal active")
                    if _syn_id_sig == "BUY_ABOVE":
                        _mo_signals.append(f"🟢 Intraday: BUY ABOVE ₹{_syn_id_entry:.2f}")
                    elif _syn_id_sig == "SELL_BELOW":
                        _mo_signals.append(f"🔴 Intraday: SELL BELOW ₹{_syn_id_entry:.2f}")
                    if _syn_sc_sig == "INITIAL_ENTRY":
                        _mo_signals.append("🏗️ Scaling: start building position")

                    _mo_trend = "🟢 Bullish" if "Advancing" in _syn_stage else "🔴 Bearish" if "Declining" in _syn_stage else "⚪ Transitioning"
                    _mo_rsi_txt = (
                        f"RSI {_syn_rsi:.0f} — overbought, trail stop" if _syn_rsi and _syn_rsi > 70
                        else f"RSI {_syn_rsi:.0f} — momentum zone" if _syn_rsi and 50 < _syn_rsi <= 70
                        else f"RSI {_syn_rsi:.0f} — below 50, caution" if _syn_rsi
                        else "RSI —"
                    )
                    _mo_ai_line = (
                        f"🤖 AI score: {_syn_ai_score:.1f}/10 · Verdict: {_syn_ai_verdict}"
                        if _syn_ai_score and not pd.isna(_syn_ai_score) else ""
                    )
                    _mo_bullets = [b for b in [
                        f"{_mo_trend} · {_syn_stage}",
                        _mo_rsi_txt,
                        _mo_ai_line,
                    ] + _mo_signals if b]
                    _mo_col = "#22c55e" if "BUY" in " ".join([_syn_sw_sig, _syn_id_sig]) else "#f59e0b" if "Advancing" in _syn_stage else "#ef4444"
                    _mo_bullets_html = "".join(
                        f'<li style="margin:4px 0;color:#e2e8f0;font-size:13px;">{b}</li>'
                        for b in _mo_bullets
                    )
                    st.markdown(
                        f'<div style="background:rgba(15,23,42,0.95);border:1px solid {_mo_col};'
                        f'border-radius:10px;padding:14px 18px;margin:8px 0 12px 0;">'
                        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;">📊 QUANT SYNTHESIS · '
                        f'<span style="color:#475569;font-size:10px;">AI brief available — add API key for full synthesis</span>'
                        f'</div>'
                        f'<ul style="margin:0;padding-left:18px;">{_mo_bullets_html}</ul>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ============================================================
            # TIMEFRAME + CANDLE INTERVAL SELECTORS
            # ============================================================
            # Kite Connect interval codes and their max history limits
            _KITE_INTERVAL_MAP = {
                "1m":  ("minute",    60),
                "3m":  ("3minute",  100),
                "5m":  ("5minute",  100),
                "10m": ("10minute", 100),
                "15m": ("15minute", 200),
                "30m": ("30minute", 200),
                "1H":  ("60minute", 400),
                "1D":  (None,       None),   # use stored daily data
                "1W":  (None,       None),   # resample weekly from daily
                "1M":  (None,       None),   # resample monthly from daily
            }
            # Suggested default TF per interval (avoids loading 100k bars)
            _CI_DEFAULT_TF = {
                "1m": "5D", "3m": "5D", "5m": "1M", "10m": "1M",
                "15m": "3M", "30m": "3M", "1H": "6M",
                "1D": "1Y", "1W": "Max", "1M": "Max",
            }
            _TF_DAYS = {
                "5D": 5, "1M": 22, "3M": 63, "6M": 126, "1Y": 252, "Max": len(chart_df),
            }

            _tf_col, _ci_col = st.columns([4, 6])
            with _tf_col:
                selected_tf = st.radio(
                    "Timeframe", list(_TF_DAYS.keys()),
                    index=4,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"tf_{selected}",
                )
            with _ci_col:
                candle_interval = st.radio(
                    "Candle",
                    list(_KITE_INTERVAL_MAP.keys()),
                    index=7,   # default: 1D
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"ci_{selected}",
                )

            # ── Build the DataFrame to plot ───────────────────────────────
            def _resample_daily_to(df: pd.DataFrame, rule: str) -> pd.DataFrame:
                """Resample daily OHLCV data to weekly or monthly candles."""
                df2 = df[["date", "open", "high", "low", "close", "volume"]].copy()
                df2 = df2.assign(date=pd.to_datetime(df2["date"]))
                df2 = df2.set_index("date").sort_index()
                resampled = df2.resample(rule).agg(
                    open=("open", "first"), high=("high", "max"),
                    low=("low", "min"),   close=("close", "last"),
                    volume=("volume", "sum"),
                ).dropna(subset=["open"])
                return resampled.reset_index()

            kite_interval, max_hist_days = _KITE_INTERVAL_MAP[candle_interval]
            _is_intraday = kite_interval is not None

            if _is_intraday:
                # Intraday — fetch from Kite API, cache in session state
                _ci_tf = _TF_DAYS.get(selected_tf, 22)
                fetch_days = min(_ci_tf, max_hist_days)
                _cache_key = f"intraday_{selected}_{candle_interval}_{fetch_days}"
                if _cache_key not in st.session_state:
                    kc = st.session_state.get("kite_client")
                    if kc is None:
                        st.info(
                            f"ℹ️ Intraday {candle_interval} candles require Kite authentication. "
                            "Please connect Kite in the sidebar first."
                        )
                        plot_df = chart_df
                        _is_intraday = False
                    else:
                        with st.spinner(f"Fetching {candle_interval} candles ({fetch_days}d)…"):
                            try:
                                from_dt = datetime.now() - timedelta(days=fetch_days)
                                raw = kc.get_historical(
                                    int(stock_row["instrument_token"]),
                                    from_dt, datetime.now(), kite_interval,
                                )
                                if raw:
                                    _ic_df = pd.DataFrame(raw)
                                    # Strip timezone info — Kite returns IST-aware timestamps;
                                    # keeping tz-aware causes Plotly to convert to UTC which
                                    # shifts x-axis times by +5:30.
                                    _ic_df["date"] = (
                                        pd.to_datetime(_ic_df["date"])
                                        .dt.tz_localize(None)  # already IST, just drop tz
                                    )
                                    st.session_state[_cache_key] = _ic_df
                                    plot_df = _ic_df
                                else:
                                    st.warning(f"No {candle_interval} data returned.")
                                    plot_df = chart_df
                            except Exception as _e:
                                st.warning(f"Could not fetch {candle_interval} data: {_e}")
                                plot_df = chart_df
                else:
                    plot_df = st.session_state[_cache_key]
                    # Show freshness + refresh button
                    _ci_info_col, _ci_btn_col = st.columns([6, 2])
                    with _ci_btn_col:
                        if st.button("↻ Refresh candles", key=f"ci_refresh_{selected}_{candle_interval}"):
                            del st.session_state[_cache_key]
                            st.rerun()
            elif candle_interval == "1W":
                plot_df = _resample_daily_to(chart_df, "W-FRI")
            elif candle_interval == "1M":
                plot_df = _resample_daily_to(chart_df, "ME")
            else:
                plot_df = chart_df  # 1D

            # Validate we have enough bars
            if len(plot_df) < 5:
                st.warning("Not enough candles for the selected interval and timeframe.")
                plot_df = chart_df

            # ── X-axis range (visible window) ────────────────────────────
            # Calendar days per TF label (used for date-arithmetic on intraday)
            _TF_CAL_DAYS = {
                "5D": 7,   # 5 trading days ≈ 7 calendar days
                "1M": 31,
                "3M": 93,
                "6M": 186,
                "1Y": 366,
                "Max": 9999,
            }
            _sorted = plot_df.sort_values("date").reset_index(drop=True)
            _x_end  = _sorted["date"].iloc[-1]

            if _is_intraday:
                # For intraday bars use date arithmetic — NOT candle count.
                # iloc[-5] on a 5m DataFrame = last 25 minutes, not 5 days.
                _cal_days = _TF_CAL_DAYS.get(selected_tf, 7)
                _x_start  = pd.Timestamp(_x_end) - pd.Timedelta(days=_cal_days)
                tf_days   = _cal_days   # used only for display_days param below
            else:
                # Daily/Weekly/Monthly: count candles (works correctly)
                tf_days  = min(_TF_DAYS[selected_tf], len(_sorted))
                _x_start = _sorted["date"].iloc[-tf_days] if tf_days < len(_sorted) else _sorted["date"].iloc[0]

            _x_range = (_x_start, _x_end)

            # Show a note when timeframe might be beyond the API limit
            if _is_intraday and _TF_CAL_DAYS.get(selected_tf, 0) > (max_hist_days or 0):
                st.caption(
                    f"⚠️ {candle_interval} candles are limited to {max_hist_days} days of history by Kite. "
                    f"Showing all available data. Try a shorter timeframe for full coverage."
                )

            # Plotly chart config — scroll to zoom, drag to pan
            _pchart_cfg = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False}

            # ──────────────────────────────────────────────────────────────────
            # PRE-COMPUTE indicators for summary cards (no chart rendering yet)
            # ──────────────────────────────────────────────────────────────────
            def _safe(v, fmt="₹{:,.2f}"):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return "—"
                try:
                    return fmt.format(float(v))
                except Exception:
                    return str(v)

            _summary_df = plot_df.copy()
            _sc = _summary_df["close"]
            _ltp_s  = float(_sc.iloc[-1]) if len(_sc) else float("nan")
            _52w_hi = float(_summary_df["high"].tail(252).max()) if len(_summary_df) >= 20 else float("nan")
            _52w_lo = float(_summary_df["low"].tail(252).min())  if len(_summary_df) >= 20 else float("nan")

            # EMA
            _ema50  = _sc.ewm(span=50,  adjust=False).mean().iloc[-1]  if len(_sc) >= 10 else float("nan")
            _ema200 = _sc.ewm(span=200, adjust=False).mean().iloc[-1]  if len(_sc) >= 20 else float("nan")
            _above200 = _ltp_s > _ema200 if not (pd.isna(_ema200) or pd.isna(_ltp_s)) else None

            # Golden / Death cross within last 30 bars
            def _cross_status(close):
                if len(close) < 30:
                    return "—"
                e50  = close.ewm(span=50,  adjust=False).mean()
                e200 = close.ewm(span=200, adjust=False).mean()
                diff = e50 - e200
                recent = diff.tail(30)
                for i in range(len(recent) - 1, 0, -1):
                    if recent.iloc[i - 1] < 0 and recent.iloc[i] >= 0:
                        return "🟡 Golden Cross (recent)"
                    if recent.iloc[i - 1] > 0 and recent.iloc[i] <= 0:
                        return "💀 Death Cross (recent)"
                return "No cross in past 30 bars"

            _cross = _cross_status(_sc)

            # Weinstein stage
            def _w_stage(close, length):
                if length < 50:
                    return "?", "Insufficient data"
                e50  = close.ewm(span=50, adjust=False).mean()
                e200 = close.ewm(span=200, adjust=False).mean()
                ltp  = close.iloc[-1]
                above_200 = ltp > e200.iloc[-1]
                e50_above = e50.iloc[-1] > e200.iloc[-1]
                lookback  = min(30, length - 1)
                slope_pct = (e200.iloc[-1] - e200.iloc[-lookback]) / e200.iloc[-lookback] * 100
                if above_200 and e50_above and slope_pct > 0.3:
                    return "2", "STAGE 2 — ADVANCING 🟢"
                elif not above_200 and not e50_above and slope_pct < -0.3:
                    return "4", "STAGE 4 — DECLINING 🔴"
                elif not above_200 and not e50_above and abs(slope_pct) <= 0.3:
                    return "1", "STAGE 1 — BASING ⚪"
                else:
                    return "3", "STAGE 3 — DISTRIBUTION 🟡"

            _stage_num, _stage_label = _w_stage(_sc, len(_sc))

            # RSI
            _rsi_val = float(stock_row.get("rsi_14", float("nan")))
            if pd.isna(_rsi_val) and len(_sc) >= 15:
                delta = _sc.diff()
                gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                _rs   = gain / loss.replace(0, float("nan"))
                _rsi_val = float(100 - (100 / (1 + _rs.iloc[-1])))

            def _rsi_label(v):
                if pd.isna(v):
                    return "—", "⚪"
                if v >= 70:
                    return f"{v:.1f} — OVERBOUGHT", "🔴"
                if v >= 55:
                    return f"{v:.1f} — Bullish zone", "🟢"
                if v >= 45:
                    return f"{v:.1f} — Neutral", "⚪"
                if v >= 30:
                    return f"{v:.1f} — Bearish zone", "🟡"
                return f"{v:.1f} — OVERSOLD", "🟢"  # oversold = potential bounce

            _rsi_txt, _rsi_ico = _rsi_label(_rsi_val)

            # Bollinger Bands position
            def _bb_position(close):
                if len(close) < 20:
                    return "—"
                mid  = close.rolling(20).mean()
                std  = close.rolling(20).std()
                upper = (mid + 2 * std).iloc[-1]
                lower = (mid - 2 * std).iloc[-1]
                ltp   = close.iloc[-1]
                pct   = (ltp - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
                if pct >= 0.95:
                    return f"At upper band ({pct*100:.0f}%) — stretched / overbought"
                if pct >= 0.7:
                    return f"Upper half ({pct*100:.0f}%) — bullish range"
                if pct >= 0.3:
                    return f"Mid-band ({pct*100:.0f}%) — consolidating"
                if pct >= 0.05:
                    return f"Lower half ({pct*100:.0f}%) — bearish range"
                return f"At lower band ({pct*100:.0f}%) — oversold / potential bounce"

            _bb_pos = _bb_position(_sc)

            # MACD histogram state
            def _macd_state(close):
                if len(close) < 26:
                    return "—"
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd  = ema12 - ema26
                sig   = macd.ewm(span=9, adjust=False).mean()
                hist  = macd - sig
                cur   = hist.iloc[-1]
                prev  = hist.iloc[-2] if len(hist) >= 2 else cur
                if cur > 0 and cur > prev:
                    return "🟢 Positive & accelerating — best long entry zone"
                if cur > 0 and cur <= prev:
                    return "🟡 Positive but slowing — hold, don't add"
                if cur < 0 and cur > prev:
                    return "🟡 Negative but improving — potential reversal building"
                return "🔴 Negative & accelerating down — avoid longs"

            _macd_state_txt = _macd_state(_sc)

            # Pivot levels (from last daily candle)
            _piv_df   = chart_df.sort_values("date")
            if len(_piv_df) >= 2:
                _yest = _piv_df.iloc[-2]
                _H, _L, _C = float(_yest["high"]), float(_yest["low"]), float(_yest["close"])
                _P  = (_H + _L + _C) / 3
                _R1 = 2 * _P - _L
                _R2 = _P + (_H - _L)
                _S1 = 2 * _P - _H
                _S2 = _P - (_H - _L)
                _atr14 = float(
                    _piv_df["close"].diff().abs().tail(14).mean()
                )  # simplified ATR
                _piv_bias = "🟢 BULLISH (price above pivot)" if _ltp_s > _P else "🔴 BEARISH (price below pivot)"
                _closest_r = "R1" if abs(_ltp_s - _R1) < abs(_ltp_s - _R2) else "R2"
                _closest_s = "S1" if abs(_ltp_s - _S1) < abs(_ltp_s - _S2) else "S2"
            else:
                _P = _R1 = _R2 = _S1 = _S2 = float("nan")
                _piv_bias = "—"
                _closest_r = _closest_s = "—"
                _atr14 = float("nan")

            # S/R zones near LTP
            def _nearest_zones(df, ltp):
                if len(df) < 10 or pd.isna(ltp):
                    return [], []
                data = df.tail(120).reset_index(drop=True)
                raw = []
                for i in range(2, len(data) - 2):
                    h = float(data["high"].iloc[i])
                    if h >= data["high"].iloc[max(0,i-2):i].max() and h >= data["high"].iloc[i+1:i+3].max():
                        raw.append(h)
                    l = float(data["low"].iloc[i])
                    if l <= data["low"].iloc[max(0,i-2):i].min() and l <= data["low"].iloc[i+1:i+3].min():
                        raw.append(l)
                if not raw:
                    return [], []
                supports    = sorted([p for p in raw if p < ltp], reverse=True)[:3]
                resistances = sorted([p for p in raw if p > ltp])[:3]
                return supports, resistances

            _sup_levels, _res_levels = _nearest_zones(plot_df, _ltp_s)

            # Volume profile POC (simplified: price bin with highest cumulative volume)
            def _vol_poc(df):
                if len(df) < 20 or df["volume"].sum() == 0:
                    return float("nan")
                data = df.tail(120)
                prices = (data["high"] + data["low"]) / 2
                lo, hi = prices.min(), prices.max()
                if hi <= lo:
                    return float("nan")
                bins = pd.cut(prices, bins=30)
                vol_by_bin = data.groupby(bins, observed=True)["volume"].sum()
                if vol_by_bin.empty:
                    return float("nan")
                poc_interval = vol_by_bin.idxmax()
                return (poc_interval.left + poc_interval.right) / 2

            _poc = _vol_poc(plot_df)
            _poc_txt = (
                f"₹{_poc:,.2f} ({'above LTP — acts as resistance' if _poc > _ltp_s else 'below LTP — acts as support'})"
                if not pd.isna(_poc) else "—"
            )

            # ============================================================
            # STOCK ANALYSIS CONSOLE — 4 purpose-built chart tabs
            # ============================================================
            tab_trend, tab_mom, tab_setup, tab_struct = st.tabs([
                "📈 Trend Canvas",
                "⚡ Momentum Lab",
                "🎯 Trade Setup",
                "🏗️ Market Structure",
            ])

            with tab_trend:
                # ── Summary card ─────────────────────────────────────────
                _trend_verdict_map = {
                    "2": ("🟢", "#22c55e", "Bullish — best time to buy swings"),
                    "4": ("🔴", "#ef4444", "Bearish — avoid longs, short setups viable"),
                    "1": ("⚪", "#94a3b8", "Basing — wait for Stage 2 breakout"),
                    "3": ("🟡", "#f59e0b", "Topping — reduce exposure, tighten stops"),
                    "?": ("⚪", "#94a3b8", "Insufficient data"),
                }
                _tv_ico, _tv_col, _tv_action = _trend_verdict_map.get(_stage_num, ("⚪", "#94a3b8", "—"))
                _dist_hi_pct = (((_ltp_s - _52w_hi) / _52w_hi) * 100) if not pd.isna(_52w_hi) and _52w_hi else float("nan")
                _dist_lo_pct = (((_ltp_s - _52w_lo) / _52w_lo) * 100) if not pd.isna(_52w_lo) and _52w_lo else float("nan")

                st.markdown(
                    f'<div style="background:rgba(30,41,59,0.9);border:1px solid {_tv_col};'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
                    f'<div style="font-size:16px;font-weight:700;color:{_tv_col};margin-bottom:10px;">'
                    f'{_tv_ico} {_stage_label} &nbsp;·&nbsp; <span style="font-size:13px;font-weight:400;color:#e2e8f0;">{_tv_action}</span>'
                    f'</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-size:13px;color:#e2e8f0;">'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;width:40%;">LTP</td>'
                    f'<td><b>{_safe(_ltp_s)}</b></td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">EMA 50</td>'
                    f'<td><b>{_safe(_ema50)}</b> &nbsp;{"🟢 Price above" if not pd.isna(_ema50) and _ltp_s > _ema50 else "🔴 Price below"}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">EMA 200</td>'
                    f'<td><b>{_safe(_ema200)}</b> &nbsp;{"🟢 Price above — long bias" if _above200 else "🔴 Price below — caution"}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">Cross signal</td>'
                    f'<td>{_cross}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">52W High</td>'
                    f'<td>{_safe(_52w_hi)} &nbsp;({_dist_hi_pct:+.1f}% from LTP)</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">52W Low</td>'
                    f'<td>{_safe(_52w_lo)} &nbsp;({_dist_lo_pct:+.1f}% from LTP)</td></tr>'
                    f'</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📊 Open Trend Canvas chart", expanded=False):
                    fig1 = _charts.chart_trend_canvas(plot_df, stock_row, x_range=_x_range, candle_label=candle_interval)
                    st.plotly_chart(fig1, use_container_width=True, config=_pchart_cfg)

                with st.expander("📖 How to interpret — Trend Canvas"):
                    st.markdown(f"""
**What this chart answers:** *"Which market phase is this stock in — bull, bear, or transitioning?"*

**Current reading for {selected}:**
- Weinstein stage: **{_stage_label}** → _{_tv_action}_
- EMA 50 is `{_safe(_ema50)}` — LTP is **{'above' if not pd.isna(_ema50) and _ltp_s > _ema50 else 'below'}**
- EMA 200 is `{_safe(_ema200)}` — LTP is **{'above → long-only stance' if _above200 else 'below → avoid fresh longs'}**
- Cross signal: **{_cross}**

| Element | What it means | How to act |
|---|---|---|
| **Stage 2 — Advancing** 🟢 | Price > EMA 200, EMA 50 rising. The profitable zone | Best time to buy swings |
| **Stage 4 — Declining** 🔴 | Price < EMA 200, EMA 50 falling | Avoid longs. Short setups viable |
| **Stage 1 — Basing** ⚪ | EMA 200 flat, price oscillating | Wait — accumulation happening |
| **Stage 3 — Distribution** 🟡 | Near highs, EMA 50 rolling over | Reduce exposure, tighten stops |
| **🟡 Golden Cross** | EMA 50 crossed above EMA 200 | Biggest gains happen after a Golden Cross |
| **💀 Death Cross** | EMA 50 crossed below EMA 200 | Exit longs immediately |
| **52W High** (dotted) | Annual ceiling — psychological resistance | Breaking above on high volume = very bullish |
""")

            with tab_mom:
                # ── Summary card ─────────────────────────────────────────
                _mom_overall = (
                    "🟢 Strong momentum — enter or hold longs" if not pd.isna(_rsi_val) and 45 <= _rsi_val <= 65 and "accelerating" in _macd_state_txt
                    else "🔴 Overextended — book partial profits" if not pd.isna(_rsi_val) and _rsi_val > 70
                    else "🟡 Momentum mixed — watch for confirmation" if not pd.isna(_rsi_val) and _rsi_val >= 40
                    else "🔴 Weak momentum — avoid longs"
                )
                st.markdown(
                    f'<div style="background:rgba(30,41,59,0.9);border:1px solid #3b82f6;'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
                    f'<div style="font-size:15px;font-weight:700;color:#3b82f6;margin-bottom:10px;">'
                    f'⚡ Momentum Overview &nbsp;·&nbsp; <span style="font-size:13px;font-weight:400;color:#e2e8f0;">{_mom_overall}</span>'
                    f'</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-size:13px;color:#e2e8f0;">'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;width:40%;">RSI-14</td>'
                    f'<td>{_rsi_ico} <b>{_rsi_txt}</b></td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">Bollinger position</td>'
                    f'<td>{_bb_pos}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">MACD histogram</td>'
                    f'<td>{_macd_state_txt}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">5D return</td>'
                    f'<td>{_safe(stock_row.get("ret_5d"), "{:+.2f}%")}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">1M return</td>'
                    f'<td>{_safe(stock_row.get("ret_1m"), "{:+.2f}%")}</td></tr>'
                    f'</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📊 Open Momentum Lab chart", expanded=False):
                    fig2 = _charts.chart_momentum_lab(plot_df, stock_row, x_range=_x_range, candle_label=candle_interval)
                    st.plotly_chart(fig2, use_container_width=True, config=_pchart_cfg)

                with st.expander("📖 How to interpret — Momentum Lab"):
                    st.markdown(f"""
**What this chart answers:** *"Is momentum accelerating into the trade, or exhausting?"*

**Current reading for {selected}:**
- RSI-14: **{_rsi_txt}** — {"book partial profits" if not pd.isna(_rsi_val) and _rsi_val > 70 else "add on pullbacks" if not pd.isna(_rsi_val) and 50 < _rsi_val < 70 else "wait for recovery" if not pd.isna(_rsi_val) and _rsi_val < 50 else "—"}
- Bollinger: **{_bb_pos}**
- MACD: **{_macd_state_txt}**

| RSI reading | Meaning | Action |
|---|---|---|
| > 70 | Overbought | Book 50% profits, trail stop on rest |
| 55–70 | Bullish momentum | Hold longs or add on pullbacks |
| 45–55 | Neutral | Wait for direction |
| 30–45 | Bearish momentum | Avoid new longs |
| < 30 | Oversold | Watch for reversal with divergence |

**Rule of thumb:** Best entries when RSI 45–65 + MACD histogram bright green.
""")

            with tab_setup:
                # Trade Setup always uses DAILY candles
                _setup_days = 40
                _daily_setup = chart_df.sort_values("date").tail(_setup_days).copy()
                _su_end   = pd.to_datetime(_daily_setup["date"].iloc[-1])
                _su_start = pd.to_datetime(_daily_setup["date"].iloc[0])
                _setup_x_range = (_su_start, _su_end)

                # Intraday signal values
                _id_sig   = stock_row.get("intraday_signal", "")
                _id_entry = stock_row.get("intraday_entry")
                _id_stop  = stock_row.get("intraday_stop")
                _id_t1    = stock_row.get("intraday_t1")
                _sw_sig   = stock_row.get("swing_signal", "")
                _sw_entry = stock_row.get("swing_entry")
                _sw_stop  = stock_row.get("swing_stop")
                _sw_t1    = stock_row.get("swing_t1")
                _sw_t2    = stock_row.get("swing_t2")

                _id_signal_line = (
                    f'🟢 <b>BUY ABOVE {_safe(_id_entry)}</b> · Stop {_safe(_id_stop)} · Target {_safe(_id_t1)}'
                    if _id_sig == "BUY_ABOVE"
                    else f'🔴 <b>SELL BELOW {_safe(_id_entry)}</b> · Stop {_safe(_id_stop)} · Target {_safe(_id_t1)}'
                    if _id_sig == "SELL_BELOW"
                    else "No intraday signal active"
                )
                _sw_signal_line = (
                    f'🟢 <b>Swing BUY @ {_safe(_sw_entry)}</b> · Stop {_safe(_sw_stop)} · T1 {_safe(_sw_t1)} · T2 {_safe(_sw_t2)}'
                    if _sw_sig == "BUY"
                    else f'🔴 <b>Swing SELL @ {_safe(_sw_entry)}</b> · Stop {_safe(_sw_stop)} · T1 {_safe(_sw_t1)}'
                    if _sw_sig == "SELL"
                    else "No swing signal active"
                )

                st.markdown(
                    f'<div style="background:rgba(30,41,59,0.9);border:1px solid #f59e0b;'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
                    f'<div style="font-size:15px;font-weight:700;color:#f59e0b;margin-bottom:10px;">'
                    f'🎯 Today\'s Trade Levels &nbsp;·&nbsp; <span style="font-size:13px;font-weight:400;color:#e2e8f0;">{_piv_bias}</span>'
                    f'</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-size:13px;color:#e2e8f0;">'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;width:40%;">Intraday signal</td>'
                    f'<td>{_id_signal_line}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">Swing signal</td>'
                    f'<td>{_sw_signal_line}</td></tr>'
                    f'<tr><td style="padding:4px 12px 4px 0;color:#94a3b8;">Pivot (P)</td>'
                    f'<td>{_safe(_P)}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#22c55e;">R1 / R2</td>'
                    f'<td>{_safe(_R1)} &nbsp;/&nbsp; {_safe(_R2)}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#ef4444;">S1 / S2</td>'
                    f'<td>{_safe(_S1)} &nbsp;/&nbsp; {_safe(_S2)}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">Nearest resistance</td>'
                    f'<td>{_closest_r}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;">Nearest support</td>'
                    f'<td>{_closest_s}</td></tr>'
                    f'</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if candle_interval not in ("1D", "1W", "1M"):
                    st.caption("ℹ️ Trade Setup always uses daily candles — pivot levels are based on yesterday's High/Low/Close.")

                with st.expander("📊 Open Trade Setup chart", expanded=False):
                    fig3 = _charts.chart_trade_setup(
                        _daily_setup, stock_row,
                        x_range=_setup_x_range,
                        display_days=_setup_days,
                        candle_label="1D",
                    )
                    st.plotly_chart(fig3, use_container_width=True, config=_pchart_cfg)

                with st.expander("📖 How to interpret — Trade Setup"):
                    st.markdown(f"""
**What this chart answers:** *"Where exactly do I place my order, stop-loss, and target today?"*

**Current levels for {selected}:**
- Pivot bias: **{_piv_bias}**
- Intraday: {_id_signal_line}
- Swing: {_sw_signal_line}
- Pivot P = {_safe(_P)} · R1 = {_safe(_R1)} · R2 = {_safe(_R2)} · S1 = {_safe(_S1)} · S2 = {_safe(_S2)}

| Level | Use |
|---|---|
| **R2** | Take 50% profits at R2 for intraday |
| **R1** | BUY ABOVE R1 = momentum entry; becomes support after breakout |
| **Pivot P** | Above P = bullish bias. Below = bearish bias |
| **S1** | SELL BELOW S1 = breakdown; bounce zone for longs |
| **S2** | Short target; strong buy-the-dip for longer-term holders |

**Key rule: HARD EXIT all intraday positions by 3:10 PM.**
""")

            with tab_struct:
                # ── Summary card ─────────────────────────────────────────
                _sup_txt = ", ".join(f"₹{p:,.2f}" for p in _sup_levels) if _sup_levels else "none detected"
                _res_txt = ", ".join(f"₹{p:,.2f}" for p in _res_levels) if _res_levels else "none detected"
                _struct_verdict = (
                    "🟢 Price near support — potential bounce zone"
                    if _sup_levels and abs(_ltp_s - _sup_levels[0]) / _ltp_s < 0.02
                    else "🔴 Price near resistance — potential rejection zone"
                    if _res_levels and abs(_ltp_s - _res_levels[0]) / _ltp_s < 0.02
                    else "⚪ Price in open range between zones"
                )

                st.markdown(
                    f'<div style="background:rgba(30,41,59,0.9);border:1px solid #8b5cf6;'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:10px;">'
                    f'<div style="font-size:15px;font-weight:700;color:#8b5cf6;margin-bottom:10px;">'
                    f'🏗️ Market Structure &nbsp;·&nbsp; <span style="font-size:13px;font-weight:400;color:#e2e8f0;">{_struct_verdict}</span>'
                    f'</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-size:13px;color:#e2e8f0;">'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#94a3b8;width:40%;">LTP</td>'
                    f'<td><b>{_safe(_ltp_s)}</b></td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#22c55e;">Support zones (below)</td>'
                    f'<td>{_sup_txt}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#ef4444;">Resistance zones (above)</td>'
                    f'<td>{_res_txt}</td></tr>'
                    f'<tr><td style="padding:3px 12px 3px 0;color:#f59e0b;">Volume POC (6M)</td>'
                    f'<td>{_poc_txt}</td></tr>'
                    f'</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📊 Open Market Structure chart", expanded=False):
                    fig4 = _charts.chart_market_structure(plot_df, stock_row, x_range=_x_range, candle_label=candle_interval)
                    st.plotly_chart(fig4, use_container_width=True, config=_pchart_cfg)

                with st.expander("📖 How to interpret — Market Structure"):
                    st.markdown(f"""
**What this chart answers:** *"Where are the real walls of supply and demand?"*

**Current reading for {selected}:**
- Support zones (below LTP): **{_sup_txt}**
- Resistance zones (above LTP): **{_res_txt}**
- Volume POC: **{_poc_txt}**
- Verdict: **{_struct_verdict}**

| Element | Meaning | Action |
|---|---|---|
| **Green bands** | Price bounced here multiple times | Buy zone on dip into green band |
| **Red bands** | Price rejected here multiple times | Take profits when price enters red band |
| **POC** (amber line) | Price where most volume was traded | Strongest magnet — price returns here often |
| **Thick zone** | More touches = stronger level | 4+ touch zones are highly reliable |

**Best setup:** Support zone + high volume POC at the same price = very strong buy zone.
""")

    # ── STOCKLENS AI Brief for selected stock ───────────────────────────────
    _ai_brief_val  = stock_row.get("ai_brief")
    _ai_score_val  = stock_row.get("ai_score")
    _ai_verdict_val = stock_row.get("ai_verdict", "")
    _ai_conf_val   = stock_row.get("ai_confidence", "")
    _ai_ts_val     = stock_row.get("ai_analyzed_at")
    _has_brief     = bool(_ai_brief_val) and str(_ai_brief_val) not in ("", "nan", "None")

    with st.expander(
        ("🤖 STOCKLENS Brief"
         + (f" · {_ai.VERDICT_EMOJI.get(str(_ai_verdict_val).upper(),'—')} {_ai_verdict_val} "
            f"({_ai_score_val:.1f}/10)" if _has_brief and _ai_score_val else "")
        ),
        expanded=False,
    ):
        if _has_brief:
            # Freshness badge
            if _ai_ts_val:
                try:
                    _age_h = (datetime.now() - datetime.fromisoformat(str(_ai_ts_val))).total_seconds() / 3600
                    _fresh = f"Analysed {_age_h:.0f}h ago"
                except Exception:
                    _fresh = ""
            else:
                _fresh = ""

            _vc = _ai.VERDICT_COLOR.get(str(_ai_verdict_val).upper(), "#64748b")
            st.markdown(
                f'<div style="display:flex;gap:12px;margin-bottom:10px;align-items:center;">'
                f'<span style="background:rgba(15,23,42,0.8);border:1px solid {_vc};'
                f'color:{_vc};font-weight:700;padding:4px 14px;border-radius:20px;font-size:13px;">'
                f'{_ai.VERDICT_EMOJI.get(str(_ai_verdict_val).upper(),"—")} {_ai_verdict_val}</span>'
                f'<span style="color:#94a3b8;font-size:12px;">AI Score: {_ai_score_val:.1f}/10 · '
                f'Confidence: {_ai_conf_val} · {_fresh}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
                f'padding:16px;font-family:monospace;font-size:12px;color:#e2e8f0;'
                f'white-space:pre-wrap;line-height:1.6;">{_ai_brief_val}</div>',
                unsafe_allow_html=True,
            )
        else:
            _detail_ai_c = st.session_state.get("ai_client")
            if _detail_ai_c:
                st.markdown(
                    '<div style="background:rgba(59,130,246,0.08);border:1px solid #3b82f6;'
                    'border-radius:8px;padding:12px 16px;margin-bottom:4px;">'
                    '🤖 <b>No STOCKLENS brief yet</b> for this stock. '
                    'Click below to run a full analysis (~1 API call).'
                    '</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"🤖 Analyse {selected} with STOCKLENS",
                    key=f"ai_single_{selected}",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running STOCKLENS on {selected} (fundamental + sentiment + macro)…"):
                        _res = _ai.run_stocklens(
                            selected, stock_row.to_dict(),
                            _detail_ai_c,
                            st.session_state["ai_provider"],
                        )
                        if _res.get("error"):
                            st.error(f"AI error: {_res['error']}")
                        else:
                            _token = int(stock_row["instrument_token"])
                            db.save_ai_result(_token, _res)
                            st.rerun()
            else:
                st.info(
                    "No AI analysis yet. Add an OpenAI or OpenRouter API key in the sidebar "
                    "to enable STOCKLENS fundamentals + sentiment research for this stock."
                )

    with st.expander("ℹ️ How the composite score works"):
        st.markdown(f"""
        **Composite Score** = `{config.W_TREND}` × Trend + `{config.W_RELATIVE_STRENGTH}` × RS + `{config.W_VOLUME_EXPANSION}` × Vol

        **Trend Score** weights (calibrated for 1-3 day swing):
        - 5D: {config.TREND_WEIGHTS['5D']}  |  1M: {config.TREND_WEIGHTS['1M']}  |  3M: {config.TREND_WEIGHTS['3M']}  |  6M: {config.TREND_WEIGHTS['6M']}  |  1Y: {config.TREND_WEIGHTS['1Y']}

        **Liquidity gate:** ≥ ₹{config.MIN_AVG_TURNOVER_CR} Cr daily turnover, ≥ {config.MIN_AVG_VOLUME:,} avg volume, price ≥ ₹{config.MIN_PRICE}
        """)


# ─── SIGNALS TAB ────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))


def _is_market_open() -> bool:
    """True only during NSE cash market hours (9:15–15:30 IST, Mon–Fri)."""
    now = datetime.now(_IST)
    if now.weekday() >= 5:           # Saturday / Sunday
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


# ── Shared styling helpers (used by signal tabs and intraday fragments) ───────
def _conf_color(val):
    """Color-code the Conf column by tier: green=STRONG, yellow=MODERATE, orange=MARGINAL, red=LOW."""
    v = str(val)
    try:
        n = int(v.split("/")[0])
    except Exception:
        return "color: #64748b"
    if n >= 8:
        return "color: #22c55e; font-weight: 700"
    if n >= 6:
        return "color: #f59e0b; font-weight: 600"
    if n >= 5:
        return "color: #fb923c; font-weight: 600"
    return "color: #ef4444"


def _long_status_color(val):
    return {
        "TRIGGERED":   "color: #22c55e; font-weight: 700",
        "APPROACHING": "color: #f59e0b; font-weight: 600",
        "BROKEN":      "color: #ef4444; font-weight: 600",
    }.get(val, "color: #94a3b8")


def _short_status_color(val):
    return {
        "TRIGGERED":   "color: #ef4444; font-weight: 700",
        "APPROACHING": "color: #f59e0b; font-weight: 600",
        "REVERSED":    "color: #22c55e; font-weight: 600",
    }.get(val, "color: #94a3b8")


def _delta_color(val):
    """Green for ▲ price rise, red for ▼ price fall."""
    v = str(val)
    if v.startswith("▲"):
        return "color: #22c55e; font-weight: 600"
    if v.startswith("▼"):
        return "color: #ef4444; font-weight: 600"
    return "color: #64748b"


def _gain_color(val):
    """Positive gain shown in green."""
    v = str(val)
    if v.startswith("+"):
        return "color: #22c55e"
    return "color: #94a3b8"


def _risk_color(val):
    """Risk always shown in amber so it stands out."""
    v = str(val)
    if v != "—":
        return "color: #f59e0b"
    return "color: #94a3b8"


def _gain_pct(entry, target) -> str:
    """Return '+X.X%' upside from entry → target, or '—'. Returns '—' if target ≤ entry."""
    try:
        e, t = float(entry or 0), float(target or 0)
        if e > 0 and t > e:
            return f"+{(t - e) / e * 100:.1f}%"
    except Exception:
        pass
    return "—"


def _risk_pct(entry, stop) -> str:
    """Return '-X.X%' downside from entry → stop, or '—'."""
    try:
        e, s = float(entry or 0), float(stop or 0)
        if e > 0 and s > 0:
            return f"-{abs(e - s) / e * 100:.1f}%"
    except Exception:
        pass
    return "—"


def _delta_str(sym: str) -> str:
    """LTP change since last fragment tick, formatted as ▲/▼ X.XX%."""
    cur  = st.session_state.get("_live_ltp",  {}).get(sym, 0)
    prev = st.session_state.get("_prev_ltp",  {}).get(sym, cur)
    if not cur or not prev or prev == cur:
        return "—"
    chg = (cur - prev) / prev * 100
    return f"▲{chg:.2f}%" if chg > 0 else f"▼{abs(chg):.2f}%"


# ── FRAGMENT 1: live header (badge + LTP fetch + metric pills) ────────────────
# Runs every 2 s. Does NOT contain any st.tabs() — tabs must live outside
# fragments to keep their selection state stable across re-renders.
@st.fragment(run_every=2)
def _live_signals_header():
    # ── Web Worker keepalive ─────────────────────────────────────────────────
    # st_autorefresh uses a Web Worker (background thread) that is NOT throttled
    # by browser tab visibility policies — unlike fragment run_every which uses
    # plain setInterval and pauses when the tab is hidden.
    # Placing it here means it fires even when the user is on another screen.
    _st_autorefresh(interval=2000, limit=None, key="_ltp_ar", debounce=True)

    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty or "swing_signal" not in base_df.columns:
        return

    _market_open = _is_market_open()
    # Market closed — render static header once, skip LTP fetch entirely
    if not _market_open:
        _hc1, _hc2 = st.columns([5, 2])
        _hc1.subheader("🎯 Trade Signals")
        _hc2.markdown(
            "<div style='text-align:right;padding-top:8px'>"
            "<span style='color:#64748b'>⏸ Market Closed</span></div>",
            unsafe_allow_html=True,
        )
        _sig_age = ""
        if "last_updated" in base_df.columns:
            try:
                _lu = pd.to_datetime(base_df["last_updated"]).max()
                if pd.notna(_lu):
                    _lu_ist = _lu.tz_localize("Asia/Kolkata") if _lu.tzinfo is None else _lu.astimezone(_IST)
                    _sig_age = f" · Signals: {_lu_ist.strftime('%d %b %H:%M')}"
            except Exception:
                pass
        st.caption(
            f"Auto-refresh paused outside market hours (9:15–15:30 IST, Mon–Fri).{_sig_age} "
            "Run Quick Scan to recompute signals."
        )
        df_c = base_df.copy()
        _n_sb = int((df_c["swing_signal"]    == "BUY").sum())
        _n_ss = int((df_c["swing_signal"]    == "SELL").sum())
        _n_il = int((df_c["intraday_signal"] == "BUY_ABOVE").sum())
        _n_is = int((df_c["intraday_signal"] == "SELL_BELOW").sum())
        _n_sc = int((df_c["scale_signal"]    == "INITIAL_ENTRY").sum())
        st.session_state["_n_intra_long"]  = _n_il
        st.session_state["_n_intra_short"] = _n_is
        _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns(5)
        _pc1.metric("Swing Buy",       _n_sb)
        _pc2.metric("Exit signals",    _n_ss)
        _pc3.metric("Intraday Long",   _n_il)
        _pc4.metric("Intraday Short",  _n_is)
        _pc5.metric("Scaling entries", _n_sc)
        return   # ← stop here; no live fetch, no further work
    _has_signal_mask = (
        base_df["swing_signal"].isin(["BUY", "SELL"]) |
        base_df["intraday_signal"].isin(["BUY_ABOVE", "SELL_BELOW"]) |
        (base_df["scale_signal"] == "INITIAL_ENTRY")
    )
    _signal_syms = base_df.loc[_has_signal_mask, "tradingsymbol"].dropna().unique().tolist()

    _ltp_err = None
    if _market_open and _signal_syms:
        try:
            _fc = st.session_state.get("kite_client") or KiteClient(
                api_key=st.session_state.get("kite_api_key", ""),
                api_secret=st.session_state.get("kite_api_secret", ""),
                access_token=st.session_state.get("kite_access_token", ""),
            )
            if _fc.authenticated:
                fresh = _fc.get_ltp_batch([f"NSE:{s}" for s in _signal_syms])
                # Snapshot current → previous BEFORE overwriting with new prices
                if "_live_ltp" in st.session_state:
                    st.session_state["_prev_ltp"] = dict(st.session_state["_live_ltp"])
                st.session_state["_live_ltp"] = {
                    k.split(":", 1)[-1]: v for k, v in fresh.items()
                }
                st.session_state["_live_ltp_ts"] = datetime.now(_IST)

                # ── Nifty 50 intraday direction gate ─────────────────────────
                # Fetch Nifty 50 LTP separately and compute today's % change vs
                # the previous close stored in computed_metrics.
                try:
                    _nf_raw = _fc.get_ltp_batch(["NSE:NIFTY 50"])
                    _nifty_ltp = _nf_raw.get("NSE:NIFTY 50") or _nf_raw.get("NIFTY 50")
                    if _nifty_ltp:
                        _nifty_ltp_f = float(_nifty_ltp)
                        st.session_state["_nifty_live_ltp"] = _nifty_ltp_f
                        _nf_prev = st.session_state.get("_nifty_prev_close")
                        if not _nf_prev or _nf_prev <= 0:
                            # Bootstrap: use OHLC to get today's open as prev-close proxy
                            try:
                                _nf_ohlc = _fc.get_today_open(["NSE:NIFTY 50"])
                                _nf_open  = _nf_ohlc.get("NSE:NIFTY 50") or _nf_ohlc.get("NIFTY 50")
                                if _nf_open and _nf_open > 0:
                                    st.session_state["_nifty_prev_close"] = float(_nf_open)
                                    _nf_prev = float(_nf_open)
                            except Exception:
                                pass
                        if _nf_prev and _nf_prev > 0:
                            _nifty_pct = (_nifty_ltp_f - _nf_prev) / _nf_prev * 100
                            st.session_state["_nifty_intraday_pct"] = round(_nifty_pct, 3)
                except Exception:
                    pass
        except Exception as exc:
            _ltp_err = str(exc)[:80]

    _hc1, _hc2 = st.columns([5, 2])
    _hc1.subheader("🎯 Trade Signals")
    _ts = st.session_state.get("_live_ltp_ts")
    if _market_open:
        if _ts:
            _hc2.markdown(
                f"<div style='text-align:right;padding-top:8px'>"
                f"<span style='color:#22c55e;font-weight:700'>● LIVE</span>"
                f"&nbsp;&nbsp;<small style='color:#94a3b8'>{_ts.strftime('%H:%M:%S')} IST</small></div>",
                unsafe_allow_html=True,
            )
        else:
            _hc2.markdown(
                "<div style='text-align:right;padding-top:8px'>"
                "<span style='color:#f59e0b'>● Market Open — awaiting auth</span></div>",
                unsafe_allow_html=True,
            )
    else:
        _hc2.markdown(
            "<div style='text-align:right;padding-top:8px'>"
            "<span style='color:#64748b'>⏸ Market Closed — last data shown</span></div>",
            unsafe_allow_html=True,
        )
    if _ltp_err:
        _is_token_err = any(k in _ltp_err.lower() for k in
                            ("invalid", "token", "access", "403", "unauthori", "expired"))
        if _is_token_err:
            _reauth_kc = KiteClient(
                api_key=st.session_state.get("kite_api_key", ""),
                api_secret=st.session_state.get("kite_api_secret", ""),
            )
            _reauth_url2 = _reauth_kc.get_login_url()
            _tc1, _tc2 = st.columns([5, 2])
            _tc1.error(
                "🔑 **Kite access token expired.** Live prices & auto-trade paused. "
                "Re-authenticate to resume — all open paper trades are safe in the database."
            )
            if _reauth_url2:
                _tc2.link_button("🔑 Re-authenticate Kite", _reauth_url2, use_container_width=True, type="primary")
        else:
            st.caption(f"⚠ LTP fetch error: {_ltp_err}")

    # Show when signals were last computed (from DB timestamp)
    _sig_age = ""
    if "last_updated" in base_df.columns:
        try:
            _lu = pd.to_datetime(base_df["last_updated"]).max()
            if pd.notna(_lu):
                _lu_ist = _lu.tz_localize("Asia/Kolkata") if _lu.tzinfo is None else _lu.astimezone(_IST)
                _sig_age = f" · Signals computed: {_lu_ist.strftime('%d %b %H:%M')}"
        except Exception:
            pass
    st.caption(
        f"Levels (entry/stop/targets) loaded from DB — persist across restarts.{_sig_age} "
        "LTP refreshes every ~2 s during market hours. Run Quick Scan to recompute signals."
    )

    # Apply live LTP for metric pill counts
    df_c = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_c.loc[df_c["tradingsymbol"] == _sym, "ltp"] = _price

    _n_sb  = int((df_c["swing_signal"]    == "BUY").sum())
    _n_ss  = int((df_c["swing_signal"]    == "SELL").sum())
    _n_il  = int((df_c["intraday_signal"] == "BUY_ABOVE").sum())
    _n_is  = int((df_c["intraday_signal"] == "SELL_BELOW").sum())
    _n_sc  = int((df_c["scale_signal"]    == "INITIAL_ENTRY").sum())
    # Persist counts so outer tabs can display them in sub-tab labels
    st.session_state["_n_intra_long"]  = _n_il
    st.session_state["_n_intra_short"] = _n_is

    _pc1, _pc2, _pc3, _pc4, _pc5, _pc6 = st.columns(6)
    _pc1.metric("Swing Buy",       _n_sb,  help="Stocks with a BUY signal (PULLBACK / BREAKOUT / NR7)")
    _pc2.metric("Exit signals",    _n_ss,  help="Existing longs where trend has broken — consider exiting")
    _pc3.metric("Intraday Long",   _n_il,  help="BUY_ABOVE R1 setups for today's session")
    _pc4.metric("Intraday Short",  _n_is,  help="SELL_BELOW S1 setups — short sell opportunities")
    _pc5.metric("Scaling entries", _n_sc,  help="Stocks in full EMA stack at EMA50 pullback — position build")
    # ── Nifty 50 intraday direction pill ──
    _nifty_pct_disp = st.session_state.get("_nifty_intraday_pct")
    _nifty_ltp_disp = st.session_state.get("_nifty_live_ltp")
    if _nifty_ltp_disp:
        _nifty_pct_str = f"{_nifty_pct_disp:+.2f}%" if _nifty_pct_disp is not None else "—"
        _pc6.metric(
            "Nifty 50",
            f"₹{_nifty_ltp_disp:,.0f}",
            delta=_nifty_pct_str,
            help=(
                "Nifty 50 live LTP and today's % change vs prev close.\n\n"
                f"Gate threshold: ±{config.NIFTY_GATE_PCT}%\n"
                "🔴 Down > 0.6%: long signals flagged with headwind warning\n"
                "🟢 Up > 0.6%: short signals flagged with headwind warning"
            ),
        )
    else:
        _pc6.metric("Nifty 50", "—", help="Nifty 50 live price (fetches when authenticated)")

    # ── Global paper trade exit monitor ──────────────────────────────────────
    # Runs here (in the header fragment) so exits are detected regardless of
    # which tab the user is viewing — not just when the Intraday Plan tab is open.
    if _market_open and st.session_state.get("paper_open"):
        _live_ltp_now = st.session_state.get("_live_ltp", {})

        # ── Pass 1: T1 / T2 / stop-loss exits ───────────────────────────────
        # Stop is triggered at planned stop price OR if LTP has blown >0.2% past
        # it (fast-exit: don't wait for the next 2-second tick, exit at actual LTP).
        # T1 hit → partial booking: close 60% at T1, move stop to break-even, trail to T2.
        _STOP_OVERSHOOT_PCT = 0.002   # 0.2% past planned stop = fast-exit at LTP
        _exits   = []   # (pid, sym, outcome, exit_price) — full close
        _partials = []  # (pid, sym, exit_price) — partial T1 exit, trail to T2
        for _pid, _pt in list(st.session_state["paper_open"].items()):
            _pt_ltp = _live_ltp_now.get(_pt.get("sym", ""), 0)
            if not _pt_ltp:
                continue
            _sig    = _pt.get("signal_type", "")
            _t1     = _pt.get("t1", 0)
            _t2     = _pt.get("t2", 0)
            _stop   = _pt.get("stop", 0)
            _entry  = _pt.get("entry", 0)
            _partial_done = _pt.get("partial_booked", False)

            if _sig == "BUY_ABOVE":
                if _t1 and _pt_ltp >= _t1:
                    if _t2 and not _partial_done:
                        # Partial booking: 60% exit at T1, trail 40% to T2
                        _partials.append((_pid, _pt["sym"], _t1))
                    else:
                        # No T2, or already partial-booked and now hit T2 → full exit
                        _exits.append((_pid, _pt["sym"], "TARGET_HIT", _t1 if not _partial_done else _pt_ltp))
                elif _stop and _pt_ltp <= _stop:
                    _exits.append((_pid, _pt["sym"], "STOPPED_OUT", _pt_ltp))
                elif _stop and _entry and _pt_ltp < _stop * (1 - _STOP_OVERSHOOT_PCT):
                    _exits.append((_pid, _pt["sym"], "STOPPED_OUT", _pt_ltp))
            elif _sig == "SELL_BELOW":
                if _t1 and _pt_ltp <= _t1:
                    if _t2 and not _partial_done:
                        _partials.append((_pid, _pt["sym"], _t1))
                    else:
                        _exits.append((_pid, _pt["sym"], "TARGET_HIT", _t1 if not _partial_done else _pt_ltp))
                elif _stop and _pt_ltp >= _stop:
                    _exits.append((_pid, _pt["sym"], "STOPPED_OUT", _pt_ltp))
                elif _stop and _pt_ltp > _stop * (1 + _STOP_OVERSHOOT_PCT):
                    _exits.append((_pid, _pt["sym"], "STOPPED_OUT", _pt_ltp))

        # Apply partial bookings — 60% at T1, trail 40% to T2 (trade stays OPEN in DB)
        for _pid, _sym_e, _t1_px in _partials:
            try:
                _pt_ref = st.session_state["paper_open"].get(_pid, {})
                _orig_entry = _pt_ref.get("entry", _t1_px)
                _t2_val     = _pt_ref.get("t2", 0)
                _note_partial = (
                    f"📊 Partial T1: 60% booked at ₹{_t1_px:.2f}. "
                    f"Stop → break-even ₹{_orig_entry:.2f}. "
                    f"Trailing 40% to T2 ₹{_t2_val:.2f}."
                )
                # Note on DB but keep trade OPEN — final close happens at T2 or stop
                db.note_partial_t1(_pid, _t1_px, _note_partial)
                # Update in-memory monitoring: new stop = break-even, new target = T2
                st.session_state["paper_open"][_pid]["stop"]           = _orig_entry
                st.session_state["paper_open"][_pid]["t1"]             = _t2_val
                st.session_state["paper_open"][_pid]["t2"]             = 0
                st.session_state["paper_open"][_pid]["partial_booked"] = True
                st.toast(f"📊 Partial T1: {_sym_e} — 60% @ ₹{_t1_px:.2f}, trailing 40% → T2 ₹{_t2_val:.2f}", icon="📊")
            except Exception:
                pass

        # Apply full exits
        for _pid, _sym_e, _outcome, _ep in _exits:
            try:
                _was_partial = st.session_state.get("paper_open", {}).get(_pid, {}).get("partial_booked", False)
                _label = "T2 hit" if _was_partial else ("closed at T1" if _outcome == "TARGET_HIT" else "stopped")
                _note = f"📄 Paper trade auto-{_label}"
                db.close_trade(_pid, _ep, _outcome, _note)
                st.session_state["paper_open"].pop(_pid, None)
                _icon = "✅" if _outcome == "TARGET_HIT" else "🛑"
                st.toast(f"{_icon} Paper {_sym_e}: {_outcome} at ₹{_ep:.2f}", icon=_icon)
            except Exception:
                pass

        # ── Pass 2: Option-C trailing-cutoff gate — close losing open trades ─
        # Trigger: total return (realized P&L + open MTM) drops to the cutoff.
        # Only NEGATIVE-MTM trades are closed; profitable open trades keep running.
        _g_hwm    = st.session_state.get("paper_day_hwm_pct", 0.0)
        _g_low    = config.DAILY_TARGET_LOW_PCT     # 2.0 %
        _g_trail  = config.DAILY_TRAIL_PCT          # 0.3 %
        _g_cutoff = (_g_hwm - _g_trail) if _g_hwm >= _g_low else None

        if _g_cutoff is not None and st.session_state.get("paper_open"):
            _g_uid  = st.session_state.get("kite_user_id", "")
            try:
                _g_real = db.get_today_closed_pnl(user_id=_g_uid, is_paper=True)
            except Exception:
                _g_real = 0.0

            # Compute per-trade MTM and total
            _g_per_trade: dict[int, tuple[float, float]] = {}  # pid → (mtm_pnl, ltp)
            _g_mtm_sum = 0.0
            for _g_pid, _g_pt in list(st.session_state["paper_open"].items()):
                _g_ltp = _live_ltp_now.get(_g_pt.get("sym", ""), 0)
                if not _g_ltp:
                    continue
                _g_dir      = -1 if _g_pt.get("signal_type") == "SELL_BELOW" else 1
                _g_slot_cap = _g_pt.get("cap", config.PAPER_CAP_MODERATE)
                _g_qty      = max(1, int(_g_slot_cap / (_g_pt.get("entry") or 1)))
                _g_trade_mtm = _g_dir * (_g_ltp - _g_pt["entry"]) * _g_qty
                _g_per_trade[_g_pid] = (_g_trade_mtm, _g_ltp)
                _g_mtm_sum += _g_trade_mtm

            _g_total_ret = ((_g_real + _g_mtm_sum) / config.PAPER_CAPITAL * 100)

            if _g_total_ret <= _g_cutoff:
                # Close only the loss-making legs; winners keep running
                for _g_pid, (_g_trade_mtm, _g_exit_ltp) in _g_per_trade.items():
                    if _g_trade_mtm < 0 and _g_pid in st.session_state["paper_open"]:
                        _g_sym = st.session_state["paper_open"][_g_pid].get("sym", "")
                        try:
                            db.close_trade(
                                _g_pid, _g_exit_ltp, "STOPPED_OUT",
                                f"📄 Gate-C: total return {_g_total_ret:.2f}% ≤ cutoff "
                                f"{_g_cutoff:.2f}% — loss position auto-closed"
                            )
                            st.session_state["paper_open"].pop(_g_pid, None)
                            st.toast(
                                f"🛡 Gate: {_g_sym} loss closed @ ₹{_g_exit_ltp:.2f} "
                                f"(total return touched cutoff {_g_cutoff:.2f}%)",
                                icon="🛡",
                            )
                        except Exception:
                            pass

        # ── Pass 2b: T1 / stop exits for scalp trades (same logic, separate dict) ─
        _scalp_exits    = []
        _scalp_partials = []
        for _scid, _sct in list(st.session_state.get("scalp_open", {}).items()):
            _sc_ltp = _live_ltp_now.get(_sct.get("sym", ""), 0)
            if not _sc_ltp:
                continue
            _sc_sig  = _sct.get("signal_type", "")
            _sc_t1   = _sct.get("t1", 0)
            _sc_stop = _sct.get("stop", 0)
            _sc_partial = _sct.get("partial_booked", False)
            if _sc_sig == "BUY_ORB":
                if _sc_t1 and _sc_ltp >= _sc_t1:
                    _scalp_exits.append((_scid, _sct["sym"], "TARGET_HIT", _sc_t1))
                elif _sc_stop and _sc_ltp <= _sc_stop:
                    _scalp_exits.append((_scid, _sct["sym"], "STOPPED_OUT", _sc_ltp))
                elif _sc_stop and _sc_ltp < _sc_stop * (1 - _STOP_OVERSHOOT_PCT):
                    _scalp_exits.append((_scid, _sct["sym"], "STOPPED_OUT", _sc_ltp))
            elif _sc_sig == "SELL_ORB":
                if _sc_t1 and _sc_ltp <= _sc_t1:
                    _scalp_exits.append((_scid, _sct["sym"], "TARGET_HIT", _sc_t1))
                elif _sc_stop and _sc_ltp >= _sc_stop:
                    _scalp_exits.append((_scid, _sct["sym"], "STOPPED_OUT", _sc_ltp))
                elif _sc_stop and _sc_ltp > _sc_stop * (1 + _STOP_OVERSHOOT_PCT):
                    _scalp_exits.append((_scid, _sct["sym"], "STOPPED_OUT", _sc_ltp))
        for _scid, _sym_sc, _out_sc, _ep_sc in _scalp_exits:
            try:
                _lbl_sc = "at T1" if _out_sc == "TARGET_HIT" else "stopped"
                db.close_trade(_scid, _ep_sc, _out_sc,
                               f"⚡ Scalp auto-{_lbl_sc} @ ₹{_ep_sc:.2f}")
                st.session_state["scalp_open"].pop(_scid, None)
                _icon_sc = "✅" if _out_sc == "TARGET_HIT" else "🛑"
                st.toast(f"{_icon_sc} Scalp {_sym_sc}: {_out_sc} @ ₹{_ep_sc:.2f}", icon=_icon_sc)
            except Exception:
                pass

        # ── Pass 2c: Scalp hard exit at 2:45 PM ──────────────────────────────
        _now_sc_exit = datetime.now(_IST)
        _past_245 = (_now_sc_exit.hour > 14 or
                     (_now_sc_exit.hour == 14 and _now_sc_exit.minute >= 45))
        _scalp_exit_key = f"_scalp_hard_exit_{_dt.date.today()}"
        if _past_245 and st.session_state.get("scalp_open") and not st.session_state.get(_scalp_exit_key):
            for _schid, _scht in list(st.session_state["scalp_open"].items()):
                _sch_ltp = _live_ltp_now.get(_scht.get("sym", ""), 0) or _scht.get("entry", 0)
                try:
                    db.close_trade(
                        _schid, _sch_ltp, "CLOSED",
                        f"⏰ Scalp hard exit 2:45 PM (LTP ₹{_sch_ltp:.2f})"
                    )
                    st.session_state["scalp_open"].pop(_schid, None)
                    st.toast(f"⏰ Scalp exit: {_scht.get('sym', '')} @ ₹{_sch_ltp:.2f}", icon="⏰")
                except Exception:
                    pass
            st.session_state[_scalp_exit_key] = True

        # ── Pass 3: Hard exit at 3:10 PM — close ALL remaining paper positions ─
        # Kite auto-squares MIS at 3:20 PM; we enforce 3:10 PM ourselves.
        # The flag is day-scoped so it only fires once per calendar day.
        _now_p3  = datetime.now(_IST)
        _past_310 = _now_p3.hour > 15 or (_now_p3.hour == 15 and _now_p3.minute >= 10)
        _hard_exit_key = f"_paper_hard_exit_{_dt.date.today()}"
        if _past_310 and st.session_state.get("paper_open") and not st.session_state.get(_hard_exit_key):
            for _hpid, _hpt in list(st.session_state["paper_open"].items()):
                _h_ltp = _live_ltp_now.get(_hpt.get("sym", ""), 0) or _hpt.get("entry", 0)
                try:
                    db.close_trade(
                        _hpid, _h_ltp, "CLOSED",
                        f"⏰ Hard exit 3:10 PM — intraday close rule (LTP ₹{_h_ltp:.2f})"
                    )
                    st.session_state["paper_open"].pop(_hpid, None)
                    st.toast(
                        f"⏰ Hard exit: {_hpt.get('sym', '')} @ ₹{_h_ltp:.2f}",
                        icon="⏰",
                    )
                except Exception:
                    pass
            st.session_state[_hard_exit_key] = True

    # ── Pass 4 (real trades): Kite position sync at 3:20 PM ──────────────────
    # After 3:20 PM, Kite has finished auto-squaring MIS positions.
    # Fetch day positions and update any DB open real trades that were squared.
    _now_p4   = datetime.now(_IST)
    _past_320 = _now_p4.hour > 15 or (_now_p4.hour == 15 and _now_p4.minute >= 20)
    _kite_sync_key = f"_real_kite_sync_{_dt.date.today()}"
    _kc_sync  = st.session_state.get("kite_client")
    if (
        _past_320
        and not st.session_state.get(_kite_sync_key)
        and _kc_sync is not None
        and getattr(_kc_sync, "authenticated", False)
    ):
        try:
            _uid_sync  = st.session_state.get("kite_user_id", "")
            _day_pos   = {}
            try:
                _positions = _kc_sync.get_positions()
                for _p in _positions.get("day", []):
                    _day_pos[_p.get("tradingsymbol", "")] = _p
            except Exception:
                pass

            # Fetch all OPEN real trades for today
            _open_real = db.get_open_real_trades(user_id=_uid_sync)
            for _rt in _open_real:
                _rt_sym  = _rt.get("tradingsymbol", "")
                _rt_id   = _rt.get("id")
                _rt_sig  = _rt.get("signal_type", "BUY_ABOVE")
                _kpos    = _day_pos.get(_rt_sym)
                if not _kpos:
                    # Not found in Kite positions — may have been fully squared silently.
                    # Close at last known LTP to avoid leaving DB in OPEN state.
                    _fallback_ltp = _live_ltp_now.get(_rt_sym, 0)
                    if _fallback_ltp:
                        try:
                            db.close_trade(
                                _rt_id, _fallback_ltp, "CLOSED",
                                "🤖 Kite sync 3:20 PM — position not found (auto-squared)"
                            )
                        except Exception:
                            pass
                    continue
                # Position quantity = 0 means fully squared off
                _net_qty = int(_kpos.get("quantity", 1) or 1)
                if _net_qty == 0:
                    # Derive exit price from Kite position data
                    if _rt_sig in ("BUY", "BUY_ABOVE"):
                        _exit_px = float(_kpos.get("sell_price") or _kpos.get("last_price") or 0)
                    else:
                        _exit_px = float(_kpos.get("buy_price") or _kpos.get("last_price") or 0)
                    if not _exit_px:
                        _exit_px = float(_kpos.get("last_price") or 0)
                    if _exit_px:
                        try:
                            _kpnl = float(_kpos.get("realised") or _kpos.get("pnl") or 0)
                            db.close_trade(
                                _rt_id, _exit_px, "CLOSED",
                                f"🤖 Kite auto-squared 3:20 PM @ ₹{_exit_px:.2f} "
                                f"(Kite P&L ₹{_kpnl:.2f})"
                            )
                        except Exception:
                            pass
            st.session_state[_kite_sync_key] = True
        except Exception:
            pass


# ── FRAGMENT 2c: trading mode control strip + kill switch ────────────────────
@st.fragment
def _trading_mode_control():
    """
    Control panel shown at the top of the Intraday Plan tab.
    Lets the user choose Paper / Real / Off and fire the kill switch.
    """
    _kc_ctrl  = st.session_state.get("kite_client")
    _kite_ok  = _kc_ctrl is not None and getattr(_kc_ctrl, "authenticated", False)
    _mode     = st.session_state.get("trading_mode", "paper")

    # ── Mode strip ────────────────────────────────────────────────────────────
    _mode_labels = {
        "paper": ("📄 Paper Auto",  "#3b82f6", "Auto-creates paper trades when a signal triggers"),
        "real":  ("💸 Real Auto",   "#22c55e", "Auto-places real Kite orders when a signal triggers"),
        "off":   ("⏸ Off",         "#64748b", "No auto-trading — manual orders only"),
    }
    _ml, _mm, _mr = st.columns([3, 3, 3])
    for _col, (_mkey, _mlabel) in zip([_ml, _mm, _mr], [
        ("paper", "📄 Paper Auto"), ("real", "💸 Real Auto"), ("off", "⏸ Off")
    ]):
        _active = (_mode == _mkey)
        _border = "3px solid " + _mode_labels[_mkey][1] if _active else "1px solid #334155"
        _bg     = "#0f172a" if _active else "#0a0f1a"
        _opacity = "" if _active else "opacity:0.55;"
        _col.markdown(
            f'<div style="background:{_bg};border:{_border};border-radius:8px;'
            f'padding:8px 14px;text-align:center;{_opacity}">'
            f'<span style="font-size:0.85rem;font-weight:700;color:{_mode_labels[_mkey][1]}">'
            f'{_mlabel}</span><br>'
            f'<span style="font-size:0.7rem;color:#64748b">{_mode_labels[_mkey][2]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if _col.button(
            "✓ Active" if _active else "Switch",
            key=f"mode_btn_{_mkey}",
            type="primary" if _active else "secondary",
            use_container_width=True,
            disabled=_active or (_mkey == "real" and not _kite_ok),
            help=None if _kite_ok else "Connect Kite in the sidebar to enable Real Auto mode",
        ):
            if _mkey == "real" and not _kite_ok:
                st.warning("Connect Kite (sidebar) before enabling Real Auto mode.")
            else:
                st.session_state["trading_mode"] = _mkey
                st.rerun()

    # Real mode warning
    if _mode == "real":
        st.warning(
            "⚠️ **Real Auto mode active.** "
            "Orders will be placed on the exchange automatically when a signal triggers. "
            "Make sure your Kite account has sufficient margin.",
            icon="⚠️",
        )
    elif _mode == "off":
        st.info("Auto-trading is **Off** — signals are shown but no trades are created automatically.", icon="⏸")

    # ── Kill Switch ───────────────────────────────────────────────────────────
    st.markdown("---")
    _ks_col, _ks_info = st.columns([2, 5])
    with _ks_col:
        _ks_pressed = st.button(
            "🔴 Kill Switch — Close All & Stop",
            type="primary",
            use_container_width=True,
            help="Immediately closes all open paper trades at LTP, cancels open Kite orders, and switches mode to Off",
        )

    _n_paper_open = len(st.session_state.get("paper_open", {}))
    _live_ltp_now = st.session_state.get("_live_ltp", {})
    _uid_ks       = st.session_state.get("kite_user_id", "")
    with _ks_info:
        st.markdown(
            f'<div style="padding-top:6px;font-size:0.8rem;color:#94a3b8">'
            f'Will close <b style="color:#f59e0b">{_n_paper_open} open paper trade(s)</b> at live LTP '
            f'+ cancel any open real Kite orders → then switches to <b>Off</b> mode.</div>',
            unsafe_allow_html=True,
        )

    if _ks_pressed:
        _closed_paper = 0
        _cancelled_real = 0
        _errors = []

        # 1. Close all open paper trades at LTP
        for _pid, _pt in list(st.session_state.get("paper_open", {}).items()):
            _exit_p = _live_ltp_now.get(_pt.get("sym", ""), _pt.get("entry", 0))
            if not _exit_p:
                _exit_p = _pt.get("entry", 0)
            try:
                db.close_trade(_pid, float(_exit_p), "CLOSED", "🔴 Kill switch — closed at LTP")
                st.session_state["paper_open"].pop(_pid, None)
                _closed_paper += 1
            except Exception as _e:
                _errors.append(f"Paper {_pt.get('sym')}: {_e}")

        # 2. Cancel open REAL Kite orders
        if _kite_ok:
            try:
                _open_real = db.load_trade_log(status_filter=["OPEN"], user_id=_uid_ks)
                _real_open_rows = _open_real[
                    (_open_real.get("is_paper_trade", False) != True) &
                    (_open_real["kite_order_id"].notna())
                ] if not _open_real.empty and "kite_order_id" in _open_real.columns else pd.DataFrame()
                for _, _rrow in _real_open_rows.iterrows():
                    _oid = _rrow.get("kite_order_id")
                    try:
                        _kc_ctrl.cancel_order(_oid)
                        db.close_trade(
                            int(_rrow["id"]), 0.0, "CANCELLED",
                            "🔴 Kill switch — Kite order cancelled"
                        )
                        _cancelled_real += 1
                    except Exception as _e:
                        _errors.append(f"Real order {_oid}: {_e}")
            except Exception as _e:
                _errors.append(f"Real order query failed: {_e}")

        # 3. Close open scalp positions at LTP
        for _sc_pid, _sc_pt in list(st.session_state.get("scalp_open", {}).items()):
            _sc_ltp_kill = st.session_state.get("_live_ltp", {}).get(_sc_pt.get("sym", ""), 0)
            if _sc_ltp_kill:
                try:
                    db.close_trade(_sc_pid, _sc_ltp_kill, "CLOSED",
                                   f"🔴 Kill switch — scalp closed @ ₹{_sc_ltp_kill:.2f}")
                    st.session_state["scalp_open"].pop(_sc_pid, None)
                    _closed_paper += 1
                except Exception:
                    pass

        # 4. Switch to Off mode + clear triggered dicts
        st.session_state["trading_mode"]    = "off"
        st.session_state["paper_triggered"] = {}
        st.session_state["real_triggered"]  = {}
        st.session_state["scalp_triggered"] = {}
        st.session_state["scalp_open"]      = {}

        _msg = f"🔴 Kill switch fired: {_closed_paper} paper trade(s) closed, {_cancelled_real} Kite order(s) cancelled."
        if _errors:
            st.warning(_msg + f"\n\nErrors: {'; '.join(_errors)}")
        else:
            st.success(_msg)
        st.rerun()


# ── FRAGMENT 2b: paper-trade banner (auto-refresh, daily gate logic) ─────────
@st.fragment(run_every=2)
def _intraday_paper_banner():
    """
    Renders the 'Paper Trades Today' dashboard strip and enforces the
    daily trailing-stop gate:

      • Below 2%  realised return  → keep allocating to new signals
      • 2% – 5%                   → trailing cutoff = peak_return − 0.3%
                                    (cutoff only ratchets UP with gains)
      • ≥ 5%                      → hard ceiling, no new entries
      • Return drops to cutoff    → block new entries for the rest of day
    """
    _uid = st.session_state.get("kite_user_id", "")

    # ── Fetch today's realised paper P&L ─────────────────────────────────────
    try:
        _today_paper_pnl = db.get_today_closed_pnl(user_id=_uid, is_paper=True)
    except Exception:
        _today_paper_pnl = 0.0

    _cap = config.PAPER_CAPITAL
    _ret_pct = (_today_paper_pnl / _cap * 100) if _cap else 0.0

    # ── Live MTM on open positions ────────────────────────────────────────────
    _open_pt       = st.session_state.get("paper_open", {})
    _n_open        = len(_open_pt)
    _n_today       = len(st.session_state.get("paper_triggered", {}))
    _live_ltp_now  = st.session_state.get("_live_ltp", {})
    _open_mtm      = 0.0
    _capital_deployed = 0
    for _ppid, _pp in _open_pt.items():
        _p_ltp    = _live_ltp_now.get(_pp["sym"], _pp.get("entry", 0))
        _dir      = -1 if _pp["signal_type"] == "SELL_BELOW" else 1
        _slot_cap = _pp.get("cap", config.PAPER_CAP_MODERATE)
        _p_qty    = max(1, int(_slot_cap / (_pp.get("entry") or 1)))
        _open_mtm         += _dir * (_p_ltp - _pp["entry"]) * _p_qty
        _capital_deployed += _slot_cap

    _total_pnl   = _today_paper_pnl + _open_mtm   # realised + unrealised
    _total_ret   = (_total_pnl / _cap * 100) if _cap else 0.0
    _pnl_color   = "#22c55e" if _total_pnl >= 0 else "#ef4444"

    # ── Update high-water mark (based on realized P&L — stable, not swung by MTM) ─
    _hwm = max(st.session_state.get("paper_day_hwm_pct", 0.0), _ret_pct)
    st.session_state["paper_day_hwm_pct"] = _hwm
    # Also persist total return so Option-C gate in header fragment can read it
    st.session_state["_paper_total_ret"] = _total_ret

    _LOW   = config.DAILY_TARGET_LOW_PCT    # 2.0
    _HIGH  = config.DAILY_TARGET_HIGH_PCT   # 5.0
    _TRAIL = config.DAILY_TRAIL_PCT         # 0.3

    # Cutoff = hwm − TRAIL (but only active once hwm ≥ LOW)
    _cutoff_pct: float | None = (_hwm - _TRAIL) if _hwm >= _LOW else None

    # New entries blocked when REALIZED return drops to cutoff (or hits ceiling)
    _blocked = (
        (_cutoff_pct is not None and _ret_pct <= _cutoff_pct)
        or _ret_pct >= _HIGH
    )
    # Persist so auto-trigger code can read it without re-computing
    st.session_state["paper_day_blocked"] = _blocked

    # ── Gate status label — show total return (realized + MTM) ───────────────
    if _blocked:
        if _ret_pct >= _HIGH:
            _gate_html = (
                f'<span style="color:#ef4444;font-weight:700;font-size:0.75rem">'
                f'🚫 CEILING HIT ({_HIGH:.0f}%) — no new entries today</span>'
            )
        else:
            _gate_html = (
                f'<span style="color:#ef4444;font-weight:700;font-size:0.75rem">'
                f'🚫 CUTOFF HIT — total return {_total_ret:.2f}% '
                f'(realized {_ret_pct:.2f}% | peak {_hwm:.2f}% − {_TRAIL:.1f}% = {_cutoff_pct:.2f}%)'
                f'</span>'
            )
    elif _cutoff_pct is not None:
        _gate_html = (
            f'<span style="color:#f59e0b;font-weight:600;font-size:0.75rem">'
            f'⚡ Trailing active · total {_total_ret:.2f}% (realized {_ret_pct:.2f}%) — '
            f'loss positions close if total ≤ <b>{_cutoff_pct:.2f}%</b> '
            f'(peak {_hwm:.2f}% − {_TRAIL:.1f}%)</span>'
        )
    elif _hwm > 0:
        _gate_html = (
            f'<span style="color:#22c55e;font-size:0.75rem">'
            f'✅ Total {_total_ret:.2f}% (realized {_ret_pct:.2f}%) — trailing activates at {_LOW:.0f}%'
            f'</span>'
        )
    else:
        _target_amt = _cap * _LOW / 100
        _gate_html = (
            f'<span style="color:#94a3b8;font-size:0.75rem">'
            f'🎯 Daily target: {_LOW:.0f}%–{_HIGH:.0f}% '
            f'(₹{_target_amt:,.0f}–₹{_cap * _HIGH / 100:,.0f})</span>'
        )

    if _n_today > 0 or _n_open > 0 or True:   # always show banner
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
            f'padding:10px 18px;margin-bottom:10px;display:flex;flex-wrap:wrap;'
            f'gap:20px;align-items:center">'
            f'<span style="font-size:0.8rem;color:#94a3b8;white-space:nowrap">'
            f'📄 <b>Paper Trades Today</b></span>'
            f'<span style="color:#f8fafc;font-weight:700">{_n_today} created</span>'
            f'<span style="color:#f59e0b;font-weight:600">{_n_open} open</span>'
            f'<span style="color:{_pnl_color};font-weight:700">'
            f'Live P&amp;L: ₹{_total_pnl:+,.0f} '
            f'(<span style="font-size:0.8em">{_total_ret:+.2f}%</span>)'
            f'</span>'
            f'<span style="font-size:0.75rem;color:#64748b;white-space:nowrap">'
            f'Capital: ₹{_cap:,} · ₹{_cap // config.PAPER_MAX_POSITIONS:,}/trade · '
            f'Deployed: ₹{_capital_deployed:,}</span>'
            f'<span>{_gate_html}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Block banner ──────────────────────────────────────────────────────────
    if _blocked:
        st.warning(
            f"**Daily trading gate closed** — paper trades will NOT be created for new signals today.  \n"
            f"Realised gain: **{_ret_pct:.2f}%** · Peak: **{_hwm:.2f}%** · "
            + (
                f"Cutoff: **{_cutoff_pct:.2f}%** (trail {_TRAIL:.1f}% from peak)"
                if _cutoff_pct is not None
                else f"Hard ceiling: **{_HIGH:.0f}%** reached"
            ),
            icon="🚫",
        )


# ── FRAGMENT 2d: scalping signals (ORB — Opening Range Breakout) ─────────────
# Refreshes every 5 s (slower than intraday because candle fetches are heavier).
# ORB is valid only after 9:30 AM (need 15-min opening range to be complete).
@st.fragment(run_every=5)
def _intraday_scalp_live():
    """
    Scalp signal tab — Opening Range Breakout with 3 confirmations:
      1. ORB breakout (LTP > ORB_high / LTP < ORB_low)
      2. VWAP alignment (above/below VWAP)
      3. 5-min RSI momentum (>55 long / <45 short)

    Needs Kite auth to fetch 5-min candles. Falls back to a static info card
    during market-closed hours or when not authenticated.
    """
    if not _is_market_open():
        st.info(
            "Scalping signals are only active during market hours (9:15–15:30 IST). "
            "ORB is computed after 9:30 AM once the first 15 minutes have completed.",
            icon="⏸",
        )
        return

    _now_ist = datetime.now(_IST)
    if _now_ist.hour == 9 and _now_ist.minute < 30:
        st.info(
            "⏳ Opening Range not ready yet. ORB scalp signals activate at 9:30 AM "
            "once the first 15 minutes close.",
            icon="⏳",
        )
        return

    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        st.info("Run a scan first to populate the watchlist.", icon="ℹ️")
        return

    _kc_scalp = st.session_state.get("kite_client")
    if not _kc_scalp or not getattr(_kc_scalp, "authenticated", False):
        st.warning("Connect Kite to enable ORB scalp signals (5-min candle data required).", icon="🔑")
        return

    # ── Build candidate list ──────────────────────────────────────────────────
    # Use intraday signal stocks as ORB candidates (they already passed pivot + RSI + R/R gates)
    _scalp_mask = base_df["intraday_signal"].isin(["BUY_ABOVE", "SELL_BELOW"])
    _scalp_cands = base_df.loc[_scalp_mask].copy()
    if _scalp_cands.empty:
        st.info("No intraday signal stocks available for ORB scalping today.", icon="ℹ️")
        return

    # Limit to top 15 by composite score to stay within API rate limits
    _scalp_cands = (
        _scalp_cands
        .sort_values("composite_score", ascending=False, na_position="last")
        .head(15)
    )

    _nifty_pct     = st.session_state.get("_nifty_intraday_pct", 0.0) or 0.0
    _scalp_results = []
    _live_ltp_now  = st.session_state.get("_live_ltp", {})
    _trade_mode    = st.session_state.get("trading_mode", "paper")
    _ltp_ts_sc     = st.session_state.get("_live_ltp_ts")
    _ltp_stale_sc  = (_ltp_ts_sc is None or
                      (datetime.now(_IST) - _ltp_ts_sc).total_seconds() > config.LTP_FRESHNESS_SECS)

    # ── Mode / status strip ───────────────────────────────────────────────────
    _sc_open_count   = len(st.session_state.get("scalp_open", {}))
    _sc_slots_left   = config.SCALP_MAX_POSITIONS - _sc_open_count
    _sc_cap_deployed = _sc_open_count * config.SCALP_CAP_PER_TRADE

    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("Open Scalps",   _sc_open_count,   help="Currently open scalp trades")
    _mc2.metric("Slots Left",    _sc_slots_left,   help=f"Max {config.SCALP_MAX_POSITIONS} concurrent scalp positions")
    _mc3.metric("Cap Deployed",  f"₹{_sc_cap_deployed:,}", help=f"₹{config.SCALP_CAP_PER_TRADE:,} per trade")
    _mc4.metric("Mode",          _trade_mode.upper(), help="Trading mode: Paper / Real / Off")

    if _ltp_stale_sc:
        st.info("⏸ Prices stale — waiting for refresh. Auto-scalp paused.", icon="🔄")

    st.caption(
        f"Scanning {len(_scalp_cands)} stocks for ORB breakouts. "
        "Requires 2/3 confirmations: **ORB break + VWAP + 5-min RSI**. "
        f"Confirm hold: {_SCALP_ENTRY_CONFIRM_SECS}s. "
        f"Hard exit: 2:45 PM. Capital per scalp: ₹{config.SCALP_CAP_PER_TRADE:,}. "
        f"Nifty: {_nifty_pct:+.2f}%"
    )

    _scalp_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter…",
        key="scalp_sym_search", label_visibility="collapsed",
    )

    import indicators as _ind  # noqa: PLC0415
    import signals as _sig_mod  # noqa: PLC0415

    _scalp_confirm_map = st.session_state.setdefault("_scalp_confirm_since", {})

    for _, _row in _scalp_cands.iterrows():
        _sym = str(_row.get("tradingsymbol", ""))
        _tok = int(_row.get("instrument_token") or 0)
        if not _tok:
            continue
        if _scalp_sym_q and _scalp_sym_q.strip().lower() not in _sym.lower():
            continue

        _ltp_now = _live_ltp_now.get(_sym, 0) or float(_row.get("ltp") or 0)
        if not _ltp_now:
            continue

        # ── Fetch / cache 5-min candles ───────────────────────────────────────
        try:
            _candle_cache = st.session_state.setdefault("_scalp_candle_cache", {})
            _cache_ts_key = f"_scalp_ts_{_sym}"
            _last_fetch   = st.session_state.get(_cache_ts_key)
            _candles_df   = _candle_cache.get(_sym, pd.DataFrame())
            if (_last_fetch is None or
                    (datetime.now(_IST) - _last_fetch).total_seconds() > 300
                    or _candles_df.empty):
                _candles_df = _kc_scalp.get_today_candles(_tok, interval="5minute")
                _candle_cache[_sym] = _candles_df
                st.session_state[_cache_ts_key] = datetime.now(_IST)
        except Exception:
            _candles_df = pd.DataFrame()

        if _candles_df.empty:
            continue

        # ── Compute intraday indicators ───────────────────────────────────────
        _orb_data   = _ind.opening_range(_candles_df, minutes=config.SCALP_ORB_MINUTES)
        _vwap_price = _ind.vwap(_candles_df)
        _rsi_5m     = _ind.rsi_intraday(_candles_df)
        _avg_vol    = float(_row.get("avg_volume") or 0)
        _vol_ratio  = _ind.intraday_volume_ratio(_candles_df, _avg_vol) if _avg_vol > 0 else None

        if not _orb_data:
            continue

        _scalp_sig = _sig_mod.scalping_signal(
            current_ltp      = _ltp_now,
            orb_high         = _orb_data["orb_high"],
            orb_low          = _orb_data["orb_low"],
            orb_range        = _orb_data["orb_range"],
            vwap_price       = _vwap_price,
            rsi_5min         = _rsi_5m,
            atr              = float(_row.get("atr_14") or 0) or None,
            nifty_pct_change = _nifty_pct,
            daily_vol_ratio  = _vol_ratio,
        )

        _status  = _scalp_sig.get("scalp_signal") or "INSIDE_ORB"
        _dir     = _scalp_sig.get("scalp_direction") or ""
        _confs   = _scalp_sig.get("scalp_confirmations") or 0
        _sc_conf = _scalp_sig.get("scalp_confidence") or 0

        # ── Confirmation timer (10 s sustained breakout) ──────────────────────
        _sc_key_long  = f"_sc_long_{_sym}"
        _sc_key_short = f"_sc_short_{_sym}"
        if _status == "LONG":
            if _sc_key_long not in _scalp_confirm_map:
                _scalp_confirm_map[_sc_key_long] = datetime.now(_IST)
            _scalp_confirm_map.pop(_sc_key_short, None)
        elif _status == "SHORT":
            if _sc_key_short not in _scalp_confirm_map:
                _scalp_confirm_map[_sc_key_short] = datetime.now(_IST)
            _scalp_confirm_map.pop(_sc_key_long, None)
        else:
            _scalp_confirm_map.pop(_sc_key_long,  None)
            _scalp_confirm_map.pop(_sc_key_short, None)

        _sc_dir_key  = _sc_key_long if _status == "LONG" else _sc_key_short
        _sc_ts       = _scalp_confirm_map.get(_sc_dir_key)
        _sc_elapsed  = (datetime.now(_IST) - _sc_ts).total_seconds() if _sc_ts else 0
        _sc_confirmed = _sc_elapsed >= _SCALP_ENTRY_CONFIRM_SECS

        if _status in ("LONG", "SHORT"):
            _remain_secs = max(0, int(_SCALP_ENTRY_CONFIRM_SECS - _sc_elapsed))
            _status_label = (
                f"🚀 {_status}" if _sc_confirmed
                else f"⏱ CONFIRMING {_remain_secs}s"
            )
        elif _status == "WATCH":
            _status_label = "👁 WATCH"
        else:
            _status_label = "⬜ INSIDE"

        # ── Capital gate ──────────────────────────────────────────────────────
        _sc_within_limit = (
            _sc_slots_left > 0
            and not st.session_state.get("paper_day_blocked", False)
        )

        # ── Signal type for DB ────────────────────────────────────────────────
        _sc_sig_type   = "BUY_ORB" if _status == "LONG" else "SELL_ORB"
        _sc_trig_key   = (_today_str, _sym, _sc_sig_type)
        _entry_px      = _scalp_sig.get("scalp_entry") or _ltp_now
        _stop_px       = _scalp_sig.get("scalp_stop")  or 0
        _t1_px         = _scalp_sig.get("scalp_t1")    or 0
        _pqty          = max(1, int(config.SCALP_CAP_PER_TRADE / (_entry_px or 1)))

        # ── Auto-trade when confirmed + within limits ─────────────────────────
        if (_status in ("LONG", "SHORT") and _sc_confirmed
                and _sc_within_limit and not _ltp_stale_sc
                and _trade_mode != "off"):

            # ── Paper scalp ───────────────────────────────────────────────────
            if (_trade_mode == "paper"
                    and _sc_trig_key not in st.session_state.get("scalp_triggered", {})):
                try:
                    _scid = db.log_trade({
                        "trade_date":          _dt.date.today(),
                        "tradingsymbol":       _sym,
                        "instrument_token":    _tok,
                        "setup_type":          "SCALP",
                        "signal_type":         _sc_sig_type,
                        "rec_entry":           _entry_px,
                        "rec_stop":            _stop_px,
                        "rec_t1":              _t1_px,
                        "rec_t2":              0,
                        "rec_rr":              float(_scalp_sig.get("scalp_rr") or 0),
                        "rec_reason":          str(_scalp_sig.get("scalp_reason") or "")[:250],
                        "rec_composite_score": float(_row.get("composite_score") or 0),
                        "kite_user_id":        _cur_user_id,
                        "quantity":            _pqty,
                        "actual_entry":        _ltp_now,
                        "status":              "OPEN",
                        "notes": (
                            f"⚡ Scalp paper — {_status} ORB auto-triggered @ ₹{_ltp_now:.2f} "
                            f"(confs {_confs}/3, score {_sc_conf}/10, qty {_pqty})"
                        ),
                        "is_paper_trade": True,
                        "intraday_confidence": _sc_conf,
                    })
                    st.session_state["scalp_triggered"][_sc_trig_key] = _scid
                    st.session_state["scalp_open"][_scid] = {
                        "sym": _sym, "stop": _stop_px, "t1": _t1_px, "t2": 0,
                        "signal_type": _sc_sig_type, "entry": _ltp_now,
                        "cap": config.SCALP_CAP_PER_TRADE, "setup_type": "SCALP",
                        "partial_booked": False,
                    }
                    _scalp_confirm_map.pop(_sc_dir_key, None)
                    _sc_slots_left -= 1
                    st.toast(
                        f"⚡ Scalp paper {_status}: {_sym} @ ₹{_ltp_now:.2f} × {_pqty} "
                        f"(confirmed {_SCALP_ENTRY_CONFIRM_SECS}s, {_confs}/3 confs)",
                        icon="⚡",
                    )
                except Exception:
                    pass

            # ── Real scalp ────────────────────────────────────────────────────
            elif (_trade_mode == "real"
                    and _sc_trig_key not in st.session_state.get("scalp_triggered", {})):
                _kc_sc_rt = st.session_state.get("kite_client")
                if _kc_sc_rt and getattr(_kc_sc_rt, "authenticated", False):
                    try:
                        _sc_tx  = "BUY" if _status == "LONG" else "SELL"
                        _sc_px  = round(_entry_px * (1.001 if _status == "LONG" else 0.999), 1)
                        _sc_oid = _kc_sc_rt.place_order(
                            tradingsymbol    = _sym,
                            qty              = _pqty,
                            transaction_type = _sc_tx,
                            order_type       = "LIMIT",
                            product          = "MIS",
                            price            = _sc_px,
                            tag              = "scr_scalp",
                        )
                        _sc_sl_oid = None
                        if _stop_px:
                            try:
                                _sc_sl_tx = "SELL" if _status == "LONG" else "BUY"
                                _sc_sl_oid = _kc_sc_rt.place_order(
                                    tradingsymbol    = _sym,
                                    qty              = _pqty,
                                    transaction_type = _sc_sl_tx,
                                    order_type       = "SL-M",
                                    product          = "MIS",
                                    trigger_price    = _stop_px,
                                    tag              = "scr_scslp",
                                )
                            except Exception:
                                pass
                        _scid_r = db.log_trade({
                            "trade_date":          _dt.date.today(),
                            "tradingsymbol":       _sym,
                            "instrument_token":    _tok,
                            "setup_type":          "SCALP",
                            "signal_type":         _sc_sig_type,
                            "rec_entry":           _entry_px,
                            "rec_stop":            _stop_px,
                            "rec_t1":              _t1_px,
                            "rec_t2":              0,
                            "rec_rr":              float(_scalp_sig.get("scalp_rr") or 0),
                            "rec_reason":          str(_scalp_sig.get("scalp_reason") or "")[:250],
                            "rec_composite_score": float(_row.get("composite_score") or 0),
                            "kite_user_id":        _cur_user_id,
                            "kite_order_id":       _sc_oid,
                            "kite_sl_order_id":    _sc_sl_oid,
                            "kite_status":         "OPEN",
                            "quantity":            _pqty,
                            "actual_entry":        _ltp_now,
                            "status":              "OPEN",
                            "notes": (
                                f"⚡ Scalp real — {_status} ORB @ ₹{_ltp_now:.2f} "
                                f"(confs {_confs}/3, score {_sc_conf}/10) | Kite {_sc_oid}"
                            ),
                            "is_paper_trade": False,
                            "intraday_confidence": _sc_conf,
                        })
                        st.session_state["scalp_triggered"][_sc_trig_key] = _scid_r
                        st.session_state["scalp_open"][_scid_r] = {
                            "sym": _sym, "stop": _stop_px, "t1": _t1_px, "t2": 0,
                            "signal_type": _sc_sig_type, "entry": _ltp_now,
                            "cap": config.SCALP_CAP_PER_TRADE, "setup_type": "SCALP",
                            "partial_booked": False,
                        }
                        _scalp_confirm_map.pop(_sc_dir_key, None)
                        _sc_slots_left -= 1
                        st.toast(
                            f"⚡ Scalp REAL {_status}: {_sym} @ ₹{_ltp_now:.2f} × {_pqty} | Kite {_sc_oid}",
                            icon="⚡",
                        )
                    except Exception as _sc_re:
                        st.toast(f"⚠ Scalp real order failed for {_sym}: {_sc_re}", icon="⚠️")

        # ── Build display row ─────────────────────────────────────────────────
        _auto_badge = ""
        if _sc_trig_key in st.session_state.get("scalp_triggered", {}):
            _auto_badge = " ✅ TRADED"

        _scalp_results.append({
            "Status":     _status_label + _auto_badge,
            "Symbol":     _sym,
            "LTP":        f"₹{_ltp_now:,.2f}",
            "Direction":  _dir or "—",
            "Confirms":   f"{_confs}/3" if _confs else "—",
            "Conf":       f"{_sc_conf}/10",
            "ORB High":   f"₹{_orb_data['orb_high']:,.2f}",
            "ORB Low":    f"₹{_orb_data['orb_low']:,.2f}",
            "VWAP":       f"₹{_vwap_price:,.2f}" if _vwap_price else "—",
            "5m RSI":     f"{_rsi_5m:.0f}" if _rsi_5m else "—",
            "Entry":      f"₹{_entry_px:,.2f}" if _entry_px else "—",
            "Stop":       f"₹{_stop_px:,.2f}"  if _stop_px  else "—",
            "Target":     f"₹{_t1_px:,.2f}"    if _t1_px    else "—",
            "R/R":        f"{_scalp_sig.get('scalp_rr', 0):.1f}×" if _scalp_sig.get("scalp_rr") else "—",
        })

    if not _scalp_results:
        if _scalp_sym_q:
            st.info(f"No ORB results for '{_scalp_sym_q}'.", icon="🔍")
        else:
            st.info(
                "No ORB breakouts active yet. LTP is inside the opening range for all stocks. "
                "Watch for breakouts above ORB_high or breakdowns below ORB_low.",
                icon="⏳",
            )
        return

    _scalp_df = pd.DataFrame(_scalp_results)

    # Sort: LONG/SHORT first, then WATCH, then INSIDE
    _order = {"🚀 LONG": 0, "🚀 SHORT": 1, "👁 WATCH": 2, "⬜ INSIDE": 3}
    _scalp_df["_sort"] = _scalp_df["Status"].map(lambda x: _order.get(x, 9))
    _scalp_df = _scalp_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    st.dataframe(
        _scalp_df,
        use_container_width=True,
        height=min(500, 50 + len(_scalp_results) * 38),
        hide_index=True,
        column_config={
            "Status":     st.column_config.TextColumn("Status", help="ORB signal state. 🚀 = live breakout, 👁 = only 1/3 confirmations, ⬜ = inside opening range"),
            "Confirms":   st.column_config.TextColumn("Confirms", help="Number of internal confirmations (max 3): ORB break + VWAP + 5-min RSI. Need ≥2 to act."),
            "Conf Score": st.column_config.TextColumn("Score", help="Scalp confidence 0–10. Adjusted down if Nifty is against direction or volume is below average."),
            "5m RSI":     st.column_config.TextColumn("5m RSI", help="RSI computed on today's 5-min candles. >55 favours longs, <45 favours shorts."),
        },
    )

    # ── Scalp strategy explainer ──────────────────────────────────────────────
    st.markdown(
        f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;margin-top:8px">
<div style="font-size:0.78rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;margin-bottom:8px">
⚡ ORB SCALP STRATEGY GUIDE
</div>
<table style="width:100%;border-collapse:collapse;font-size:0.8rem">
<tr>
  <td style="padding:4px 14px 4px 0;color:#38bdf8;font-weight:700;white-space:nowrap">Entry</td>
  <td style="color:#cbd5e1">LTP crosses above ORB_high (long) / below ORB_low (short) with ≥2/3 confirmations.</td>
</tr>
<tr>
  <td style="padding:4px 14px 4px 0;color:#f59e0b;font-weight:700;white-space:nowrap">Target</td>
  <td style="color:#cbd5e1">Breakout + {config.SCALP_TARGET_MULT}× ORB range. Typical: 0.5–1.2% move.</td>
</tr>
<tr>
  <td style="padding:4px 14px 4px 0;color:#ef4444;font-weight:700;white-space:nowrap">Stop</td>
  <td style="color:#cbd5e1">Breakout − {config.SCALP_STOP_MULT}× ORB range (tight). If price re-enters ORB → exit immediately.</td>
</tr>
<tr>
  <td style="padding:4px 14px 4px 0;color:#a78bfa;font-weight:700;white-space:nowrap">Hard Exit</td>
  <td style="color:#cbd5e1"><b>2:45 PM IST</b> — earlier than intraday (3:10 PM) because scalps need liquidity window.</td>
</tr>
<tr>
  <td style="padding:4px 14px 4px 0;color:#22c55e;font-weight:700;white-space:nowrap">Capital</td>
  <td style="color:#cbd5e1">₹{config.SCALP_CAP_PER_TRADE:,} per scalp trade. Max {config.SCALP_MAX_POSITIONS} concurrent scalp positions.</td>
</tr>
</table>
<div style="margin-top:8px;padding-top:6px;border-top:1px solid #1e293b;color:#64748b;font-size:0.74rem">
3 Confirmations: <b>①</b> ORB breakout (price outside range) · <b>②</b> VWAP alignment (above/below) · <b>③</b> 5-min RSI momentum (&gt;55 long / &lt;45 short)
</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ── FRAGMENT 2a: intraday LONG table (live status column) ────────────────────
# Runs every 2 s inside the Long sub-tab. No st.tabs() here — the sub-tabs
# that contain this fragment are created OUTSIDE in _signals_main().
@st.fragment(run_every=2)
def _intraday_long_live():
    # Skip all work outside market hours — fragment still runs but is instant
    if not _is_market_open():
        return
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        return
    df_l = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_l.loc[df_l["tradingsymbol"] == _sym, "ltp"] = _price

    _si = (
        df_l[df_l["intraday_signal"] == "BUY_ABOVE"]
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    if _si.empty:
        st.info("No intraday long setups for today's session.")
        return

    # ── LTP freshness guard ───────────────────────────────────────────────────
    _ltp_ts    = st.session_state.get("_live_ltp_ts")
    _ltp_age   = (datetime.now(_IST) - _ltp_ts).total_seconds() if _ltp_ts else 9999
    _ltp_stale = _ltp_age > config.LTP_FRESHNESS_SECS
    if _ltp_stale:
        _age_str = f"{int(_ltp_age)}s" if _ltp_age < 120 else f"{int(_ltp_age/60)}m"
        st.info(
            f"⏸ Prices paused while tab was inactive ({_age_str} ago). "
            "Refreshing now — auto-trading resumes once fresh prices arrive.",
            icon="🔄",
        )

    st.caption("Watch for price to trade **above R1**. Enter with stop just below Pivot. T1=R2 (60% exit), T2=R3 (trail 40%).")

    # ── Nifty direction gate banner ──────────────────────────────────────────
    _nifty_pct = st.session_state.get("_nifty_intraday_pct", 0.0) or 0.0
    if _nifty_pct <= -config.NIFTY_GATE_PCT:
        st.warning(
            f"⚠️ **Nifty headwind:** Nifty 50 is down **{abs(_nifty_pct):.2f}%** today "
            f"(threshold: −{config.NIFTY_GATE_PCT}%). Long signals have headwind — "
            "consider reducing position size or waiting for Nifty stabilisation.",
            icon="🔴",
        )
    elif _nifty_pct >= config.NIFTY_GATE_PCT:
        st.success(
            f"✅ **Nifty tailwind:** Nifty 50 is up **{_nifty_pct:.2f}%** today — "
            "market breadth favours long positions.",
            icon="🟢",
        )

    _long_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter…",
        key="intra_long_sym_search", label_visibility="collapsed",
    )
    if _long_sym_q:
        _si = _si[_si["tradingsymbol"].str.contains(_long_sym_q.strip(), case=False, na=False, regex=False)]

    _si_rows = []
    for _, r in _si.iterrows():
        sym        = r.get("tradingsymbol", "")
        ltp_now    = r.get("ltp") or 0
        r1_val     = r.get("intraday_r1") or 0
        entry_val  = r.get("intraday_entry") or 0
        s1_val     = float(r.get("intraday_s1") or 0)
        _conf_raw  = r.get("intraday_confidence")
        confidence = int(_conf_raw) if not _isna(_conf_raw) else 0
        if r1_val and ltp_now:
            if ltp_now >= entry_val:
                # Show confirming countdown until 30s are up
                _cs_map = st.session_state.get("_entry_confirm_since", {})
                _cs_ts  = _cs_map.get(sym)
                if _cs_ts is not None:
                    _elapsed = (datetime.now(_IST) - _cs_ts).total_seconds()
                    if _elapsed < _ENTRY_CONFIRM_SECS:
                        _remaining = int(_ENTRY_CONFIRM_SECS - _elapsed)
                        live_status = f"CONFIRMING {_remaining}s"
                    else:
                        live_status = "TRIGGERED"
                else:
                    live_status = "TRIGGERED"
            elif ltp_now >= r1_val * 0.995:
                live_status = "APPROACHING"
            elif s1_val and ltp_now < s1_val:
                live_status = "BROKEN"
            else:
                live_status = "WATCHING"
        else:
            live_status = "—"

        # ── Confidence-based capital tier ─────────────────────────────────────
        if confidence >= config.CONFIDENCE_STRONG_MIN:
            _cap_this_trade = config.PAPER_CAP_STRONG
            _conf_tier = "STRONG"
        elif confidence >= config.CONFIDENCE_MODERATE_MIN:
            _cap_this_trade = config.PAPER_CAP_MODERATE
            _conf_tier = "MODERATE"
        elif confidence >= config.CONFIDENCE_MARGINAL_MIN:
            _cap_this_trade = config.PAPER_CAP_MARGINAL
            _conf_tier = "MARGINAL"
        else:
            _cap_this_trade = 0
            _conf_tier = "LOW"

        # ── Capital availability gate (replaces fixed slot count) ─────────────
        _deployed = sum(
            v.get("cap", config.PAPER_CAP_MODERATE)
            for v in st.session_state.get("paper_open", {}).values()
        )
        _remaining = config.PAPER_CAPITAL - _deployed
        _within_limit = (_cap_this_trade > 0) and (_remaining >= _cap_this_trade)

        # Quantity based on tier capital and actual trigger price
        _trigger_price = ltp_now if ltp_now > 0 else entry_val
        _pqty = max(1, int(_cap_this_trade / (_trigger_price or 1))) if _cap_this_trade else 1

        # ── Auto-trade when TRIGGERED ─────────────────────────────────────────
        _trade_mode = st.session_state.get("trading_mode", "paper")
        _paper_key  = (_today_str, sym)
        _real_key   = (_today_str, sym)

        # ── False-breakout confirmation timer ────────────────────────────────
        # Record the first moment LTP crossed entry. Only auto-trade after the
        # price has stayed above entry for _ENTRY_CONFIRM_SECS seconds.
        _confirm_map = st.session_state.setdefault("_entry_confirm_since", {})
        if live_status == "TRIGGERED":
            if sym not in _confirm_map:
                _confirm_map[sym] = datetime.now(_IST)
        else:
            # Price fell back below entry — reset timer
            _confirm_map.pop(sym, None)

        _confirm_ts  = _confirm_map.get(sym)
        _confirmed   = (
            _confirm_ts is not None and
            (datetime.now(_IST) - _confirm_ts).total_seconds() >= _ENTRY_CONFIRM_SECS
        )

        if live_status == "TRIGGERED" and _confirmed and _within_limit and not _ltp_stale:

            # ── PAPER mode ────────────────────────────────────────────────────
            if (_trade_mode == "paper"
                    and not st.session_state.get("paper_day_blocked", False)
                    and _paper_key not in st.session_state.get("paper_triggered", {})):
                try:
                    _t2_val = float(r.get("intraday_t2") or r.get("intraday_r3") or 0)
                    _pid = db.log_trade({
                        "trade_date":          _dt.date.today(),
                        "tradingsymbol":       sym,
                        "instrument_token":    int(r.get("instrument_token") or 0),
                        "setup_type":          "INTRADAY",
                        "signal_type":         "BUY_ABOVE",
                        "rec_entry":           entry_val,
                        "rec_stop":            float(r.get("intraday_stop") or 0),
                        "rec_t1":              float(r.get("intraday_t1") or 0),
                        "rec_t2":              _t2_val,
                        "rec_rr":              None,
                        "rec_reason":          str(r.get("intraday_reason") or "")[:200],
                        "rec_composite_score": r.get("composite_score"),
                        "kite_user_id":        _cur_user_id,
                        "quantity":            _pqty,
                        "actual_entry":        _trigger_price,
                        "status":              "OPEN",
                        "notes":               f"📄 Paper — auto TRIGGERED @ ₹{_trigger_price:.2f} (conf {confidence}/10, ₹{_cap_this_trade:,})",
                        "is_paper_trade":      True,
                    })
                    st.session_state["paper_triggered"][_paper_key] = _pid
                    st.session_state["paper_open"][_pid] = {
                        "sym": sym, "stop": float(r.get("intraday_stop") or 0),
                        "t1": float(r.get("intraday_t1") or 0),
                        "t2": _t2_val,
                        "signal_type": "BUY_ABOVE", "entry": _trigger_price,
                        "cap": _cap_this_trade,
                        "partial_booked": False,
                    }
                    _confirm_map.pop(sym, None)  # reset timer after firing
                    st.toast(f"📄 Paper BUY [{_conf_tier}]: {sym} @ ₹{_trigger_price:.2f} × {_pqty} (confirmed {_ENTRY_CONFIRM_SECS}s)", icon="📄")
                except Exception:
                    pass

            # ── REAL mode ─────────────────────────────────────────────────────
            elif (_trade_mode == "real"
                    and not st.session_state.get("real_day_blocked", False)
                    and _real_key not in st.session_state.get("real_triggered", {})):
                _kc_rt = st.session_state.get("kite_client")
                if _kc_rt and getattr(_kc_rt, "authenticated", False):
                    try:
                        _stop_val = float(r.get("intraday_stop") or 0)
                        # Place entry LIMIT order first; SL placed only after fill
                        _oid = _kc_rt.place_order(
                            tradingsymbol    = sym,
                            qty              = _pqty,
                            transaction_type = "BUY",
                            order_type       = "LIMIT",
                            product          = "MIS",
                            price            = round(_trigger_price * 1.001, 1),
                            tag              = "scr_intra",
                        )
                        # SL-M companion order — placed immediately after entry order.
                        # In practice Kite fills near-instantly for liquid stocks so
                        # this is safe. For a stricter guarantee, poll order status
                        # and place SL only once status = "COMPLETE".
                        _sl_oid = None
                        if _stop_val:
                            try:
                                _sl_oid = _kc_rt.place_order(
                                    tradingsymbol    = sym,
                                    qty              = _pqty,
                                    transaction_type = "SELL",
                                    order_type       = "SL-M",
                                    product          = "MIS",
                                    trigger_price    = _stop_val,
                                    tag              = "scr_sl",
                                )
                            except Exception:
                                pass
                        _rid = db.log_trade({
                            "trade_date":          _dt.date.today(),
                            "tradingsymbol":       sym,
                            "instrument_token":    int(r.get("instrument_token") or 0),
                            "setup_type":          "INTRADAY",
                            "signal_type":         "BUY_ABOVE",
                            "rec_entry":           entry_val,
                            "rec_stop":            _stop_val,
                            "rec_t1":              float(r.get("intraday_t1") or 0),
                            "rec_rr":              None,
                            "rec_reason":          str(r.get("intraday_reason") or "")[:200],
                            "rec_composite_score": r.get("composite_score"),
                            "kite_user_id":        _cur_user_id,
                            "kite_order_id":       _oid,
                            "kite_sl_order_id":    _sl_oid,
                            "kite_status":         "OPEN",
                            "quantity":            _pqty,
                            "actual_entry":        _trigger_price,
                            "status":              "OPEN",
                            "notes":               f"💸 Real — auto TRIGGERED @ ₹{_trigger_price:.2f} (conf {confidence}/10, ₹{_cap_this_trade:,})",
                            "is_paper_trade":      False,
                        })
                        st.session_state["real_triggered"][_real_key] = _rid
                        _confirm_map.pop(sym, None)  # reset timer after firing
                        st.toast(f"💸 Real BUY [{_conf_tier}]: {sym} @ ₹{_trigger_price:.2f} × {_pqty} | Kite {_oid}", icon="💸")
                    except Exception as _re:
                        st.toast(f"⚠ Real order failed for {sym}: {_re}", icon="⚠️")

        # Compute R/R for display
        try:
            _e = float(r.get("intraday_entry") or 0)
            _s = float(r.get("intraday_stop")  or 0)
            _t = float(r.get("intraday_t1")    or 0)
            _rr_val = round((_t - _e) / (_e - _s), 1) if (_e > _s > 0 and _t > _e) else None
            _rr_str = f"{_rr_val:.1f}×" if _rr_val else "—"
        except Exception:
            _rr_str = "—"

        _conf_str = f"{confidence}/10" if confidence else "—"
        _gap_flag = r.get("intraday_gap_flag") or ""
        _nifty_gate = r.get("intraday_nifty_gate") or ""
        _flags = []
        if _gap_flag == "GAP_WARN":    _flags.append("⚠️GAP")
        if _nifty_gate:                _flags.append("🔴NF")
        _flag_str = " ".join(_flags) if _flags else "✓"

        _si_rows.append([
            live_status,
            sym,
            r.get("company_name", ""),
            _fmt(ltp_now,                  "₹{:,.2f}"),
            _delta_str(sym),
            _conf_str,
            _flag_str,
            _gain_pct(r.get("intraday_entry"), r.get("intraday_t1")),
            _risk_pct(r.get("intraday_entry"), r.get("intraday_stop")),
            _rr_str,
            _fmt(r.get("intraday_entry"),  "₹{:,.2f}"),
            _fmt(r.get("intraday_stop"),   "₹{:,.2f}"),
            _fmt(r.get("intraday_t1"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_t2") or r.get("intraday_r3"), "₹{:,.2f}"),
            _fmt(r.get("intraday_pivot"),  "₹{:,.2f}"),
            _fmt(r.get("intraday_r1"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_r2"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_s1"),     "₹{:,.2f}"),
        ])

    _si_df = pd.DataFrame(_si_rows, columns=[
        "Status", "Symbol", "Company", "LTP", "Δ",
        "Conf", "Flags", "Gain %", "Risk %", "R/R",
        "Buy Above", "Stop", "T1 (R2)", "T2 (R3)", "Pivot", "R1", "R2", "S1",
    ])
    st.dataframe(
        _si_df.style
            .map(_long_status_color, subset=["Status"])
            .map(_delta_color,       subset=["Δ"])
            .map(_conf_color,        subset=["Conf"])
            .map(_gain_color,        subset=["Gain %"])
            .map(_risk_color,        subset=["Risk %"])
            .map(_gain_color,        subset=["R/R"]),
        use_container_width=True,
        height=min(450, 50 + len(_si_rows) * 38),
        hide_index=True,
        column_config={
            "Status":    st.column_config.TextColumn("Status",
                help=(
                    "Live signal state (refreshes every 2 s):\n\n"
                    "🟢 TRIGGERED — LTP is already above the buy-stop entry. "
                    "The breakout has fired. Enter if momentum is still strong; "
                    "do NOT chase if it has run >1% past entry.\n\n"
                    "🟡 APPROACHING — LTP is within 0.5% of R1. "
                    "Breakout could happen any moment. "
                    "Set your buy-stop order NOW.\n\n"
                    "🔴 BROKEN — LTP fell below S1. "
                    "Bullish structure has failed for today. Do not enter long.\n\n"
                    "⚪ WATCHING — LTP is between S1 and R1. "
                    "No action yet. Monitor for approach to R1."
                )),
            "Δ":         st.column_config.TextColumn("Δ",
                help="LTP change since last 2-second refresh tick. "
                     "▲ green = price rising, ▼ red = price falling."),
            "Conf":      st.column_config.TextColumn("Conf",
                help=(
                    "Signal confidence score (0–10) — composite of R/R, RSI zone, "
                    "volume expansion, relative strength vs Nifty, and composite score.\n\n"
                    "🟢 8–10 STRONG   → auto-trade ₹2,00,000/slot\n"
                    "🟡 6–7 MODERATE  → auto-trade ₹1,50,000/slot\n"
                    "🟠 5   MARGINAL  → auto-trade ₹1,00,000/slot (if capital available)\n"
                    "🔴 <5  LOW       → shown only, NOT auto-traded"
                )),
            "Flags":     st.column_config.TextColumn("Flags",
                help=(
                    "Context warnings for this signal:\n\n"
                    "⚠️GAP — Today's open is 0.8–1.5% above R1 (gap-up risk: "
                    "price may fade after open gap). Reduce size.\n"
                    "🔴NF — Nifty 50 is down >0.6% today (market headwind for longs). "
                    "Consider skipping or waiting for Nifty to stabilise.\n"
                    "✓ — No flags, all context signals are favourable."
                )),
            "Gain %":    st.column_config.TextColumn("Gain %",
                help="Potential upside from Buy-Above entry to T1 (R2) target. "
                     "Typical range: 0.8%–3%."),
            "Risk %":    st.column_config.TextColumn("Risk %",
                help="Distance from entry to stop-loss (0.3% below R1). "
                     "Tight stop — if price falls back below R1 the breakout failed."),
            "R/R":       st.column_config.TextColumn("R/R",
                help="Risk/Reward ratio = Gain ÷ Risk. "
                     "Only signals with R/R ≥ 1.5× are shown. "
                     "Aim for 2×+ for best setups."),
            "Buy Above": st.column_config.TextColumn("Buy Above",
                help="Your entry trigger. Place a buy-stop order 0.1% above R1. "
                     "Execute only when price actually trades above this level."),
            "Stop":      st.column_config.TextColumn("Stop",
                help="Hard intraday stop-loss, set just below Pivot. "
                     "If price closes below Pivot after you enter, exit immediately."),
            "T1 (R2)":  st.column_config.TextColumn("T1 (R2)",
                help="First target = R2. On auto-trade: 60% of position exits here. "
                     "Stop moves to break-even. Remaining 40% trails to T2."),
            "T2 (R3)":  st.column_config.TextColumn("T2 (R3)",
                help="Second target = R3. Remaining 40% trails here after T1 is hit. "
                     "Hard exit at 3:10 PM regardless."),
            "Pivot":     st.column_config.TextColumn("Pivot",
                help="(Prev H + L + C) / 3. Above Pivot = bullish bias."),
            "R1":        st.column_config.TextColumn("R1",
                help="Resistance 1. Break above = intraday long trigger."),
            "R2":        st.column_config.TextColumn("R2",
                help="Resistance 2. T1 intraday target."),
            "S1":        st.column_config.TextColumn("S1",
                help="Support 1. Break below invalidates the long setup."),
        },
    )
    st.markdown(
        """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px 20px;margin-top:8px">
<div style="font-size:0.78rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;margin-bottom:10px">STATUS GUIDE — INTRADAY LONG</div>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
<tr>
  <td style="padding:5px 12px 5px 0;color:#22c55e;font-weight:700;white-space:nowrap">🟢 TRIGGERED</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP already <b>above Buy-Above entry</b>. Trade is live. Enter only if still within 1% of entry — otherwise wait for next pullback.</td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#f59e0b;font-weight:700;white-space:nowrap">🟡 APPROACHING</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP is <b>within 0.5% of R1</b>. Breakout is imminent. <b>Place your buy-stop order now</b> at the "Buy Above" price.</td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#ef4444;font-weight:700;white-space:nowrap">🔴 BROKEN</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP fell <b>below S1</b>. Intraday bullish structure has collapsed. <b>Do not enter long today.</b></td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#94a3b8;font-weight:700;white-space:nowrap">⚪ WATCHING</td>
  <td style="padding:5px 0;color:#cbd5e1">Price is consolidating between S1 and R1. No action yet — monitor and wait for R1 approach.</td>
</tr>
</table>
<div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e293b;color:#64748b;font-size:0.76rem">
⏰ <b>Hard exit rule:</b> Close ALL intraday positions by <b>3:10 PM IST</b> regardless of P&L. Kite auto-squares at 3:20 PM. &nbsp;|&nbsp; Levels refresh live every 2 s during market hours.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    _si_log = _si.assign(
        rec_entry=_si.get("intraday_entry"),
        rec_stop=_si.get("intraday_stop"),
        rec_t1=_si.get("intraday_t1"),
        rec_t2=None,
        rec_rr=None,
        rec_reason=_si.get("intraday_reason"),
        signal_type="BUY_ABOVE",
    ) if not _si.empty else _si
    _render_order_panel(_si_log, setup_type="INTRADAY", form_key="intra_long")

    # ── Market Intel BUY overlay ──────────────────────────────────────────────
    _intel_stks = st.session_state.get("_intel_stocks_cache", [])
    _buy_intel  = [s for s in _intel_stks if s.get("stance") in (_mi.BUY, _mi.BUY_ON_COND)]
    _avoid_intel_syms = {s["tradingsymbol"].upper() for s in _intel_stks if s.get("stance") == _mi.AVOID}

    # Annotate screener symbols with intel overlap badge (shown in caption above table)
    _screener_syms = set(_si["tradingsymbol"].str.upper().tolist()) if not _si.empty else set()
    _overlap_buy   = [s for s in _buy_intel if s["tradingsymbol"].upper() in _screener_syms]
    _intel_only    = [s for s in _buy_intel if s["tradingsymbol"].upper() not in _screener_syms]
    _avoid_overlap = [s for s in _intel_stks if s.get("stance") == _mi.AVOID and s["tradingsymbol"].upper() in _screener_syms]

    if _buy_intel or _avoid_overlap:
        st.markdown("---")
        st.markdown("#### 🧠 Market Intel — Long Signals")

        if _overlap_buy:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #22c55e33;border-radius:6px;'
                f'padding:6px 14px;margin-bottom:6px;font-size:0.78rem;color:#86efac">'
                f'🔥 <b>High-confidence overlap:</b> '
                + ", ".join(f"<b>{s['tradingsymbol']}</b>" for s in _overlap_buy) +
                f' — in BOTH screener signals AND Market Intel BUY</div>',
                unsafe_allow_html=True,
            )

        if _avoid_overlap:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #ef444433;border-radius:6px;'
                f'padding:6px 14px;margin-bottom:6px;font-size:0.78rem;color:#f87171">'
                f'⚠️ <b>AVOID warning (confidence reduced −2):</b> '
                + ", ".join(
                    f"<b>{s['tradingsymbol']}</b> — {s['fundamental_reason'][:60]}"
                    for s in _avoid_overlap
                ) +
                f'</div>',
                unsafe_allow_html=True,
            )

        if _intel_only or _buy_intel:
            _intel_rows = []
            for s in (_overlap_buy + _intel_only):
                _badge = "🔥 Both++" if s["tradingsymbol"].upper() in _screener_syms else "🧠 Intel only"
                _stance_lbl = "BUY" if s["stance"] == _mi.BUY else "BUY ON COND"
                _intel_rows.append({
                    "Source":         _badge,
                    "Symbol":         s["tradingsymbol"],
                    "Stance":         _stance_lbl,
                    "Sector":         s.get("sector", "")[:25],
                    "Why":            s.get("fundamental_reason", "")[:100],
                    "Entry Trigger":  s.get("entry_trigger", "")[:80],
                    "Stop Loss":      s.get("stop_loss", "")[:40],
                    "Conviction":     s.get("conviction", ""),
                    "Condition":      s.get("condition_required", "")[:60] if s["stance"] == _mi.BUY_ON_COND else "—",
                })
            if _intel_rows:
                st.dataframe(pd.DataFrame(_intel_rows), hide_index=True, use_container_width=True)
                st.caption(
                    "🧠 = Market Intel only  ·  🔥 = Both screener + intel (Very High confidence)  "
                    "·  Intel signals do NOT auto-trade — execute manually based on your judgement."
                )


# ── FRAGMENT 2b: intraday SHORT table (live status column) ───────────────────
@st.fragment(run_every=2)
def _intraday_short_live():
    # Skip all work outside market hours — fragment still runs but is instant
    if not _is_market_open():
        return
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        return
    df_s = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_s.loc[df_s["tradingsymbol"] == _sym, "ltp"] = _price

    _ss = (
        df_s[df_s["intraday_signal"] == "SELL_BELOW"]
        .sort_values("composite_score", ascending=True)
        .reset_index(drop=True)
    )
    if _ss.empty:
        st.info("No intraday short setups for today's session.")
        return

    # ── LTP freshness guard ───────────────────────────────────────────────────
    _ltp_ts_s  = st.session_state.get("_live_ltp_ts")
    _ltp_age_s = (datetime.now(_IST) - _ltp_ts_s).total_seconds() if _ltp_ts_s else 9999
    _ltp_stale = _ltp_age_s > config.LTP_FRESHNESS_SECS
    if _ltp_stale:
        _age_str_s = f"{int(_ltp_age_s)}s" if _ltp_age_s < 120 else f"{int(_ltp_age_s/60)}m"
        st.info(
            f"⏸ Prices paused while tab was inactive ({_age_str_s} ago). "
            "Refreshing now — auto-trading resumes once fresh prices arrive.",
            icon="🔄",
        )

    _short_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter…",
        key="intra_short_sym_search", label_visibility="collapsed",
    )
    if _short_sym_q:
        _ss = _ss[_ss["tradingsymbol"].str.contains(_short_sym_q.strip(), case=False, na=False, regex=False)]

    st.caption(
        "Watch for price to break **below S1**. "
        "Short with cover-stop just above Pivot. "
        "T1=S2 (60% exit), T2=S3 (trail 40%). "
        "**Only for stocks eligible for intraday short selling (check Kite margin).**"
    )

    # ── Nifty direction gate banner ──────────────────────────────────────────
    _nifty_pct_s = st.session_state.get("_nifty_intraday_pct", 0.0) or 0.0
    if _nifty_pct_s >= config.NIFTY_GATE_PCT:
        st.warning(
            f"⚠️ **Nifty headwind for shorts:** Nifty 50 is up **{_nifty_pct_s:.2f}%** today "
            f"(threshold: +{config.NIFTY_GATE_PCT}%). Short signals have headwind — "
            "consider reducing size or waiting for market reversal.",
            icon="🔴",
        )
    elif _nifty_pct_s <= -config.NIFTY_GATE_PCT:
        st.success(
            f"✅ **Nifty tailwind for shorts:** Nifty 50 is down **{abs(_nifty_pct_s):.2f}%** today — "
            "market breadth favours short positions.",
            icon="🟢",
        )

    _ss_rows = []
    for _, r in _ss.iterrows():
        sym        = r.get("tradingsymbol", "")
        ltp_now    = r.get("ltp") or 0
        s1_val     = r.get("intraday_s1") or 0
        entry_val  = r.get("intraday_entry") or 0
        piv_val    = float(r.get("intraday_pivot") or 0)
        _conf_raw  = r.get("intraday_confidence")
        confidence = int(_conf_raw) if not _isna(_conf_raw) else 0
        if s1_val and ltp_now:
            if ltp_now <= entry_val:
                _cs_map_s = st.session_state.get("_entry_confirm_since", {})
                _cs_ts_s  = _cs_map_s.get(f"_short_{sym}")
                if _cs_ts_s is not None:
                    _elapsed_s = (datetime.now(_IST) - _cs_ts_s).total_seconds()
                    if _elapsed_s < _ENTRY_CONFIRM_SECS:
                        _rem_s = int(_ENTRY_CONFIRM_SECS - _elapsed_s)
                        short_status = f"CONFIRMING {_rem_s}s"
                    else:
                        short_status = "TRIGGERED"
                else:
                    short_status = "TRIGGERED"
            elif ltp_now <= s1_val * 1.005:
                short_status = "APPROACHING"
            elif piv_val and ltp_now > piv_val:
                short_status = "REVERSED"
            else:
                short_status = "WATCHING"
        else:
            short_status = "—"

        # ── Confidence-based capital tier ─────────────────────────────────────
        if confidence >= config.CONFIDENCE_STRONG_MIN:
            _cap_this_trade = config.PAPER_CAP_STRONG
            _conf_tier = "STRONG"
        elif confidence >= config.CONFIDENCE_MODERATE_MIN:
            _cap_this_trade = config.PAPER_CAP_MODERATE
            _conf_tier = "MODERATE"
        elif confidence >= config.CONFIDENCE_MARGINAL_MIN:
            _cap_this_trade = config.PAPER_CAP_MARGINAL
            _conf_tier = "MARGINAL"
        else:
            _cap_this_trade = 0
            _conf_tier = "LOW"

        # ── Capital availability gate ─────────────────────────────────────────
        _deployed = sum(
            v.get("cap", config.PAPER_CAP_MODERATE)
            for v in st.session_state.get("paper_open", {}).values()
        )
        _remaining = config.PAPER_CAPITAL - _deployed
        _within_limit = (_cap_this_trade > 0) and (_remaining >= _cap_this_trade)

        _trigger_price = ltp_now if ltp_now > 0 else entry_val
        _pqty = max(1, int(_cap_this_trade / (_trigger_price or 1))) if _cap_this_trade else 1

        # ── Auto-trade when TRIGGERED ─────────────────────────────────────────
        _trade_mode = st.session_state.get("trading_mode", "paper")
        _paper_key  = (_today_str, sym)
        _real_key   = (_today_str, sym)

        # ── False-breakout confirmation timer (shorts) ───────────────────────
        _confirm_map_s = st.session_state.setdefault("_entry_confirm_since", {})
        _short_key = f"_short_{sym}"
        if short_status == "TRIGGERED":
            if _short_key not in _confirm_map_s:
                _confirm_map_s[_short_key] = datetime.now(_IST)
        else:
            _confirm_map_s.pop(_short_key, None)

        _confirm_ts_s = _confirm_map_s.get(_short_key)
        _confirmed_s  = (
            _confirm_ts_s is not None and
            (datetime.now(_IST) - _confirm_ts_s).total_seconds() >= _ENTRY_CONFIRM_SECS
        )

        if short_status == "TRIGGERED" and _confirmed_s and _within_limit and not _ltp_stale:

            # ── PAPER mode ────────────────────────────────────────────────────
            if (_trade_mode == "paper"
                    and not st.session_state.get("paper_day_blocked", False)
                    and _paper_key not in st.session_state.get("paper_triggered", {})):
                try:
                    _t2_val_s = float(r.get("intraday_t2") or r.get("intraday_s3") or 0)
                    _pid = db.log_trade({
                        "trade_date":          _dt.date.today(),
                        "tradingsymbol":       sym,
                        "instrument_token":    int(r.get("instrument_token") or 0),
                        "setup_type":          "INTRADAY",
                        "signal_type":         "SELL_BELOW",
                        "rec_entry":           entry_val,
                        "rec_stop":            float(r.get("intraday_stop") or 0),
                        "rec_t1":              float(r.get("intraday_t1") or 0),
                        "rec_t2":              _t2_val_s,
                        "rec_rr":              None,
                        "rec_reason":          str(r.get("intraday_reason") or "")[:200],
                        "rec_composite_score": r.get("composite_score"),
                        "kite_user_id":        _cur_user_id,
                        "quantity":            _pqty,
                        "actual_entry":        _trigger_price,
                        "status":              "OPEN",
                        "notes":               f"📄 Paper — auto TRIGGERED @ ₹{_trigger_price:.2f} (conf {confidence}/10, ₹{_cap_this_trade:,})",
                        "is_paper_trade":      True,
                    })
                    st.session_state["paper_triggered"][_paper_key] = _pid
                    st.session_state["paper_open"][_pid] = {
                        "sym": sym, "stop": float(r.get("intraday_stop") or 0),
                        "t1": float(r.get("intraday_t1") or 0),
                        "t2": _t2_val_s,
                        "signal_type": "SELL_BELOW", "entry": _trigger_price,
                        "cap": _cap_this_trade,
                        "partial_booked": False,
                    }
                    _confirm_map_s.pop(_short_key, None)
                    st.toast(f"📄 Paper SHORT [{_conf_tier}]: {sym} @ ₹{_trigger_price:.2f} × {_pqty} (confirmed {_ENTRY_CONFIRM_SECS}s)", icon="📄")
                except Exception:
                    pass

            # ── REAL mode ─────────────────────────────────────────────────────
            elif (_trade_mode == "real"
                    and not st.session_state.get("real_day_blocked", False)
                    and _real_key not in st.session_state.get("real_triggered", {})):
                _kc_rt = st.session_state.get("kite_client")
                if _kc_rt and getattr(_kc_rt, "authenticated", False):
                    try:
                        _stop_val = float(r.get("intraday_stop") or 0)
                        # Place entry LIMIT sell order first
                        _oid = _kc_rt.place_order(
                            tradingsymbol    = sym,
                            qty              = _pqty,
                            transaction_type = "SELL",
                            order_type       = "LIMIT",
                            product          = "MIS",
                            price            = round(_trigger_price * 0.999, 1),
                            tag              = "scr_intra",
                        )
                        # SL-M cover order placed after entry
                        _sl_oid = None
                        if _stop_val:
                            try:
                                _sl_oid = _kc_rt.place_order(
                                    tradingsymbol    = sym,
                                    qty              = _pqty,
                                    transaction_type = "BUY",
                                    order_type       = "SL-M",
                                    product          = "MIS",
                                    trigger_price    = _stop_val,
                                    tag              = "scr_sl",
                                )
                            except Exception:
                                pass
                        _rid = db.log_trade({
                            "trade_date":          _dt.date.today(),
                            "tradingsymbol":       sym,
                            "instrument_token":    int(r.get("instrument_token") or 0),
                            "setup_type":          "INTRADAY",
                            "signal_type":         "SELL_BELOW",
                            "rec_entry":           entry_val,
                            "rec_stop":            _stop_val,
                            "rec_t1":              float(r.get("intraday_t1") or 0),
                            "rec_rr":              None,
                            "rec_reason":          str(r.get("intraday_reason") or "")[:200],
                            "rec_composite_score": r.get("composite_score"),
                            "kite_user_id":        _cur_user_id,
                            "kite_order_id":       _oid,
                            "kite_sl_order_id":    _sl_oid,
                            "kite_status":         "OPEN",
                            "quantity":            _pqty,
                            "actual_entry":        _trigger_price,
                            "status":              "OPEN",
                            "notes":               f"💸 Real — auto TRIGGERED @ ₹{_trigger_price:.2f} (conf {confidence}/10, ₹{_cap_this_trade:,})",
                            "is_paper_trade":      False,
                        })
                        st.session_state["real_triggered"][_real_key] = _rid
                        _confirm_map_s.pop(_short_key, None)
                        st.toast(f"💸 Real SHORT [{_conf_tier}]: {sym} @ ₹{_trigger_price:.2f} × {_pqty} | Kite {_oid}", icon="💸")
                    except Exception as _re:
                        st.toast(f"⚠ Real short order failed for {sym}: {_re}", icon="⚠️")

        # For shorts: gain = entry → T1 (price drops), risk = entry → stop (price rises)
        try:
            _se = float(r.get("intraday_entry") or 0)
            _ss2 = float(r.get("intraday_stop")  or 0)
            _st1 = float(r.get("intraday_t1")    or 0)
            _rr_val = round((_se - _st1) / (_ss2 - _se), 1) if (_ss2 > _se > _st1 > 0) else None
            _rr_str = f"{_rr_val:.1f}×" if _rr_val else "—"
        except Exception:
            _rr_str = "—"

        _conf_str = f"{confidence}/10" if confidence else "—"
        _ss_rows.append([
            short_status,
            sym,
            r.get("company_name", ""),
            _fmt(ltp_now,                  "₹{:,.2f}"),
            _delta_str(sym),
            _conf_str,
            _gain_pct(r.get("intraday_t1"), r.get("intraday_entry")),   # short: t1 < entry
            _risk_pct(r.get("intraday_stop"), r.get("intraday_entry")), # short: stop > entry
            _rr_str,
            _fmt(r.get("intraday_entry"),  "₹{:,.2f}"),
            _fmt(r.get("intraday_stop"),   "₹{:,.2f}"),
            _fmt(r.get("intraday_t1"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_pivot"),  "₹{:,.2f}"),
            _fmt(r.get("intraday_s1"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_s2"),     "₹{:,.2f}"),
            _fmt(r.get("intraday_r1"),     "₹{:,.2f}"),
        ])

    _ss_df = pd.DataFrame(_ss_rows, columns=[
        "Status", "Symbol", "Company", "LTP", "Δ",
        "Conf", "Gain %", "Risk %", "R/R",
        "Sell Below", "Cover Stop", "T1 (S2)", "Pivot", "S1", "S2", "R1",
    ])
    st.dataframe(
        _ss_df.style
            .map(_short_status_color, subset=["Status"])
            .map(_delta_color,        subset=["Δ"])
            .map(_conf_color,         subset=["Conf"])
            .map(_gain_color,         subset=["Gain %"])
            .map(_risk_color,         subset=["Risk %"])
            .map(_gain_color,         subset=["R/R"]),
        use_container_width=True,
        height=min(450, 50 + len(_ss_rows) * 38),
        hide_index=True,
        column_config={
            "Status":     st.column_config.TextColumn("Status",
                help=(
                    "Live signal state for the short setup (refreshes every 2 s):\n\n"
                    "🔴 TRIGGERED — LTP is already below the Sell-Below entry. "
                    "Breakdown confirmed. Short is live. Trail your cover-stop downward.\n\n"
                    "🟡 APPROACHING — LTP is within 0.5% above S1. "
                    "Breakdown is imminent. Be ready to place a sell/short order.\n\n"
                    "🟢 REVERSED — LTP recovered above Pivot. "
                    "Bearish setup has failed. Do NOT short — cover any existing short.\n\n"
                    "⚪ WATCHING — LTP is between Pivot and S1. "
                    "Bearish bias but no trigger yet. Monitor only."
                )),
            "Δ":          st.column_config.TextColumn("Δ",
                help="LTP change since last 2-second refresh tick. "
                     "▲ green = price rising (bad for shorts), ▼ red = price falling (good)."),
            "Conf":       st.column_config.TextColumn("Conf",
                help=(
                    "Signal confidence score (0–10) — composite of R/R, RSI zone, "
                    "volume expansion, relative strength vs Nifty, and composite score.\n\n"
                    "🟢 8–10 STRONG   → auto-trade ₹2,00,000/slot\n"
                    "🟡 6–7 MODERATE  → auto-trade ₹1,50,000/slot\n"
                    "🟠 5   MARGINAL  → auto-trade ₹1,00,000/slot (if capital available)\n"
                    "🔴 <5  LOW       → shown only, NOT auto-traded"
                )),
            "Gain %":     st.column_config.TextColumn("Gain %",
                help="Potential profit from Sell-Below entry to T1/S2 target (short direction)."),
            "Risk %":     st.column_config.TextColumn("Risk %",
                help="Distance from entry to cover-stop (0.3% above S1). "
                     "Tight — if price recovers above S1 the breakdown failed."),
            "R/R":        st.column_config.TextColumn("R/R",
                help="Risk/Reward ratio for the short trade. "
                     "Only signals with R/R ≥ 1.5× are shown."),
            "Sell Below": st.column_config.TextColumn("Sell Below",
                help="Short entry trigger — place a sell-stop order 0.1% below S1."),
            "Cover Stop": st.column_config.TextColumn("Cover Stop",
                help="Buy-to-cover stop just above Pivot. Cover if price recovers above Pivot."),
            "T1 (S2)":   st.column_config.TextColumn("T1 (S2)",
                help="First intraday short target = S2. Cover full short here."),
            "Pivot":      st.column_config.TextColumn("Pivot",
                help="(Prev H + L + C) / 3. Below Pivot = bearish bias."),
            "S1":         st.column_config.TextColumn("S1",
                help="Support 1. Break below = short trigger."),
            "S2":         st.column_config.TextColumn("S2",
                help="Support 2. Intraday short target."),
            "R1":         st.column_config.TextColumn("R1",
                help="Resistance 1. If price runs above R1, cover the short."),
        },
    )
    st.markdown(
        """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px 20px;margin-top:8px">
<div style="font-size:0.78rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;margin-bottom:10px">STATUS GUIDE — INTRADAY SHORT</div>
<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
<tr>
  <td style="padding:5px 12px 5px 0;color:#ef4444;font-weight:700;white-space:nowrap">🔴 TRIGGERED</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP already <b>below Sell-Below entry</b>. Breakdown is confirmed, short is live. Enter only if still within 1% of entry — trail cover-stop down as price falls.</td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#f59e0b;font-weight:700;white-space:nowrap">🟡 APPROACHING</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP is <b>within 0.5% above S1</b>. Breakdown is imminent. <b>Place your sell-stop order now</b> at the "Sell Below" price.</td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#22c55e;font-weight:700;white-space:nowrap">🟢 REVERSED</td>
  <td style="padding:5px 0;color:#cbd5e1">LTP recovered <b>above Pivot</b>. Bearish setup has failed. <b>Do not short.</b> Cover any existing short immediately.</td>
</tr>
<tr>
  <td style="padding:5px 12px 5px 0;color:#94a3b8;font-weight:700;white-space:nowrap">⚪ WATCHING</td>
  <td style="padding:5px 0;color:#cbd5e1">Price is between Pivot and S1. Bearish bias exists but no trigger yet — monitor for approach to S1.</td>
</tr>
</table>
<div style="margin-top:10px;padding-top:8px;border-top:1px solid #1e293b;color:#64748b;font-size:0.76rem">
⏰ <b>Hard exit rule:</b> Cover ALL short positions by <b>3:10 PM IST</b>. &nbsp;|&nbsp;
⚠ <b>Check Kite margin:</b> Not all stocks are eligible for intraday short selling. Verify availability in your Kite account before placing orders. &nbsp;|&nbsp; Levels refresh live every 2 s.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    _ss_log = _ss.assign(
        rec_entry=_ss["intraday_entry"],
        rec_stop=_ss["intraday_stop"],
        rec_t1=_ss["intraday_t1"],
        rec_t2=None,
        rec_rr=None,
        rec_reason=_ss["intraday_reason"],
        signal_type="SELL_BELOW",
    ) if not _ss.empty else _ss
    _render_order_panel(_ss_log, setup_type="INTRADAY", form_key="intra_short")

    # ── Market Intel SHORT overlay ────────────────────────────────────────────
    _intel_stks_s   = st.session_state.get("_intel_stocks_cache", [])
    _short_intel    = [s for s in _intel_stks_s if s.get("stance") == _mi.SHORT]
    _avoid_intel_s  = {s["tradingsymbol"].upper() for s in _intel_stks_s if s.get("stance") == _mi.AVOID}
    _screener_syms_s = set(_ss["tradingsymbol"].str.upper().tolist()) if not _ss.empty else set()
    _overlap_short   = [s for s in _short_intel if s["tradingsymbol"].upper() in _screener_syms_s]
    _intel_only_s    = [s for s in _short_intel if s["tradingsymbol"].upper() not in _screener_syms_s]
    _avoid_overlap_s = [s for s in _intel_stks_s if s.get("stance") == _mi.AVOID and s["tradingsymbol"].upper() in _screener_syms_s]

    if _short_intel or _avoid_overlap_s:
        st.markdown("---")
        st.markdown("#### 🧠 Market Intel — Short Signals")

        if _overlap_short:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #ef444433;border-radius:6px;'
                f'padding:6px 14px;margin-bottom:6px;font-size:0.78rem;color:#f87171">'
                f'🔥 <b>High-confidence overlap:</b> '
                + ", ".join(f"<b>{s['tradingsymbol']}</b>" for s in _overlap_short) +
                f' — in BOTH screener SELL_BELOW AND Market Intel SHORT</div>',
                unsafe_allow_html=True,
            )

        if _avoid_overlap_s:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #f59e0b33;border-radius:6px;'
                f'padding:6px 14px;margin-bottom:6px;font-size:0.78rem;color:#f59e0b">'
                f'⚠️ <b>AVOID warning (confidence reduced):</b> '
                + ", ".join(
                    f"<b>{s['tradingsymbol']}</b> — {s['fundamental_reason'][:60]}"
                    for s in _avoid_overlap_s
                ) +
                f'</div>',
                unsafe_allow_html=True,
            )

        _s_intel_rows = []
        for s in (_overlap_short + _intel_only_s):
            _badge_s = "🔥 Both++" if s["tradingsymbol"].upper() in _screener_syms_s else "🧠 Intel only"
            _s_intel_rows.append({
                "Source":          _badge_s,
                "Symbol":          s["tradingsymbol"],
                "Sector":          s.get("sector", "")[:25],
                "Why Short":       s.get("fundamental_reason", "")[:100],
                "Breakdown Trigger": s.get("entry_trigger", "")[:80],
                "Stop Loss":       s.get("stop_loss", "")[:40],
                "Conviction":      s.get("conviction", ""),
            })
        if _s_intel_rows:
            st.dataframe(pd.DataFrame(_s_intel_rows), hide_index=True, use_container_width=True)
            st.caption(
                "🧠 = Market Intel only  ·  🔥 = Both screener + intel (Very High confidence)  "
                "·  Intel signals do NOT auto-trade — execute manually."
            )


# ── EMA-dist colour helper (used by Exit/Sell fragment) ──────────────────────
def _dist_color(val):
    return "color:#f59e0b" if str(val).startswith("+") else "color:#94a3b8"


# ────────────────────────────────────────────────────────────────────────────
# Per-tab fragments — each one re-runs only when its own search input changes,
# so filtering is truly real-time without triggering a full-page rerun.
# ────────────────────────────────────────────────────────────────────────────

@st.fragment
def _swing_buy_tab_content():
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        return
    df_live = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_live.loc[df_live["tradingsymbol"] == _sym, "ltp"] = _price

    _sb = (
        df_live[df_live["swing_signal"] == "BUY"]
        .sort_values(["swing_quality", "composite_score"], ascending=False)
        .reset_index(drop=True)
    )
    if _sb.empty:
        st.info("No swing buy setups detected in the current universe.")
        return
    st.caption(
        "**How to use:** Place a limit/SL-M order at the Entry price. "
        "Set a hard stop-loss at Stop. Book 50% at T1, let the rest run to T2 "
        "with a trailing stop. Minimum R/R to consider: 1.5×."
    )
    _swing_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter...",
        key="swing_buy_sym_search", label_visibility="collapsed",
    )
    if _swing_sym_q:
        _sb = _sb[_sb["tradingsymbol"].str.contains(
            _swing_sym_q.strip(), case=False, na=False, regex=False)]
    _sb_rows = []
    for _, r in _sb.iterrows():
        _sb_rows.append([
            r.get("tradingsymbol", ""), r.get("company_name", ""),
            r.get("swing_setup", ""),
            _fmt(r.get("ltp"),         "₹{:,.2f}"),
            _gain_pct(r.get("swing_entry"), r.get("swing_t1")),
            _risk_pct(r.get("swing_entry"), r.get("swing_stop")),
            _fmt(r.get("swing_entry"),  "₹{:,.2f}"),
            _fmt(r.get("swing_stop"),   "₹{:,.2f}"),
            _fmt(r.get("swing_t1"),     "₹{:,.2f}"),
            _fmt(r.get("swing_t2"),     "₹{:,.2f}"),
            _fmt(r.get("swing_rr"),     "{:.2f}×"),
            _stars(r.get("swing_quality")),
            str(r.get("swing_reason", ""))[:120],
        ])
    _sb_df = pd.DataFrame(_sb_rows, columns=[
        "Symbol", "Company", "Setup", "LTP",
        "Gain %", "Risk %", "Entry", "Stop", "T1", "T2", "R/R", "Quality", "Reason",
    ])
    st.dataframe(
        _sb_df.style.map(_gain_color, subset=["Gain %"]).map(_risk_color, subset=["Risk %"]),
        use_container_width=True, height=min(400, 50 + len(_sb_rows) * 38), hide_index=True,
        column_config={
            "Setup":   st.column_config.TextColumn("Setup",   help="PULLBACK = retracement to EMA20 | BREAKOUT = 20D high break | NR7 = narrowest range coil"),
            "Gain %":  st.column_config.TextColumn("Gain %",  help="Upside to T1 from entry price. Swing trades typically target 5–15%."),
            "Risk %":  st.column_config.TextColumn("Risk %",  help="Downside from entry to hard stop-loss. Ideal risk per trade: 2–4%."),
            "Entry":   st.column_config.TextColumn("Entry",   help="Suggested entry price. For NR7 this is a buy-stop order above today's high."),
            "Stop":    st.column_config.TextColumn("Stop",    help="Hard stop-loss. Exit the full position if price closes below this level."),
            "T1":      st.column_config.TextColumn("T1",      help="First target = Entry + 2×ATR. Book 50% here."),
            "T2":      st.column_config.TextColumn("T2",      help="Second target = Entry + 4×ATR. Trail stop on the remaining 50%."),
            "R/R":     st.column_config.TextColumn("R/R",     help="Risk/Reward ratio at T1. Minimum 1.5× recommended."),
            "Quality": st.column_config.TextColumn("Quality", help="★★★★★ = best (full EMA stack + ADX confirmed + strong volume)"),
        },
    )
    _render_order_panel(_sb, setup_type="SWING", form_key="sw_buy")


@st.fragment
def _exit_sell_tab_content():
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        return
    df_live = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_live.loc[df_live["tradingsymbol"] == _sym, "ltp"] = _price

    _se = (
        df_live[df_live["swing_signal"] == "SELL"]
        .sort_values("composite_score", ascending=True)
        .reset_index(drop=True)
    )
    if _se.empty:
        st.info("No exit signals currently. All trend structures intact.")
        return
    st.caption(
        "**How to use:** These are stocks where the uptrend structure has broken or "
        "the stock is severely overbought. If you hold any of these, review your position."
    )
    _exit_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter...",
        key="exit_sym_search", label_visibility="collapsed",
    )
    if _exit_sym_q:
        _se = _se[_se["tradingsymbol"].str.contains(
            _exit_sym_q.strip(), case=False, na=False, regex=False)]
    _se_rows = []
    for _, r in _se.iterrows():
        ltp_v = r.get("ltp") or 0
        ema_v = r.get("ema_20") or 0
        try:
            dist_ema = f"+{(ltp_v - ema_v) / ema_v * 100:.1f}%" if ema_v > 0 else "—"
        except Exception:
            dist_ema = "—"
        _se_rows.append([
            r.get("tradingsymbol", ""), r.get("company_name", ""),
            _fmt(ltp_v, "₹{:,.2f}"), _fmt(ema_v, "₹{:,.2f}"),
            dist_ema, _fmt(r.get("rsi_14"), "{:.1f}"),
            str(r.get("swing_reason", "")),
        ])
    _se_df = pd.DataFrame(_se_rows, columns=[
        "Symbol", "Company", "LTP", "EMA20", "EMA20 Dist", "RSI", "Exit Reason",
    ])
    st.dataframe(
        _se_df.style.map(_dist_color, subset=["EMA20 Dist"]),
        use_container_width=True, height=min(400, 50 + len(_se_rows) * 38), hide_index=True,
        column_config={
            "EMA20 Dist": st.column_config.TextColumn("EMA20 Dist", help=(
                "How far LTP is above EMA20. A large positive value (+10%+) means the stock is "
                "stretched and likely to mean-revert — this is what triggers the SELL signal.")),
            "RSI": st.column_config.TextColumn("RSI", help=(
                "RSI > 70 = overbought. RSI > 80 = strongly overbought. "
                "Combined with a high EMA20 Dist, this confirms an exit is warranted.")),
        },
    )
    _se_log = _se.assign(
        rec_entry=_se.get("swing_entry"), rec_stop=_se.get("swing_stop"),
        rec_t1=_se.get("swing_t1"),       rec_t2=_se.get("swing_t2"),
        rec_rr=_se.get("swing_rr"),       rec_reason=_se.get("swing_reason"),
        signal_type=_se.get("swing_signal"),
    ) if not _se.empty else _se
    _render_order_panel(_se_log, setup_type="SWING", form_key="sw_exit")


@st.fragment
def _scaling_tab_content():
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        return
    df_live = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_live.loc[df_live["tradingsymbol"] == _sym, "ltp"] = _price

    _ssc = (
        df_live[df_live["scale_signal"] == "INITIAL_ENTRY"]
        .sort_values(["scale_quality", "composite_score"], ascending=False)
        .reset_index(drop=True)
    )
    if _ssc.empty:
        st.info("No scaling entry setups. Stocks need full EMA stack + EMA50 pullback + positive RS.")
        return
    st.caption(
        "**How to use:** Deploy **40% of intended position** at Entry 1 (EMA50 pullback). "
        "Add **30% more** when price breaks above the 20-day high. "
        "Add the final **30%** on a new 52W high with volume. "
        "Trailing stop: close below EMA50 on any day."
    )
    _scale_sym_q = st.text_input(
        "🔍 Search symbol", placeholder="type to filter...",
        key="scaling_sym_search", label_visibility="collapsed",
    )
    if _scale_sym_q:
        _ssc = _ssc[_ssc["tradingsymbol"].str.contains(
            _scale_sym_q.strip(), case=False, na=False, regex=False)]
    _ssc_rows = []
    for _, r in _ssc.iterrows():
        _ssc_rows.append([
            r.get("tradingsymbol", ""), r.get("company_name", ""),
            _fmt(r.get("ltp"),                 "₹{:,.2f}"),
            _gain_pct(r.get("scale_entry_1"), r.get("scale_target")),
            _risk_pct(r.get("scale_entry_1"), r.get("scale_stop")),
            _fmt(r.get("scale_entry_1"),       "₹{:,.2f}"),
            _fmt(r.get("scale_stop"),          "₹{:,.2f}"),
            _fmt(r.get("scale_trailing_stop"), "₹{:,.2f}"),
            _fmt(r.get("scale_target"),        "₹{:,.2f}"),
            _fmt(r.get("ret_6m"),              "{:+.1f}%"),
            _fmt(r.get("rs_vs_nifty_3m"),      "{:+.1f}%"),
            _stars(r.get("scale_quality")),
            str(r.get("scale_reason", ""))[:120],
        ])
    _ssc_df = pd.DataFrame(_ssc_rows, columns=[
        "Symbol", "Company", "LTP",
        "Gain %", "Risk %",
        "Entry 1 (40%)", "Hard Stop", "Trail Stop (EMA50)",
        "Target (+18%)", "6M Return", "RS vs Nifty",
        "Quality", "Reason",
    ])
    st.dataframe(
        _ssc_df.style.map(_gain_color, subset=["Gain %"]).map(_risk_color, subset=["Risk %"]),
        use_container_width=True, height=min(450, 50 + len(_ssc_rows) * 38), hide_index=True,
        column_config={
            "Gain %":             st.column_config.TextColumn("Gain %",     help="Upside from Entry 1 to the 18% target. Scaling trades typically run 15–25%."),
            "Risk %":             st.column_config.TextColumn("Risk %",     help="Downside from Entry 1 to hard stop. Acceptable for a position you intend to scale into."),
            "Entry 1 (40%)":      st.column_config.TextColumn("Entry 1",    help="0.5% above EMA50 — confirms EMA50 held as support. Deploy 40% of position here."),
            "Hard Stop":          st.column_config.TextColumn("Hard Stop",  help="1.5×ATR below EMA50. Full exit if price closes below this."),
            "Trail Stop (EMA50)": st.column_config.TextColumn("Trail Stop", help="Move this stop up every week as EMA50 rises. Exit if day close is below EMA50."),
            "Target (+18%)":      st.column_config.TextColumn("Target",     help="18% measured-move from Entry 1. Take 25–50% off here and trail the rest."),
            "Quality":            st.column_config.TextColumn("Quality",    help="★★★★★ = best. Requires full EMA stack + positive RS + strong 6M return."),
        },
    )
    _ssc_log = _ssc.assign(
        rec_entry=_ssc.get("scale_entry_1"), rec_stop=_ssc.get("scale_stop"),
        rec_t1=_ssc.get("scale_target"),     rec_t2=None,
        rec_rr=None,                          rec_reason=_ssc.get("scale_reason"),
        signal_type=_ssc.get("scale_signal"),
    ) if not _ssc.empty else _ssc
    _render_order_panel(_ssc_log, setup_type="SCALING", form_key="scaling")


# ── STABLE signal tables (no fragment — tabs never re-create themselves) ──────
def _signals_main():
    """
    Renders all 4 signal tabs. Deliberately NOT a fragment so that tab widgets
    are never re-created and their selected-tab state is always preserved.
    LTP is read from st.session_state["_live_ltp"] which the header fragment
    keeps fresh every 2 s.
    """
    base_df = st.session_state.get("_signals_base_df", pd.DataFrame())
    if base_df.empty:
        st.info("No data yet. Click 'Full Rescan' in the sidebar to bootstrap.", icon="ℹ️")
        return
    if "swing_signal" not in base_df.columns:
        st.info(
            "Signal columns not found. Run a Full Rescan to compute entry/exit signals.",
            icon="ℹ️",
        )
        return

    # Apply most-recent live LTP (updated by the header fragment every 2 s)
    df_live = base_df.copy()
    for _sym, _price in st.session_state.get("_live_ltp", {}).items():
        df_live.loc[df_live["tradingsymbol"] == _sym, "ltp"] = _price

    st.markdown("---")

    _sig_t1, _sig_t2, _sig_t3, _sig_t4 = st.tabs(
        ["Swing Buy", "Exit / Sell", "Intraday Plan", "Scaling"]
    )

    # ── SWING BUY ─────────────────────────────────────────────────
    with _sig_t1:
        _swing_buy_tab_content()   # fragment: live search, reruns only on input change

    # ── EXIT / SELL ───────────────────────────────────────────────
    with _sig_t2:
        _exit_sell_tab_content()   # fragment: live search

    # ── INTRADAY PLAN ─────────────────────────────────────────────
    with _sig_t3:
        st.caption(
            "**Hard rule: close ALL intraday positions by 3:10 PM regardless of P&L.** "
            "LTP and Status update automatically every 2 s during market hours."
        )

        # ── Trading mode control + kill switch ────────────────────────────────
        _trading_mode_control()

        # ── Paper trade live dashboard (auto-refresh fragment) ────────────────
        _intraday_paper_banner()

        _n_il = st.session_state.get("_n_intra_long",  0)
        _n_is = st.session_state.get("_n_intra_short", 0)
        _it_long_tab, _it_short_tab, _it_scalp_tab = st.tabs([
            f"📈 Long (BUY_ABOVE)  {_n_il}",
            f"📉 Short (SELL_BELOW)  {_n_is}",
            "⚡ Scalp (ORB)",
        ])
        with _it_long_tab:
            _intraday_long_live()   # fragment: live status, runs every 2 s
        with _it_short_tab:
            _intraday_short_live()  # fragment: live status, runs every 2 s
        with _it_scalp_tab:
            _intraday_scalp_live()  # fragment: ORB scalp signals, runs every 5 s

    # ── SCALING ───────────────────────────────────────────────────
    with _sig_t4:
        _scaling_tab_content()     # fragment: live search


with tab_signals:
    if st.session_state.get("_signals_base_df", pd.DataFrame()).empty:
        st.info("No data yet. Click 'Full Rescan' in the sidebar to bootstrap.", icon="ℹ️")
    elif "swing_signal" not in st.session_state.get("_signals_base_df", pd.DataFrame()).columns:
        st.info("Signal columns not found. Run a Full Rescan to compute entry/exit signals.", icon="ℹ️")
    else:
        _intel_poller()           # Fragment: checks background market intel job every 5 s
        _live_signals_header()   # Fragment: live clock + LTP fetch + metric pills (no tabs)
        _signals_main()          # Stable: all 4 tabs + tables (never re-creates tab widgets)


# ============================================================
# ALGORITHM READJUSTMENT DIALOG
# ============================================================
@st.dialog("🔬 Algorithm Readjustment Insights", width="large")
def _show_algo_readjust_dialog(uid: str) -> None:
    """Analyse archived paper trades and propose signal-config changes."""
    import datetime as _dt_mod

    df = db.get_archived_paper_trades_for_analysis(user_id=uid)
    if df.empty:
        st.info(
            "No archived paper trades found yet. "
            "This section populates from completed trades on past trading days.",
            icon="📭",
        )
        return

    # ── Numeric coercion ────────────────────────────────────────────────
    for _nc in ("pnl_amount", "pnl_pct", "rr_realised", "intraday_confidence",
                "actual_entry", "actual_exit", "rec_stop", "rec_t1", "intraday_rr"):
        if _nc in df.columns:
            df[_nc] = pd.to_numeric(df[_nc], errors="coerce")

    total      = len(df)
    wins       = int((df["pnl_amount"] > 0).sum())
    stopped    = int((df["status"] == "STOPPED_OUT").sum())
    target_hit = int((df["status"] == "TARGET_HIT").sum())
    win_rate   = wins / total * 100 if total else 0.0
    stop_rate  = stopped / total * 100 if total else 0.0
    th_rate    = target_hit / total * 100 if total else 0.0

    cur_cfg   = db.get_signal_config(user_id=uid)
    proposals = dict(cur_cfg)

    # ── Gap analysis ────────────────────────────────────────────────────
    gaps = []

    # 1. Stop-out rate
    if stop_rate > 55:
        gaps.append(
            f"🔴 **High stop-out rate ({stop_rate:.0f}%)** — more than half of trades are being "
            f"stopped out before reaching the target. Causes: stops too tight, ATR volatility higher "
            f"than expected, or entries on false breakouts. Consider raising the min R/R filter so only "
            f"wider-stop setups qualify."
        )
        proposals["intraday_min_rr"] = round(min(2.5, cur_cfg["intraday_min_rr"] + 0.25), 2)
    elif stop_rate < 20 and win_rate < 40:
        gaps.append(
            f"⚠️ **Stop-out rate is low ({stop_rate:.0f}%) but win rate is also low ({win_rate:.0f}%)** — "
            f"trades are not being stopped out but are closing at a loss (CLOSED status). "
            f"Check if exits at 3:10 PM hard-exit are realising losses instead of letting winners run."
        )

    # 2. Signal-type breakdown
    _buys  = df[df["signal_type"] == "BUY_ABOVE"]
    _sells = df[df["signal_type"] == "SELL_BELOW"]
    _buy_wr  = float((_buys["pnl_amount"] > 0).mean() * 100)  if len(_buys)  >= 3 else None
    _sell_wr = float((_sells["pnl_amount"] > 0).mean() * 100) if len(_sells) >= 3 else None

    if _buy_wr is not None:
        if _buy_wr < 40:
            gaps.append(
                f"🔴 **Long signals (BUY_ABOVE) win rate is {_buy_wr:.0f}%** — below acceptable threshold. "
                f"Market may be in a downtrend or RSI ceiling ({cur_cfg['intraday_rsi_buy_max']:.0f}) is "
                f"allowing overbought entries. Tightening RSI threshold."
            )
            proposals["intraday_rsi_buy_max"] = round(max(55.0, cur_cfg["intraday_rsi_buy_max"] - 5.0), 1)
        elif _buy_wr > 65:
            gaps.append(
                f"✅ **Long signals performing well ({_buy_wr:.0f}% win rate)** — "
                f"RSI buy threshold ({cur_cfg['intraday_rsi_buy_max']:.0f}) is well-calibrated."
            )

    if _sell_wr is not None:
        if _sell_wr < 40:
            gaps.append(
                f"🔴 **Short signals (SELL_BELOW) win rate is {_sell_wr:.0f}%** — "
                f"RSI floor ({cur_cfg['intraday_rsi_sell_min']:.0f}) may be too permissive. "
                f"Raising threshold to reduce oversold bounce traps."
            )
            proposals["intraday_rsi_sell_min"] = round(min(45.0, cur_cfg["intraday_rsi_sell_min"] + 5.0), 1)
        elif _sell_wr > 65:
            gaps.append(
                f"✅ **Short signals performing well ({_sell_wr:.0f}% win rate)** — threshold calibrated correctly."
            )

    # 3. Avg realised R/R vs planned
    _rrv = df["rr_realised"].dropna()
    _planned_rr = df["intraday_rr"].dropna() if "intraday_rr" in df.columns else pd.Series(dtype=float)
    if len(_rrv) >= 3:
        _avg_rr = float(_rrv.mean())
        _avg_plan = float(_planned_rr.mean()) if len(_planned_rr) >= 3 else None
        if _avg_rr < 1.0:
            gaps.append(
                f"⚠️ **Average realised R/R is {_avg_rr:.2f}x** (below 1:1) — "
                f"trades are exiting with losses larger than their gains. "
                f"{'Planned avg R/R was ' + f'{_avg_plan:.1f}x' + ', meaning targets are not being reached.' if _avg_plan else ''} "
                f"Raising the minimum R/R filter will pre-screen out lower-probability setups."
            )
            proposals["intraday_min_rr"] = round(min(2.5, max(proposals["intraday_min_rr"], cur_cfg["intraday_min_rr"] + 0.25)), 2)
        elif _avg_rr > 1.8 and win_rate > 55:
            gaps.append(
                f"✅ **Strong realised R/R of {_avg_rr:.2f}x** with {win_rate:.0f}% win rate — "
                f"current parameters are effective. Consider slightly relaxing min R/R to capture more setups."
            )
            if cur_cfg["intraday_min_rr"] > 1.5:
                proposals["intraday_min_rr"] = round(max(1.5, cur_cfg["intraday_min_rr"] - 0.25), 2)

    # 4. Confidence correlation
    if "intraday_confidence" in df.columns and df["intraday_confidence"].notna().sum() >= 4:
        _hc = df[df["intraday_confidence"] >= 7]
        _lc = df[df["intraday_confidence"] < 7]
        _hcwr = float((_hc["pnl_amount"] > 0).mean() * 100) if len(_hc) >= 2 else None
        _lcwr = float((_lc["pnl_amount"] > 0).mean() * 100) if len(_lc) >= 2 else None
        if _hcwr is not None and _lcwr is not None:
            if _hcwr - _lcwr > 15:
                gaps.append(
                    f"✅ **Confidence score is predictive** — high-confidence signals (≥7) win at "
                    f"{_hcwr:.0f}% vs {_lcwr:.0f}% for lower confidence. "
                    f"The confidence gate is working correctly."
                )
            elif _hcwr < _lcwr - 10:
                gaps.append(
                    f"⚠️ **Confidence scoring may need recalibration** — high-confidence signals "
                    f"({_hcwr:.0f}% win) underperform lower-confidence ones ({_lcwr:.0f}% win). "
                    f"Review the RSI, ATR, and volume factors in signals.py confidence calculation."
                )

    # 5. Stop overshoot (actual_exit vs rec_stop for STOPPED_OUT trades)
    _stopped_df = df[df["status"] == "STOPPED_OUT"].copy()
    if len(_stopped_df) >= 3 and "actual_exit" in _stopped_df.columns and "rec_stop" in _stopped_df.columns:
        _long_stops = _stopped_df[_stopped_df["signal_type"] == "BUY_ABOVE"]
        if len(_long_stops) >= 2:
            _long_stops = _long_stops.dropna(subset=["actual_exit", "rec_stop"])
            if not _long_stops.empty:
                _overshoot = float(
                    ((_long_stops["rec_stop"] - _long_stops["actual_exit"]) / _long_stops["rec_stop"]).mean() * 100
                )
                if _overshoot > 0.3:
                    gaps.append(
                        f"⚠️ **Stop-loss overshoot of {_overshoot:.2f}% on long trades** — "
                        f"actual exit is averaging {_overshoot:.2f}% below the planned stop, "
                        f"indicating the fast-exit overshoot buffer may need tightening, "
                        f"or slippage on volatile small-caps is high."
                    )

    if not gaps:
        gaps.append(
            "✅ **Algorithm is performing within expected parameters.** "
            "Win rate and R/R metrics are healthy based on the archived trade data."
        )

    # ── Display ─────────────────────────────────────────────────────────
    st.markdown("#### 📊 Archive Performance Summary")
    _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
    _mc1.metric("Total Trades", total)
    _mc2.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{'↑ good' if win_rate >= 50 else '↓ low'}", delta_color="normal" if win_rate >= 50 else "inverse")
    _mc3.metric("Stop-Out Rate", f"{stop_rate:.1f}%", delta=f"{'↑ high' if stop_rate > 55 else '✓ ok'}", delta_color="inverse" if stop_rate > 55 else "off")
    _mc4.metric("Target Hit Rate", f"{th_rate:.1f}%")
    _avg_pnl_pct = float(df["pnl_pct"].dropna().mean()) if "pnl_pct" in df.columns and df["pnl_pct"].notna().any() else 0.0
    _mc5.metric("Avg P&L %", f"{_avg_pnl_pct:+.2f}%")

    # Signal breakdown table
    _br_rows = []
    for _sk, _sdf in [("BUY_ABOVE (Long)", _buys), ("SELL_BELOW (Short)", _sells)]:
        if len(_sdf) > 0:
            _sw = int((_sdf["pnl_amount"] > 0).sum())
            _wr_s = _sw / len(_sdf) * 100
            _arr  = float(_sdf["rr_realised"].dropna().mean()) if _sdf["rr_realised"].notna().any() else None
            _br_rows.append({
                "Signal":     _sk,
                "Trades":     len(_sdf),
                "Wins":       _sw,
                "Win Rate":   f"{_wr_s:.1f}%",
                "Avg R/R":    f"{_arr:.2f}×" if _arr is not None else "—",
                "Avg P&L %":  f"{float(_sdf['pnl_pct'].dropna().mean()):+.2f}%" if _sdf["pnl_pct"].notna().any() else "—",
            })
    if _br_rows:
        st.dataframe(pd.DataFrame(_br_rows), hide_index=True, use_container_width=True)

    st.markdown("#### 🔍 Gaps Identified")
    for _g in gaps:
        st.markdown(_g)

    # ── Proposed changes ────────────────────────────────────────────────
    st.markdown("#### ⚙️ Proposed Algorithm Adjustments")
    _changed = {k: v for k, v in proposals.items() if abs(float(v) - float(cur_cfg.get(k, v))) > 1e-6}
    if _changed:
        _ch_rows = []
        _param_labels = {
            "intraday_rsi_buy_max":  "RSI Buy Max",
            "intraday_rsi_sell_min": "RSI Sell Min",
            "intraday_min_rr":       "Min R/R",
        }
        for _k, _new_v in _changed.items():
            _old_v = cur_cfg.get(_k)
            _ch_rows.append({
                "Parameter": _param_labels.get(_k, _k),
                "Current":   f"{_old_v:.2f}",
                "Proposed":  f"{_new_v:.2f}",
                "Direction": "↓ Tighter" if float(_new_v) < float(_old_v) else "↑ Relaxed",
            })
        st.dataframe(pd.DataFrame(_ch_rows), hide_index=True, use_container_width=True)
    else:
        st.success("No parameter changes needed — current settings are well-tuned.", icon="✅")

    # ── What to expect next day ──────────────────────────────────────────
    st.markdown("#### 🔭 What to Expect Next Day")
    _expectations = []
    if "intraday_rsi_buy_max" in _changed:
        _new_rsi_b = _changed["intraday_rsi_buy_max"]
        _dir = "fewer" if _new_rsi_b < cur_cfg["intraday_rsi_buy_max"] else "more"
        _expectations.append(
            f"• RSI buy ceiling → `{_new_rsi_b:.0f}`: expect **{_dir} BUY_ABOVE signals** per day, "
            f"{'filtering overbought entries in weak markets' if _dir == 'fewer' else 'capturing more trending setups'}."
        )
    if "intraday_rsi_sell_min" in _changed:
        _new_rsi_s = _changed["intraday_rsi_sell_min"]
        _dir = "fewer" if _new_rsi_s > cur_cfg["intraday_rsi_sell_min"] else "more"
        _expectations.append(
            f"• RSI sell floor → `{_new_rsi_s:.0f}`: expect **{_dir} SELL_BELOW signals**, "
            f"{'reducing mean-reversion bounce traps' if _dir == 'fewer' else 'capturing more short setups'}."
        )
    if "intraday_min_rr" in _changed:
        _new_rr = _changed["intraday_min_rr"]
        _dir_rr = "fewer" if _new_rr > cur_cfg["intraday_min_rr"] else "more"
        _expectations.append(
            f"• Min R/R → `{_new_rr:.2f}×`: expect **{_dir_rr} total signals**, "
            f"{'focusing on higher-quality wide-stop setups with better expected value' if _dir_rr == 'fewer' else 'more signals qualifying the R/R gate'}."
        )
    if not _expectations:
        _expectations.append("• No changes applied → signal frequency and quality unchanged from today.")
    for _exp in _expectations:
        st.markdown(_exp)

    st.markdown("---")
    _dc1, _dc2 = st.columns([2, 1])
    if _changed:
        if _dc1.button("✅ Confirm & Apply Adjustments", type="primary", use_container_width=True):
            db.save_signal_config(proposals, user_id=uid)
            st.success(
                "Algorithm updated! Run **Refresh Signals** or **Full Rescan** on the "
                "Trade Signals tab to see updated signals tomorrow.",
                icon="✅",
            )
    else:
        _dc1.info("No changes to apply.", icon="ℹ️")
    if _dc2.button("❌ Cancel", use_container_width=True):
        st.rerun()


# ============================================================
# ACTIVITY LOG — live fragment (stats + table + paper perf)
# ============================================================
@st.fragment(run_every=2)
def _activity_log_live():
    """Auto-refreshes every 2 s: portfolio snapshot, summary banners, trade table."""
    # Web Worker keepalive — fires even when tab is in background
    _st_autorefresh(interval=2000, limit=None, key="_actlog_ar", debounce=True)
    _uid = st.session_state.get("kite_user_id", "")

    # ── PORTFOLIO SNAPSHOT — live Kite margin + holdings + positions ────────
    _pf_kc = st.session_state.get("kite_client")
    _pf_ok = _pf_kc is not None and getattr(_pf_kc, "authenticated", False)
    if _pf_ok:
        try:
            _margins   = _pf_kc.get_margins("equity")
            _eq        = _margins.get("equity", _margins)
            _avail     = _eq.get("available", {})
            _used      = _eq.get("used", {})
            _net_bal   = float(_eq.get("net", 0) or 0)
            _live_bal  = float(_avail.get("live_balance", _avail.get("cash", 0)) or 0)
            _used_deb  = float(_used.get("debits", 0) or 0)
            _holdings  = _pf_kc.get_holdings()
            _h_value   = sum(float(h.get("last_price", 0)) * float(h.get("quantity", 0))
                             for h in _holdings if h.get("quantity", 0) > 0)
            _h_pnl     = sum(float(h.get("pnl", 0)) for h in _holdings)
            _h_day_pnl = sum(float(h.get("day_change", 0)) * float(h.get("quantity", 0))
                             for h in _holdings if h.get("quantity", 0) > 0)
            _h_count   = sum(1 for h in _holdings if h.get("quantity", 0) > 0)
            _positions = _pf_kc.get_positions()
            _net_pos   = _positions.get("net", []) if isinstance(_positions, dict) else []
            _pos_open  = [p for p in _net_pos if p.get("quantity", 0) != 0]
            _pos_value = sum(abs(float(p.get("value", 0))) for p in _pos_open)
            _pos_m2m   = sum(float(p.get("m2m", 0)) for p in _net_pos)
            _pf_m2m_c  = "#22c55e" if _pos_m2m >= 0 else "#ef4444"
            _pf_hp_c   = "#22c55e" if _h_pnl   >= 0 else "#ef4444"
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
                f'padding:10px 18px;margin-bottom:8px;display:flex;flex-wrap:wrap;'
                f'gap:22px;align-items:center">'
                f'<span style="font-size:0.78rem;color:#64748b;font-weight:600;'
                f'letter-spacing:0.05em;text-transform:uppercase;white-space:nowrap">'
                f'🏦 Portfolio Snapshot</span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Available '
                f'<b style="color:#f8fafc">₹{_live_bal:,.0f}</b></span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Net Balance '
                f'<b style="color:#f8fafc">₹{_net_bal:,.0f}</b></span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Margin Used '
                f'<b style="color:#f8fafc">₹{_used_deb:,.0f}</b></span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Holdings ({_h_count}) '
                f'<b style="color:#f8fafc">₹{_h_value:,.0f}</b> '
                f'<span style="color:{_pf_hp_c};font-size:0.75rem">{_h_pnl:+,.0f}</span></span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Positions ({len(_pos_open)}) '
                f'<b style="color:#f8fafc">₹{_pos_value:,.0f}</b></span>'
                f'<span style="font-size:0.8rem;color:#94a3b8">Today\'s P&amp;L '
                f'<b style="color:{_pf_m2m_c};font-weight:700">₹{_pos_m2m:+,.0f}</b>'
                f'<span style="font-size:0.7rem;color:#64748b"> holdings ₹{_h_day_pnl:+,.0f}</span>'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception as _pf_err:
            st.caption(f"⚠ Portfolio data unavailable: {_pf_err}")

    # ── Summary banners: Paper | Real ──────────────────────────────────────
    _stats_paper = db.get_trade_stats(user_id=_uid, is_paper=True)
    _stats_real  = db.get_trade_stats(user_id=_uid, is_paper=False)
    _stats_all   = db.get_trade_stats(user_id=_uid)

    if _stats_all.get("total", 0) == 0:
        st.info(
            "No trades yet. Go to **🎯 Trade Signals** tab and click "
            "**🚀 Place Order via Kite** (or **📝 Log a trade** if not connected) below any signal table.",
            icon="📒",
        )
        return

    _cap = config.PAPER_CAPITAL  # reference capital for both paper and real

    def _stat_banner(label: str, icon: str, s: dict, accent: str,
                     charges: float = 0.0, extra_html: str = "") -> str:
        """Build a compact dark banner for one category of trades."""
        if not s or s.get("total", 0) == 0:
            return (
                f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
                f'padding:10px 18px;flex:1;min-width:280px">'
                f'<span style="font-size:0.78rem;color:#64748b;font-weight:600;'
                f'text-transform:uppercase">{icon} {label}</span>'
                f'<span style="color:#475569;font-size:0.8rem;margin-left:16px">No trades yet</span>'
                f'</div>'
            )
        _gross    = s["total_pnl"]
        _net      = _gross - charges
        _net_c    = "#22c55e" if _net >= 0 else "#ef4444"
        _gross_c  = "#22c55e" if _gross >= 0 else "#ef4444"
        _net_pct  = (_net / _cap * 100) if _cap else 0.0
        _wr       = f"{s['win_rate']:.1f}%" if s["closed"] > 0 else "—"
        _rr       = f"{s['avg_rr']:.2f}×" if s.get("avg_rr") else "—"
        _best     = f"₹{s['best_trade']:+,.0f}" if s.get("best_trade") else "—"
        return (
            f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
            f'padding:10px 18px;flex:1;min-width:280px;display:flex;flex-wrap:wrap;gap:16px;align-items:center">'
            f'<span style="font-size:0.78rem;color:{accent};font-weight:700;'
            f'text-transform:uppercase;white-space:nowrap">{icon} {label}</span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Trades '
            f'<b style="color:#f8fafc">{s["total"]}</b></span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Open '
            f'<b style="color:#f59e0b">{s["open"]}</b></span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Win rate '
            f'<b style="color:#f8fafc">{_wr}</b></span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Gross P&amp;L '
            f'<b style="color:{_gross_c}">₹{_gross:+,.0f}</b></span>'
            f'<span style="font-size:0.8rem;color:#64748b" title="Brokerage + STT + NSE txn + GST + Stamp">'
            f'Charges <b style="color:#f59e0b">₹{charges:,.0f}</b></span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Net P&amp;L '
            f'<b style="color:{_net_c}">₹{_net:+,.0f}</b> '
            f'<span style="font-size:0.75rem;color:{_net_c}">({_net_pct:+.2f}% of ₹{_cap//100_000}L)</span>'
            f'</span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Best '
            f'<b style="color:#f8fafc">{_best}</b></span>'
            f'<span style="font-size:0.8rem;color:#94a3b8">Avg R/R '
            f'<b style="color:#f8fafc">{_rr}</b></span>'
            f'{extra_html}'
            f'</div>'
        )

    # ── Charges and net P&L for banners ─────────────────────────────────────
    _paper_charges = 0.0
    _real_charges  = 0.0
    try:
        _paper_charges = db.get_total_charges(user_id=_uid, is_paper=True)
        _real_charges  = db.get_total_charges(user_id=_uid, is_paper=False)
    except Exception:
        pass

    # Extra info on the paper banner: today's realised return + gate status
    _p_today_pnl = 0.0
    try:
        _p_today_pnl = db.get_today_closed_pnl(user_id=_uid, is_paper=True)
    except Exception:
        pass
    _p_today_ret   = (_p_today_pnl / _cap * 100) if _cap else 0.0
    _p_hwm         = st.session_state.get("paper_day_hwm_pct", 0.0)
    _p_cutoff      = (_p_hwm - config.DAILY_TRAIL_PCT) if _p_hwm >= config.DAILY_TARGET_LOW_PCT else None
    _p_blocked_now = st.session_state.get("paper_day_blocked", False)

    # Real: today's realised return
    _r_today_pnl = 0.0
    try:
        _r_today_pnl = db.get_today_closed_pnl(user_id=_uid, is_paper=False)
    except Exception:
        pass
    _r_today_ret = (_r_today_pnl / _cap * 100) if _cap else 0.0

    if _p_blocked_now:
        _paper_extra = (
            f'<span style="font-size:0.73rem;color:#ef4444;font-weight:600">'
            f'🚫 Gate closed &nbsp;·&nbsp; Today: <b>{_p_today_ret:+.2f}%</b> of ₹{_cap//100_000}L</span>'
        )
    elif _p_cutoff is not None:
        _paper_extra = (
            f'<span style="font-size:0.73rem;color:#f59e0b">'
            f'⚡ Today: <b>{_p_today_ret:+.2f}%</b> &nbsp;·&nbsp; Cutoff: {_p_cutoff:.2f}%</span>'
        )
    else:
        _paper_extra = (
            f'<span style="font-size:0.73rem;color:#64748b">'
            f'Today: <b style="color:#f8fafc">{_p_today_ret:+.2f}%</b> of ₹{_cap//100_000}L '
            f'&nbsp;·&nbsp; Target ≥{config.DAILY_TARGET_LOW_PCT:.0f}%</span>'
        )

    _real_extra = (
        f'<span style="font-size:0.73rem;color:#64748b">'
        f'Today: <b style="color:#f8fafc">{_r_today_ret:+.2f}%</b> of ₹{_cap//100_000}L</span>'
    ) if _stats_real.get("total", 0) > 0 else ""

    st.markdown(
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px">'
        + _stat_banner("Paper Trades", "📄", _stats_paper, "#94a3b8", _paper_charges, _paper_extra)
        + _stat_banner("Real Trades",  "💸", _stats_real,  "#22c55e", _real_charges,  _real_extra)
        + f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Load all trades once; split into today (active) vs past (archive) ──
    _log_df_all = db.load_trade_log(user_id=_uid)
    _today_pd   = pd.Timestamp.today().normalize()
    if not _log_df_all.empty and "trade_date" in _log_df_all.columns:
        _tdate_col    = pd.to_datetime(_log_df_all["trade_date"], errors="coerce")
        _active_mask  = (_tdate_col >= _today_pd) | (_log_df_all["status"] == "OPEN")
        _archive_mask = (_tdate_col < _today_pd)  & (_log_df_all["status"] != "OPEN")
        _log_df_today = _log_df_all[_active_mask].copy()
        _log_df_arch  = _log_df_all[_archive_mask].copy()
    else:
        _log_df_today = _log_df_all.copy()
        _log_df_arch  = pd.DataFrame()

    # ── Shared helpers ──────────────────────────────────────────────────────
    _live_ltp_now = st.session_state.get("_live_ltp", {})

    _TRADE_COL_CFG = {
        "status":             st.column_config.TextColumn("Status",       width="small"),
        "id":                 st.column_config.NumberColumn("ID",         width="small"),
        "logged_at":          st.column_config.DatetimeColumn(
                                  "Opened At", format="DD/MM/YY HH:mm:ss",
                                  help="Exact timestamp when the trade was triggered and logged",
                              ),
        "tradingsymbol":      st.column_config.TextColumn("Symbol"),
        "ltp":                st.column_config.TextColumn("LTP",
            help="Last traded price — live from market during trading hours"),
        "trade_type":         st.column_config.TextColumn("Type",
            help="📄 Paper = virtual paper trade\n💸 Real = actual Kite order"),
        "setup_type":         st.column_config.TextColumn("Setup"),
        "signal_type":        st.column_config.TextColumn("Signal"),
        "quantity":           st.column_config.NumberColumn("Qty"),
        "rec_entry":          st.column_config.TextColumn("Rec Entry",    help="Screener-recommended entry price"),
        "actual_entry":       st.column_config.TextColumn("Actual Entry", help="Your actual execution price"),
        "rec_stop":           st.column_config.TextColumn("Rec Stop",     help="Recommended stop-loss"),
        "rec_t1":             st.column_config.TextColumn("Rec T1",       help="Recommended first target"),
        "actual_exit":        st.column_config.TextColumn("Actual Exit",  help="Your actual exit price"),
        "pnl_amount":         st.column_config.TextColumn("Gross P&L ₹",  help="(Exit − Entry) × Qty — live MTM for open trades"),
        "pnl_pct":            st.column_config.TextColumn("Gross P&L %",  help="(Exit − Entry) / Entry × 100, before charges"),
        "charges":            st.column_config.TextColumn("Charges ₹",    help=(
            "Zerodha statutory charges per round-trip (intraday):\n"
            "• Brokerage: min(₹20, 0.03%) × 2 orders\n"
            "• STT: 0.025% on sell value\n"
            "• NSE txn: 0.00307% on total turnover\n"
            "• SEBI: ₹10/crore\n"
            "• GST: 18% on (brokerage + txn + SEBI)\n"
            "• Stamp: 0.003% on buy value\n"
            "For open trades, provisional charges based on LTP."
        )),
        "net_pnl":            st.column_config.TextColumn("Net P&L ₹",    help="Gross P&L minus all statutory charges"),
        "net_pnl_pct":        st.column_config.TextColumn("Net P&L %",    help="Net P&L as % of capital deployed (entry × qty)"),
        "rr_realised":        st.column_config.TextColumn("R/R actual",   help="Actual gain ÷ actual risk"),
        "slippage_entry_pct": st.column_config.TextColumn("Entry slip %", help="How far your entry was from the recommended entry"),
        "kite_order_id":      st.column_config.TextColumn("Kite Order ID"),
        "kite_status":        st.column_config.TextColumn("Kite Status"),
        "notes":              st.column_config.TextColumn("Notes",        width="large"),
    }

    _FMT_MAP = {
        "ltp": "₹{:,.2f}", "rec_entry": "₹{:,.2f}", "actual_entry": "₹{:,.2f}",
        "rec_stop": "₹{:,.2f}", "rec_t1": "₹{:,.2f}", "actual_exit": "₹{:,.2f}",
        "pnl_amount": "₹{:+,.2f}", "pnl_pct": "{:+.2f}%",
        "charges": "₹{:,.2f}", "net_pnl": "₹{:+,.2f}", "net_pnl_pct": "{:+.2f}%",
        "rr_realised": "{:.2f}×", "slippage_entry_pct": "{:+.2f}%",
    }

    def _pnl_color(val):
        try:
            v = float(str(val).replace("₹","").replace(",","").replace("+",""))
            if v > 0: return "color:#22c55e;font-weight:600"
            if v < 0: return "color:#ef4444;font-weight:600"
        except Exception:
            pass
        return ""

    def _status_badge_color(val):
        return {
            "TARGET_HIT":  "color:#22c55e;font-weight:700",
            "STOPPED_OUT": "color:#ef4444;font-weight:700",
            "OPEN":        "color:#f59e0b;font-weight:600",
            "CANCELLED":   "color:#94a3b8",
            "CLOSED":      "color:#3b82f6;font-weight:600",
        }.get(str(val).upper(), "")

    def _enrich_df(df: pd.DataFrame, inject_mtm: bool = False) -> pd.DataFrame:
        """Add trade_type, ltp, live MTM, charges, net P&L columns."""
        df = df.copy()
        if "is_paper_trade" in df.columns:
            df["trade_type"] = df["is_paper_trade"].apply(
                lambda v: "📄 Paper" if v is True or v == 1 else "💸 Real"
            )
        else:
            df["trade_type"] = "💸 Real"

        df["ltp"] = df["tradingsymbol"].map(lambda s: _live_ltp_now.get(s, None))

        if inject_mtm:
            _om = df["status"] == "OPEN"
            for _idx, _row in df[_om].iterrows():
                _ltp_v = _live_ltp_now.get(str(_row.get("tradingsymbol") or ""), 0)
                if _ltp_v and not _isna(_row.get("actual_entry")) and _row.get("actual_entry"):
                    _ep   = float(_row["actual_entry"])
                    _qp   = float(_row.get("quantity", 0) or 0) if not _isna(_row.get("quantity")) else 0
                    _mult = -1 if _row.get("signal_type","") == "SELL_BELOW" else 1
                    df.at[_idx, "pnl_amount"] = (_ltp_v - _ep) * _qp * _mult
                    if _ep:
                        df.at[_idx, "pnl_pct"] = (_ltp_v - _ep) / _ep * 100 * _mult

        def _charges_row(row):
            entry  = 0.0 if _isna(row.get("actual_entry")) else float(row.get("actual_entry") or 0)
            exit_p = 0.0 if _isna(row.get("actual_exit"))  else float(row.get("actual_exit")  or 0)
            qty    = 0   if _isna(row.get("quantity"))      else int(row.get("quantity") or 0)
            stype  = str(row.get("setup_type") or "INTRADAY")
            if str(row.get("status","")) == "OPEN" and not exit_p:
                exit_p = float(_live_ltp_now.get(str(row.get("tradingsymbol") or ""), 0) or 0)
            if entry and exit_p and qty:
                return db.compute_trade_charges(entry, exit_p, qty, stype).get("total", 0.0)
            return 0.0

        df["charges"]     = df.apply(_charges_row, axis=1)
        df["net_pnl"]     = df["pnl_amount"].fillna(0) - df["charges"]
        df["net_pnl_pct"] = df.apply(
            lambda r: (
                float(r["net_pnl"]) / (float(r.get("actual_entry") or 0) * float(r.get("quantity") or 1)) * 100
                if not _isna(r.get("actual_entry")) and not _isna(r.get("quantity"))
                   and r.get("actual_entry") and r.get("quantity") else None
            ), axis=1,
        )
        _no_pnl = df["pnl_amount"].isna() & (df["status"] != "OPEN")
        df.loc[_no_pnl, ["charges", "net_pnl", "net_pnl_pct"]] = None
        return df

    def _render_trade_table(df: pd.DataFrame, key_sfx: str) -> None:
        """Render a styled trade dataframe."""
        if df.empty:
            return
        _disp_cols = [
            "status", "id", "logged_at", "tradingsymbol", "ltp",
            "pnl_amount", "pnl_pct", "charges", "net_pnl", "net_pnl_pct",
            "trade_type", "setup_type", "signal_type", "quantity",
            "rec_entry", "actual_entry", "rec_stop", "rec_t1", "actual_exit",
            "rr_realised", "slippage_entry_pct", "kite_order_id", "kite_status", "notes",
        ]
        _disp_cols = [c for c in _disp_cols if c in df.columns]
        _show = df[_disp_cols].copy()
        _fmt  = {k: v for k, v in _FMT_MAP.items() if k in _show.columns}
        _styled = _show.style.format(_fmt, na_rep="—")
        _pnl_s  = [c for c in ["pnl_amount","pnl_pct","net_pnl","net_pnl_pct","rr_realised"] if c in _show.columns]
        _stat_s = ["status"] if "status" in _show.columns else []
        if _pnl_s:  _styled = _styled.map(_pnl_color,         subset=_pnl_s)
        if _stat_s: _styled = _styled.map(_status_badge_color, subset=_stat_s)
        if "ltp" in _show.columns:
            _styled = _styled.map(
                lambda v: "color:#60a5fa;font-weight:600" if str(v) not in ("—","nan","") else "",
                subset=["ltp"],
            )
        st.dataframe(
            _styled,
            use_container_width=True,
            height=min(600, 60 + len(_show) * 38),
            hide_index=True,
            column_config=_TRADE_COL_CFG,
        )

    # ── TWO TABS ───────────────────────────────────────────────────────────
    _act_tab, _arch_tab = st.tabs(["📊 Active — Today", "📁 Archive — Past Days"])

    # ════════════════════════════════════════════════════════════════════════
    # ACTIVE TAB  — today's trades with live MTM
    # ════════════════════════════════════════════════════════════════════════
    with _act_tab:
        # ── Filters ────────────────────────────────────────────────────────
        _af1, _af2, _af3, _af4, _af5 = st.columns(5)
        _a_flt_status = _af1.multiselect(
            "Status", ["OPEN","CLOSED","TARGET_HIT","STOPPED_OUT","CANCELLED"],
            default=[], placeholder="All statuses", key="act_flt_status",
        )
        _a_flt_setup = _af2.multiselect(
            "Setup type", ["SWING","INTRADAY","SCALING"],
            default=[], placeholder="All setups", key="act_flt_setup",
        )
        _a_flt_sym   = _af3.text_input("Symbol search", placeholder="e.g. RELIANCE", key="act_flt_sym")
        _a_sort_by   = _af4.selectbox("Sort by", ["Newest first","Oldest first","P&L ↓","P&L ↑"], key="act_sort")
        _a_flt_type  = _af5.selectbox("Trade type", ["All","Real only","Paper only"], key="act_flt_type")

        _cur = _log_df_today.copy()
        if _a_flt_status:
            _cur = _cur[_cur["status"].isin(_a_flt_status)]
        if _a_flt_setup:
            _cur = _cur[_cur["setup_type"].isin(_a_flt_setup)]
        if _a_flt_sym:
            _cur = _cur[_cur["tradingsymbol"].str.contains(_a_flt_sym.strip(), case=False, na=False, regex=False)]
        if _a_flt_type == "Paper only" and "is_paper_trade" in _cur.columns:
            _cur = _cur[_cur["is_paper_trade"] == True]
        elif _a_flt_type == "Real only" and "is_paper_trade" in _cur.columns:
            _cur = _cur[(_cur["is_paper_trade"] != True) | _cur["is_paper_trade"].isna()]

        _cur["_so"] = (_cur["status"] != "OPEN").astype(int)
        if   _a_sort_by == "Oldest first": _cur = _cur.sort_values(["_so","logged_at"],   ascending=[True, True])
        elif _a_sort_by == "P&L ↓":        _cur = _cur.sort_values(["_so","pnl_amount"],  ascending=[True, False], na_position="last")
        elif _a_sort_by == "P&L ↑":        _cur = _cur.sort_values(["_so","pnl_amount"],  ascending=[True, True],  na_position="last")
        else:                               _cur = _cur.sort_values(["_so","logged_at"],   ascending=[True, False])
        _cur = _cur.drop(columns=["_so"])

        st.caption(f"Showing {len(_cur)} trade(s) for today · auto-refreshes every 2 s")

        if not _cur.empty:
            _cur = _enrich_df(_cur, inject_mtm=True)
            _render_trade_table(_cur, key_sfx="act")

        # ── Close an open trade ────────────────────────────────────────────
        _open_trades = _cur[_cur["status"] == "OPEN"] if not _cur.empty else pd.DataFrame()
        if not _open_trades.empty:
            st.markdown("---")
            st.subheader("📌 Close an open trade")
            _cl_labels = []
            for _, r in _open_trades.iterrows():
                _ae = r.get("actual_entry")
                _ae_str = f"₹{float(_ae):.2f}" if _ae and not pd.isna(_ae) else "entry pending"
                _kid = r.get("kite_order_id")
                _cl_labels.append(
                    f"#{r['id']} · {r['tradingsymbol']} ({r['setup_type']}) — {_ae_str}"
                    + (f" · Kite#{_kid}" if _kid else "")
                )
            _id_map = dict(zip(_cl_labels, _open_trades["id"].tolist()))
            _sel_lbl = st.selectbox("Select open trade to close", _cl_labels, key="close_trade_sel")
            _sel_id  = _id_map.get(_sel_lbl)
            if _sel_id:
                _ct1, _ct2, _ct3 = st.columns(3)
                _close_exit   = _ct1.number_input("Exit price ₹", min_value=0.01, value=0.01, step=0.05, format="%.2f", key="close_exit")
                _close_status = _ct2.selectbox("Outcome", ["CLOSED","TARGET_HIT","STOPPED_OUT","CANCELLED"], key="close_status")
                _close_notes  = _ct3.text_input("Notes", key="close_notes")
                if st.button("✅ Close Trade", type="primary", key="close_trade_btn"):
                    db.close_trade(_sel_id, float(_close_exit), _close_status, _close_notes or None)
                    st.success(f"Trade #{_sel_id} closed as {_close_status} at ₹{_close_exit:.2f}")
                    st.rerun()

        # ── Delete a trade ─────────────────────────────────────────────────
        with st.expander("🗑️ Delete a trade entry", expanded=False):
            _del_id = st.number_input("Trade ID to delete", min_value=1, step=1, key="del_trade_id")
            if st.button("Delete", type="secondary", key="del_trade_btn"):
                db.delete_trade(int(_del_id))
                st.success(f"Trade #{_del_id} deleted.")
                st.rerun()

        # ── Current signal thresholds ──────────────────────────────────────
        _cur_cfg = db.get_signal_config(user_id=_uid)
        _cfg_html = (
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
            f'padding:8px 16px;margin:6px 0;font-size:0.78rem;color:#94a3b8">'
            f'⚙️ <b>Signal thresholds (tuned from paper):</b> &nbsp;'
            f'RSI buy ≤ <b style="color:#f8fafc">{_cur_cfg["intraday_rsi_buy_max"]:.0f}</b> &nbsp;·&nbsp; '
            f'RSI sell ≥ <b style="color:#f8fafc">{_cur_cfg["intraday_rsi_sell_min"]:.0f}</b> &nbsp;·&nbsp; '
            f'Min R/R <b style="color:#f8fafc">{_cur_cfg["intraday_min_rr"]:.1f}×</b> &nbsp;·&nbsp; '
            f'<span style="color:#64748b">Run <b>Full Rescan</b> or <b>Refresh Signals</b> to apply</span>'
            f'</div>'
        )
        _paper_perf  = db.get_paper_trade_perf(user_id=_uid, days=30)
        _paper_long  = _paper_perf.get("BUY_ABOVE",  {})
        _paper_short = _paper_perf.get("SELL_BELOW", {})
        _pp_rows = []
        for _sig_k, _sig_d in [("BUY_ABOVE (Long)", _paper_long), ("SELL_BELOW (Short)", _paper_short)]:
            if _sig_d.get("total", 0) > 0:
                _pp_rows.append({
                    "Signal":    _sig_k,
                    "Trades":    _sig_d["total"],
                    "Wins":      _sig_d["wins"],
                    "Losses":    _sig_d["losses"],
                    "Win Rate":  f"{_sig_d['win_rate']:.1f}%",
                    "Avg R/R":   f"{_sig_d['avg_rr']:.2f}×" if _sig_d.get("avg_rr") else "—",
                    "Avg P&L %": f"{_sig_d['avg_pnl_pct']:+.2f}%" if _sig_d.get("avg_pnl_pct") is not None else "—",
                })
        if _pp_rows:
            with st.expander("📄 Paper trade breakdown by signal (30d)", expanded=False):
                st.dataframe(pd.DataFrame(_pp_rows), hide_index=True, use_container_width=True)
                st.markdown(_cfg_html, unsafe_allow_html=True)
        else:
            st.markdown(_cfg_html, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # ARCHIVE TAB  — past days' completed trades + re-adjust button
    # ════════════════════════════════════════════════════════════════════════
    with _arch_tab:
        _arch_hdr, _arch_btn_col = st.columns([3, 1])
        with _arch_hdr:
            st.caption(
                "All completed trades from previous trading days. "
                "Use **Re-adjust Algorithm** to evaluate gaps and tune signal thresholds "
                "based on this archive."
            )
        with _arch_btn_col:
            if st.button(
                "🔬 Re-adjust Algorithm",
                type="primary",
                use_container_width=True,
                help="Analyse archived paper trades and propose + apply algorithm improvements",
                key="btn_readjust_algo",
            ):
                _show_algo_readjust_dialog(uid=_uid)

        if _log_df_arch.empty:
            st.info(
                "No archived trades yet. "
                "Completed trades from past days will appear here automatically.",
                icon="📭",
            )
        else:
            # ── Archive filters ────────────────────────────────────────────
            _xf1, _xf2, _xf3, _xf4, _xf5 = st.columns(5)
            _x_flt_status = _xf1.multiselect(
                "Status", ["CLOSED","TARGET_HIT","STOPPED_OUT","CANCELLED"],
                default=[], placeholder="All statuses", key="arch_flt_status",
            )
            _x_flt_setup = _xf2.multiselect(
                "Setup type", ["SWING","INTRADAY","SCALING"],
                default=[], placeholder="All setups", key="arch_flt_setup",
            )
            _x_flt_sym  = _xf3.text_input("Symbol search", placeholder="e.g. RELIANCE", key="arch_flt_sym")
            _x_sort_by  = _xf4.selectbox("Sort by", ["Newest first","Oldest first","P&L ↓","P&L ↑"], key="arch_sort")
            _x_flt_type = _xf5.selectbox("Trade type", ["All","Real only","Paper only"], key="arch_flt_type")

            _xdf = _log_df_arch.copy()
            if _x_flt_status:
                _xdf = _xdf[_xdf["status"].isin(_x_flt_status)]
            if _x_flt_setup:
                _xdf = _xdf[_xdf["setup_type"].isin(_x_flt_setup)]
            if _x_flt_sym:
                _xdf = _xdf[_xdf["tradingsymbol"].str.contains(_x_flt_sym.strip(), case=False, na=False, regex=False)]
            if _x_flt_type == "Paper only" and "is_paper_trade" in _xdf.columns:
                _xdf = _xdf[_xdf["is_paper_trade"] == True]
            elif _x_flt_type == "Real only" and "is_paper_trade" in _xdf.columns:
                _xdf = _xdf[(_xdf["is_paper_trade"] != True) | _xdf["is_paper_trade"].isna()]

            if   _x_sort_by == "Oldest first": _xdf = _xdf.sort_values("logged_at",  ascending=True)
            elif _x_sort_by == "P&L ↓":        _xdf = _xdf.sort_values("pnl_amount", ascending=False, na_position="last")
            elif _x_sort_by == "P&L ↑":        _xdf = _xdf.sort_values("pnl_amount", ascending=True,  na_position="last")
            else:                               _xdf = _xdf.sort_values("logged_at",  ascending=False)

            st.caption(f"Showing {len(_xdf)} archived trade(s) across all past days")

            _xdf = _enrich_df(_xdf, inject_mtm=False)
            _render_trade_table(_xdf, key_sfx="arch")

            # ── Export archive ─────────────────────────────────────────────
            st.download_button(
                "⬇️ Export archive as CSV",
                data=_log_df_arch.to_csv(index=False).encode(),
                file_name=f"trade_archive_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="arch_export_csv",
            )

            # ── Strategy Insights (archive) ────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Strategy Insights (Archive)")
            _closed = _xdf[_xdf["status"].isin(["CLOSED","TARGET_HIT","STOPPED_OUT"])]
            if len(_closed) >= 3:
                _ins1, _ins2, _ins3 = st.columns(3)
                _wr_setup = (
                    _closed.groupby("setup_type")
                    .apply(lambda g: (g["pnl_amount"] > 0).mean() * 100, include_groups=False)
                    .reset_index(name="Win %")
                )
                _ins1.markdown("**Win rate by setup**")
                _ins1.dataframe(_wr_setup, hide_index=True, use_container_width=True)

                _avg_pnl_arch = (
                    _closed.groupby("signal_type")["pnl_pct"]
                    .mean()
                    .reset_index(name="Avg P&L %")
                    .sort_values("Avg P&L %", ascending=False)
                )
                _ins2.markdown("**Avg P&L % by signal**")
                _ins2.dataframe(
                    _avg_pnl_arch.style.format({"Avg P&L %": "{:+.2f}%"}),
                    hide_index=True, use_container_width=True,
                )

                _slip = _closed["slippage_entry_pct"].dropna() if "slippage_entry_pct" in _closed.columns else pd.Series(dtype=float)
                if not _slip.empty:
                    _ins3.markdown("**Entry slippage**")
                    _ins3.metric("Avg slippage", f"{_slip.mean():+.2f}%")
                    _ins3.metric("Max slippage", f"{_slip.max():+.2f}%")
            else:
                st.info("Need at least 3 closed archived trades to show strategy insights.", icon="📊")


# ============================================================
# ACTIVITY LOG TAB
# ============================================================
with tab_activity:
    st.subheader("📒 Trade Activity Log")

    _act_kc = st.session_state.get("kite_client")
    _act_kite_ok = _act_kc is not None and getattr(_act_kc, "authenticated", False)

    # ── Header row: description + Sync button ──────────────────────────────
    _act_hdr, _act_btn_col = st.columns([4, 1])
    with _act_hdr:
        if _act_kite_ok:
            st.caption(
                "Orders placed via Kite appear here automatically. "
                "Hit **🔄 Sync from Kite** to pull the latest fill status for open orders. "
                "Once filled, record your exit price to compute P&L."
            )
        else:
            st.caption(
                "A journal of every trade logged against a recommendation. "
                "Connect Kite (sidebar) to place orders directly from the Trade Signals tab."
            )
    with _act_btn_col:
        if _act_kite_ok and st.button("🔄 Sync from Kite", use_container_width=True,
                                       help="Pulls today's Kite orders and updates fill status for open trades"):
            with st.spinner("Syncing with Kite…"):
                try:
                    _today_orders = _act_kc.get_orders()
                    _n_synced     = db.sync_from_kite_orders(_today_orders, user_id=_cur_user_id)
                    st.success(f"Synced {_n_synced} trade(s) from Kite." if _n_synced
                               else "All open trades already up-to-date.")
                    st.rerun()
                except Exception as _sync_err:
                    st.error(f"Sync failed: {_sync_err}")

    # ── Live-refreshing portfolio snapshot + stats + table (fragment) ────────
    _activity_log_live()
