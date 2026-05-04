"""
ls_store.py — Thin Python wrapper around the _ls_frontend Streamlit component.

Provides get / set / delete operations on the user's browser localStorage
so that Kite API credentials survive across browser sessions (per-domain,
per-browser — completely isolated between different users).

Usage in app.py:
    from ls_store import ls_get, ls_set, ls_delete

    # Read (returns "" or None on the very first Streamlit render cycle;
    #        returns the stored string value on subsequent cycles)
    val = ls_get("kite_api_key")

    # Write (fire-and-forget; value visible from the next rerun onward)
    ls_set("kite_api_key", "my_key", expires_days=365)

    # Delete
    ls_delete("kite_access_token")
"""
import os
import json
from datetime import datetime, timedelta

import streamlit.components.v1 as _cv1

# Resolve the frontend directory relative to this file
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ls_frontend")

# Declare the component once at module import time.
# In production (path=), Streamlit serves the static HTML from disk.
_ls = _cv1.declare_component("kite_ls", path=_FRONTEND_DIR)


def ls_get(key: str) -> str | None:
    """
    Read a value from localStorage.

    Returns:
        str  — the stored value (may be "")
        None — on the first render cycle before the component has responded
    """
    raw = _ls(action="get", key=key, default=None)
    if raw is None:
        return None
    # Check expiry if the value was stored as JSON with exp field
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "v" in obj and "exp" in obj:
            if datetime.now().timestamp() * 1000 > obj["exp"]:
                ls_delete(key)   # expired — clean up
                return ""
            return obj["v"]
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def ls_set(key: str, value: str, expires_days: int = 0) -> None:
    """
    Write a value to localStorage.
    If expires_days > 0, the value will be treated as expired after that many days.
    (Expiry is enforced by ls_get — localStorage itself doesn't expire entries.)
    """
    expires_ms = 0
    if expires_days > 0:
        expires_ms = int(
            (datetime.now() + timedelta(days=expires_days)).timestamp() * 1000
        )
    _ls(action="set", key=key, value=value, expires_ms=expires_ms, default=None)


def ls_delete(key: str) -> None:
    """Remove a key from localStorage."""
    _ls(action="delete", key=key, default=None)
