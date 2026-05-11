"""
auth.py — Password hashing and session token helpers.

Uses bcrypt (work factor 12) for passwords.
Session tokens are 256-bit URL-safe random strings stored in browser localStorage
and validated against the user_sessions DB table.
"""

import secrets
import bcrypt


# ── Passwords ─────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plaintext password (work factor 12)."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Session tokens ────────────────────────────────────────────────────────────

def new_session_token() -> str:
    """Generate a cryptographically random 64-char URL-safe session token."""
    return secrets.token_urlsafe(48)   # 48 bytes → 64-char base64url string


# ── Kite token validity ───────────────────────────────────────────────────────

def is_kite_token_fresh(kite_token_updated_at) -> bool:
    """
    Kite access tokens expire at midnight IST every day.
    A token is considered fresh if it was issued today (IST).
    """
    if not kite_token_updated_at:
        return False
    from datetime import datetime, timezone, timedelta, date
    _IST = timezone(timedelta(hours=5, minutes=30))
    today_ist = datetime.now(_IST).date()
    if hasattr(kite_token_updated_at, "date"):
        token_date = kite_token_updated_at.date()
    else:
        try:
            token_date = datetime.fromisoformat(str(kite_token_updated_at)).date()
        except Exception:
            return False
    return token_date >= today_ist
