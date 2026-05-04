"""
kite_client.py — Wraps the Kite SDK with two things it doesn't give you:
  1. Persistent access token (so you don't re-login every script run)
  2. Rate limiting that respects Kite's actual caps

Why this matters: KiteConnect's SDK is a thin REST wrapper. It will
happily fire 100 requests/sec at the API and get you 429-ed and
temp-banned. We add the safety rails.

Auth flow (frontend-driven):
  1. app.py calls client.get_login_url() and opens it in the browser.
  2. Zerodha redirects back to the app with ?request_token=xxx.
  3. app.py calls client.complete_auth(request_token) which exchanges it
     for an access_token and caches it to data/access_token.json.
  4. Subsequent runs load the cached token until it expires (~6 AM IST).
"""
import json
import time
from datetime import datetime, date
from pathlib import Path

from kiteconnect import KiteConnect
import config


class RateLimiter:
    """
    Simple token-bucket rate limiter.
    Why not the `ratelimit` library decorator? Because we want to share
    one limiter across multiple call sites (historical fetches happen
    in a loop) and decorators make that awkward.
    """
    def __init__(self, calls_per_sec: float):
        self.min_interval = 1.0 / calls_per_sec
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        sleep_for = self.min_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        self.last_call = time.time()


class KiteClient:
    """
    Wrapper around KiteConnect. On init, tries to load a cached access token.
    If none exists or it's stale, `self.authenticated` is False — the caller
    (app.py) is responsible for driving the OAuth flow via get_login_url() /
    complete_auth(). No blocking CLI prompts.
    """
    def __init__(self):
        if not config.KITE_API_KEY:
            raise RuntimeError(
                "KITE_API_KEY missing. Create a .env file with "
                "KITE_API_KEY=xxx and KITE_API_SECRET=xxx"
            )
        self.kite = KiteConnect(api_key=config.KITE_API_KEY)
        self._patch_session()
        self.hist_limiter = RateLimiter(config.HIST_REQUESTS_PER_SEC)
        self.quote_limiter = RateLimiter(config.QUOTE_REQUESTS_PER_SEC)
        self.authenticated = self._try_load_cached_token()

    # ----------------------------------------------------------
    # SESSION PATCH — replace requests with curl_cffi so that
    # Cloudflare's TLS fingerprint check sees Chrome, not Python.
    # We preserve all headers the SDK already set (X-Kite-Version,
    # User-Agent, etc.) by copying them onto the new session.
    # ----------------------------------------------------------
    def _patch_session(self):
        try:
            from curl_cffi.requests import Session as CurlSession
            old_headers = dict(self.kite.reqsession.headers)
            new_session = CurlSession(impersonate="chrome120")
            new_session.headers.update(old_headers)
            self.kite.reqsession = new_session
        except ImportError:
            # Fallback: at least stop leaking the Python User-Agent
            self.kite.reqsession.trust_env = False
            self.kite.reqsession.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })

    # ----------------------------------------------------------
    # AUTH
    # ----------------------------------------------------------
    def _try_load_cached_token(self) -> bool:
        """
        Load today's access token from disk. Returns True if a valid
        token was found and set, False otherwise.
        Kite tokens expire daily at ~6 AM IST, so we key on today's date.
        """
        token_path = Path(config.TOKEN_CACHE)
        if not token_path.exists():
            return False
        try:
            data = json.loads(token_path.read_text())
            if date.fromisoformat(data["date"]) == date.today():
                self.kite.set_access_token(data["access_token"])
                return True
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
        return False

    def get_login_url(self) -> str:
        """Return the Zerodha OAuth login URL to open in the browser."""
        return self.kite.login_url()

    def complete_auth(self, request_token: str) -> str:
        """
        Exchange a request_token (from Kite's OAuth redirect) for an
        access_token. Caches it to disk and marks this client as authenticated.
        Returns the access_token string.
        """
        session = self.kite.generate_session(
            request_token, api_secret=config.KITE_API_SECRET
        )
        access_token = session["access_token"]
        self.kite.set_access_token(access_token)
        self.authenticated = True

        token_path = Path(config.TOKEN_CACHE)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps({
            "access_token": access_token,
            "date": date.today().isoformat(),
        }))
        return access_token

    # ----------------------------------------------------------
    # INSTRUMENTS — full master list of NSE securities
    # ----------------------------------------------------------
    def get_instruments(self, exchange: str = "NSE") -> list:
        """Returns full instruments list. ~10K rows for NSE."""
        return self.kite.instruments(exchange)

    # ----------------------------------------------------------
    # HISTORICAL — single instrument, rate-limited
    # ----------------------------------------------------------
    def get_historical(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day"
    ) -> list:
        """
        Returns list of candles: [{date, open, high, low, close, volume}]
        Rate-limited to stay under 3 req/sec.
        """
        self.hist_limiter.wait()
        return self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
        )

    # ----------------------------------------------------------
    # QUOTE — batch OHLC for many instruments at once
    # ----------------------------------------------------------
    def get_ohlc_batch(self, instruments: list[str]) -> dict:
        """
        instruments format: ['NSE:RELIANCE', 'NSE:TCS', ...]
        Returns dict keyed by 'NSE:SYMBOL' with last_price + ohlc.
        Max 1000 per call per Kite docs; we chunk at 250 for safety.
        """
        result = {}
        chunk_size = config.QUOTE_BATCH_SIZE
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            result.update(self.kite.ohlc(chunk))
        return result

    def get_full_quote_batch(self, instruments: list[str]) -> dict:
        """
        Like get_ohlc_batch but returns full quote (volume, depth, etc).
        Limit is 500 per call.
        """
        result = {}
        chunk_size = 250
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            result.update(self.kite.quote(chunk))
        return result

    def get_ltp_batch(self, instruments: list[str]) -> dict[str, float]:
        """
        Fastest possible price fetch — returns only the last traded price.

        Kite's ltp() endpoint is lighter than ohlc() or quote():
          • Single field per symbol (last_price only)
          • Up to 1000 instruments per call
          • Typically responds in 100–300 ms for 50–100 symbols

        Returns: {"NSE:RELIANCE": 2451.5, "NSE:TCS": 3842.0, ...}
        """
        result = {}
        chunk_size = 500   # ltp() supports up to 1000; keep 500 for headroom
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            self.quote_limiter.wait()
            raw = self.kite.ltp(chunk)
            # raw format: {"NSE:RELIANCE": {"instrument_token": ..., "last_price": 2451.5}}
            for key, val in raw.items():
                result[key] = float(val["last_price"])
        return result
