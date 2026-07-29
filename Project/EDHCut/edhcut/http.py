"""Shared HTTP session factory: requests-cache + tenacity retry + rate limiting + UA.

Every ingest module gets its session via `get_session(source_name)` instead of building
requests directly, so caching/backoff/politeness are consistent across sources (plan §5.1).
"""

from __future__ import annotations

import time
from typing import Any

import requests
import requests_cache
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from edhcut.config import CONFIG

# Last request time per source, process-wide (enforces the minimum delay between requests
# even across multiple session instances for the same source).
_last_request_at: dict[str, float] = {}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
        return response is not None and (response.status_code == 429 or response.status_code >= 500)
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    return False


class RateLimitedSession(requests_cache.CachedSession):
    """CachedSession that sleeps to enforce a minimum delay between non-cached requests."""

    def __init__(self, *args: Any, source_name: str, min_delay_seconds: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_name = source_name
        self._min_delay_seconds = min_delay_seconds

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        if not self._is_cached(request):
            self._throttle()
        return super().send(request, **kwargs)

    def _is_cached(self, request: requests.PreparedRequest) -> bool:
        try:
            return self.cache.contains(request=request)
        except Exception:
            return False

    def _throttle(self) -> None:
        last = _last_request_at.get(self._source_name, 0.0)
        wait = self._min_delay_seconds - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _last_request_at[self._source_name] = time.monotonic()


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception(_is_retryable),
)
def request_with_retry(session: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
    """`session.request` wrapped with exponential-backoff retry on 429/5xx."""
    response = session.request(method, url, **kwargs)
    response.raise_for_status()
    return response


def get_session(source_name: str) -> RateLimitedSession:
    """Build the shared cached/rate-limited/retrying session for one source.

    `source_name` must be a key in `CONFIG.rate_limits` (currently: scryfall, archidekt,
    edhrec) — this ties caching/delay policy to the plan §5 decision per source.
    """
    rate_limit = CONFIG.rate_limits[source_name]
    session = RateLimitedSession(
        cache_name=str(CONFIG.paths.http_cache_path),
        backend="sqlite",
        expire_after=rate_limit.cache_expire_seconds,
        allowable_codes=(200,),
        source_name=source_name,
        min_delay_seconds=rate_limit.min_delay_seconds,
    )
    session.headers.update({"User-Agent": CONFIG.user_agent})
    return session
