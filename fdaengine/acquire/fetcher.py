"""HTTP fetching with an on-disk cache, per-host rate limiting and retries.

Every public source goes through `Fetcher`, so repeated runs of the same case are cheap and polite.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from fdaengine import config

# Minimum seconds between requests per host (SEC asks for <= 10 req/s; others are generous).
_HOST_INTERVAL = {
    "efts.sec.gov": 0.15,
    "www.sec.gov": 0.15,
    "api.fda.gov": 0.25,
    "clinicaltrials.gov": 0.1,
    "rxnav.nlm.nih.gov": 0.06,
    "www.accessdata.fda.gov": 0.3,
    "eutils.ncbi.nlm.nih.gov": 0.35,     # NCBI: <= 3 requests/s without an API key
    "www.federalregister.gov": 1.0,      # returns 429 when hit faster
}
_DEFAULT_UA = "fdaengine/0.1 (research tool)"


class FetchError(RuntimeError):
    pass


@dataclass
class Response:
    url: str
    status: int
    content: bytes
    content_type: str
    from_cache: bool

    def json(self):
        return json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, FetchError) and "retryable" in str(exc)


class Fetcher:
    def __init__(self, cache_dir: Path = config.CACHE_DIR, use_cache: bool = True):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=60, follow_redirects=True)
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def _headers_for(self, host: str) -> dict[str, str]:
        if host.endswith("sec.gov"):
            ua = config.sec_user_agent()
            if not ua:
                raise FetchError(
                    "SEC_USER_AGENT is not set. SEC EDGAR requires a User-Agent with your name and email; "
                    "add it to .env (see .env.example)."
                )
            return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        return {"User-Agent": _DEFAULT_UA}

    def _throttle(self, host: str) -> None:
        interval = _HOST_INTERVAL.get(host, 0.2)
        with self._lock:
            wait = self._last_call.get(host, 0) + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[host] = time.monotonic()

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode()).hexdigest()
        sub = self.cache_dir / key[:2]
        return sub / f"{key}.bin", sub / f"{key}.meta.json"

    def get(self, url: str, params: dict | None = None, allow_404: bool = False) -> Response:
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        body_path, meta_path = self._cache_paths(url)
        if self.use_cache and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return Response(url, meta["status"], body_path.read_bytes(), meta["content_type"], True)
        resp = self._get_live(url, allow_404)
        if resp.status == 200 or (allow_404 and resp.status == 404):
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(resp.content)
            meta_path.write_text(json.dumps({"url": url, "status": resp.status, "content_type": resp.content_type,
                                             "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        return resp

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    def _get_live(self, url: str, allow_404: bool) -> Response:
        host = urlparse(url).netloc
        self._throttle(host)
        r = self._client.get(url, headers=self._headers_for(host))
        if r.status_code in (429, 500, 502, 503, 504):
            raise FetchError(f"retryable HTTP {r.status_code} for {url}")
        if r.status_code == 404 and allow_404:
            return Response(url, 404, r.content, r.headers.get("content-type", ""), False)
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code} for {url}")
        return Response(url, 200, r.content, r.headers.get("content-type", ""), False)


_default: Fetcher | None = None


def default_fetcher() -> Fetcher:
    global _default
    if _default is None:
        _default = Fetcher()
    return _default
