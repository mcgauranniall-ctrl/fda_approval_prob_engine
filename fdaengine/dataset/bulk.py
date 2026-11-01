"""Bulk raw-data store for the offline training pipeline (PLAN.md §5): streaming downloads + sha256 manifest.

Everything lands under data/raw/<source>/. `data/raw/manifest.json` records, for every file, where it came from,
when, its size and sha256, and the source's own export date. Later scripts read files only through the manifest,
so a dataset build can always say exactly which snapshot it was built from.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from fdaengine import config

RAW_DIR = config.DATA_DIR / "raw"
MANIFEST = RAW_DIR / "manifest.json"
_UA = "fdaengine/0.1 (research tool; bulk download)"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, MANIFEST)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_current(manifest: dict, rel: str, version: str | None = None) -> bool:
    """True if `rel` is in the manifest, exists with the recorded size, and (if given) has the same version."""
    entry = manifest["files"].get(rel)
    path = RAW_DIR / rel
    if not entry or not path.exists() or path.stat().st_size != entry["bytes"]:
        return False
    return version is None or entry.get("source_version") == version


def record(manifest: dict, rel: str, source: str, url: str, sha: str, version: str | None = None,
           **extra) -> dict:
    path = RAW_DIR / rel
    entry = {"source": source, "url": url, "bytes": path.stat().st_size, "sha256": sha,
             "downloaded_at": now_iso(), "source_version": version, **extra}
    manifest["files"][rel] = entry
    return entry


@retry(retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
       stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
def download(url: str, dest: Path, headers: dict | None = None) -> tuple[str, dict]:
    """Stream `url` to `dest` atomically. Returns (sha256, response headers)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    with httpx.stream("GET", url, headers={"User-Agent": _UA, **(headers or {})}, timeout=120,
                      follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                h.update(chunk)
        resp_headers = dict(r.headers)
    os.replace(tmp, dest)
    return h.hexdigest(), resp_headers


class JsonlGzWriter:
    """Write JSON records to a .jsonl.gz file atomically (rename on close), hashing the bytes on disk."""

    def __init__(self, dest: Path):
        self.dest = dest
        self.tmp = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0 keeps the gzip bytes, and so the sha256, reproducible for identical content.
        self._fh = gzip.GzipFile(self.tmp, "wb", compresslevel=6, mtime=0)
        self.n = 0

    def write(self, obj) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self._fh.write(b"\n")
        self.n += 1

    def close(self) -> str:
        self._fh.close()
        os.replace(self.tmp, self.dest)
        return sha256_file(self.dest)

    def abort(self) -> None:
        self._fh.close()
        self.tmp.unlink(missing_ok=True)


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def raw_files(source: str, manifest: dict | None = None) -> list[Path]:
    """Manifest-registered files of one source, in path order."""
    manifest = manifest or load_manifest()
    return sorted(RAW_DIR / rel for rel, e in manifest["files"].items() if e["source"] == source)


class Throttle:
    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()
