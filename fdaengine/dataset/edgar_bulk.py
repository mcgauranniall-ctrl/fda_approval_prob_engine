"""Bulk SEC EDGAR full-text search crawl for the training timelines (PLAN.md §5 script 03). No LLM calls (D6).

Queries are by *event phrase and year across all filers* (not per drug), so the crawl is the same for every
programme and does not depend on knowing outcomes. Results live in one resumable SQLite store:
  hits(doc_id, adsh, filename, cik, company, form, file_type, file_date, url)  one row per matching document
  hit_phrases(doc_id, phrase)                                                 which phrase queries found it
  queries(phrase, start, end, n_hits, done_at)                                completed index queries
  docs(doc_id, sha256, n_bytes, text_z, fetched_at, error)                    zlib-compressed document text
"""

from __future__ import annotations

import calendar
import hashlib
import sqlite3
import zlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from fdaengine.acquire.fetcher import Fetcher, FetchError
from fdaengine.dataset.bulk import RAW_DIR, now_iso
from fdaengine.sources.edgar import FTS, _company, html_to_text

DB_PATH = RAW_DIR / "edgar" / "edgar_fts.sqlite"
FORMS = "8-K,6-K"
FIRST_YEAR = 2001                    # EDGAR full-text search coverage starts in 2001
# Press releases are filed as EX-99.x exhibits or as the 8-K/6-K body; contracts (EX-10), merger agreements (EX-2),
# charters etc. also match the phrases but are not event announcements.
PRESS_RELEASE_TYPES = "(file_type LIKE 'EX-99%' OR file_type IN ('8-K', '8-K/A', '6-K', '6-K/A'))"
FTS_CAP = 9_900                      # FTS returns at most 10,000 hits per query; split into months above this

# Pre-registered event phrases (fixed before looking at any outcome).
PHRASES = ["PDUFA", "complete response letter", "accepted for filing", "accepted for review", "topline",
           "top-line results", "top-line data", "primary endpoint", "advisory committee", "priority review",
           "new drug application", "biologics license application", "Breakthrough Therapy", "Fast Track",
           "orphan drug designation", "FDA approval", "Special Protocol Assessment", "action date",
           "refuse to file"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hits (doc_id TEXT PRIMARY KEY, adsh TEXT, filename TEXT, cik TEXT, company TEXT,
    ticker TEXT, form TEXT, file_type TEXT, file_date TEXT, url TEXT);
CREATE TABLE IF NOT EXISTS hit_phrases (doc_id TEXT, phrase TEXT, PRIMARY KEY (doc_id, phrase));
CREATE TABLE IF NOT EXISTS queries (phrase TEXT, start TEXT, "end" TEXT, n_hits INTEGER, done_at TEXT,
    PRIMARY KEY (phrase, start, "end"));
CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, sha256 TEXT, n_bytes INTEGER, text_z BLOB,
    fetched_at TEXT, error TEXT);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript(_SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _windows(year: int, split: bool) -> list[tuple[str, str]]:
    if not split:
        return [(f"{year}-01-01", f"{year}-12-31")]
    return [(f"{year}-{m:02d}-01", f"{year}-{m:02d}-{calendar.monthrange(year, m)[1]:02d}") for m in range(1, 13)]


def _search_page(fetcher: Fetcher, phrase: str, start: str, end: str, offset: int) -> dict:
    return fetcher.get(FTS, params={"q": f'"{phrase}"', "forms": FORMS, "dateRange": "custom", "startdt": start,
                                    "enddt": end, "from": offset}).json()


def index_query(con: sqlite3.Connection, fetcher: Fetcher, phrase: str, start: str, end: str) -> int:
    """Page through one (phrase, window) query and store its hits. Returns the hit count."""
    offset, n = 0, 0
    while True:
        data = _search_page(fetcher, phrase, start, end, offset)
        hits = data.get("hits", {}).get("hits", [])
        rows, prow = [], []
        for h in hits:
            src = h["_source"]
            adsh, _, filename = h["_id"].partition(":")
            company, cik, ticker = _company(src.get("display_names", []))
            cik = cik or (src.get("ciks") or [""])[0].lstrip("0")
            doc_id = f"{adsh}_{filename}"
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{filename}"
            rows.append((doc_id, adsh, filename, cik, company, ticker, src.get("form", ""),
                         src.get("file_type", ""), src.get("file_date", ""), url))
            prow.append((doc_id, phrase))
        with con:
            con.executemany("INSERT OR IGNORE INTO hits VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            con.executemany("INSERT OR IGNORE INTO hit_phrases VALUES (?,?)", prow)
        n += len(hits)
        if len(hits) < 100 or offset + 100 >= 10_000:
            break
        offset += 100
    with con:
        con.execute("INSERT OR REPLACE INTO queries VALUES (?,?,?,?,?)", (phrase, start, end, n, now_iso()))
    return n


def plan_index(con: sqlite3.Connection, fetcher: Fetcher, phrases: list[str], last_year: int) -> Iterator[tuple]:
    """(phrase, start, end) windows still to run. Past years are final once done; the current year is re-run."""
    done = {(p, s, e) for p, s, e in con.execute('SELECT phrase, start, "end" FROM queries')}
    this_year = date.today().year
    for phrase in phrases:
        for year in range(FIRST_YEAR, last_year + 1):
            year_win = (f"{year}-01-01", f"{year}-12-31")
            months = _windows(year, True)
            if year != this_year and ((phrase, *year_win) in done or all((phrase, *m) in done for m in months)):
                continue
            total = _search_page(fetcher, phrase, *year_win, 0)["hits"]["total"]["value"]
            for win in _windows(year, total >= FTS_CAP):
                yield phrase, *win


def pending_docs(con: sqlite3.Connection) -> list[tuple[str, str]]:
    return con.execute("SELECT h.doc_id, h.url FROM hits h LEFT JOIN docs d USING (doc_id) "
                       f"WHERE d.doc_id IS NULL AND {PRESS_RELEASE_TYPES} ORDER BY h.file_date").fetchall()


def fetch_docs(con: sqlite3.Connection, fetcher: Fetcher, todo: list[tuple[str, str]], workers: int = 4,
               progress=print) -> None:
    """Download documents (the fetcher's per-host throttle keeps the rate within SEC's limit)."""

    def one(item):
        doc_id, url = item
        try:
            raw = fetcher.get(url).content
            text = html_to_text(raw)
            return (doc_id, hashlib.sha256(raw).hexdigest(), len(raw), zlib.compress(text.encode("utf-8"), 6),
                    now_iso(), None)
        except (FetchError, OSError, ValueError) as e:
            return (doc_id, None, 0, None, now_iso(), str(e)[:300])

    batch = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(one, todo), 1):
            batch.append(row)
            if len(batch) >= 200 or i == len(todo):
                with con:
                    con.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)", batch)
                batch = []
                progress(f"[docs] {i:,}/{len(todo):,}")


def doc_text(blob: bytes | None) -> str:
    return zlib.decompress(blob).decode("utf-8") if blob else ""
