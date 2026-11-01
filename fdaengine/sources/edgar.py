"""SEC EDGAR full-text search + document retrieval (press releases filed as 8-K exhibits)."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from fdaengine.acquire.fetcher import Fetcher, default_fetcher

FTS = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class Filing:
    adsh: str
    filename: str
    cik: str
    company: str
    ticker: str | None
    form: str
    file_type: str
    file_date: str
    url: str

    @property
    def doc_id(self) -> str:
        return f"{self.adsh}_{self.filename}"


def _company(display_names: list[str]) -> tuple[str, str, str | None]:
    """'VERTEX PHARMACEUTICALS INC / MA  (VRTX)  (CIK 0000875320)' -> (name, cik, ticker)."""
    name = display_names[0] if display_names else ""
    m = re.search(r"\(CIK (\d+)\)", name)
    cik = str(int(m.group(1))) if m else ""
    t = re.search(r"\(([A-Z.\-]{1,6})(?:,\s*[A-Z.\-]+)*\)\s*\(CIK", name)
    return re.sub(r"\s*\(.*$", "", name).strip(), cik, t.group(1) if t else None


def search(terms: list[str], forms: str = "8-K,6-K", start: str | None = None, end: str | None = None,
           fetcher: Fetcher | None = None, max_hits: int = 300) -> list[Filing]:
    """Full-text search for exact phrases; results are merged across terms and de-duplicated."""
    fetcher = fetcher or default_fetcher()
    found: dict[str, Filing] = {}
    for term in terms:
        offset = 0
        while offset < max_hits:
            params = {"q": f'"{term}"', "forms": forms, "from": offset}   # 6-K covers foreign filers
            if start or end:
                params.update({"dateRange": "custom", "startdt": start or "2001-01-01", "enddt": end or "2100-01-01"})
            data = fetcher.get(FTS, params=params).json()
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                src = h["_source"]
                adsh, _, filename = h["_id"].partition(":")
                company, cik, ticker = _company(src.get("display_names", []))
                cik = cik or (src.get("ciks") or [""])[0].lstrip("0")
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{filename}"
                found.setdefault(f"{adsh}:{filename}", Filing(adsh, filename, cik, company, ticker, src.get("form", ""),
                                                              src.get("file_type", ""), src.get("file_date", ""), url))
            if len(hits) < 100:
                break
            offset += 100
    return sorted(found.values(), key=lambda f: f.file_date)


def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def fetch_text(filing: Filing, fetcher: Fetcher | None = None) -> tuple[bytes, str]:
    fetcher = fetcher or default_fetcher()
    raw = fetcher.get(filing.url).content
    return raw, html_to_text(raw)
