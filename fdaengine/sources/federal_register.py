"""Federal Register notices of FDA advisory committee meetings (federalregister.gov API, public, no key).

FDA must announce each advisory committee meeting in the Federal Register, normally at least 15 days ahead, with an
agenda naming the application (NDA/BLA number), product, applicant and proposed indication. A notice is public on
its publication date, so it can date an "AdCom scheduled" event without hindsight; the meeting date is kept as the
event's value. Votes are not in the notices (they come from press releases).
"""

from __future__ import annotations

import re
from datetime import date

from fdaengine.acquire.fetcher import Fetcher

API = "https://www.federalregister.gov/api/v1/documents.json"
FIELDS = ["title", "publication_date", "document_number", "raw_text_url", "html_url"]
TITLE = re.compile(r"advisory committees?.*notice of (a )?meetings?", re.IGNORECASE)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
MEETING_DATE = re.compile(r"(?:held on|meeting will be held|date(?:s)? and time[s]?[:.]?)[^.]{0,60}?\b("
                          + "|".join(m.capitalize() for m in _MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})", re.IGNORECASE)
APPLICATION = re.compile(r"\b(?:NDA|BLA|sNDA|sBLA|new drug application|biologics license application)\s*\(?"
                         r"(?:NDA|BLA)?\)?\s*(\d{2,3}[-\s]?\d{3,4})", re.IGNORECASE)


def list_notices(fetcher: Fetcher, start_year: int = 1994, end_year: int | None = None) -> list[dict]:
    """FDA advisory-committee meeting notices, one API query per year (each well under the 10,000-result cap)."""
    end_year = end_year or date.today().year
    out = []
    for y in range(start_year, end_year + 1):
        page = 1
        while True:
            r = fetcher.get(API, {"conditions[agencies][]": "food-and-drug-administration",
                                  "conditions[type][]": "NOTICE", "conditions[term]": '"advisory committee"',
                                  "conditions[publication_date][gte]": f"{y}-01-01",
                                  "conditions[publication_date][lte]": f"{y}-12-31",
                                  "per_page": 1000, "page": page, "order": "oldest", "fields[]": FIELDS}).json()
            out += [d for d in r.get("results", []) if TITLE.search(d.get("title") or "")]
            if page >= (r.get("total_pages") or 1):
                break
            page += 1
    return out


def meeting_dates(text: str) -> list[tuple[int, date]]:
    """(position, date) of each meeting date stated in the notice."""
    out = []
    for m in MEETING_DATE.finditer(text):
        try:
            out.append((m.start(), date(int(m.group(3)), _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2)))))
        except ValueError:
            continue
    return out


def meeting_date_near(dates: list[tuple[int, date]], pos: int) -> date | None:
    """The meeting date governing a mention: the last one stated before it, else the first after it (notices of
    several meetings give each committee's date ahead of its agenda)."""
    before = [d for p, d in dates if p <= pos]
    if before:
        return before[-1]
    return dates[0][1] if dates else None


def agenda_segments(text: str, max_len: int = 3000) -> list[tuple[int, str]]:
    """(position, text) of each agenda section ('Agenda: The committee will discuss NDA ...'), up to the next
    'Procedure' heading or agenda; the whole notice when it has no agenda heading."""
    starts = [m.start() for m in re.finditer(r"\bAgenda\s*[:.]", text)]
    if not starts:
        return [(0, text[:20000])]
    out = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        proc = re.search(r"\bProcedure\s*[:.]", text[s:end])
        out.append((s, text[s:s + min(proc.start() if proc else end - s, max_len)]))
    return out


BIOSIMILAR_OF = re.compile(r"(biosimilar|interchangeable)\s+(product\s+)?to\b.*", re.IGNORECASE)


def application_sentences(segment: str) -> list[tuple[str, set[str]]]:
    """(sentence, application numbers) for agenda sentences that cite an application. A drug only counts as the
    subject of a meeting when it is named in such a sentence; the reference product of a proposed biosimilar
    ('ABP 215, a proposed biosimilar to Genentech's AVASTIN (bevacizumab)') is cut off."""
    out = []
    for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", " ".join(segment.split())):
        apps = applications(s)
        if apps:
            out.append((BIOSIMILAR_OF.sub("", s), apps))
    return out


def applications(text: str) -> set[str]:
    return {re.sub(r"\D", "", m.group(1)).zfill(6) for m in APPLICATION.finditer(text)}
