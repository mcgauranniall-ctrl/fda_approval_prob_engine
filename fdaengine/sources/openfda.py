"""openFDA client: Drugs@FDA applications and SPL labels.

Drugs@FDA submissions give FDA action dates and outcomes. They are used for timelines and outcome labels, never
as model inputs for the decision they describe.
"""

from __future__ import annotations

from dataclasses import dataclass

from fdaengine import config
from fdaengine.acquire.fetcher import Fetcher, default_fetcher
from fdaengine.dates import parse_partial

BASE = "https://api.fda.gov/drug"


@dataclass
class Submission:
    application_number: str
    submission_type: str          # ORIG / SUPPL
    submission_number: str
    status: str                   # AP = approved, TA = tentative approval, ...
    status_date: str | None       # ISO date of the FDA action
    review_priority: str | None   # PRIORITY / STANDARD
    class_code: str | None
    class_description: str | None


def _query(fetcher: Fetcher, endpoint: str, search: str, limit: int = 20) -> list[dict]:
    params = {"search": search, "limit": limit}
    if key := config.openfda_api_key():
        params["api_key"] = key
    resp = fetcher.get(f"{BASE}/{endpoint}.json", params=params, allow_404=True)
    if resp.status == 404:
        return []
    return resp.json().get("results", [])


def applications_for(names: list[str], fetcher: Fetcher | None = None) -> list[dict]:
    """Drugs@FDA application records whose active ingredient or brand matches any name."""
    fetcher = fetcher or default_fetcher()
    seen: dict[str, dict] = {}
    for name in names:
        n = name.upper().replace('"', "")
        for search in (f'products.active_ingredients.name:"{n}"', f'products.brand_name:"{n}"'):
            for rec in _query(fetcher, "drugsfda", search):
                seen.setdefault(rec["application_number"], rec)
    return list(seen.values())


def submissions(app: dict) -> list[Submission]:
    out = []
    for s in app.get("submissions", []):
        d = parse_partial(s.get("submission_status_date"))
        out.append(Submission(
            application_number=app["application_number"],
            submission_type=s.get("submission_type", ""),
            submission_number=str(s.get("submission_number", "")),
            status=s.get("submission_status", ""),
            status_date=d.isoformat() if d else None,
            review_priority=s.get("review_priority"),
            class_code=s.get("submission_class_code"),
            class_description=s.get("submission_class_code_description"),
        ))
    return sorted(out, key=lambda s: s.status_date or "")


def is_prescription(app: dict) -> bool:
    return any(p.get("marketing_status") == "Prescription" for p in app.get("products", []))


def labels_for(names: list[str], fetcher: Fetcher | None = None) -> list[dict]:
    fetcher = fetcher or default_fetcher()
    for name in names:
        n = name.upper().replace('"', "")
        for search in (f'openfda.generic_name:"{n}"', f'openfda.brand_name:"{n}"'):
            results = _query(fetcher, "label", search, limit=1)
            if results:
                return results
    return []
