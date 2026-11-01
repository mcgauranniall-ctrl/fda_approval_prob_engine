"""ClinicalTrials.gov API v2 client (https://clinicaltrials.gov/data-api/api)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from fdaengine.acquire.fetcher import Fetcher, default_fetcher
from fdaengine.dates import parse_partial

BASE = "https://clinicaltrials.gov/api/v2"
PHASE_ORDER = {"EARLY_PHASE1": 0.5, "PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}


@dataclass
class TrialSummary:
    nct_id: str
    title: str
    phases: list[str]
    status: str
    sponsor: str
    sponsor_class: str
    conditions: list[str]
    start_date: date | None
    primary_completion_date: date | None
    results_first_post_date: date | None
    first_post_date: date | None
    has_results: bool
    enrollment: int | None
    enrollment_type: str | None
    allocation: str | None
    masking: str | None
    arm_types: list[str]
    intervention_names: list[str] = field(default_factory=list)

    @property
    def max_phase(self) -> float:
        return max((PHASE_ORDER.get(p, 0) for p in self.phases), default=0)

    @property
    def phase_label(self) -> str:
        return "/".join(p.replace("PHASE", "P").replace("EARLY_P", "Early P") for p in self.phases) or "N/A"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, date):
                d[k] = v.isoformat()
        return d


def summarise(study: dict) -> TrialSummary:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    sponsor = p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    arms = p.get("armsInterventionsModule", {})
    names: list[str] = []
    for iv in arms.get("interventions", []):
        names.append(iv.get("name", ""))
        names.extend(iv.get("otherNames", []))
    return TrialSummary(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        phases=design.get("phases", []),
        status=status.get("overallStatus", ""),
        sponsor=sponsor.get("name", ""),
        sponsor_class=sponsor.get("class", ""),
        conditions=p.get("conditionsModule", {}).get("conditions", []),
        start_date=parse_partial(status.get("startDateStruct", {}).get("date")),
        primary_completion_date=parse_partial(status.get("primaryCompletionDateStruct", {}).get("date"), True),
        results_first_post_date=parse_partial(status.get("resultsFirstPostDateStruct", {}).get("date")),
        first_post_date=parse_partial(status.get("studyFirstPostDateStruct", {}).get("date")),
        has_results=bool(study.get("hasResults")),
        enrollment=design.get("enrollmentInfo", {}).get("count"),
        enrollment_type=design.get("enrollmentInfo", {}).get("type"),
        allocation=info.get("allocation"),
        masking=info.get("maskingInfo", {}).get("masking"),
        arm_types=[a.get("type", "") for a in arms.get("armGroups", [])],
        intervention_names=[n for n in names if n],
    )


def search_by_intervention(terms: list[str], fetcher: Fetcher | None = None, max_studies: int = 500) -> list[dict]:
    """Full study records whose intervention matches any of `terms`."""
    fetcher = fetcher or default_fetcher()
    query = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
    studies: list[dict] = []
    token = None
    while len(studies) < max_studies:
        params = {"query.intr": query, "pageSize": 100, "format": "json"}
        if token:
            params["pageToken"] = token
        data = fetcher.get(f"{BASE}/studies", params=params).json()
        studies.extend(data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break
    return studies


def get_study(nct_id: str, fetcher: Fetcher | None = None) -> dict:
    fetcher = fetcher or default_fetcher()
    return fetcher.get(f"{BASE}/studies/{nct_id}").json()


def large_documents(study: dict) -> list[dict]:
    """Protocol / SAP / ICF PDFs attached to a record, with their document and upload dates."""
    nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
    docs = study.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
    out = []
    for d in docs:
        out.append({
            "filename": d["filename"],
            "label": d.get("label", d.get("typeAbbrev", "")),
            "doc_date": d.get("date"),
            "upload_date": (d.get("uploadDate") or "")[:10] or None,
            "url": f"https://cdn.clinicaltrials.gov/large-docs/{nct[-2:]}/{nct}/{d['filename']}",
        })
    return out
