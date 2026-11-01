"""Resolve a drug + indication into a programme: synonyms, candidate trials, FDA applications, sponsor.

The output is a set of *suggestions*. The user confirms which trials belong to the programme (UI step 2), which
avoids v1's habit of mixing up trials and indications of the same drug.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from fdaengine.acquire.fetcher import Fetcher, FetchError, default_fetcher
from fdaengine.sources import ctgov, edgar, openfda, rxnorm

_STOP = {"the", "of", "and", "or", "with", "in", "for", "to", "a", "an", "associated", "disease", "disorder",
         "patients", "adults", "adult", "treatment", "moderate", "severe", "mild", "including", "due"}
_CODE_NAME = re.compile(r"^[A-Z]{1,6}[- ]?\d{2,7}[A-Z]?$")


@dataclass
class Candidate:
    trial: ctgov.TrialSummary
    indication_score: float
    sponsor_match: bool
    suggested: bool
    reason: str


@dataclass
class Resolution:
    drug: str
    indication: str
    names: list[str]
    sponsor: str | None
    candidates: list[Candidate]
    studies: dict[str, dict]                     # nct_id -> full record
    applications: list[dict]
    ticker: str | None = None
    cik: str | None = None
    warnings: list[str] = field(default_factory=list)


def indication_keywords(indication: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", indication.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


def indication_score(indication: str, trial: ctgov.TrialSummary) -> float:
    """Percent of the indication's keywords found (fuzzily) in the trial's conditions or title."""
    keywords = indication_keywords(indication)
    if not keywords:
        return 0.0
    words = set(re.findall(r"[a-z0-9]+", " ".join(trial.conditions + [trial.title]).lower()))
    hits = sum(any(fuzz.ratio(k, w) >= 85 for w in words) for k in keywords)
    return 100.0 * hits / len(keywords)


def _expand_names(drug: str, studies: list[dict], seed: set[str]) -> set[str]:
    """Add code names / aliases that ClinicalTrials.gov lists alongside a known name."""
    lowered = {n.lower() for n in seed}
    found = set(seed)
    for s in studies:
        for iv in s.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", []):
            aliases = [iv.get("name", "")] + iv.get("otherNames", [])
            if any(a.lower() in lowered or any(n in a.lower() for n in lowered) for a in aliases):
                for a in aliases:
                    a = a.strip()
                    if _CODE_NAME.match(a.upper()) or a.lower() in lowered:
                        found.add(a)
    return found


def _sponsor_from_edgar(names: list[str], sponsor: str | None,
                        fetcher: Fetcher) -> tuple[str | None, str | None, str | None]:
    """(company, cik, ticker) of the SEC filer that most likely sponsors the programme."""
    filings = edgar.search(names[:3], fetcher=fetcher, max_hits=100)
    if not filings:
        return None, None, None
    companies = Counter((f.company, f.cik, f.ticker) for f in filings)
    if sponsor:
        first_word = sponsor.split()[0].lower()
        for key, _ in companies.most_common():
            if key[0].lower().startswith(first_word):
                return key
    return companies.most_common(1)[0][0]


def resolve(drug: str, indication: str, extra_ncts: list[str] | None = None,
            fetcher: Fetcher | None = None) -> Resolution:
    fetcher = fetcher or default_fetcher()
    warnings: list[str] = []
    names = {drug.strip()}
    try:
        names.update(rxnorm.related_names(drug, fetcher))
    except FetchError as e:
        warnings.append(f"RxNorm lookup failed: {e}")

    studies = ctgov.search_by_intervention(sorted(names), fetcher)
    expanded = _expand_names(drug, studies, names)
    if expanded - names:
        names = expanded
        studies = ctgov.search_by_intervention(sorted(names), fetcher)
    for nct in extra_ncts or []:
        if nct and nct not in {s["protocolSection"]["identificationModule"]["nctId"] for s in studies}:
            studies.append(ctgov.get_study(nct, fetcher))

    by_id = {s["protocolSection"]["identificationModule"]["nctId"]: s for s in studies}
    summaries = [ctgov.summarise(s) for s in studies]
    industry = Counter(t.sponsor for t in summaries if t.sponsor_class == "INDUSTRY" and 0 < t.max_phase <= 3)
    sponsor = industry.most_common(1)[0][0] if industry else None

    candidates = []
    for t in summaries:
        score = indication_score(indication, t)
        sponsor_match = bool(sponsor) and t.sponsor == sponsor
        if t.max_phase == 0 or t.max_phase >= 4:
            suggested, reason = False, "Phase 4 / not applicable"
        elif not sponsor_match:
            suggested, reason = False, f"not run by the lead sponsor ({t.sponsor})"
        elif t.max_phase < 2:
            suggested, reason = True, "sponsor Phase 1 (drug-level safety/PK)"
        elif score >= 70:
            suggested, reason = True, f"sponsor trial in matching indication ({score:.0f}% match)"
        else:
            suggested, reason = False, f"different indication ({score:.0f}% match)"
        candidates.append(Candidate(t, score, sponsor_match, suggested, reason))
    candidates.sort(key=lambda c: (not c.suggested, -c.trial.max_phase, c.trial.start_date or date.max))

    applications: list[dict] = []
    try:
        applications = openfda.applications_for(sorted(names), fetcher)
    except FetchError as e:
        warnings.append(f"openFDA lookup failed: {e}")

    company = cik = ticker = None
    try:
        company, cik, ticker = _sponsor_from_edgar(sorted(names, key=len, reverse=True), sponsor, fetcher)
        if company is None:
            warnings.append("No SEC filings mention this drug (private sponsor?); press releases unavailable.")
    except FetchError as e:
        warnings.append(f"SEC EDGAR unavailable: {e}")
    if not any(c.suggested and c.trial.max_phase >= 2 for c in candidates):
        warnings.append("No sponsor Phase 2/3 trials matched the indication; check the indication wording or add NCT IDs.")

    return Resolution(drug=drug, indication=indication, names=sorted(names), sponsor=sponsor,
                      candidates=candidates, studies=by_id, applications=applications,
                      ticker=ticker, cik=cik, warnings=warnings)
