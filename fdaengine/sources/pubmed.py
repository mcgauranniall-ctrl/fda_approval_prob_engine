"""PubMed (NCBI E-utilities): trial publications linked by ClinicalTrials.gov ID, and the primary-endpoint result
reported in their abstracts. Shared by the historical dataset (script 01b) and live cases.

Linking: PubMed indexes registry numbers in the secondary-source field, so `NCT01234567[si]` finds articles that
report that trial. A publication is usable as of its **availability date**: the electronic publication date when
PubMed records one, else the journal issue date taken at the end of its period (month → last day, year → 31 Dec),
so it is never earlier than the article was public.

Extraction is rule-based (D6): in the abstract's results (structured RESULTS section, else the whole abstract), the
first sentence about the primary endpoint that reports a p-value gives the worst p-value in it; failing that, a
ratio (HR/OR/RR) with a 95% CI in such a sentence gives |z| from the CI. Design papers, protocols, reviews,
meta-analyses, pooled/post-hoc/subgroup and extension reports are skipped.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date

from lxml import etree
from scipy import stats

from fdaengine.acquire.fetcher import Fetcher
from fdaengine.features import stats_transforms as st
from fdaengine.timeline.press_release_parser import PVALUE, _pvalue_number, split_sentences

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "fdaengine"
SEARCH_BATCH = 100
FETCH_BATCH = 200
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}

SKIP_TYPES = {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Comment", "Letter", "News",
              "Published Erratum", "Retraction of Publication", "Clinical Trial Protocol", "Case Reports"}
SKIP_TITLE = re.compile(r"\b(design|rationale|protocol|baseline characteristics|methodology|post.?hoc|sub.?group|"
                        r"pooled|subanalys|sub-analys|secondary analys|exploratory|open.label extension|"
                        r"long.term extension|extension study|pharmacokinetic|cost.effective|economic|"
                        r"quality of life|patient.reported|resource|utili[sz]ation|productivity|satisfaction|"
                        r"biomarkers?|markers?|genetic|polymorphisms?|predictors?|predictive|association|"
                        r"correlat\w*|real.world|budget|plain language|psychometric|validation|trajectory|"
                        r"status|by age|in subgroups|according to|stratified|update|follow.?up|final results|"
                        r"validity|reliability|prognostic|after the end of|comparison of)\b",
                        re.IGNORECASE)
PRIMARY = re.compile(r"\bprimary\b", re.IGNORECASE)
# The fallback (first p-value of a main paper) must compare arms and not be a safety statement.
COMPARISON = re.compile(r"\b(versus|vs\.?|compared|placebo|than|hazard ratio|odds ratio|relative risk|difference)\b",
                        re.IGNORECASE)
SAFETY = re.compile(r"adverse|discontinu|tolerab|safety|toxicit|at baseline|baseline (characteristics|values?)|"
                    r"completed the (trial|study)",
                    re.IGNORECASE)
NOT_PRIMARY = re.compile(r"secondary|post.?hoc|sub.?group|exploratory|sensitivity analys|per.protocol", re.IGNORECASE)
RATIO_CI = re.compile(r"\b(hazard ratio|odds ratio|risk ratio|relative risk|rate ratio|HR|OR|RR)\b[^0-9]{0,25}"
                      r"(\d+\.\d+)\s*[;,(\[]?\s*(?:95\s*%\s*(?:CI|confidence interval)[:,]?\s*)?\(?\s*"
                      r"(\d+\.\d+)\s*(?:-|–|to|,)\s*(\d+\.\d+)", re.IGNORECASE)


@dataclass
class Publication:
    pmid: str
    ncts: list[str]
    title: str
    journal: str
    available: str                  # ISO date the article was public
    date_basis: str                 # 'epub' | 'issue_day' | 'issue_month' | 'issue_year'
    pub_types: list[str]
    first_p_value: float | None = None     # first p-value in the results (fallback for a trial's main paper)
    first_quote: str | None = None
    p_value: float | None = None
    z_abs: float | None = None
    result_basis: str | None = None  # 'p_value' | 'ratio_ci'
    quote: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ network
def search_by_nct(fetcher: Fetcher, ncts: list[str]) -> list[str]:
    """PMIDs of articles that list any of these trial IDs (one request per SEARCH_BATCH IDs)."""
    pmids: list[str] = []
    for i in range(0, len(ncts), SEARCH_BATCH):
        term = " OR ".join(f"{n}[si]" for n in ncts[i:i + SEARCH_BATCH])
        r = fetcher.get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": term, "retmax": 10000,
                                                    "retmode": "json", "tool": TOOL})
        pmids += r.json()["esearchresult"].get("idlist", [])
    return sorted(set(pmids), key=int)


def fetch(fetcher: Fetcher, pmids: list[str]) -> list[Publication]:
    out = []
    for i in range(0, len(pmids), FETCH_BATCH):
        r = fetcher.get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids[i:i + FETCH_BATCH]),
                                                   "retmode": "xml", "tool": TOOL})
        out += parse_xml(r.content)
    return out


# ------------------------------------------------------------------------------------------------ parsing
def _txt(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _issue_date(pd_el) -> tuple[date, str] | None:
    if pd_el is None:
        return None
    y = pd_el.findtext("Year")
    if not y:
        m = re.search(r"\b(19|20)\d{2}\b", pd_el.findtext("MedlineDate") or "")
        return (date(int(m.group(0)), 12, 31), "issue_year") if m else None
    y = int(y)
    mon = (pd_el.findtext("Month") or "").strip()
    mi = int(mon) if mon.isdigit() else _MONTHS.get(mon[:3].lower())
    if not mi:
        return date(y, 12, 31), "issue_year"
    day = pd_el.findtext("Day")
    if day and day.isdigit():
        return date(y, mi, int(day)), "issue_day"
    return date(y, mi, calendar.monthrange(y, mi)[1]), "issue_month"


def availability(article) -> tuple[date, str] | None:
    for ad in article.findall(".//ArticleDate"):
        try:
            return date(int(ad.findtext("Year")), int(ad.findtext("Month")), int(ad.findtext("Day"))), "epub"
        except (TypeError, ValueError):
            continue
    return _issue_date(article.find(".//Journal/JournalIssue/PubDate"))


def parse_xml(xml: bytes) -> list[Publication]:
    root = etree.fromstring(xml)
    out = []
    for pa in root.findall("PubmedArticle"):
        cit = pa.find("MedlineCitation")
        art = cit.find("Article")
        av = availability(art)
        if av is None:
            continue
        ncts = sorted({_txt(a).upper() for db in art.findall(".//DataBank")
                       if (db.findtext("DataBankName") or "").lower().startswith("clinicaltrials")
                       for a in db.findall(".//AccessionNumber")})
        pub = Publication(pmid=cit.findtext("PMID"), ncts=ncts, title=_txt(art.find("ArticleTitle")),
                          journal=_txt(art.find(".//Journal/ISOAbbreviation")), available=av[0].isoformat(),
                          date_basis=av[1], pub_types=[_txt(t) for t in art.findall(".//PublicationType")])
        sections = [(t.get("Label") or t.get("NlmCategory") or "", _txt(t).replace("·", "."))
                    for t in art.findall(".//Abstract/AbstractText")]
        res = primary_result(pub, sections)
        if res:
            pub.p_value, pub.z_abs, pub.result_basis, pub.quote = res
        first = first_result_p(pub, sections)
        if first:
            pub.first_p_value, pub.first_quote = first
        out.append(pub)
    return out


def is_results_report(pub: Publication) -> bool:
    return not (set(pub.pub_types) & SKIP_TYPES) and not SKIP_TITLE.search(pub.title)


def _results_text(sections: list[tuple[str, str]]) -> str:
    results = [t for label, t in sections if re.match(r"(?i)results?|findings", label)]
    return " ".join(results) if results else " ".join(t for _l, t in sections)


def first_result_p(pub: Publication, sections: list[tuple[str, str]]) -> tuple[float, str] | None:
    """Worst p-value of the first results sentence reporting one (not about secondary/subgroup analyses)."""
    if not sections or not is_results_report(pub):
        return None
    for s in split_sentences(_results_text(sections)):
        ps = [p for p in (_pvalue_number(m.group(1)) for m in PVALUE.finditer(s)) if p is not None and 0 < p <= 1]
        if ps:
            ok = COMPARISON.search(s) and not NOT_PRIMARY.search(s) and not SAFETY.search(s)
            return (max(ps), s[:600]) if ok else None
    return None


def trial_result(pubs: list[Publication]) -> dict | None:
    """The trial's result from its main paper: the earliest results report (by availability). A sentence about the
    primary endpoint is preferred; otherwise the first p-value in its results. Returns None when the main paper
    reports neither (later secondary papers are never used)."""
    reports = sorted((p for p in pubs if is_results_report(p)), key=lambda p: (p.available, int(p.pmid)))
    for p in reports[:1]:
        if p.z_abs is not None:
            return {"pmid": p.pmid, "available": p.available, "p_value": p.p_value, "z_abs": p.z_abs,
                    "basis": p.result_basis, "quote": p.quote, "title": p.title}
        if p.first_p_value is not None:
            return {"pmid": p.pmid, "available": p.available, "p_value": p.first_p_value,
                    "z_abs": st.z_from_p(p.first_p_value), "basis": "first_p_value", "quote": p.first_quote,
                    "title": p.title}
    return None


def primary_result(pub: Publication, sections: list[tuple[str, str]]) -> tuple | None:
    """(p, |z|, basis, quote) for the primary endpoint, or None."""
    if not sections or not is_results_report(pub):
        return None
    for s in split_sentences(_results_text(sections)):
        if not PRIMARY.search(s) or NOT_PRIMARY.search(s):
            continue
        ps = [p for p in (_pvalue_number(m.group(1)) for m in PVALUE.finditer(s)) if p is not None and 0 < p <= 1]
        if ps:
            p = max(ps)
            return p, st.z_from_p(p), "p_value", s[:600]
        m = RATIO_CI.search(s)
        if m:
            est, lo, hi = (float(m.group(k)) for k in (2, 3, 4))
            if 0 < lo <= est <= hi:
                z = abs(st.z_from_ratio_ci(est, lo, hi))
                return float(2 * stats.norm.sf(z)), z, "ratio_ci", s[:600]
    return None
