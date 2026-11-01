"""Training timelines (PLAN.md §5 script 03): documents → programme events → levels and first decision.

Everything reuses the live-case code: `press_release_parser.parse_document` for events, `timeline.events.build`
for assembly and `timeline.levels.compute` / `decision` for cut-offs. Only the inputs differ: the trials come from
the bulk ClinicalTrials.gov tables, the press releases from the bulk EDGAR crawl, and FDA actions from the
Drugs@FDA bulk file (converted to the openFDA record shape the live code expects).

Drug mentions are matched on INNs, development codes and ClinicalTrials.gov intervention names, never on brand
names (a brand name only exists after approval). Events count only when the filer is a sponsor or collaborator
of the programme's trials, or discusses the drug in at least `REPEAT_FILINGS` filings (licensees, acquirers);
competitor mentions are kept in the table but flagged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

from fdaengine.dataset import drugs
from fdaengine.resolve.programme import indication_keywords
from fdaengine.sources import ctgov
from fdaengine.timeline.press_release_parser import _keyword_coverage, body_sentences, parse_document

REPEAT_FILINGS = 3
WINDOW_BEFORE_YEARS, WINDOW_AFTER_YEARS = 3, 12       # documents considered around a programme's anchor
MAX_NGRAM = 4
_CORP = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|plc|ag|sa|se|nv|bv|gmbh|kk|"
                   r"holdings?|group|pharmaceuticals?|pharma|therapeutics|biosciences?|biotherapeutics|"
                   r"biopharma(ceuticals)?|biotech(nology)?|laboratories|labs?|research|development|r&d|"
                   r"the|of|and|us|usa|international|global)\b|/[a-z]{2}/|[^a-z0-9 ]", re.IGNORECASE)


def company_key(name: str) -> str:
    return re.sub(r"\s+", " ", _CORP.sub(" ", name.lower())).strip()


def same_company(a: str, b: str) -> bool:
    ka, kb = company_key(a), company_key(b)
    if not ka or not kb:
        return False
    first_a, first_b = ka.split()[0], kb.split()[0]
    return ka == kb or fuzz.token_set_ratio(ka, kb) >= 90 or (first_a == first_b and len(first_a) >= 5)


# ------------------------------------------------------------------------------------------------ aliases
def alias_ok(s: str) -> bool:
    k = drugs.key(s)
    if len(k) < 5 and not (len(k) >= 4 and any(c.isdigit() for c in k)):
        return False
    return not (k in drugs.GENERIC_WORDS or k in drugs.SALT_WORDS or drugs.NON_DRUG.search(s))


CODE_NAME = re.compile(r"^[A-Z]{1,6}[- ]?\d{2,7}[A-Z]?$")
MAX_ALIAS_DF = 0.03                     # an alias in >3% of documents is an ordinary word, not a drug name


def build_aliases(drug_ids: set[str], ctgov_names: dict[str, list[str]], idx: drugs.GsrsIndex,
                  common_keys: set[str] = frozenset()) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(alias key → drug id, drug id → alias strings).

    Aliases are GSRS common/official names and codes, plus ClinicalTrials.gov intervention names that GSRS confirms
    for the drug or that are code-shaped ('VX-548'). Brand names, ambiguous aliases, numbers and `common_keys`
    (words too frequent in the corpus to identify a drug) are dropped.
    """
    brand_keys = set()
    per_drug: dict[str, set[str]] = defaultdict(set)
    for did in drug_ids:
        moieties = did.removeprefix("UNII:").split("+") if did.startswith("UNII:") else []
        gsrs_keys: set[str] = set()
        if len(moieties) == 1:
            for name, ntype in idx.names.get(moieties[0], ()):
                clean = re.sub(r"\s*\[[^\]]*\]", "", name).strip()
                if ntype == "bn":
                    brand_keys.add(drugs.key(clean))
                elif ntype in ("cn", "cd", "of") and len(clean) <= 40:
                    per_drug[did].add(clean)
                    gsrs_keys.add(drugs.key(clean))
        for n in ctgov_names.get(did, []):
            for s in [n, drugs.strip_dose(n), *re.findall(r"\(([^()]*)\)", n)]:
                s = s.strip()
                if s and (drugs.key(s) in gsrs_keys or CODE_NAME.match(s.upper())):
                    per_drug[did].add(s)
        if did.startswith("NAME:"):
            per_drug[did].add(did.removeprefix("NAME:"))
    owners: dict[str, set[str]] = defaultdict(set)
    for did, names in per_drug.items():
        for s in names:
            k = drugs.key(s)
            if alias_ok(s) and k not in brand_keys and k not in common_keys and not k.isdigit():
                owners[k].add(did)
    key_to_drug = {k: next(iter(ds)) for k, ds in owners.items() if len(ds) == 1}
    names_out: dict[str, list[str]] = defaultdict(list)
    for did, names in per_drug.items():
        names_out[did] = sorted({s for s in names if key_to_drug.get(drugs.key(s)) == did}, key=len)
    return key_to_drug, dict(names_out)


def common_alias_keys(texts, key_to_drug: dict[str, str], max_df: float = MAX_ALIAS_DF) -> set[str]:
    """Alias keys found in more than `max_df` of the sample documents."""
    df: dict[str, int] = defaultdict(int)
    n = 0
    for text in texts:
        n += 1
        for keys in drugs_mentioned(text, key_to_drug).values():
            for k in keys:
                df[k] += 1
    return {k for k, c in df.items() if n and c / n > max_df}


def drugs_mentioned(text: str, key_to_drug: dict[str, str]) -> dict[str, set[str]]:
    """drug id → alias keys found in `text` (word n-grams compacted the same way as alias keys)."""
    toks = re.findall(r"[A-Za-z0-9]+", text.upper())
    found: dict[str, set[str]] = defaultdict(set)
    for i in range(len(toks)):
        k = ""
        for j in range(i, min(i + MAX_NGRAM, len(toks))):
            k += toks[j]
            did = key_to_drug.get(k)
            if did:
                found[did].add(k)
    return found


# ------------------------------------------------------------------------------------------------ programmes
@dataclass
class ProgInfo:
    programme_id: str
    drug_id: str
    indication: str
    keywords: list[str]
    anchor: str | None
    sponsors: set[str] = field(default_factory=set)


def programme_keywords(indication_label: str, condition_texts: list[str]) -> list[str]:
    kw = indication_keywords(indication_label)
    if len(kw) < 2:
        for t in condition_texts:
            kw += [w for w in indication_keywords(t) if w not in kw]
            if len(kw) >= 2:
                break
    return kw[:6]


def programme_infos(pg, pt, st, cond) -> dict[str, list[ProgInfo]]:
    """drug id → ProgInfo for each of its programmes (sponsors and collaborators of its Phase 3 trials, indication
    keywords from the programme label or, if too few, the trials' first-listed conditions)."""
    sponsors = st.set_index("nct_id")[["lead_sponsor", "collaborators"]]
    first_cond = cond[cond["pos"] == 0].set_index("nct_id")["condition"]
    by_prog = pt[pt["relation"].isin(["p3_seed", "p3_linked"])].groupby("programme_id")["nct_id"].apply(list)
    out: dict[str, list[ProgInfo]] = defaultdict(list)
    for r in pg.itertuples(index=False):
        ncts = by_prog.get(r.programme_id, [])
        sp = set()
        for n in ncts:
            if n in sponsors.index:
                sp.add(sponsors.at[n, "lead_sponsor"])
                sp.update(x for x in (sponsors.at[n, "collaborators"] or "").split("|") if x)
        texts = [first_cond[n] for n in ncts if n in first_cond.index]
        out[r.drug_id].append(ProgInfo(r.programme_id, r.drug_id, r.indication_label,
                                       programme_keywords(r.indication_label, texts),
                                       r.anchor_p3_start if isinstance(r.anchor_p3_start, str) else None, sp))
    return out


def in_window(file_date: str, anchor: str | None) -> bool:
    if not anchor:
        return True
    y = int(anchor[:4])
    return f"{y - WINDOW_BEFORE_YEARS}{anchor[4:]}" <= file_date <= f"{y + WINDOW_AFTER_YEARS}{anchor[4:]}"


MAX_DOC_CHARS = 400_000                 # press releases are far shorter; huge exhibits are decks / reports
_GAP = ["[...]"] * 3                    # separates blocks so the parser's 3-sentence look-back can't cross them


def mention_blocks(sentences: list[str], names: list[str]) -> list[str]:
    """The sentences the parser can act on: each mention of the drug, 3 sentences before (indication context)
    and 2 after (the parser looks back 2 sentences for the drug name)."""
    pats = [re.compile(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])") for n in names]
    hits = [i for i, s in enumerate(sentences) if any(p.search(s.lower()) for p in pats)]
    keep: list[tuple[int, int]] = []
    for i in hits:
        lo, hi = max(0, i - 3), min(len(sentences), i + 3)
        if keep and lo <= keep[-1][1]:
            keep[-1] = (keep[-1][0], hi)
        else:
            keep.append((lo, hi))
    out: list[str] = []
    for lo, hi in keep:
        out += (_GAP if out else []) + sentences[lo:hi]
    return out


# Worker state for multiprocessing (set once per process by `init_worker`).
_W: dict = {}


def init_worker(key_to_drug, drug_names, progs_by_drug):
    _W.update(key_to_drug=key_to_drug, drug_names=drug_names, progs_by_drug=progs_by_drug)


def process_doc(row: tuple) -> tuple[list[dict], list[tuple]]:
    """(events, mentions) for one document: every programme of every drug it mentions, inside the time window."""
    doc_id, file_date, cik, company, text = row
    events, mentions = [], []
    if not text:
        return events, mentions
    filed = date.fromisoformat(file_date)
    found = drugs_mentioned(text[:MAX_DOC_CHARS], _W["key_to_drug"])
    if not found:
        return events, mentions
    all_sentences = body_sentences(text[:MAX_DOC_CHARS])
    for did, keys in found.items():
        mentions.append((did, doc_id, cik, company, file_date))
        # Only the aliases this document uses: the parser runs one regex per name per sentence.
        names = [s for s in _W["drug_names"].get(did, []) if drugs.key(s) in keys]
        progs = [p for p in _W["progs_by_drug"].get(did, []) if in_window(file_date, p.anchor)]
        if not progs or not names:
            continue
        sentences = mention_blocks(all_sentences, names)
        for p in progs:
            for e in parse_document("", filed, doc_id, names, p.indication, p.keywords, sentences=sentences):
                d = e.to_dict()
                d.update(programme_id=p.programme_id, drug_id=did, cik=cik, company=company,
                         filer_is_sponsor=any(same_company(company, s) for s in p.sponsors))
                events.append(d)
    return events, mentions


def best_indication(events: list[dict], keywords: dict[str, list[str]], types: set[str]) -> list[dict]:
    """One press-release event parsed for several programmes of a drug stays with the programmes whose indication
    its own sentence names (at least half the keywords, the parser's bar). The parser also accepts an indication
    named only in the preceding sentences, so 'accepted the BLA for fremanezumab for migraine' reached the
    cluster-headache programme and a urothelial filing a head-and-neck one: such context-only matches are dropped
    when another programme matches in the sentence. Fragmented indications ('Osteoarthritis' / 'Osteoarthritis,
    Hip') all keep it, and every programme keeps it when none matches in the sentence."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    out = []
    for e in events:
        if e["event_type"] in types:
            groups[(e["drug_id"], e["source_id"], e["event_type"], e["event_date"], e["quote"])].append(e)
        else:
            out.append(e)
    for group in groups.values():
        if len(group) == 1:
            out += group
            continue
        cov = [_keyword_coverage(e["quote"], keywords.get(e["programme_id"], [])) for e in group]
        out += [e for e, c in zip(group, cov) if max(cov) < 0.5 or c >= 0.5]
    return out


# ------------------------------------------------------------------------------------------------ assembly
def trial_summary(row: dict) -> ctgov.TrialSummary:
    def d(v):
        return date.fromisoformat(v) if isinstance(v, str) and v else None

    return ctgov.TrialSummary(
        nct_id=row["nct_id"], title=row.get("title") or "", phases=(row.get("phases") or "").split("|"),
        status=row.get("status") or "", sponsor=row.get("lead_sponsor") or "",
        sponsor_class=row.get("lead_sponsor_class") or "", conditions=[],
        start_date=d(row.get("start_date")), primary_completion_date=d(row.get("primary_completion_date")),
        results_first_post_date=d(row.get("results_first_post_date")), first_post_date=d(row.get("first_post_date")),
        has_results=bool(row.get("has_results")), enrollment=row.get("enrollment"),
        enrollment_type=row.get("enrollment_type"), allocation=row.get("allocation"), masking=row.get("masking"),
        arm_types=[])


def openfda_app(appl_no: str, appl_type: str, subs: list[dict]) -> dict:
    """A Drugs@FDA bulk application in the openFDA `drug/drugsfda` record shape used by `timeline.events`."""
    return {"application_number": f"{appl_type}{appl_no}",
            "submissions": [{"submission_type": s["SubmissionType"], "submission_number": s["SubmissionNo"],
                             "submission_status": s["SubmissionStatus"],
                             "submission_status_date": (s["status_date"] or "").replace("-", ""),
                             "review_priority": s["ReviewPriority"], "submission_class_code": s["class_code"],
                             "submission_class_code_description": s["class_desc"]} for s in subs]}
