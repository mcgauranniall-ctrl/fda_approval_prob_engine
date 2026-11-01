"""Label outcomes given approval (PLAN.md §3 "Label outcomes", phase P6c). Labels only: nothing here is a model input.

From documents the FDA releases at approval (post-decision, so training labels only, D2):
  * **accelerated approval**: FDA's list of CDER accelerated approvals (application, supplement, date), with the
    approval letter's wording ("approved under ... subpart H / 21 CFR 314.510 / 601.41") as a cross-check;
  * **REMS**: the approval letter ("we have determined that a REMS is necessary", "your proposed REMS ... is
    approved"); `etasu` marks a REMS with elements to assure safe use (restricted distribution, certification);
  * **boxed warning**: the label as approved (Drugs@FDA label PDF of the approving submission); openFDA's label is
    today's version and can carry a boxed warning added years later;
  * **label narrower than the trial population**: a restriction in the approved indication (prior therapy, no
    alternative, a companion-diagnostic subgroup, higher severity, older age) that the pivotal Phase 3 trials'
    registered inclusion criteria do not have. A rule-based proxy; script 11 reports its measured precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------ accelerated approval
_AA_ENTRY = re.compile(r"\b(NDA|BLA)[\s-]*(\d{6})\s*(?:Supplement\s*(\d+))?")
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def parse_accelerated_list(text: str) -> list[dict]:
    """FDA's 'CDER Drug and Biologic Accelerated Approvals Based on a Surrogate Endpoint' (text of the PDF) →
    one record per accelerated approval: application, supplement (None = original), accelerated approval date and
    the conversion/withdrawal status. Columns are application, names, applicant, received date, approval date."""
    ents = list(_AA_ENTRY.finditer(text))
    out = []
    for i, m in enumerate(ents):
        seg = text[m.end(): ents[i + 1].start() if i + 1 < len(ents) else len(text)]
        ds = _DATE.findall(seg)
        if len(ds) < 2:
            continue
        mo, d, y = ds[1]
        status = re.search(r"Not Yet Converted|Converted|Withdrawn", seg)
        out.append({"ApplNo": m.group(2), "appl_type": m.group(1), "supplement": m.group(3),
                    "aa_date": f"{y}-{int(mo):02d}-{int(d):02d}", "aa_status": status.group(0) if status else None})
    return out


# Letters: the approving action granted under the accelerated approval regulations (21 CFR 314.510 / 601.41; not
# Subpart H's restricted distribution, 314.520, which Xyrem and Revlimid were approved under). A letter converting an
# earlier accelerated approval, an open commitment of another application, or the standard paragraph on promotional
# materials mentions the same words, so that wording is excluded. Used only after the date of FDA's list.
_LETTER_AA = re.compile(r"accelerated approval|314\.510|601\.41|506\(c\)", re.IGNORECASE)
_LETTER_CONVERSION = re.compile(r"conver(t|sion)|verif(y|ied|ies) (and describe )?(the )?clinical benefit .{0,80}"
                                r"(fulfill|satisf)|fulfills? .{0,60}(subpart H|subpart E|314\.510|601\.41)|"
                                r"previously (granted|approved under) accelerated|promotional materials for accelerated|"
                                r"commitments? (required )?for accelerated approval of", re.IGNORECASE)


def letter_accelerated(letter: str) -> bool | None:
    """True when the letter grants accelerated approval; None without letter text."""
    if len(letter or "") < 500:
        return None
    hits = [m for m in _LETTER_AA.finditer(letter)]
    return any(not _LETTER_CONVERSION.search(letter[max(0, m.start() - 300): m.end() + 300]) for m in hits)


# ------------------------------------------------------------------------------------------------ REMS
# A sponsor's *proposed* REMS is not evidence: FDA may answer 'a REMS is not necessary' (Benlysta, Mepsevii).
_REMS_YES = re.compile(r"(determined|concluded) that a (REMS|risk evaluation and mitigation strategy)\b[^.]{0,40} "
                       r"(is|are) (necessary|required)|"
                       r"(approved|modified|existing|current) (REMS|risk evaluation and mitigation strategy)|"
                       r"(REMS|risk evaluation and mitigation strategy)[^.]{0,120}\b(is|are|was) (hereby )?approved|"
                       r"REMS (assessment|modification)s?", re.IGNORECASE)
_REMS_APPROVED = re.compile(r"(REMS|risk evaluation and mitigation strategy)[^.]{0,120}\b(is|are|was) (hereby )?approved",
                            re.IGNORECASE)
_REMS_NO = re.compile(r"(REMS|risk evaluation and mitigation strategy)[^.]{0,60}\b(is|was) not (necessary|required)|"
                      r"not (necessary|required)[^.]{0,40}\b(REMS|risk evaluation and mitigation strategy)",
                      re.IGNORECASE)
_ETASU = re.compile(r"elements? (necessary )?to assure safe use|ETASU|505-1\(f\)", re.IGNORECASE)


def letter_rems(letter: str) -> tuple[bool | None, bool | None]:
    """(REMS required at this approval, REMS includes elements to assure safe use); (None, None) without text. An
    explicit 'a REMS is not necessary' decides unless the letter also approves a REMS."""
    if len(letter or "") < 500:
        return None, None
    if _REMS_NO.search(letter) and not _REMS_APPROVED.search(letter):
        return False, False
    yes = bool(_REMS_YES.search(letter))
    return yes, yes and bool(_ETASU.search(letter))


# ------------------------------------------------------------------------------------------------ boxed warning
_PLR = re.compile(r"HIGHLIGHTS OF PRESCRIBING INFORMATION|FULL PRESCRIBING INFORMATION")
_BOXED_ANY = re.compile(r"complete boxed warning|boxed warning", re.IGNORECASE)
# Case-sensitive: the box heading is capitalised ('WARNING: BLEEDING RISK'), unlike 'WARNINGS AND PRECAUTIONS'.
_BOXED_TITLE = re.compile(r"\bWARNING:\s*['\"]?[A-Z]{3}")
_OLD_BOX = re.compile(r"\bWARNING\b(?!S)|\bUSE IN PREGNANCY\b")  # pre-2006 box ('USE IN PREGNANCY': ACE/ARB fetal risk)


def label_boxed_warning(label: str) -> bool | None:
    """Boxed warning in the label as approved; None without label text. Physician-labeling-rule labels (2006 on)
    open Highlights with 'WARNING: <TITLE>' and usually 'See full prescribing information for complete boxed
    warning'; older labels open with a capitalised 'WARNING' (or 'USE IN PREGNANCY') box, unlike the 'WARNINGS'
    section every label has."""
    t = label or ""
    if len(t) < 500:
        return None
    if _BOXED_ANY.search(t) or _BOXED_TITLE.search(t[:8000]):
        return True
    if _PLR.search(t):
        return False
    return bool(_OLD_BOX.search(t[:2500]))


# ------------------------------------------------------------------------------------------------ indication text
_IND_START = re.compile(r"INDICATIONS\s+(AND|&)\s+USAGE")
# Old-format labels describe the supporting studies inside the Indications section ('This indication is based on ...
# Patients Without Prior Antiretroviral Therapy'): the indication statement ends there.
_IND_END = re.compile(r"DOSAGE\s+AND\s+ADMINISTRATION|CONTRAINDICATIONS|DOSAGE FORMS AND STRENGTHS|"
                      r"This indication is based|Description of Clinical Stud|CLINICAL STUDIES")


def label_indications(label: str, max_chars: int = 4000) -> str:
    """The Indications and Usage section of a label (Highlights version on physician-labeling-rule labels)."""
    t = label or ""
    m = _IND_START.search(t)
    if not m:
        return ""
    rest = t[m.end(): m.end() + max_chars]
    e = _IND_END.search(rest)
    return rest[: e.start()] if e else rest


_PARTS = re.compile(r"\s•\s|\s•\s|(?<=[.;])\s+|\(\s*1\.\d+\s*\)|\s-{3,}\s")


def indication_passage(ind: str, keyword_sets: list[list[str]], min_cov: float = 0.5) -> str:
    """The parts of a multi-indication label that name the programme's indication (Yervoy's NSCLC bullet says
    nothing about its melanoma programme); the whole text when no part does."""
    from fdaengine.dataset.labels import indication_coverage
    parts = [p for p in _PARTS.split(ind) if p and p.strip()]
    keep = [p for p in parts if max((indication_coverage(k, p) for k in keyword_sets if k), default=0) >= min_cov]
    return " ".join(keep) if keep else ind


# ------------------------------------------------------------------------------------------------ narrower label
_PRIOR = (r"previously[- ]treated|prior (systemic |platinum[- ]based |anti[- ]?\w+ )?(therapy|therapies|treatment|"
          r"chemotherapy|regimen|lines?)|after (failure|progression|at least|two or more|one or more|three or more|"
          r"treatment with|prior|receiving)|who have (received|failed|progressed|had an inadequate|not responded)|"
          r"inadequate(ly)? respon|intoleran|refractory|not (adequately )?controlled|unable to tolerate|"
          r"(second|third)[- ]line|lines? of (systemic )?therapy|relapsed|resistant to|treatment[- ]experienced")
# A label that also covers untreated patients is not restricted to pretreated ones.
_UNRESTRICTED = (r"untreated|treatment[- ]na[iï]ve|(without|no) prior|regardless of prior|first[- ]line|newly diagnosed|"
                 r"initial therapy")
# Registered wording for a pretreated or uncontrolled trial population.
_PRIOR_TRIAL = _PRIOR + r"|uncontrolled|resistant|persistent|despite|insufficient(ly)? respon|on stable|failure of"
_ALTERNATIVES = (r"no (other |alternative |satisfactory )?(treatment|therapeutic) options|alternative (treatment|"
                 r"therap)\w* (options )?(are|is) (not|in)|not (a )?candidates? for|ineligible for|not eligible for|"
                 r"not suitable for|for whom .{0,40}(not appropriate|not suitable|unsuitable)|limited or no "
                 r"(alternative )?(treatment|options)")
_TEST = r"FDA[- ](approved|authorized|cleared) (test|assay|device)|as (detected|determined|identified|selected) by"
_BIOMARKER = (r"mutation|mutated|positive|express|amplif|fusion|rearrange|deficien|deletion|variant|genotype|"
              r"biomarker|PD-?L1|HER2|EGFR|ALK|BRAF|KRAS|BRCA|carrier|homozygous|heterozygous")
_SEVERE = r"moderate(ly)?[- ]to[- ]severe(ly)?|\bsevere\b"
_MILD = r"\bmild\b"


@dataclass
class Restriction:
    category: str
    label_quote: str


def _quote(rx: str, text: str) -> str | None:
    m = re.search(rx, text, re.IGNORECASE)
    return None if not m else text[max(0, m.start() - 80): m.end() + 80].strip()


def inclusion_text(eligibility: str) -> str:
    """Inclusion part of registered eligibility criteria (exclusion criteria mention prior therapy too)."""
    parts = re.split(r"exclusion criteria|key exclusion|exclusion:", eligibility or "", maxsplit=1, flags=re.IGNORECASE)
    return parts[0]


_AGE = re.compile(r"(\d{1,2})\s*years (of age )?(and|or) older|aged? (\d{1,2}) (years )?(and|or) older|"
                  r"(\d{1,2}) years (of age )?and above", re.IGNORECASE)


def label_min_age(ind: str) -> float | None:
    """Youngest age the indication covers: an explicit 'N years and older', else 18 for adult-only wording."""
    ages = [int(next(g for g in m.groups() if g and g.isdigit())) for m in _AGE.finditer(ind)]
    if ages:
        return float(min(ages))
    if re.search(r"pediatric|children|adolescent|infant|neonat", ind, re.IGNORECASE):
        return 0.0
    return 18.0 if re.search(r"\badults?\b", ind, re.IGNORECASE) else None


def trial_min_age(s: str | None) -> float | None:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(year|month|week|day)", s if isinstance(s, str) else "", re.IGNORECASE)
    if not m:
        return None
    v = float(m.group(1))
    return v / {"year": 1, "month": 12, "week": 52, "day": 365}[m.group(2).lower()]


def narrower(ind: str, trials: list[dict]) -> tuple[bool | None, list[Restriction]]:
    """Is the approved indication narrower than the pivotal trials' registered population?

    `ind`: the indication passage for the programme (`indication_passage`). `trials`: the Phase 3 trials completed
    before the approval, [{'text': title + conditions + inclusion criteria, 'min_age': years or None}] (a
    confirmatory trial finishing later is not the population behind the approval). A restriction counts only when no
    trial's registered text has the same kind of restriction (a trial in refractory patients makes a 'refractory'
    label as broad as the trial). None when the indication or the trial text is missing."""
    if len(ind.strip()) < 30 or not trials or not any(len(t.get("text") or "") > 100 for t in trials):
        return None, []
    trial_text = " ".join(t.get("text") or "" for t in trials)
    found = []
    for cat, rx, trial_rx in (("prior_therapy", _PRIOR, _PRIOR_TRIAL),
                              ("no_alternative", _ALTERNATIVES, _ALTERNATIVES + "|" + _PRIOR_TRIAL),
                              ("diagnostic_subgroup", _TEST, _BIOMARKER)):
        q = _quote(rx, ind)
        if cat == "prior_therapy" and re.search(_UNRESTRICTED, ind, re.IGNORECASE):
            continue
        if q and not re.search(trial_rx, trial_text, re.IGNORECASE):
            found.append(Restriction(cat, q))
    q = _quote(_SEVERE, ind)
    if q and not re.search(_MILD, ind, re.IGNORECASE) and re.search(_MILD, trial_text, re.IGNORECASE):
        found.append(Restriction("severity", q))
    # Age: only when every Phase 3 enrolled younger patients than the label allows (a programme's separate
    # pediatric study does not make an adult label narrow).
    la = label_min_age(ind)
    ta = [t.get("min_age") for t in trials]
    if la is not None and ta and None not in ta and la >= max(ta) + 2:
        found.append(Restriction("age", f"label from age {la:g}; every Phase 3 from age {max(ta):g} or younger"))
    return bool(found), found
