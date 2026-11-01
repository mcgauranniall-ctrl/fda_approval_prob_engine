"""Outcome labels (PLAN.md §3, §5 script 04). Labels only: nothing here is ever a model input.

Evidence for "this programme was approved, on date d" comes from three sources, combined by fixed rules:
  * Drugs@FDA approval actions on NDA/BLA applications whose products are exactly the programme's drug;
  * the application's current FDA label (openFDA `drug/label`), whose Indications section must cover the
    programme's indication: a drug approved for acute pain is not approved for diabetic neuropathy;
  * EDGAR press-release APPROVED events (rule-based parser, indication-checked).

Drugs@FDA does not say which indication a supplement added, and the label is today's version, so approvals of
already-marketed moieties (lifecycle programmes) carry lower confidence than first approvals.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import ijson
from rapidfuzz import fuzz

WINDOW_YEARS = 7                 # D1: approval within 7 years of the first Phase 3 start
LABEL_COVERAGE = 0.5             # share of the programme's indication keywords found in the label's indications
NEAR_DAYS = 45                   # a press-release approval within this many days of an FDA action is that action
NEXT_ACTION_YEARS = 2            # an FDA action this long after a cut-off is no longer "the next action"
LATE_LEVELS = {"L3", "L4", "L5", "L6"}
LATE_HORIZON_YEARS = 3           # from filing on, Y_LEVEL = approved within this long of the cut-off (83% of L3 and
                                 # 96% of L5 eventual approvals arrive within 3 years; later ones are new episodes)
NEGATIVE_QUIET_YEARS = 2         # a failed Phase 3 with no further progress for this long counts as a failure
TERMINAL_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
OPEN_STATUSES = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}
# Original applications that can approve a new use of an already-marketed moiety.
LIFECYCLE_ORIG_CLASSES = {"TYPE 2/3", "TYPE 3", "TYPE 3/4", "TYPE 5", "TYPE 4/5", "TYPE 6", "TYPE 9", "TYPE 9- BLA",
                          "TYPE 10", "TYPE 10- BLA", "BIOSIMILAR"}
FAILURE_CONFIDENCE = {"all_p3_terminated": "high", "p3_negative_no_progress": "medium", "window_closed": "medium",
                      "approved_after_window": "high"}

# Label wording differs from MeSH wording; these count as the same word.
_SYNONYMS = {w: "cancer" for w in ("neoplasm", "neoplasms", "carcinoma", "carcinomas", "cancers", "tumor", "tumors",
                                    "tumour", "tumours", "malignancy", "malignancies", "malignant")}


# Words that say nothing about *which* disease ('non-small cell' matches any label mentioning cells): coverage is
# computed on the remaining, disease-specific words.
_UNSPECIFIC = {"non", "small", "cell", "cells", "type", "chronic", "acute", "disease", "diseases", "disorder",
               "disorders", "syndrome", "infection", "infections", "cancer", "advanced", "metastatic", "primary",
               "secondary", "severe", "moderate", "related", "associated", "adult", "pediatric", "failure",
               "insufficiency", "hereditary", "refractory", "recurrent", "relapsed", "stage", "high", "risk",
               "locally", "first", "line"}
# Label sentences that exclude a use ('Efficacy ... has not been established for ... fibromyalgia').
_NEGATED_SENTENCE = re.compile(r"\bnot\b[^.;]{0,60}\b(established|indicated|approved|recommended|studied)\b|"
                               r"has not been|limitations? of use", re.IGNORECASE)


def indicated_text(label: str) -> str:
    """A label's Indications section without the sentences that rule uses out."""
    parts = re.split(r"(?<=[.;•])\s+|\s+•\s+|\(\s*\d+(?:\.\d+)?\s*\)", label)
    return " ".join(p for p in parts if not _NEGATED_SENTENCE.search(p))


def indication_coverage(keywords: list[str], text: str) -> float:
    """Share of the disease-specific `keywords` found (fuzzily) in `text`."""
    kws = [_SYNONYMS.get(k, k) for k in keywords]
    kws = [k for k in kws if k not in _UNSPECIFIC] or kws
    if not kws:
        return 0.0
    words = {_SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower())}
    return sum(k in words or any(fuzz.ratio(k, w) >= 85 for w in words) for k in kws) / len(kws)


_GENERIC_KEYWORDS = {"pain", "infection", "infections", "cancer", "neoplasms", "disease", "diseases", "syndrome",
                     "chronic", "acute", "disorders", "healthy", "volunteers", "tumors", "solid", "advanced"}


def keyword_sets(indication_keywords: list[str], condition_keywords: list[list[str]]) -> list[list[str]]:
    """Ways of naming a programme's indication: its MeSH label, plus each trial's own first-listed condition
    ('Acute Bacterial Skin and Skin Structure Infections' where the MeSH label says 'Soft Tissue Infections').
    A one-word condition counts only if the word is specific ('psoriasis', not 'pain')."""
    out = [indication_keywords] if indication_keywords else []
    for kw in condition_keywords:
        kw = [k for k in kw if k not in ("healthy", "volunteers")]
        if (len(kw) >= 2 or (len(kw) == 1 and kw[0] not in _GENERIC_KEYWORDS)) and kw not in out:
            out.append(kw)
    return out


def best_coverage(sets: list[list[str]], label: str) -> float:
    text = indicated_text(label)
    return max((indication_coverage(kw, text) for kw in sets), default=0.0)


# Products licensed by CBER (vaccines, blood and plasma products, allergenics, cell and gene therapies) are not in
# Drugs@FDA; their licences come from the Purple Book (`purplebook.py`).
_CBER = re.compile(r"vaccin|virus|toxoid|antigen|immune globulin|immunoglobulin|\bplasma\b|albumin human|fibrin|"
                   r"factor (viii|ix|vii|xiii)|antihemophilic|allergen|pollen|calmette|\bbcg\b|gene therapy|"
                   r"antithrombin|c1 esterase|proteinase inhibitor|prothrombin complex|\b\w+(cel|vec|cog)\b",
                   re.IGNORECASE)


def is_cber_product(name: str) -> bool:
    return bool(_CBER.search(name or ""))


def label_rows(zip_path: Path) -> Iterator[dict]:
    """NDA/BLA labels in one openFDA `drug/label` bulk file: application, indications text, boxed warning flag."""
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as fh:
        for r in ijson.items(fh, "results.item"):
            of = r.get("openfda") or {}
            for appl in of.get("application_number") or []:
                m = re.match(r"(NDA|BLA)(\d+)$", appl)
                if m:
                    yield {"ApplNo": m.group(2).zfill(6), "appl_type": m.group(1), "set_id": r.get("set_id"),
                           "effective_time": r.get("effective_time"),
                           "product_type": "|".join(of.get("product_type") or []),
                           "indications": " ".join(r.get("indications_and_usage") or [])[:20_000],
                           "boxed_warning": bool(r.get("boxed_warning"))}


# ------------------------------------------------------------------------------------------------ programme facts
@dataclass
class Action:
    """An FDA approval action on an application for exactly this programme's drug."""
    date: str
    appl: str
    kind: str                        # 'ORIG' | 'SUPPL'
    class_code: str
    holder_match: bool               # application holder is one of the programme's sponsors
    label_cov: float | None          # indication coverage of the application's current label (None: no label)


@dataclass
class Facts:
    programme_id: str
    anchor: str | None               # first Phase 3 start
    lifecycle: bool                  # moiety approved before the anchor
    n_rival_programmes: int = 0      # other programmes of the drug that could own an unlabelled first approval
    actions: list[Action] = field(default_factory=list)
    press_approvals: list[str] = field(default_factory=list)       # APPROVED press-release event dates
    p3_statuses: list[str] = field(default_factory=list)           # seed Phase 3 trial statuses
    p3_end_dates: list[str] = field(default_factory=list)          # their primary completion (or start) dates
    rival_p3_ends: list[str] = field(default_factory=list)         # other programmes' earliest Phase 3 end dates
    p3_negative: str | None = None                                 # first PHASE3_NEGATIVE event
    progress_dates: list[str] = field(default_factory=list)        # P3 starts, positive readouts, filings
    cber: bool = False                                             # CBER product: licences from the Purple Book

    def progress_after(self, d: str) -> bool:
        return any(x > d for x in self.progress_dates)


def _add_years(d: str, years: int) -> str:
    y = int(d[:4]) + years
    return f"{y}{d[4:]}" if d[5:10] != "02-29" else f"{y}-02-28"


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def approval(f: Facts) -> tuple[str | None, str, str, str | None]:
    """(approval date, label_source, label_confidence, application) for the programme's indication; the date and
    application are None when no approval is attributable."""
    if not f.anchor:
        return None, "no_anchor", "none", None
    floor = f.anchor
    own_end = min((d for d in f.p3_end_dates if d), default=f.anchor)
    if f.lifecycle:
        # A supplement approved before this programme's own Phase 3 ended is another use.
        floor = max(f.anchor, own_end)

    def owned_elsewhere(a: Action) -> bool:
        # An approval before this programme's own Phase 3 ended belongs to another programme of the drug whose Phase 3
        # had already finished (gold set: an ABSSSI approval was credited to the later pneumonia programme). With no
        # such rival it is this programme's: accelerated or interim approvals precede the registry completion date
        # (adagrasib's Krazati, exa-cel's Casgevy).
        return bool(f.n_rival_programmes) and a.date < own_end and any(e <= a.date for e in f.rival_p3_ends)
    if f.lifecycle:
        acts = [a for a in f.actions if a.holder_match and (a.kind == "SUPPL" and a.class_code == "EFFICACY" or
                                                             a.kind == "ORIG" and a.class_code in LIFECYCLE_ORIG_CLASSES)]
    else:
        acts = [a for a in f.actions if a.kind == "ORIG" or a.class_code == "EFFICACY"]
    acts = sorted((a for a in acts if a.date >= floor and not owned_elsewhere(a)), key=lambda a: a.date)
    covered = [a for a in acts if a.label_cov is not None and a.label_cov >= LABEL_COVERAGE]
    press = sorted(d for d in f.press_approvals if d >= f.anchor)

    # Candidates in rule order; the earliest date wins (gold set: a 2020 press-matched supplement had beaten the
    # drug's labelled 2018 first approval).
    cands: list[tuple[str, str, str, str]] = []
    # 1. A press-release approval (indication-checked) that coincides with an FDA action on this drug.
    for p in press:
        near = [a for a in acts if abs(_days(a.date, p)) <= NEAR_DAYS]
        if near:
            a = near[0]
            sure = not f.lifecycle or (a.label_cov or 0) >= LABEL_COVERAGE
            cands.append((a.date, "edgar+drugsfda", "high" if sure else "medium", a.appl))
            break
    if not f.lifecycle:
        first = [a for a in acts if a.kind == "ORIG"]
        # 2. First approval of a new drug whose current label covers the indication; an earlier first approval
        #    with no label on file (a discontinued original, e.g. Orbactiv before Kimyrsa) counts from its date.
        orig = [a for a in covered if a.kind == "ORIG"]
        if orig:
            a = orig[0]
            cands.append((a.date, "drugsfda_orig+label", "high" if a.holder_match else "medium", a.appl))
            earlier = [b for b in first if b.label_cov is None and b.date < a.date]
            if earlier:
                cands.append((earlier[0].date, "drugsfda_orig_unlabelled_earlier", "low", earlier[0].appl))
        # 3. First approval with no label on file: attributable only if no other programme of the drug could own
        #    it (a label that exists but does not cover the indication means another use: ChiRhoStim, a
        #    diagnostic, is not an autism approval). A CBER licence (no indications on file) goes to every
        #    programme of the product that was under way before it, at low confidence when there are several.
        elif first and first[0].label_cov is None and (f.n_rival_programmes == 0 or f.cber):
            a = first[0]
            cands.append((a.date, "purplebook_cber" if f.cber else "drugsfda_orig_sole_programme",
                          "medium" if f.n_rival_programmes == 0 else "low", a.appl))
    # 4. A supplement (or new-use application) whose label covers the indication: the first such action.
    if covered and not cands:
        a = covered[0]
        cands.append((a.date, "drugsfda_suppl+label", "low", a.appl))
    if cands:
        return min(cands, key=lambda c: c[0])
    # 5. A press-release approval with no FDA record behind it (about 40% precise on review: negations, boilerplate,
    #    unresolved drug names): neither approved nor failed.
    if press:
        return None, "edgar_only_unverified", "none", None
    if not f.lifecycle and any(a.kind == "ORIG" and a.label_cov is None for a in acts):
        return None, "ambiguous_unlabelled_approval", "none", None
    return None, "no_approval_found", "none", None


def failure(f: Facts, as_of: str) -> tuple[str | None, str]:
    """(date, basis) if the programme is already known to have failed, else (None, '')."""
    ends = [d for d in f.p3_end_dates if d]
    if f.p3_statuses and all(s in TERMINAL_STATUSES for s in f.p3_statuses) and ends \
            and not f.progress_after(max(ends)):
        return max(ends), "all_p3_terminated"
    if f.p3_negative and not f.progress_after(f.p3_negative) and not any(s in OPEN_STATUSES for s in f.p3_statuses) \
            and _days(f.p3_negative, as_of) >= 365 * NEGATIVE_QUIET_YEARS:
        return f.p3_negative, "p3_negative_no_progress"
    return None, ""


def programme_label(f: Facts, as_of: str) -> dict:
    """Y_PROG with its competing-risks coding.

    event 1 = approved within the window; 2 = failed (window closed without approval, or a terminal signal);
    0 = censored (window still open, no outcome yet); None = excluded (an approval exists that cannot be attributed).
    y_prog is 1/0 for events 1/2 and None otherwise.
    """
    app_date, source, conf, appl = approval(f)
    out = {"programme_id": f.programme_id, "approval_date": app_date, "approval_appl": appl,
           "label_source": source, "label_confidence": conf, "window_end": None, "window_closed": None,
           "event": None, "event_date": None, "t_event_days": None, "y_prog": None, "failure_basis": None}
    if not f.anchor:
        return out
    end = _add_years(f.anchor, WINDOW_YEARS)
    out.update(window_end=end, window_closed=end <= as_of)
    if app_date and app_date <= end:
        out.update(event=1, event_date=app_date, y_prog=1)
    elif source in ("ambiguous_unlabelled_approval", "edgar_only_unverified"):
        return out
    elif app_date:
        out.update(event=2, event_date=end, y_prog=0, failure_basis="approved_after_window",
                   label_confidence=FAILURE_CONFIDENCE["approved_after_window"])
    else:
        f_date, basis = failure(f, as_of)
        if end <= as_of or f_date:
            basis = basis or "window_closed"
            out.update(event=2, event_date=min(f_date or end, end), y_prog=0, failure_basis=basis,
                       label_confidence=FAILURE_CONFIDENCE[basis])
        else:
            out.update(event=0, event_date=as_of)
    out["t_event_days"] = _days(f.anchor, out["event_date"])
    return out


def level_target(level: str, cutoff: str, approval_date: str | None, y_prog, event, t_event_days, as_of: str) -> dict:
    """Y_LEVEL, the per-level training target. L1-L2: Y_PROG (D1, 7 years from the first Phase 3 start). From filing
    on, the 7-year clock mostly penalises late filers, so the target is approval within LATE_HORIZON_YEARS of the
    cut-off, with the same competing-risks coding (event 1 approved, 2 not approved by the horizon, 0 censored)."""
    if level not in LATE_LEVELS:
        return {"y_level": y_prog, "event_level": event, "t_level_days": t_event_days}
    if event is None or event != event:           # no trustworthy programme label (e.g. press-release-only)
        return {"y_level": None, "event_level": None, "t_level_days": None}
    horizon = _add_years(cutoff, LATE_HORIZON_YEARS)
    if approval_date and cutoff < approval_date <= horizon:
        return {"y_level": 1, "event_level": 1, "t_level_days": _days(cutoff, approval_date)}
    if horizon <= as_of:
        return {"y_level": 0, "event_level": 2, "t_level_days": _days(cutoff, horizon)}
    return {"y_level": None, "event_level": 0, "t_level_days": _days(cutoff, as_of)}


def next_action(cutoff: str, approval_date: str | None, crl_dates: list[str], as_of: str) -> dict:
    """Y_NEXT at a level cut-off: did the FDA approve at its next action (1) or issue a CRL (0)?"""
    horizon = _add_years(cutoff, NEXT_ACTION_YEARS)
    acts = sorted([(d, 0) for d in crl_dates if cutoff < d <= horizon]
                  + ([(approval_date, 1)] if approval_date and cutoff < approval_date <= horizon else []))
    if acts:
        d, y = acts[0]
        return {"y_next": y, "next_action_date": d, "next_action_status": "approved" if y else "crl"}
    return {"y_next": None, "next_action_date": None,
            "next_action_status": "pending" if horizon > as_of else "no_action_found"}
