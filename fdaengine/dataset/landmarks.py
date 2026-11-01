"""Landmark feature rows (PLAN.md §5 script 05): one row per programme per level it reached, built strictly from
information public on or before that level's cut-off. No LLM calls (D6): Claude-only fields stay empty.

Features come from the live-case code (`features.build.compute`) plus `context_features`, fed with:
  * trials registered (first posted) by the cut-off, via `structure_trial(..., cutoff)`, which withholds results
    posted after it; an *actual* enrolment of a trial still running at the cut-off is withheld too (the count
    reveals early stops);
  * timeline events dated by the cut-off (press releases dated no earlier than they were public, see
    `press_release_parser._event_date`), excluding the FDA CRL database, which was published in 2025 for
    historical letters and is a label source only.

Every row carries `inputs_max_date`, the latest availability date of anything it used; `assert_no_leak` fails the
build if that date is after the cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fdaengine.features.build import compute
from fdaengine.prepare.ctgov_structured import structure_trial
from fdaengine.sources import ctgov

LABEL_ONLY_SOURCES = {"fda_crl"}              # published after the decisions they describe
STRUCTURED_RELATIONS = {"p3_seed", "p3_linked", "p2"}


class LeakageError(AssertionError):
    pass


def visible_trial(study: dict, names: list[str], cutoff: date, publication: dict | None = None
                  ) -> tuple[dict, str] | None:
    """(structured record, availability date) of a trial as it could be known at `cutoff`, or None if it was
    not yet registered. `publication` is the trial's PubMed main-paper result (script 01b); it is attached only if
    the paper was public by the cut-off."""
    s = ctgov.summarise(study)
    if not s.first_post_date or s.first_post_date > cutoff:
        return None
    rec = structure_trial(study, names, cutoff)
    available = s.first_post_date.isoformat()
    if rec["results_status"] == "available" and s.results_first_post_date:
        available = max(available, s.results_first_post_date.isoformat())
    if publication and publication["available"] <= cutoff.isoformat():
        rec["publication_result"] = publication
        available = max(available, publication["available"])
    design = rec["design"]
    if design.get("enrollment_type") == "ACTUAL" and (not s.primary_completion_date
                                                      or s.primary_completion_date > cutoff):
        design["enrollment"], design["enrollment_type"] = None, None
    return rec, available


def visible_events(events: list[dict], cutoff: str) -> list[dict]:
    return [e for e in events if e["source"] not in LABEL_ONLY_SOURCES and e["date"] <= cutoff]


@dataclass
class Context:
    """Programme facts known at the cut-off (computed by the caller with the same as-of rule)."""
    therapeutic_area: str | None = None
    modality: str | None = None                       # 'biologic' | 'small_molecule' (registered intervention type)
    moiety_approved: bool | None = None               # moiety already FDA-approved (another use) by the cut-off
    sponsor_prior_approvals: int | None = None        # sponsor's NDA/BLA original approvals before the cut-off
    sponsor_prior_crls: int | None = None             # complete response letters the sponsor announced before it
    ta_prior_rate: float | None = None                # smoothed approval rate of TA programmes resolved before it
    anchor_year: int | None = None
    is_combination: bool | None = None
    n_sponsors: int | None = None
    dates: list[str] = field(default_factory=list)    # availability dates of the facts above


def context_features(ctx: Context) -> dict:
    out = {}
    for name in ("therapeutic_area", "modality", "moiety_approved", "sponsor_prior_approvals", "sponsor_prior_crls",
                 "ta_prior_rate",
                 "anchor_year", "is_combination", "n_sponsors"):
        v = getattr(ctx, name)
        if v is not None:
            out[name] = {"value": v, "source": "derived", "detail": "structured data as of the cut-off"}
    return out


def build_row(level: str, cutoff: str, meta: dict, trials: list[tuple[dict, str]], events: list[dict],
              ctx: Context) -> tuple[dict, str]:
    """(features, inputs_max_date) for one landmark. `trials` are `visible_trial` results for this cut-off."""
    structured = [rec for rec, _ in trials]
    dates = list(ctx.dates) + [available for _, available in trials]
    tl = visible_events(events, cutoff)
    dates += [e["date"] for e in tl]
    feats, _ = compute(level, {"level": level, "cutoff": cutoff, **meta}, structured, tl, {})
    feats.update(context_features(ctx))
    feats.update(_extra(level, cutoff, structured, tl))
    return feats, max(dates, default="0000-00-00")


def _extra(level: str, cutoff: str, structured: list[dict], tl: list[dict]) -> dict:
    """Features in the plan that `compute` does not produce (missing results, filing delay, AdCom scheduled)."""
    out = {}

    def put(name, value, detail):
        out[name] = {"value": value, "source": "derived", "detail": detail}

    done = [t for t in structured if t["primary_completion_date"] and t["primary_completion_date"] <= cutoff]
    p2 = [t for t in done if 2 <= t["max_phase"] < 3]
    p3 = [t for t in structured if t["max_phase"] >= 3]
    if p2:
        put("p2_results_posted", any(t["results_status"] == "available" for t in p2),
            "any completed Phase 2 with results on ClinicalTrials.gov by the cut-off")
    if level != "L1" and p3:
        put("p3_results_posted", any(t["results_status"] == "available" for t in p3),
            "any Phase 3 with results on ClinicalTrials.gov by the cut-off")
    put("n_p3_trials", len(p3), "Phase 3 trials registered by the cut-off")
    types = {e["event_type"]: e["date"] for e in sorted(tl, key=lambda e: e["date"], reverse=True)}
    readout = min((e["date"] for e in tl if e["event_type"] in ("PHASE3_POSITIVE", "PHASE3_NEGATIVE")), default=None)
    filed = min((e["date"] for e in tl if e["event_type"] in ("NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW",
                                                               "PDUFA_DATE")), default=None)
    if level in ("L3", "L4", "L5", "L6") and readout and filed and filed >= readout:
        put("months_readout_to_filing", round((date.fromisoformat(filed) - date.fromisoformat(readout)).days / 30.44, 1),
            "first Phase 3 topline to first filing-acceptance news")
    if level in ("L3", "L4", "L5", "L6"):
        put("adcom_scheduled", "ADCOM_SCHEDULED" in types or "ADCOM_VOTE" in types, "AdCom announced by the cut-off")
    return out


def assert_no_leak(rows: list[dict]) -> None:
    """Fail if any row used information dated after its cut-off, or sits on/after its own decision."""
    bad = [r for r in rows if r["inputs_max_date"] > r["cutoff"]]
    if bad:
        r = bad[0]
        raise LeakageError(f"{len(bad)} rows use post-cut-off information, e.g. {r['programme_id']} {r['level']}: "
                           f"input dated {r['inputs_max_date']} > cut-off {r['cutoff']}")
    late = [r for r in rows if isinstance(r.get("approval_date"), str) and r["cutoff"] >= r["approval_date"]]
    if late:
        r = late[0]
        raise LeakageError(f"{len(late)} rows are on/after the approval, e.g. {r['programme_id']} {r['level']}")
