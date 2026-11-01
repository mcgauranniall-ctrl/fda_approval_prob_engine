"""Structured trial facts computed directly from ClinicalTrials.gov records (no LLM involved).

These values outrank anything parsed from text when sources disagree (see validate/reconcile.py).
"""

from __future__ import annotations

from datetime import date

from rapidfuzz import fuzz

from fdaengine.features import stats_transforms as st
from fdaengine.sources import ctgov

_MASKING = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 2, "QUADRUPLE": 2}


def _arm_role(title: str, arms: list[dict], drug_names: list[str]) -> str:
    """'drug', 'placebo', 'active' or 'other' for a results group, via its matching registered arm."""
    t = title.lower()
    if "placebo" in t or "sham" in t:
        return "placebo"
    best = max(arms, key=lambda a: fuzz.token_set_ratio(t, a.get("label", "").lower()), default=None)
    if best is not None and fuzz.token_set_ratio(t, best.get("label", "").lower()) >= 80:
        role = {"EXPERIMENTAL": "drug", "PLACEBO_COMPARATOR": "placebo", "SHAM_COMPARATOR": "placebo",
                "ACTIVE_COMPARATOR": "active"}.get(best.get("type", ""), "other")
        if role != "other":
            return role
    if any(n.lower() in t for n in drug_names):
        return "drug"
    return "other"


def _design(study: dict) -> dict:
    p = study["protocolSection"]
    d = p.get("designModule", {})
    info = d.get("designInfo", {})
    arm_types = [a.get("type", "") for a in p.get("armsInterventionsModule", {}).get("armGroups", [])]
    return {
        "randomized": info.get("allocation") == "RANDOMIZED",
        "blinding_level": _MASKING.get(info.get("maskingInfo", {}).get("masking", "NONE"), 0),
        "placebo_controlled": any(t in ("PLACEBO_COMPARATOR", "SHAM_COMPARATOR") for t in arm_types),
        "active_controlled": "ACTIVE_COMPARATOR" in arm_types,
        "single_arm": len(arm_types) <= 1 or all(t == "EXPERIMENTAL" for t in arm_types),
        "n_experimental_arms": arm_types.count("EXPERIMENTAL"),
        "enrollment": d.get("enrollmentInfo", {}).get("count"),
        "enrollment_type": d.get("enrollmentInfo", {}).get("type"),
        "primary_outcomes_registered": [o.get("measure", "") for o in p.get("outcomesModule", {}).get("primaryOutcomes", [])],
        "has_dmc": p.get("oversightModule", {}).get("oversightHasDmc"),
    }


def _primary_analyses(results: dict, arms: list[dict], drug_names: list[str]) -> list[dict]:
    out = []
    for om in results.get("outcomeMeasuresModule", {}).get("outcomeMeasures", []):
        if om.get("type") != "PRIMARY":
            continue
        groups = {g["id"]: g.get("title", "") for g in om.get("groups", [])}
        for a in om.get("analyses", []):
            p, op = st.parse_pvalue(a.get("pValue"))
            roles = sorted({_arm_role(groups.get(g, ""), arms, drug_names) for g in a.get("groupIds", [])})
            entry = {
                "outcome": om.get("title", ""),
                "groups": [groups.get(g, g) for g in a.get("groupIds", [])],
                "comparison_roles": roles,
                "p_value": p, "p_operator": op,
                "method": a.get("statisticalMethod"),
                "param_type": a.get("paramType"),
                "param_value": _num(a.get("paramValue")),
                "ci_lower": _num(a.get("ciLowerLimit")),
                "ci_upper": _num(a.get("ciUpperLimit")),
                "ci_pct": _num(a.get("ciPctValue")),
                "non_inferiority_type": a.get("nonInferiorityType"),
            }
            entry["z_abs"] = st.z_from_p(p) if p is not None else None
            out.append(entry)
    return out


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safety(results: dict, arms: list[dict], drug_names: list[str]) -> dict:
    groups = results.get("adverseEventsModule", {}).get("eventGroups", [])
    pooled = {r: {"sae": 0, "sae_n": 0, "deaths": 0, "deaths_n": 0} for r in ("drug", "placebo", "active")}
    for g in groups:
        role = _arm_role(g.get("title", ""), arms, drug_names)
        if role not in pooled:
            continue
        pooled[role]["sae"] += g.get("seriousNumAffected") or 0
        pooled[role]["sae_n"] += g.get("seriousNumAtRisk") or 0
        pooled[role]["deaths"] += g.get("deathsNumAffected") or 0
        pooled[role]["deaths_n"] += g.get("deathsNumAtRisk") or 0
    disc = {"drug": [0, 0], "placebo": [0, 0], "active": [0, 0]}
    flow = results.get("participantFlowModule", {})
    fg_titles = {g["id"]: g.get("title", "") for g in flow.get("groups", [])}
    for period in flow.get("periods", [])[:1]:
        started = {}
        for m in period.get("milestones", []):
            if m.get("type") == "STARTED":
                started = {a["groupId"]: int(a.get("numSubjects", 0) or 0) for a in m.get("achievements", [])}
        for gid, n in started.items():
            role = _arm_role(fg_titles.get(gid, ""), arms, drug_names)
            if role in disc:
                disc[role][1] += n
        for dw in period.get("dropWithdraws", []):
            if "adverse" not in dw.get("type", "").lower():
                continue
            for r in dw.get("reasons", []):
                role = _arm_role(fg_titles.get(r.get("groupId"), ""), arms, drug_names)
                if role in disc:
                    disc[role][0] += int(r.get("numSubjects", 0) or 0)

    drug = pooled["drug"]
    control_role = "placebo" if pooled["placebo"]["sae_n"] else ("active" if pooled["active"]["sae_n"] else None)
    out: dict = {"control": control_role, "arms": pooled,
                 "ae_discontinuation": {k: v for k, v in disc.items() if v[1]}}
    if drug["sae_n"]:
        out["sae_rate_drug"] = drug["sae"] / drug["sae_n"]
    if control_role:
        c = pooled[control_role]
        out["sae_rate_control"] = c["sae"] / c["sae_n"]
        if drug["sae_n"]:
            if drug["sae"] or c["sae"]:   # RR is undefined when neither arm had an SAE
                rr, lo, hi = st.risk_ratio_ci(drug["sae"], drug["sae_n"], c["sae"], c["sae_n"])
                out["sae_rr"], out["sae_rr_ci"] = rr, [lo, hi]
            out["sae_fisher_p"] = st.fisher_p(drug["sae"], drug["sae_n"], c["sae"], c["sae_n"])
        if drug["deaths_n"] and c["deaths_n"]:
            d, lo, hi = st.newcombe_diff_ci(drug["deaths"], drug["deaths_n"], c["deaths"], c["deaths_n"])
            out["death_rd"], out["death_rd_ci"] = d, [lo, hi]
            out["death_fisher_p"] = st.fisher_p(drug["deaths"], drug["deaths_n"], c["deaths"], c["deaths_n"])
    if disc["drug"][1]:
        out["ae_discontinuation_rate_drug"] = disc["drug"][0] / disc["drug"][1]
    return out


def structure_trial(study: dict, drug_names: list[str], cutoff: date | None) -> dict:
    summary = ctgov.summarise(study)
    p = study["protocolSection"]
    arms = p.get("armsInterventionsModule", {}).get("armGroups", [])
    rec = {"nct_id": summary.nct_id, "title": summary.title, "phases": summary.phases,
           "max_phase": summary.max_phase, "status": summary.status, "sponsor": summary.sponsor,
           "conditions": summary.conditions,
           "start_date": summary.start_date.isoformat() if summary.start_date else None,
           "primary_completion_date": (summary.primary_completion_date.isoformat()
                                       if summary.primary_completion_date else None),
           "design": _design(study)}
    results = study.get("resultsSection")
    posted = summary.results_first_post_date
    if not results:
        rec["results_status"] = "not_posted"
    elif cutoff and posted and posted > cutoff:
        rec["results_status"] = "posted_after_cutoff"
        rec["results_posted_date"] = posted.isoformat()
    else:
        rec["results_status"] = "available"
        rec["results_posted_date"] = posted.isoformat() if posted else None
        rec["primary_analyses"] = _primary_analyses(results, arms, drug_names)
        rec["safety"] = _safety(results, arms, drug_names)
        key = key_analysis(rec["primary_analyses"])
        if key:
            rec["key_analysis"] = key
    return rec


def key_analysis(analyses: list[dict]) -> dict | None:
    """The primary drug-vs-control comparison with the smallest p-value (prefers placebo control)."""
    vs_control = [a for a in analyses if "drug" in a["comparison_roles"] and a["p_value"] is not None
                  and ({"placebo", "active"} & set(a["comparison_roles"]))]
    if not vs_control:
        return None
    vs_control.sort(key=lambda a: ("placebo" not in a["comparison_roles"], a["p_value"]))
    return vs_control[0]
