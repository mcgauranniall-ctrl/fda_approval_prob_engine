"""Risk flags. Thresholds are provisional until the fitted model locates where effects become meaningful."""

from __future__ import annotations

from fdaengine.timeline.levels import ORDER


def _v(features: dict, name: str):
    f = features.get(name)
    return None if f is None else f["value"]


def model_flags(level: str, features: dict) -> list[dict]:
    li = ORDER.index(level)
    out: list[dict] = []

    def add(sev, text, basis=""):
        out.append({"severity": sev, "text": text, "basis": basis})

    met2 = _v(features, "p2_primary_endpoint_met")
    if met2 == "no":
        add("critical", "Phase 2 missed its primary endpoint; the Phase 3 rationale rests on secondary or subgroup data.")
    p2p = _v(features, "p2_best_pvalue")
    if isinstance(p2p, (int, float)) and 0.01 <= p2p < 0.05 and li == 0:
        add("warning", f"Phase 2 significance is modest (p = {p2p:.3g}); with winner's-curse shrinkage the Phase 3 "
                       "effect is likely smaller.")
    a = _v(features, "assurance")
    if isinstance(a, (int, float)) and li == 0:
        sev = "warning" if a < 0.6 else "info"
        add(sev, f"Phase 2→3 assurance is {a:.0%}: the predicted chance this Phase 3 design reaches significance "
                 "given the (shrunk) Phase 2 effect.", "O'Hagan et al. 2005; shrinkage 0.75 provisional")
    if _v(features, "p2_single_arm"):
        add("warning", "Phase 2 had no concurrent control arm; single-arm Phase 2s more often fail in Phase 3.")
    if _v(features, "p3_same_endpoint_as_p2") is False:
        add("warning", "The Phase 3 primary endpoint differs from Phase 2's, so Phase 2 is a weaker predictor.")
    if _v(features, "p3_comparator") == "active" or _v(features, "p3_non_inferiority"):
        add("info", "Phase 3 uses an active comparator / non-inferiority design.")

    if li >= 1:
        met3 = _v(features, "p3_primary_endpoint_met")
        if met3 == "no":
            add("critical", "PIVOTAL PHASE 3 MISSED ITS PRIMARY ENDPOINT. Approval after a failed pivotal trial is rare.")
        elif met3 == "mixed":
            add("critical", "Pivotal Phase 3 results are mixed (not all pivotal trials met the primary endpoint).")
        p3p = _v(features, "p3_best_pvalue")
        if isinstance(p3p, (int, float)) and 0.01 <= p3p < 0.05:
            add("warning", f"Weakest pivotal p-value is borderline (p = {p3p:.3g}); FDA may weigh replication heavily.")
        rr = _v(features, "p3_sae_rr")
        if isinstance(rr, (int, float)):
            if rr > 2:
                add("critical", f"Serious adverse events {rr:.1f}× the control rate.")
            elif rr > 1.3:
                add("warning", f"Serious adverse events {rr:.2f}× the control rate; labelling restrictions possible.")
        lo = _v(features, "p3_death_rd_lower")
        if isinstance(lo, (int, float)) and lo > 0:
            add("critical", "Excess deaths in the drug arm (risk-difference CI excludes zero).")
    if li >= 2:
        if _v(features, "crl_received") and level != "L6":
            add("critical", "A Complete Response Letter has been issued.")
        if _v(features, "pdufa_extended"):
            add("warning", "The PDUFA goal date was extended (usually a major amendment during review).")
        if _v(features, "accelerated_pathway"):
            add("info", "Accelerated approval pathway: the approval would require confirmatory trials.")
    f = _v(features, "adcom_yes_fraction")
    if isinstance(f, (int, float)) and li >= 3:
        if f < 0.5:
            add("critical", f"Advisory Committee voted against ({f:.0%} in favour).")
        elif f < 0.65:
            add("warning", f"Advisory Committee vote was close ({f:.0%} in favour).")
    sig = _v(features, "safety_signal")
    if sig == "serious":
        add("critical", "A serious safety signal has been reported.")
    elif sig == "moderate":
        add("warning", "A moderate safety signal has been reported.")
    if _v(features, "btd"):
        add("info", "Breakthrough Therapy designation: FDA judged early evidence promising and engages more closely.")
    if _v(features, "sponsor_size") == "private/unknown":
        add("info", "Sponsor is private or not an SEC filer: fewer public disclosures, and timelines may be incomplete.")
    return out


KEY_FEATURES = {
    "L1": ["therapeutic_area", "p2_primary_endpoint_met", "p2_best_pvalue", "p2_randomized", "p3_n_pivotal_trials",
           "p3_comparator", "assurance"],
    "L2": ["therapeutic_area", "p3_primary_endpoint_met", "p3_best_pvalue", "p3_n_pivotal_trials", "p3_sae_rr"],
}
KEY_FEATURES.update({lv: KEY_FEATURES["L2"] for lv in ("L3", "L4", "L5", "L6")})


def data_flags(level: str, features: dict, conflicts: list[dict], rejected: list[str], case_meta: dict,
               manifest: dict) -> list[dict]:
    out = []
    missing = [k for k in KEY_FEATURES[level] if k not in features]
    if missing:
        out.append({"severity": "warning", "text": "Key inputs not found: " + ", ".join(m.replace('_', ' ') for m in missing)
                    + ". The model treats them as unknown, which widens the interval."})
    for c in conflicts:
        out.append({"severity": "warning", "text": f"Sources disagree on {c['feature'].replace('_', ' ')}: used "
                    f"{c['used']!r} ({c['used_source']}), Claude extracted {c['claude']!r}."})
    if rejected:
        out.append({"severity": "info", "text": f"{len(rejected)} extracted values were discarded "
                    "(unverifiable quote, or knowable only at a later level)."})
    removed = [f for f in manifest.get("files", []) if "after cut-off" in (f.get("note") or "")]
    if removed:
        out.append({"severity": "info", "text": f"{len(removed)} trial result sets were posted after the cut-off and "
                    "were excluded."})
    lv = case_meta.get("level_info")
    if lv and lv.get("estimated"):
        out.append({"severity": "warning", "text": f"The {level} cut-off date is estimated ({lv['basis']})."})
    return out
