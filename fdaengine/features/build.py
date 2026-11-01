"""Feature vector for a case at its level, with provenance for every value.

Source priority (PLAN.md §10.3): structured ClinicalTrials.gov values > rule-based timeline > Claude extraction.
Disagreements between sources are recorded as conflicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fdaengine.acquire.case_folder import Case
from fdaengine.features import stats_transforms as st
from fdaengine.timeline.levels import ORDER

LARGE_SPONSORS = ("pfizer", "merck", "johnson", "janssen", "roche", "genentech", "novartis", "sanofi",
                  "astrazeneca", "glaxosmithkline", "gsk", "abbvie", "bristol", "eli lilly", "lilly", "amgen",
                  "gilead", "novo nordisk", "bayer", "boehringer", "takeda", "regeneron", "vertex", "biogen",
                  "daiichi", "astellas", "eisai", "otsuka", "teva", "ucb", "csl", "moderna", "alexion")


@dataclass
class Feature:
    value: object
    source: str            # 'ctgov' | 'timeline' | 'claude' | 'derived' | 'heuristic'
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _yes_no_from_p(p: float | None) -> str | None:
    if p is None:
        return None
    return "yes" if p < 0.05 else "no"


def build(case: Case) -> dict:
    meta = case.read_json("case.json")
    structured = case.read_json("extracted/structured.json", {"trials": []})["trials"]
    timeline = case.read_json("timeline.json", [])
    claude = (case.read_json("extracted/fields.json") or {}).get("fields", {})
    feats, conflicts = compute(meta["level"], meta, structured, timeline, claude)
    out = {"level": meta["level"], "features": feats, "conflicts": conflicts}
    case.write_json("features.json", out)
    case.write_json("extracted/conflicts.json", conflicts)
    return out


def compute(level: str, meta: dict, structured: list[dict], timeline: list[dict],
            claude: dict) -> tuple[dict, list[dict]]:
    """Pure feature computation, so the same case can be re-scored at earlier levels (probability path)."""
    li = ORDER.index(level)
    feats: dict[str, Feature] = {}
    conflicts: list[dict] = []

    def cval(name):
        f = claude.get(name)
        return None if f is None else f.get("value")

    def put(name, value, source, detail=""):
        if value is None:
            return
        feats[name] = Feature(value, source, detail)
        other = cval(name)
        if source != "claude" and other is not None and _disagree(value, other):
            conflicts.append({"feature": name, "used": value, "used_source": source, "claude": other,
                              "claude_quote": (claude.get(name) or {}).get("quote")})

    events = {e["event_type"]: e for e in sorted(timeline, key=lambda e: e["date"], reverse=True)}

    # --- context --------------------------------------------------------------------------------------
    for name in ("therapeutic_area", "modality", "first_in_class", "approved_other_indication", "mechanism"):
        if cval(name) is not None:
            put(name, cval(name), "claude", claude[name].get("quote") or "general knowledge")
    sponsor = (meta.get("sponsor") or "").lower()
    size = "large" if any(s in sponsor for s in LARGE_SPONSORS) else ("public" if meta.get("cik") else "private/unknown")
    put("sponsor_size", size, "heuristic", meta.get("sponsor") or "")

    # --- Phase 2 ----------------------------------------------------------------------------------------
    p2 = [t for t in structured if 2 <= t["max_phase"] < 3]
    p3 = [t for t in structured if t["max_phase"] >= 3]
    if p2:
        put("p2_randomized", any(t["design"]["randomized"] for t in p2), "ctgov", ", ".join(t["nct_id"] for t in p2))
        put("p2_placebo_controlled", any(t["design"]["placebo_controlled"] for t in p2), "ctgov")
        put("p2_single_arm", all(t["design"]["single_arm"] for t in p2), "ctgov")
        put("p2_n_total", sum(t["design"]["enrollment"] or 0 for t in p2), "ctgov")
        keys = [t["key_analysis"] for t in p2 if t.get("key_analysis")]
        if keys:
            best = min(keys, key=lambda k: k["p_value"])
            put("p2_best_pvalue", best["p_value"], "ctgov", f"{best['outcome'][:80]} ({' vs '.join(best['groups'])})")
            put("p2_z", best["z_abs"], "derived", "|z| from the best Phase 2 primary p-value")
            put("p2_primary_endpoint_met", "yes" if any(k["p_value"] < 0.05 for k in keys) else "no", "ctgov")
    if "p2_primary_endpoint_met" not in feats:
        if "PHASE2_NEGATIVE" in events:
            put("p2_primary_endpoint_met", "no", "timeline", events["PHASE2_NEGATIVE"]["quote"])
        elif "PHASE2_POSITIVE" in events:
            put("p2_primary_endpoint_met", "yes", "timeline", events["PHASE2_POSITIVE"]["quote"])
    for name in ("p2_primary_endpoint_met", "p2_best_pvalue", "p2_randomized", "p2_placebo_controlled",
                 "p2_n_total", "p2_dose_response"):
        if name not in feats and cval(name) is not None:
            put(name, cval(name), "claude", claude[name].get("quote") or "")
    if "p2_z" not in feats and isinstance(feats.get("p2_best_pvalue", Feature(None, "")).value, (int, float)):
        put("p2_z", st.z_from_p(float(feats["p2_best_pvalue"].value)), "derived", "|z| from Phase 2 p-value")

    # --- Phase 3 design (registered) ----------------------------------------------------------------------
    pivotal = [t for t in p3 if t["design"]["randomized"] and not t["design"]["single_arm"]]
    if p3:
        put("p3_n_pivotal_trials", len(pivotal), "ctgov", ", ".join(t["nct_id"] for t in pivotal))
        planned = sum(t["design"]["enrollment"] or 0 for t in pivotal)
        if planned:
            put("p3_planned_n", planned, "ctgov", "enrolment of randomized Phase 3 trials (current registration)")
        if pivotal:
            comp = ("placebo" if any(t["design"]["placebo_controlled"] for t in pivotal)
                    else "active" if any(t["design"]["active_controlled"] for t in pivotal) else "mixed")
            put("p3_comparator", comp, "ctgov")
    for name in ("p3_n_pivotal_trials", "p3_comparator", "p3_same_endpoint_as_p2", "p3_non_inferiority", "p3_planned_n"):
        if name not in feats and cval(name) is not None:
            put(name, cval(name), "claude", claude[name].get("quote") or "")
    z2, n2, n3 = (feats.get(k).value if k in feats else None for k in ("p2_z", "p2_n_total", "p3_planned_n"))
    if z2 and n2 and n3:
        put("assurance", st.assurance(float(z2), int(n2), int(n3)), "derived",
            f"P2 |z|={float(z2):.2f}, n2={n2}, n3={n3}, winner's-curse shrinkage 0.75 (provisional)")

    # --- Phase 3 results (L2+) -----------------------------------------------------------------------------
    if li >= ORDER.index("L2"):
        keys3 = [t["key_analysis"] for t in pivotal if t.get("key_analysis")]
        if keys3:
            worst = max(keys3, key=lambda k: k["p_value"])
            put("p3_best_pvalue", worst["p_value"], "ctgov", "worst pivotal primary p-value")
            put("p3_z", worst["z_abs"], "derived", "|z| from the worst pivotal primary p-value")
            put("p3_n_positive_pivotal", sum(k["p_value"] < 0.05 for k in keys3), "ctgov")
            put("p3_primary_endpoint_met", "yes" if all(k["p_value"] < 0.05 for k in keys3) else
                ("mixed" if any(k["p_value"] < 0.05 for k in keys3) else "no"), "ctgov")
        elif "PHASE3_NEGATIVE" in events:
            put("p3_primary_endpoint_met", "no", "timeline", events["PHASE3_NEGATIVE"]["quote"])
        elif "PHASE3_POSITIVE" in events:
            put("p3_primary_endpoint_met", "yes", "timeline", events["PHASE3_POSITIVE"]["quote"])
        safety = [t["safety"] for t in pivotal if t.get("safety")]
        rr = [s["sae_rr"] for s in safety if s.get("sae_rr")]
        if rr:
            put("p3_sae_rr", max(rr), "ctgov", "largest SAE risk ratio across pivotal trials (drug vs control)")
        rd = [(s["death_rd"], s["death_rd_ci"]) for s in safety if s.get("death_rd") is not None]
        if rd:
            worst_rd = max(rd, key=lambda x: x[0])
            put("p3_death_rd", worst_rd[0], "ctgov", f"Newcombe 95% CI {worst_rd[1][0]:.3f} to {worst_rd[1][1]:.3f}")
            put("p3_death_rd_lower", worst_rd[1][0], "ctgov")
        for name in ("p3_primary_endpoint_met", "p3_best_pvalue", "p3_n_positive_pivotal", "p3_key_secondaries_met",
                     "p3_discontinuation_ae_drug"):
            if name not in feats and cval(name) is not None:
                put(name, cval(name), "claude", claude[name].get("quote") or "")
        if "p3_z" not in feats and isinstance(feats.get("p3_best_pvalue", Feature(None, "")).value, (int, float)):
            put("p3_z", st.z_from_p(float(feats["p3_best_pvalue"].value)), "derived", "|z| from Phase 3 p-value")
        if "p3_sae_rr" not in feats:
            d, c = cval("p3_sae_rate_drug"), cval("p3_sae_rate_control")
            if isinstance(d, (int, float)) and isinstance(c, (int, float)) and c > 0:
                put("p3_sae_rr", d / c, "claude", "SAE rate ratio from reported rates (no CI)")

    # --- Regulatory --------------------------------------------------------------------------------------
    for flag, ev in (("btd", "BTD"), ("fast_track", "FAST_TRACK"), ("orphan", "ORPHAN"), ("spa", "SPA"),
                     ("priority_review", "PRIORITY_REVIEW"), ("accelerated_pathway", "ACCELERATED_PATHWAY"),
                     ("pdufa_extended", "PDUFA_EXTENDED"), ("crl_received", "CRL")):
        if flag in ("priority_review", "accelerated_pathway", "pdufa_extended", "crl_received") and li < ORDER.index("L3"):
            continue
        if ev in events:
            put(flag, True, "timeline", events[ev]["quote"])
        elif cval(flag) is not None:
            put(flag, cval(flag), "claude", claude[flag].get("quote") or "")
    if li >= ORDER.index("L4"):
        if "ADCOM_VOTE" in events:
            yes, no = (int(x) for x in events["ADCOM_VOTE"]["value"].split("-"))
            put("adcom_yes_fraction", yes / (yes + no), "timeline", events["ADCOM_VOTE"]["quote"])
        elif isinstance(cval("adcom_votes_yes"), int) and isinstance(cval("adcom_votes_no"), int):
            y, n = cval("adcom_votes_yes"), cval("adcom_votes_no")
            if y + n:
                put("adcom_yes_fraction", y / (y + n), "claude", claude["adcom_votes_yes"].get("quote") or "")
    if cval("safety_signal") is not None:
        put("safety_signal", cval("safety_signal"), "claude", claude["safety_signal"].get("quote") or "")

    return {k: v.to_dict() for k, v in feats.items()}, conflicts


def _disagree(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) != bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > max(0.15 * abs(a), 0.005)
    return str(a).lower() != str(b).lower()
