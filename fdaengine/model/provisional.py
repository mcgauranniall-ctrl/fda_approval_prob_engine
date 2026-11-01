"""PROVISIONAL prior-only model (PLAN.md §14 "fast first win").

Nothing here is fitted to data yet. Base rates are approximate literature values (BIO/Informa/QLS phase-transition
reports; Wong, Siah & Lo 2019), and effects are prior guesses expressed as Normal(mean, sd) on the log-odds scale.
Uncertainty is propagated by Monte Carlo, so the interval reflects how unsure these priors are; it is not a
fitted posterior. The trained Bayesian landmark model (P3–P5) replaces this module behind the same `predict()`
interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fdaengine.timeline.levels import ORDER

MODEL_VERSION = "provisional-0.1 (prior-only, not fitted)"
N_DRAWS = 4000

# P(eventual approval | programme has reached this level), all therapeutic areas. (mean probability, sd on logit)
BASE = {"L1": (0.52, 0.20), "L2": (0.52, 0.20), "L3": (0.88, 0.25), "L4": (0.82, 0.30), "L5": (0.88, 0.25),
        "L6": (0.70, 0.35)}
# First-cycle approval is lower than eventual approval at filing-stage levels (logit offset).
NEXT_ACTION_OFFSET = (-1.0, 0.3)

TA_OFFSET = {"oncology": -0.45, "hematology": 0.25, "infectious_disease": 0.25, "metabolic_endocrine": -0.1,
             "neurology": -0.25, "psychiatry": -0.3, "cardiovascular": -0.25, "respiratory": 0.0,
             "immunology": 0.05, "dermatology": 0.15, "ophthalmology": 0.0, "gastroenterology": 0.0,
             "musculoskeletal": -0.1, "pain": -0.2, "rare_disease": 0.3, "nephrology": -0.1,
             "womens_health": 0.0, "other": 0.0}


@dataclass
class Term:
    key: str
    label: str
    mean: float
    sd: float
    detail: str
    kind: str = "linear"          # 'linear' | 'asymmetric' | 'missing'


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _f(features: dict, name: str):
    f = features.get(name)
    return None if f is None else f["value"]


def terms_for(level: str, features: dict) -> list[Term]:
    li = ORDER.index(level)
    t: list[Term] = []
    ta = _f(features, "therapeutic_area")
    if ta in TA_OFFSET:
        scale = 1.0 if li <= 1 else 0.5        # TA differences shrink once efficacy is known
        t.append(Term("ta", f"Therapeutic area: {ta.replace('_', ' ')}", TA_OFFSET[ta] * scale, 0.25 * scale,
                      "historical phase-transition differences by TA"))
    else:
        t.append(Term("ta_missing", "Therapeutic area unknown", 0.0, 0.3, "adds uncertainty", "missing"))

    # Phase 2 evidence: dominant at L1, mostly superseded once Phase 3 has read out.
    w2 = 1.0 if li == 0 else 0.35
    met2 = _f(features, "p2_primary_endpoint_met")
    if met2 is not None:
        m = {"yes": 0.35, "no": -0.9, "mixed": -0.3}.get(met2, 0.0)
        t.append(Term("p2_met", f"Phase 2 primary endpoint met: {met2}", m * w2, 0.2 * w2, "asymmetric: a miss hurts more",
                      "asymmetric"))
    elif li == 0:
        t.append(Term("p2_missing", "Phase 2 result unknown", 0.0, 0.35, "adds uncertainty", "missing"))
    assurance = _f(features, "assurance")
    z2 = _f(features, "p2_z")
    if assurance is not None:
        t.append(Term("assurance", f"Phase 2→3 assurance {assurance:.0%}", 1.2 * (assurance - 0.5) * w2, 0.25 * w2,
                      "predictive power of the Phase 3 from the shrunk Phase 2 effect"))
    elif z2 is not None:
        t.append(Term("p2_z", f"Phase 2 strength |z| = {z2:.2f}", 0.15 * (min(z2, 6) - 2.5) * w2, 0.08 * w2,
                      "per unit of z above 2.5"))
    if _f(features, "p2_single_arm"):
        t.append(Term("p2_single_arm", "Phase 2 was single-arm", -0.3 * w2, 0.15, "no concurrent control"))
    elif _f(features, "p2_randomized") and _f(features, "p2_placebo_controlled"):
        t.append(Term("p2_rct", "Randomised, placebo-controlled Phase 2", 0.15 * w2, 0.1, "cleaner signal"))
    dr = _f(features, "p2_dose_response")
    if dr in ("yes", "no"):
        t.append(Term("dose_resp", f"Phase 2 dose-response: {dr}", (0.15 if dr == "yes" else -0.15) * w2, 0.1, ""))
    same = _f(features, "p3_same_endpoint_as_p2")
    if same is not None:
        t.append(Term("concordance", "Phase 3 uses the Phase 2 endpoint" if same else "Endpoint changed P2→P3",
                      0.15 if same else -0.3, 0.12, "endpoint concordance"))

    # Drug / sponsor / designations
    fic = _f(features, "first_in_class")
    if fic == "yes":
        t.append(Term("fic", "First-in-class mechanism", -0.15, 0.12, "no validated precedent"))
    if _f(features, "approved_other_indication"):
        t.append(Term("repurposed", "Already approved for another indication", 0.25, 0.12, "known safety/CMC"))
    size = _f(features, "sponsor_size")
    if size == "large":
        t.append(Term("sponsor", "Large, experienced sponsor", 0.2, 0.15, "regulatory track record"))
    elif size == "private/unknown":
        t.append(Term("sponsor", "Small/private or unknown sponsor", -0.2, 0.15, ""))
    # Designations partly re-express the Phase 2 evidence (BTD is granted on it), so their priors are modest.
    for key, label, m in (("btd", "Breakthrough Therapy designation", 0.25), ("orphan", "Orphan drug designation", 0.2),
                          ("fast_track", "Fast Track designation", 0.1), ("spa", "Special Protocol Assessment", 0.15)):
        if _f(features, key):
            t.append(Term(key, label, m * (1.0 if li <= 1 else 0.5), 0.12, "designation granted before cut-off"))
    sig = _f(features, "safety_signal")
    if sig in ("moderate", "serious"):
        t.append(Term("safety_sig", f"Safety signal reported: {sig}", -0.3 if sig == "moderate" else -0.8, 0.25,
                      "", "asymmetric"))
    if _f(features, "p3_comparator") == "active" or _f(features, "p3_non_inferiority"):
        t.append(Term("p3_design", "Active-comparator / non-inferiority Phase 3", -0.15, 0.12, "harder to win"))

    # Phase 3 results (L2+)
    if li >= 1:
        met3 = _f(features, "p3_primary_endpoint_met")
        if met3 is not None:
            m = {"yes": 1.65, "no": -3.0, "mixed": -1.0}.get(met3, 0.0)
            t.append(Term("p3_met", f"Phase 3 primary endpoint met: {met3}", m if li == 1 else m * 0.3, 0.35,
                          "the single largest driver at the readout", "asymmetric"))
        elif li == 1:
            t.append(Term("p3_missing", "Phase 3 result not found", 0.0, 1.2, "large uncertainty", "missing"))
        z3 = _f(features, "p3_z")
        if z3 is not None:
            t.append(Term("p3_z", f"Phase 3 strength |z| = {z3:.2f}", 0.1 * (min(z3, 8) - 3.0), 0.06,
                          "per unit of z above 3"))
        npos, npiv = _f(features, "p3_n_positive_pivotal"), _f(features, "p3_n_pivotal_trials")
        if isinstance(npos, int) and isinstance(npiv, int) and npiv >= 2:
            t.append(Term("replication", f"{npos} of {npiv} pivotal trials positive",
                          0.3 if npos >= 2 else -0.6, 0.15, "replication across pivotal trials", "asymmetric"))
        rr = _f(features, "p3_sae_rr")
        if isinstance(rr, (int, float)) and rr > 1.3:
            t.append(Term("sae", f"SAE risk ratio {rr:.2f}", -min(0.6 * (rr - 1.3), 1.5), 0.2,
                          "penalty only above RR 1.3 (v1 assumption, to be tested: H8)", "asymmetric"))
        lo = _f(features, "p3_death_rd_lower")
        if isinstance(lo, (int, float)) and lo > 0:
            t.append(Term("deaths", "Excess deaths in drug arm (CI excludes 0)", -1.0, 0.3, "", "asymmetric"))

    # Filing-stage signals (L3+)
    if li >= 2:
        if _f(features, "priority_review"):
            t.append(Term("priority", "Priority Review", 0.2, 0.12, ""))
        if _f(features, "accelerated_pathway"):
            t.append(Term("accel", "Accelerated approval pathway", -0.2, 0.15, "surrogate endpoint risk"))
        if _f(features, "pdufa_extended"):
            t.append(Term("pdufa_ext", "PDUFA date extended", -0.3, 0.3, "direction to be tested (H12)"))
        if _f(features, "crl_received") and level != "L6":
            t.append(Term("crl", "Complete Response Letter received", -1.5, 0.4, "", "asymmetric"))
    if li >= 3:
        f = _f(features, "adcom_yes_fraction")
        if isinstance(f, (int, float)):
            m = 1.2 * (f - 0.5) * 2 if f >= 0.5 else -2.6 * (0.5 - f) * 2
            t.append(Term("adcom", f"AdCom vote {f:.0%} in favour", m, 0.3,
                          "asymmetric: negative votes weigh more (H11)", "asymmetric"))
    return t


def predict(level: str, features: dict, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    base_p, base_sd = BASE[level]
    base = rng.normal(_logit(base_p), base_sd, N_DRAWS)
    terms = terms_for(level, features)
    total = base.copy()
    for term in terms:
        total += rng.normal(term.mean, term.sd, N_DRAWS)
    p = _sig(total)
    out = {
        "model_version": MODEL_VERSION,
        "level": level,
        "p_approval": _summary(p),
        "base_rate": base_p,
        "contributions": [{"key": t.key, "label": t.label, "logit": round(t.mean, 3), "sd": t.sd,
                           "detail": t.detail, "kind": t.kind} for t in terms],
        "draws_histogram": np.histogram(p, bins=20, range=(0, 1))[0].tolist(),
    }
    if ORDER.index(level) >= 2:
        nxt = _sig(total + rng.normal(*NEXT_ACTION_OFFSET, N_DRAWS))
        out["p_next_action"] = _summary(nxt)
    return out


def _summary(p: np.ndarray) -> dict:
    q = np.quantile(p, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {"median": float(q[2]), "mean": float(p.mean()), "ci50": [float(q[1]), float(q[3])],
            "ci95": [float(q[0]), float(q[4])]}
