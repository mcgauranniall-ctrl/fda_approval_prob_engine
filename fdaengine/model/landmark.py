"""Fitted landmark model (PLAN.md §7.5): P(approval | the information public at a milestone level).

One Bayesian logistic model across levels L1–L5, fitted to Y_LEVEL on the training population (novel drugs, final
outcomes). Structure:

    η_i   = α_level + α_TA + Σ_j (β_j + δ_{j,stage}) x_ij
    α_level ~ Normal(logit(literature rate), 1)        α_TA ~ Normal(0, σ_TA)
    β_j     ~ Normal(0, λ)                              δ_{j,stage} ~ Normal(0, τ_j)   (stage: L1 | L2 | L3–L5)
    P(observed label = 1) = sens_level · σ(η) + (1 − spec_level) · (1 − σ(η))

so a feature's effect may differ between the early and late levels but borrows strength across them, L4/L5 share
the filed-stage effects (too few rows alone), and the label is treated as a noisy measurement of the truth with
sensitivity/specificity priors from the gold set (`data/processed/label_error.json`). Predictions are σ(η): the
probability of the *true* outcome.

Inputs are the same feature names `features.build.compute` produces for live cases, so training rows and case
folders go through one `design_row`. A missing value contributes nothing (its standardised value is 0, the
training mean): missingness itself is never a feature, because press-release fields are found more often for
drugs that were later approved (P4 missingness diagnostic).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from fdaengine.features.indication import TA_RULES

FIT_LEVELS = ["L1", "L2", "L3", "L4", "L5"]
STAGE = {"L1": 0, "L2": 1, "L3": 2, "L4": 2, "L5": 2}
TAS = [ta for ta, _ in TA_RULES] + ["other"]
# Literature anchors for the level intercepts (weak: sd 1 on the logit scale). Phase 3 start → approval ≈ 50%,
# topline readout ≈ 55%, NDA/BLA accepted → approval ≈ 85–90% (BIO/Informa 2021; Wong, Siah & Lo 2019).
LITERATURE_RATE = {"L1": 0.50, "L2": 0.55, "L3": 0.85, "L4": 0.85, "L5": 0.88}
# Levels where the therapeutic-area effect applies. TA shapes clinical-development risk; at FDA review (L3–L5) the
# TA prior rate predicts worse than the level base rate (script 09 benchmarks), so by default TA is not used there.
TA_LATE = False
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def _flag(v) -> float | None:
    return None if v is None else float(v is True or str(v).lower() in ("true", "yes", "1"))


def _signed(v) -> float | None:
    s = None if v is None else str(v).lower()
    return {"yes": 1.0, "no": -1.0, "mixed": 0.0}.get(s) if s else None


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    first_level: str
    kind: str                                   # 'flag' (0/1, missing = 0) | 'cont' (standardised, missing = 0)
    fn: Callable[[dict], float | None]
    last_level: str = "L5"
    late: bool = False                          # also used at L3–L5 (filed stage)


def _get(name: str) -> Callable[[dict], object]:
    return lambda f: f.get(name)


SPECS: list[Spec] = [
    # Context (all levels)
    Spec("large_sponsor", "Large sponsor", "L1", "flag", lambda f: _flag(f.get("sponsor_size") == "large"), late=True),
    Spec("public_sponsor", "Public small/mid-cap sponsor", "L1", "flag",
         lambda f: _flag(f.get("sponsor_size") == "public")),
    Spec("sponsor_approvals", "Sponsor's prior FDA approvals (log)", "L1", "cont",
         lambda f: None if _num(f.get("sponsor_prior_approvals")) is None else math.log1p(_num(f["sponsor_prior_approvals"]))),
    Spec("sponsor_crls", "Sponsor's prior complete response letters (log)", "L1", "cont",
         lambda f: None if _num(f.get("sponsor_prior_crls")) is None else math.log1p(_num(f["sponsor_prior_crls"]))),
    Spec("era", "Era (Phase 3 start year)", "L1", "cont", lambda f: _num(f.get("anchor_year"))),
    Spec("biologic", "Biologic", "L1", "flag", lambda f: _flag(f.get("modality") == "biologic")),
    Spec("moiety_approved", "Active ingredient already approved", "L1", "flag",
         lambda f: _flag(f.get("moiety_approved"))),
    Spec("orphan", "Orphan designation reported", "L1", "flag", lambda f: _flag(f.get("orphan"))),
    Spec("btd", "Breakthrough therapy designation reported", "L1", "flag", lambda f: _flag(f.get("btd"))),
    Spec("fast_track", "Fast Track reported", "L1", "flag", lambda f: _flag(f.get("fast_track"))),
    Spec("spa", "Special protocol assessment reported", "L1", "flag", lambda f: _flag(f.get("spa"))),
    # Phase 2 evidence
    Spec("p2_met", "Phase 2 primary endpoint (met +1 / missed −1)", "L1", "flag",
         lambda f: _signed(f.get("p2_primary_endpoint_met")), late=True),
    Spec("p2_z", "Phase 2 strength of evidence (|z|)", "L1", "cont",
         lambda f: None if _num(f.get("p2_z")) is None else min(_num(f["p2_z"]), 6.0)),
    # Phase 3 design (registered)
    Spec("pivotal2", "Two or more randomized Phase 3 trials", "L1", "flag",
         lambda f: None if _num(f.get("p3_n_pivotal_trials")) is None else float(_num(f["p3_n_pivotal_trials"]) >= 2), late=True),
    Spec("placebo", "Placebo-controlled Phase 3", "L1", "flag", lambda f: _flag(f.get("p3_comparator") == "placebo")),
    Spec("double_blind", "Double-blind Phase 3", "L1", "flag", lambda f: _flag(f.get("p3_double_blind"))),
    Spec("ep_survival", "Primary endpoint: overall survival", "L1", "flag",
         lambda f: _flag(f.get("p3_endpoint_class") == "survival")),
    Spec("ep_time_to_event", "Primary endpoint: progression/event-free time", "L1", "flag",
         lambda f: _flag(f.get("p3_endpoint_class") == "time_to_event")),
    Spec("ep_biomarker", "Primary endpoint: biomarker", "L1", "flag",
         lambda f: _flag(f.get("p3_endpoint_class") == "biomarker")),
    # Phase 3 results (L2+)
    Spec("p3_met", "Phase 3 primary endpoint (met +1 / missed −1)", "L2", "flag",
         lambda f: _signed(f.get("p3_primary_endpoint_met")), late=True),
    Spec("p3_z", "Phase 3 strength of evidence (|z|)", "L2", "cont",
         lambda f: None if _num(f.get("p3_z")) is None else min(_num(f["p3_z"]), 8.0), late=True),
    # Regulatory (L3+)
    Spec("priority_review", "Priority review", "L3", "flag", lambda f: _flag(f.get("priority_review")), late=True),
    Spec("accelerated", "Accelerated approval pathway", "L3", "flag", lambda f: _flag(f.get("accelerated_pathway")), late=True),
    Spec("adcom_scheduled", "AdCom scheduled", "L3", "flag", lambda f: _flag(f.get("adcom_scheduled")), late=True),
    Spec("filing_delay", "Months from Phase 3 readout to filing", "L3", "cont",
         lambda f: None if _num(f.get("months_readout_to_filing")) is None else min(_num(f["months_readout_to_filing"]), 48.0)),
    Spec("pdufa_extended", "PDUFA date extended", "L5", "flag", lambda f: _flag(f.get("pdufa_extended")), late=True),
]
KEYS = [s.key for s in SPECS]


def training_population(lm):
    """Landmark rows the model is fitted on: novel drugs (primary population) with a final Y_LEVEL — the 7-year
    window closed at L1–L2, the 3-year horizon passed or approved from L3 on — and, at L2, a press-release-dated
    readout (the live-case situation)."""
    d = lm[lm["in_cohort"] & lm["in_primary"] & lm["y_level"].notna()]
    d = d[d["window_closed"] | d["level"].isin(["L3", "L4", "L5", "L6"])]
    return d[(d["level"] != "L2") | ~d["cutoff_estimated"]]


def flat(features: dict) -> dict:
    """Case `features.json` style {name: {"value": ...}} → {name: value}; plain dicts pass through."""
    return {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in features.items()}


def ta_index(ta: str | None) -> int:
    return TAS.index(ta) if ta in TAS else TAS.index("other")


def raw_row(level: str, features: dict) -> list[float | None]:
    """Untransformed values, None where missing or not applicable at this level."""
    li = FIT_LEVELS.index(level)
    out = []
    for s in SPECS:
        ok = (FIT_LEVELS.index(s.first_level) <= li <= FIT_LEVELS.index(s.last_level)
              and (STAGE[level] < 2 or s.late))
        out.append(s.fn(features) if ok else None)
    return out


def standardiser(raw: np.ndarray) -> dict:
    """Mean/sd of each continuous feature over the rows where it is known (flags are left as 0/1)."""
    out = {}
    for j, s in enumerate(SPECS):
        if s.kind == "cont":
            col = raw[:, j][~np.isnan(raw[:, j])]
            out[s.key] = [float(col.mean()), float(col.std() or 1.0)] if len(col) else [0.0, 1.0]
    return out


def design(raw: np.ndarray, scale: dict) -> np.ndarray:
    x = raw.copy()
    for j, s in enumerate(SPECS):
        if s.kind == "cont":
            m, sd = scale[s.key]
            x[:, j] = (x[:, j] - m) / sd
    return np.nan_to_num(x, nan=0.0)


def raw_matrix(levels: list[str], feature_dicts: list[dict]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in raw_row(lv, f)] for lv, f in zip(levels, feature_dicts)],
                    dtype=float)


# ------------------------------------------------------------------------------------------------ fitting
def label_priors(label_error: dict, n_mc: int = 20000, seed: int = 7) -> dict[str, dict]:
    """Beta(a, b) priors for each level's sensitivity and specificity, moment-matched to the gold-set PPV/NPV
    posteriors re-weighted by the training population's pipeline approval share. L4 borrows L3's."""
    out = {}
    for lv in FIT_LEVELS:
        rng = np.random.default_rng(seed)
        e = label_error.get(lv if lv != "L4" else "L3", {})
        if "ppv_beta" not in e:
            e = label_error["L3"]
        ppv, npv, pi = rng.beta(*e["ppv_beta"], n_mc), rng.beta(*e["npv_beta"], n_mc), e["pipeline_rate"]
        r = pi * ppv + (1 - pi) * (1 - npv)
        sens, spec = pi * ppv / r, (1 - pi) * npv / (1 - r)
        out[lv] = {"sens": _beta_mm(sens), "spec": _beta_mm(spec)}
    return out


def _beta_mm(x: np.ndarray) -> list[float]:
    m, v = float(x.mean()), float(x.var())
    k = m * (1 - m) / v - 1
    return [round(m * k, 2), round((1 - m) * k, 2)]


def fit(X: np.ndarray, level_idx: np.ndarray, ta_idx: np.ndarray, y: np.ndarray, priors: dict[str, dict],
        draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 11, label_noise: bool = True,
        ta_late: bool = TA_LATE) -> dict:
    """NUTS fit; returns posterior draws (chains × draws flattened) as numpy arrays plus diagnostics."""
    import arviz as az
    import pymc as pm

    J, L, T = X.shape[1], len(FIT_LEVELS), len(TAS)
    stage_of_level = np.array([STAGE[lv] for lv in FIT_LEVELS])
    ta_on = np.array([ta_late or STAGE[lv] < 2 for lv in FIT_LEVELS], dtype=float)
    with pm.Model():
        alpha = pm.Normal("alpha", mu=[math.log(LITERATURE_RATE[lv] / (1 - LITERATURE_RATE[lv])) for lv in FIT_LEVELS],
                          sigma=1.0, shape=L)
        sigma_ta = pm.HalfNormal("sigma_ta", 0.5)
        a_ta = pm.Deterministic("a_ta", sigma_ta * pm.Normal("z_ta", 0, 1, shape=T))
        lam = pm.HalfNormal("lam", 1.0)
        beta = pm.Normal("beta", 0, lam, shape=J)
        tau = pm.HalfNormal("tau", 0.3, shape=J)
        delta = pm.Deterministic("delta", tau * pm.Normal("z_delta", 0, 1, shape=(3, J)))
        coef = beta[None, :] + delta[stage_of_level]                       # (L, J)
        eta = alpha[level_idx] + a_ta[ta_idx] * ta_on[level_idx] + (X * coef[level_idx]).sum(axis=1)
        p = pm.math.sigmoid(eta)
        if label_noise:
            sens = pm.Beta("sens", alpha=[priors[lv]["sens"][0] for lv in FIT_LEVELS],
                           beta=[priors[lv]["sens"][1] for lv in FIT_LEVELS], shape=L)
            spec = pm.Beta("spec", alpha=[priors[lv]["spec"][0] for lv in FIT_LEVELS],
                           beta=[priors[lv]["spec"][1] for lv in FIT_LEVELS], shape=L)
            q = sens[level_idx] * p + (1 - spec[level_idx]) * (1 - p)
        else:
            q = p
        pm.Bernoulli("y", p=q, observed=y)
        # nutpie (numba backend): PyTensor's C backend needs g++, which this Windows setup lacks.
        idata = pm.sample(draws, tune=tune, chains=chains, nuts_sampler="nutpie", random_seed=seed, progressbar=False)
    post = idata.posterior
    keep = ["alpha", "a_ta", "beta", "delta", "lam", "sigma_ta", "tau"] + (["sens", "spec"] if label_noise else [])
    out = {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in keep}
    out["ta_on"] = ta_on
    summ = az.summary(idata, var_names=keep, round_to=4)
    out["_diagnostics"] = {
        "max_rhat": float(summ["r_hat"].max()), "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()), "draws": int(chains * draws),
    }
    return out


# ------------------------------------------------------------------------------------------------ prediction
def eta_draws(post: dict, X: np.ndarray, level_idx: np.ndarray, ta_idx: np.ndarray) -> np.ndarray:
    """(draws, n) linear predictor."""
    stage = np.array([STAGE[FIT_LEVELS[i]] for i in level_idx])
    coef = post["beta"][:, None, :] + post["delta"][:, stage, :]          # (D, n, J)
    ta_on = post.get("ta_on", np.ones(len(FIT_LEVELS)))[level_idx]
    return post["alpha"][:, level_idx] + post["a_ta"][:, ta_idx] * ta_on + np.einsum("dnj,nj->dn", coef, X)


def observed_scale(post: dict, p: np.ndarray, level_idx: np.ndarray) -> np.ndarray:
    """P(observed label = 1) from true-outcome probabilities (for scoring against the noisy training labels)."""
    if "sens" not in post:
        return p
    return post["sens"][:, level_idx] * p + (1 - post["spec"][:, level_idx]) * (1 - p)


@dataclass
class FittedModel:
    version: str
    post: dict
    scale: dict
    meta: dict

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "posterior.npz", **{k: v for k, v in self.post.items() if not k.startswith("_")})
        (directory / "model.json").write_text(json.dumps({"version": self.version, "scale": self.scale,
                                                          "specs": KEYS, "tas": TAS, "levels": FIT_LEVELS,
                                                          **self.meta}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> FittedModel:
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if meta["specs"] != KEYS or meta["tas"] != TAS:
            raise ValueError(f"{directory} was fitted with a different feature set; refit it")
        with np.load(directory / "posterior.npz") as z:
            post = {k: z[k] for k in z.files}
        return cls(meta["version"], post, meta["scale"], meta)

    def p_draws(self, level: str, features: dict) -> np.ndarray:
        """Posterior draws of P(true approval within the level's horizon)."""
        f = flat(features)
        x = design(np.array([[np.nan if v is None else v for v in raw_row(level, f)]], dtype=float), self.scale)
        li, ta = FIT_LEVELS.index(level), f.get("therapeutic_area")
        ta_on = float(self.post.get("ta_on", np.ones(len(FIT_LEVELS)))[li])
        a_ta = self.post["a_ta"][:, ta_index(ta)] * ta_on if ta else 0.0
        eta = eta_draws(self.post, x, np.array([li]), np.array([0]))[:, 0] - self.post["a_ta"][:, 0] * ta_on + a_ta
        return 1 / (1 + np.exp(-eta))

    def predict(self, level: str, features: dict) -> dict:
        """Same shape as `model.provisional.predict`: summary of P(approval), base rate and per-feature
        contributions on the logit scale."""
        f = flat(features)
        raw = np.array([[np.nan if v is None else v for v in raw_row(level, f)]], dtype=float)
        x = design(raw, self.scale)
        li, ta = FIT_LEVELS.index(level), f.get("therapeutic_area")
        ta_on = float(self.post.get("ta_on", np.ones(len(FIT_LEVELS)))[li])
        a_ta = self.post["a_ta"][:, ta_index(ta)] * ta_on if ta else 0.0  # unknown TA: the average area
        eta = eta_draws(self.post, x, np.array([li]), np.array([0]))[:, 0] - self.post["a_ta"][:, 0] * ta_on + a_ta
        p = 1 / (1 + np.exp(-eta))
        base = 1 / (1 + np.exp(-(self.post["alpha"][:, li] + a_ta)))
        coef = self.post["beta"] + self.post["delta"][:, STAGE[level], :]
        contributions = []
        for j, s in enumerate(SPECS):
            if raw[0, j] != raw[0, j] or x[0, j] == 0:              # missing / not applicable / zero
                continue
            c = coef[:, j] * x[0, j]
            contributions.append({"key": s.key, "label": s.label, "logit": round(float(c.mean()), 3),
                                  "sd": round(float(c.std()), 3), "detail": f"value {raw[0, j]:g}", "kind": "fitted"})
        q = np.quantile(p, [0.025, 0.25, 0.5, 0.75, 0.975])
        return {
            "model_version": self.version, "level": level,
            "p_approval": {"median": float(q[2]), "mean": float(p.mean()), "ci50": [float(q[1]), float(q[3])],
                           "ci95": [float(q[0]), float(q[4])]},
            # The level's average fitted probability (contributions are relative to the all-features-off reference).
            "base_rate": float(self.meta.get("level_mean_p", {}).get(level, base.mean())),
            "contributions": contributions,
            "missing_inputs": [s.label for j, s in enumerate(SPECS)
                               if raw[0, j] != raw[0, j] and FIT_LEVELS.index(s.first_level) <= li
                               and (STAGE[level] < 2 or s.late)],
            "draws_histogram": np.histogram(p, bins=20, range=(0, 1))[0].tolist(),
        }
