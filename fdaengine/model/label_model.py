"""Label outcomes given approval (PLAN.md §3, phase P6c): P(boxed warning), P(REMS) and P(accelerated approval),
each conditional on the programme being approved.

One penalised logistic regression per outcome on approved training rows (novel drugs, any level L1–L5):

    logit P(outcome | approved, information at the level) = α_stage + Σ_j β_j x_j        (stage: L1 | L2 | L3–L5)

Inputs are pre-decision features cut off at the level (the same `features.build` names as the landmark model); a
missing value contributes nothing (standardised to the training mean). The penalty is chosen by cross-validation
grouped by drug inside the training data, and 50 refits on bootstrap resamples of drugs give 90% intervals.
Training labels come from post-decision documents (`dataset/label_outcomes.py`); they are never inputs.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fdaengine.model.landmark import FIT_LEVELS, MODEL_DIR, STAGE, TAS, flat

# "Label narrower than the trial population" is not modelled: its rule-based label was right for 13 of 25 flagged
# programmes on a fresh review (half the errors are upstream trial grouping), too noisy to train on (PLAN.md P6c).
OUTCOMES = {"boxed_warning": "Boxed warning", "rems": "REMS required",
            "accelerated": "Accelerated approval (confirmatory trial required)"}
CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _flag(v) -> float | None:
    return None if v is None else float(v is True or str(v).lower() in ("true", "yes", "1"))


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _log(v) -> float | None:
    x = _num(v)
    return None if x is None or x < 0 else math.log1p(x)


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    first_level: str
    kind: str                                   # 'flag' (0/1, missing = 0) | 'cont' (standardised, missing = 0)
    fn: Callable[[dict], float | None]


def _era(start: int) -> Callable[[dict], float | None]:
    return lambda f: None if _num(f.get("anchor_year")) is None else float(_num(f["anchor_year"]) >= start)


def _ta(ta: str) -> Callable[[dict], float | None]:
    return lambda f: _flag(f.get("therapeutic_area") == ta)


# Era as cumulative steps of the first Phase 3 start ('started 2015 or later' ...), not a line: label practice
# shifts in steps (Medication-Guide-only REMS 2008–2011; accelerated approvals rising since 2015), and a linear term
# extrapolated to recent programmes predicted a 1% REMS rate against 8% observed in the temporal holdout. Each step
# is the change from the previous period, so a sparse recent period shrinks towards the one before it.
ERAS = [2005, 2010, 2015, 2020]
FEATURES: list[Feature] = [
    *[Feature(f"era_{a}", f"Phase 3 started {a} or later", "L1", "flag", _era(a)) for a in ERAS],
    Feature("biologic", "Biologic", "L1", "flag", lambda f: _flag(f.get("modality") == "biologic")),
    Feature("large_sponsor", "Large sponsor", "L1", "flag", lambda f: _flag(f.get("sponsor_size") == "large")),
    Feature("combination", "Combination product", "L1", "flag", lambda f: _flag(f.get("is_combination"))),
    Feature("orphan", "Orphan designation reported", "L1", "flag", lambda f: _flag(f.get("orphan"))),
    Feature("btd", "Breakthrough therapy designation reported", "L1", "flag", lambda f: _flag(f.get("btd"))),
    Feature("fast_track", "Fast Track reported", "L1", "flag", lambda f: _flag(f.get("fast_track"))),
    Feature("placebo", "Placebo-controlled Phase 3", "L1", "flag", lambda f: _flag(f.get("p3_comparator") == "placebo")),
    Feature("double_blind", "Double-blind Phase 3", "L1", "flag", lambda f: _flag(f.get("p3_double_blind"))),
    Feature("pivotal2", "Two or more randomized Phase 3 trials", "L1", "flag",
            lambda f: None if _num(f.get("p3_n_pivotal_trials")) is None else float(_num(f["p3_n_pivotal_trials"]) >= 2)),
    Feature("p3_size", "Phase 3 planned enrolment (log)", "L1", "cont", lambda f: _log(f.get("p3_planned_n"))),
    Feature("ep_survival", "Primary endpoint: overall survival", "L1", "flag",
            lambda f: _flag(f.get("p3_endpoint_class") == "survival")),
    Feature("ep_time_to_event", "Primary endpoint: progression/event-free time", "L1", "flag",
            lambda f: _flag(f.get("p3_endpoint_class") == "time_to_event")),
    Feature("ep_response", "Primary endpoint: response rate", "L1", "flag",
            lambda f: _flag(f.get("p3_endpoint_class") == "response_rate")),
    Feature("ep_biomarker", "Primary endpoint: biomarker", "L1", "flag",
            lambda f: _flag(f.get("p3_endpoint_class") == "biomarker")),
    Feature("p3_z", "Phase 3 strength of evidence (|z|)", "L2", "cont",
            lambda f: None if _num(f.get("p3_z")) is None else min(_num(f["p3_z"]), 8.0)),
    Feature("p3_deaths", "Phase 3 excess deaths (risk difference)", "L2", "cont", lambda f: _num(f.get("p3_death_rd"))),
    Feature("p3_sae", "Phase 3 serious adverse event ratio (log)", "L2", "cont",
            lambda f: None if not _num(f.get("p3_sae_rr")) else math.log(max(_num(f["p3_sae_rr"]), 0.05))),
    Feature("priority_review", "Priority review", "L3", "flag", lambda f: _flag(f.get("priority_review"))),
    Feature("accelerated_pathway", "Accelerated approval pathway reported", "L3", "flag",
            lambda f: _flag(f.get("accelerated_pathway"))),
    Feature("adcom_scheduled", "AdCom scheduled", "L3", "flag", lambda f: _flag(f.get("adcom_scheduled"))),
] + [Feature(f"ta_{ta}", f"Therapeutic area: {ta.replace('_', ' ')}", "L1", "flag", _ta(ta)) for ta in TAS]
KEYS = [x.key for x in FEATURES]


def raw_row(level: str, features: dict) -> list[float | None]:
    f = flat(features)
    li = FIT_LEVELS.index(level)
    return [x.fn(f) if FIT_LEVELS.index(x.first_level) <= li else None for x in FEATURES]


def raw_matrix(levels: list[str], feature_dicts: list[dict]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in raw_row(lv, f)] for lv, f in zip(levels, feature_dicts)],
                    dtype=float)


def standardiser(raw: np.ndarray) -> dict:
    out = {}
    for j, x in enumerate(FEATURES):
        if x.kind == "cont":
            col = raw[:, j][~np.isnan(raw[:, j])]
            out[x.key] = [float(col.mean()), float(col.std() or 1.0)] if len(col) else [0.0, 1.0]
    return out


def design(levels: list[str], raw: np.ndarray, scale: dict) -> np.ndarray:
    x = raw.copy()
    for j, f in enumerate(FEATURES):
        if f.kind == "cont":
            m, sd = scale[f.key]
            x[:, j] = (x[:, j] - m) / sd
    stage = np.eye(3)[[STAGE[lv] for lv in levels]][:, 1:]             # L2 and L3–L5 as deviations from L1
    return np.column_stack([stage, np.nan_to_num(x, nan=0.0)])


def _fit(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    # The intercept is unpenalised, so a strong penalty shrinks towards the outcome's base rate, not towards 50%.
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def choose_c(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> float:
    """The penalty with the best log loss in cross-validation grouped by drug (training data only)."""
    from sklearn.model_selection import GroupKFold
    best, best_ll = CS[-1], np.inf
    for C in CS:
        ll = []
        for tr, te in GroupKFold(folds).split(X, y, groups):
            if y[tr].min() == y[tr].max():
                continue
            p = np.clip(_fit(X[tr], y[tr], C).predict_proba(X[te])[:, 1], 1e-4, 1 - 1e-4)
            ll.append(-np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        if ll and sum(ll) < best_ll:
            best, best_ll = C, sum(ll)
    return best


@dataclass
class LabelModel:
    version: str
    models: dict                                 # outcome → [point fit, bootstrap fits...]
    scale: dict
    meta: dict

    @classmethod
    def fit(cls, levels: list[str], raw: np.ndarray, outcomes: dict[str, np.ndarray], groups: np.ndarray,
            n_boot: int = 50, seed: int = 9, version: str = "label-v1") -> LabelModel:
        """`outcomes`: outcome → 0/1 array with NaN where the outcome is unknown for that row."""
        models, meta = {}, {"n_rows": {}, "rate": {}, "rate_by_stage": {}, "C": {}, "n_boot": n_boot}
        rng = np.random.default_rng(seed)
        scale = standardiser(raw)
        for name, y_all in outcomes.items():
            keep = ~np.isnan(y_all)
            r, y, g = raw[keep], y_all[keep].astype(int), groups[keep]
            X = design([lv for lv, k in zip(levels, keep) if k], r, scale)
            C = choose_c(X, y, g)
            fits = [_fit(X, y, C)]
            ug = np.unique(g)
            by = {u: np.flatnonzero(g == u) for u in ug}
            for _ in range(n_boot):
                idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug))])
                if y[idx].min() != y[idx].max():
                    fits.append(_fit(X[idx], y[idx], C))
            models[name] = fits
            meta["n_rows"][name], meta["rate"][name], meta["C"][name] = int(keep.sum()), float(y.mean()), C
            st = np.array([STAGE[lv] for lv, k in zip(levels, keep) if k])
            meta["rate_by_stage"][name] = {str(s): float(y[st == s].mean()) for s in np.unique(st)}
        return cls(version, models, scale, meta)

    def p(self, outcome: str, levels: list[str], raw: np.ndarray) -> np.ndarray:
        """Point predictions for many rows."""
        return self.models[outcome][0].predict_proba(design(levels, raw, self.scale))[:, 1]

    def predict(self, level: str, features: dict) -> dict:
        """{outcome: {label, p, ci90, training_rate}} for one case, all conditional on approval. `training_rate` is
        the share among approved training rows at the level's stage (L1 | L2 | L3–L5)."""
        raw = np.array([[np.nan if v is None else v for v in raw_row(level, features)]], dtype=float)
        X = design([level], raw, self.scale)
        out = {}
        for name, fits in self.models.items():
            p = float(fits[0].predict_proba(X)[0, 1])
            boot = np.array([m.predict_proba(X)[0, 1] for m in fits[1:]]) if len(fits) > 1 else np.array([p])
            rate = self.meta.get("rate_by_stage", {}).get(name, {}).get(str(STAGE[level]), self.meta["rate"][name])
            out[name] = {"label": OUTCOMES[name], "p": p, "ci90": [float(np.quantile(boot, 0.05)),
                                                                  float(np.quantile(boot, 0.95))],
                         "training_rate": rate}
        return out

    def coefficients(self, outcome: str) -> list[dict]:
        """Odds ratios (per 1 SD for continuous inputs) with 90% bootstrap intervals."""
        fits = self.models[outcome]
        names = ["Stage L2 vs L1", "Stage L3–L5 vs L1"] + [f.label for f in FEATURES]
        co = np.array([m.coef_[0] for m in fits])
        return [{"feature": n, "or": float(np.exp(co[0, j])), "or_lo": float(np.exp(np.quantile(co[1:, j], 0.05))),
                 "or_hi": float(np.exp(np.quantile(co[1:, j], 0.95)))} for j, n in enumerate(names)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "label_models.pkl", "wb") as fh:
            pickle.dump(self.models, fh)
        (directory / "label_model.json").write_text(json.dumps({"version": self.version, "features": KEYS,
                                                                "scale": self.scale, **self.meta}, indent=1,
                                                               default=float), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = MODEL_DIR / "label-v1") -> LabelModel:
        meta = json.loads((directory / "label_model.json").read_text(encoding="utf-8"))
        if meta["features"] != KEYS:
            raise ValueError(f"{directory} was fitted with a different feature set; refit it")
        with open(directory / "label_models.pkl", "rb") as fh:
            models = pickle.load(fh)
        return cls(meta["version"], models, meta["scale"], meta)
