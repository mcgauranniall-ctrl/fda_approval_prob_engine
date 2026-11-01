"""Approval timing (PLAN.md §3 T_APP, phase P6a): P(approved by month h after the cut-off).

    P(approved by h) = P(approved within the level's horizon)  ×  G(h | approved)

The first factor is the landmark model (`model/landmark.py`). G is the time-to-approval distribution among
programmes approved within the horizon (L1–L2: 7 years from the first Phase 3 start; L3+: 3 years from the
cut-off), estimated directly as a CDF: every approved training row contributes one observation per month h on a
grid (h ≤ its own horizon), labelled "approved by h", and one penalised logistic regression is fitted on

    stage-specific smooth terms in log(h) + covariates × log(h) + "the announced PDUFA date has passed by h"

The PDUFA term lets the curve jump at the goal date (first-cycle approvals) while later cycles keep a tail. The
prediction is made monotone in h. Uncertainty: refits on bootstrap resamples of drugs. G conditions on approval,
so knowing whether a PDUFA date was announced is allowed here (it is never a feature of P(approval)).
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fdaengine.model.landmark import FIT_LEVELS, MODEL_DIR, STAGE, flat

GRID = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 21, 24, 30, 36, 42, 48, 60, 72, 84], dtype=float)
KNOTS = np.log([2.0, 6.0, 12.0, 24.0, 48.0])
COVARIATES = ["large_sponsor", "btd", "orphan", "priority_review", "accelerated", "adcom_scheduled", "pdufa_known"]


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _flag(v) -> float:
    return float(v is True or str(v).lower() in ("true", "yes", "1"))


def covariates(level: str, features: dict) -> dict:
    f = flat(features)
    late = STAGE[level] == 2
    mp = _num(f.get("months_to_pdufa")) if late else None
    return {"large_sponsor": float(f.get("sponsor_size") == "large"), "btd": _flag(f.get("btd")),
            "orphan": _flag(f.get("orphan")),
            "priority_review": _flag(f.get("priority_review")) if late else 0.0,
            "accelerated": _flag(f.get("accelerated_pathway")) if late else 0.0,
            "adcom_scheduled": _flag(f.get("adcom_scheduled")) if late else 0.0,
            "pdufa_known": float(mp is not None), "months_to_pdufa": mp}


def _spline(lh: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis in log(h) (linear term + len(KNOTS) − 2 non-linear terms)."""
    k = KNOTS
    def pos3(x):
        return np.clip(x, 0, None) ** 3
    cols = [lh]
    for j in range(len(k) - 2):
        cols.append(pos3(lh - k[j]) - pos3(lh - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                    + pos3(lh - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
    return np.column_stack(cols) / 10.0


def design(level_idx: np.ndarray, cov: list[dict], h: np.ndarray) -> np.ndarray:
    """One row per (programme row, month h)."""
    n = len(h)
    lh = np.log(h)
    S = _spline(lh)
    stage = np.array([STAGE[FIT_LEVELS[i]] for i in level_idx])
    cols = [np.eye(len(FIT_LEVELS))[level_idx]]                         # level intercepts
    for s in range(3):                                                  # stage-specific shape in log(h)
        cols.append(S * (stage == s)[:, None])
    C = np.array([[c[k] for k in COVARIATES] for c in cov], dtype=float)
    cols += [C, C * lh[:, None]]
    mp = np.array([np.nan if c["months_to_pdufa"] is None else c["months_to_pdufa"] for c in cov])
    passed = np.where(np.isnan(mp), 0.0, (h >= mp + 0.5).astype(float))  # goal date (plus ~2 weeks) has passed
    cols += [passed[:, None], (passed * lh)[:, None]]
    return np.column_stack(cols).reshape(n, -1)


def stack(level_idx: np.ndarray, cov: list[dict], t_months: np.ndarray, horizon_months: np.ndarray):
    """Approved rows → (design, target) over the grid, keeping h ≤ each row's horizon."""
    ri, hh = [], []
    for i, H in enumerate(horizon_months):
        g = GRID[GRID <= H + 1e-9]
        ri += [i] * len(g)
        hh += list(g)
    ri, hh = np.array(ri), np.array(hh)
    X = design(level_idx[ri], [cov[i] for i in ri], hh)
    return X, (t_months[ri] <= hh).astype(int)


def _fit_one(X, y, C: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def cdf(models: list, level_idx: int, cov: dict, h: np.ndarray) -> np.ndarray:
    """(n_models, len(h)) G(h | approved), monotone in h."""
    X = design(np.full(len(h), level_idx), [cov] * len(h), h)
    out = np.array([m.predict_proba(X)[:, 1] for m in models])
    return np.maximum.accumulate(out, axis=1)


@dataclass
class TimingModel:
    version: str
    models: list
    meta: dict

    @classmethod
    def fit(cls, level_idx, cov, t_months, horizon_months, groups, n_boot: int = 50, C: float = 1.0, seed: int = 5,
            version: str = "timing-v1") -> TimingModel:
        X, y = stack(level_idx, cov, t_months, horizon_months)
        models = [_fit_one(X, y, C)]
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        by = {g: np.flatnonzero(groups == g) for g in ug}
        for _ in range(n_boot):
            idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
            Xb, yb = stack(level_idx[idx], [cov[i] for i in idx], t_months[idx], horizon_months[idx])
            models.append(_fit_one(Xb, yb, C))
        return cls(version, models, {"n_approved_rows": int(len(t_months)), "n_boot": n_boot, "C": C})

    def g(self, level: str, features: dict, h: np.ndarray) -> np.ndarray:
        return cdf(self.models, FIT_LEVELS.index(level), covariates(level, features), np.asarray(h, dtype=float))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "timing.pkl", "wb") as fh:
            pickle.dump(self.models, fh)
        (directory / "timing.json").write_text(json.dumps({"version": self.version, "covariates": COVARIATES,
                                                           "grid": GRID.tolist(), **self.meta}, indent=1),
                                               encoding="utf-8")

    @classmethod
    def load(cls, directory: Path = MODEL_DIR / "timing-v1") -> TimingModel:
        meta = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
        with open(directory / "timing.pkl", "rb") as fh:
            models = pickle.load(fh)
        return cls(meta["version"], models, meta)


def approval_by_month(p_draws: np.ndarray, g_draws: np.ndarray, months: np.ndarray) -> list[dict]:
    """Combine P(approved within horizon) draws with G draws → P(approved by month h) summaries."""
    rng = np.random.default_rng(3)
    k = rng.integers(0, len(g_draws), len(p_draws))
    curve = p_draws[:, None] * g_draws[k]
    out = []
    for j, h in enumerate(months):
        c = curve[:, j]
        q = np.quantile(c, [0.05, 0.5, 0.95])
        out.append({"months": float(h), "p": float(c.mean()), "median": float(q[1]), "ci90": [float(q[0]), float(q[2])]})
    return out
