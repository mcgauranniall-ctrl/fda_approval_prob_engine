"""Which model scores cases: the provisional prior-only model (default) or the fitted landmark model.

Set FDAENGINE_MODEL=fitted to use `models/landmark-v1` (scripts/09_production_model.py), with approval timing
(`timing-v1`, script 10) and label outcomes (`label-v1`, script 11). Levels the fitted model does not cover (L6), or
a missing model directory, fall back to the provisional model.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

import numpy as np

from fdaengine.model import label_model, landmark, provisional, timing

FITTED_VERSION = "landmark-v1"


@lru_cache(maxsize=1)
def _fitted() -> landmark.FittedModel | None:
    path = landmark.MODEL_DIR / FITTED_VERSION
    return landmark.FittedModel.load(path) if (path / "model.json").exists() else None


def use_fitted() -> bool:
    return os.environ.get("FDAENGINE_MODEL", "provisional").lower() == "fitted" and _fitted() is not None


def predict(level: str, features: dict) -> dict:
    if use_fitted() and level in landmark.FIT_LEVELS:
        return _fitted().predict(level, features)
    return provisional.predict(level, features)


@lru_cache(maxsize=1)
def _timing() -> timing.TimingModel | None:
    path = landmark.MODEL_DIR / "timing-v1"
    return timing.TimingModel.load(path) if (path / "timing.json").exists() else None


TIMING_MONTHS = [6, 12, 24, 36]


def approval_timing(level: str, features: dict, cutoff: str) -> dict | None:
    """P(approved by 6/12/24/36 months after the cut-off and by 1 January of the next two plan years), with 90%
    intervals. Fitted models only (the provisional model has no timing component)."""
    if not (use_fitted() and level in landmark.FIT_LEVELS and _timing() is not None):
        return None
    c = date.fromisoformat(cutoff)
    plan_years = [date(c.year + k, 1, 1) for k in (1, 2)]
    months = TIMING_MONTHS + [round((d - c).days / 30.44, 2) for d in plan_years]
    grid = np.array(sorted(set(months)), dtype=float)                 # the curve is made monotone along the grid
    p = _fitted().p_draws(level, features)
    by_m = dict(zip(grid.tolist(), timing.approval_by_month(p, _timing().g(level, features, grid), grid)))
    rows = [by_m[float(m)] for m in months]
    return {"model_version": _timing().version,
            "by_months": rows[:len(TIMING_MONTHS)],
            "by_plan_year": [{"date": d.isoformat(), **r} for d, r in zip(plan_years, rows[len(TIMING_MONTHS):])]}


@lru_cache(maxsize=1)
def _labels() -> label_model.LabelModel | None:
    path = landmark.MODEL_DIR / "label-v1"
    return label_model.LabelModel.load(path) if (path / "label_model.json").exists() else None


def label_outcomes(level: str, features: dict) -> dict | None:
    """P(boxed warning / REMS / accelerated approval / narrower label | approved), with 90% intervals. Fitted
    models only; outcomes the validation found no better than the base rate at this stage (L1 | L2 | L3–L5) are
    marked `beats_base_rate: False`."""
    if not (use_fitted() and level in landmark.FIT_LEVELS and _labels() is not None):
        return None
    m = _labels()
    stage = {0: "L1", 1: "L2", 2: "L3–L5"}[landmark.STAGE[level]]
    verdict = {v["outcome"]: v for v in m.meta.get("verdict", []) if v.get("stage") == stage}
    out = m.predict(level, features)
    for k, v in out.items():
        v["beats_base_rate"] = bool(verdict.get(k, {}).get("beats_stage_rate", False))
    return {"model_version": m.version, "conditional_on": "approval", "outcomes": out}


def is_provisional(pred: dict) -> bool:
    return str(pred.get("model_version", "")).startswith("provisional")
