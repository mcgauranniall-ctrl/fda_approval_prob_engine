"""Which model scores cases: the provisional prior-only model (default) or the fitted landmark model.

Set FDAENGINE_MODEL=fitted to use `models/landmark-v1` (scripts/09_production_model.py). Levels the fitted model
does not cover (L6), or a missing model directory, fall back to the provisional model.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

import numpy as np

from fdaengine.model import landmark, provisional, timing

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


def is_provisional(pred: dict) -> bool:
    return str(pred.get("model_version", "")).startswith("provisional")
