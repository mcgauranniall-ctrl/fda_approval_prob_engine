"""10 — Approval timing model (PLAN.md §3 T_APP, phase P6a): fit, validate, save.

P(approved by month h) = P(approved within the horizon) [landmark model, script 09] × G(h | approved)
[`model/timing.py`]. Validation on the training population (novel drugs, final outcomes):
  * 5-fold cross-validation, folds hashed by drug; P(approved within the horizon) is script 09's out-of-fold
    probability of the observed label, so the comparison with observed approvals is like with like;
  * at h = 6, 12, 24 and 36 months (rows whose horizon reaches h): observed share approved by h vs mean predicted,
    Brier score; benchmarks: the level's base rate × the level's empirical time-to-approval curve, and the same
    probability × the empirical curve (isolates what the timing model adds);
  * P6a target: P(approved by 12/24/36 months) calibrated per level (mean predicted within the observed share's
    95% interval) and a lower Brier score than the empirical-curve benchmarks.

  uv run python scripts/10_timing_model.py
Outputs: models/timing-v1/, data/processed/timing_validation.json, reports/timing.html
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.model import landmark as lmk
from fdaengine.model import timing as tm

PROCESSED = config.DATA_DIR / "processed"
REPORTS = config.PROJECT_ROOT / "reports"
HORIZONS = [6, 12, 24, 36]
FOLDS = 5


def load() -> pd.DataFrame:
    lm = pd.read_parquet(PROCESSED / "landmarks.parquet")
    pop = lmk.training_population(lm)
    pop = pop[pop["level"].isin(lmk.FIT_LEVELS)].reset_index(drop=True)
    oof = pd.read_parquet(PROCESSED / "oof_predictions.parquet")[["programme_id", "level", "bayes (observed label)"]]
    pop = pop.merge(oof.rename(columns={"bayes (observed label)": "p_obs"}), on=["programme_id", "level"], how="left")
    fcols = [c for c in pop.columns if c.startswith("f_")]
    pop["features"] = [{c[2:]: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for c, v in zip(fcols, r)}
                       for r in pop[fcols].itertuples(index=False)]
    pop["cov"] = [tm.covariates(lv, f) for lv, f in zip(pop["level"], pop["features"])]
    pop["level_idx"] = pop["level"].map(lmk.FIT_LEVELS.index)
    pop["y"] = pop["y_level"].astype(int)
    pop["t_months"] = pop["t_level_days"] / 30.44

    def horizon(r):
        if lmk.STAGE[r.level] == 2:
            return 36.0
        a = date.fromisoformat(r.anchor_p3_start)
        end = date(a.year + 7, a.month, min(a.day, 28))
        return (end - date.fromisoformat(r.cutoff)).days / 30.44
    pop["horizon"] = [horizon(r) for r in pop.itertuples()]
    pop["fold"] = pop["drug_id"].map(lambda d: int(hashlib.md5(str(d).encode()).hexdigest(), 16) % FOLDS)
    return pop


def empirical_cdf(train: pd.DataFrame, level: str, h: float) -> float:
    a = train[(train["level"] == level) & (train["y"] == 1) & (train["horizon"] >= h)]
    return float((a["t_months"] <= h).mean()) if len(a) else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    pop = load()
    print(f"[10] {len(pop):,} rows, {int(pop['y'].sum()):,} approved; months to PDUFA known at L3+: "
          f"{int(sum(c['pdufa_known'] for c, lv in zip(pop['cov'], pop['level']) if lmk.STAGE[lv] == 2))}")
    rows = []
    for k in range(FOLDS):
        tr, te = pop[pop["fold"] != k], pop[pop["fold"] == k]
        a = tr[tr["y"] == 1]
        model = tm.TimingModel.fit(a["level_idx"].to_numpy(), a["cov"].tolist(), a["t_months"].to_numpy(),
                                   a["horizon"].to_numpy(), a["drug_id"].to_numpy(), n_boot=0)
        base = tr.groupby("level")["y"].mean()
        for r in te.itertuples():
            hs = np.array([h for h in HORIZONS if h <= r.horizon + 1e-9], dtype=float)
            if not len(hs):
                continue
            g = tm.cdf(model.models, r.level_idx, r.cov, hs)[0]
            for h, gv in zip(hs, g):
                emp = empirical_cdf(tr, r.level, h)
                rows.append({"level": r.level, "h": int(h), "observed": int(r.y == 1 and r.t_months <= h),
                             "timing model": r.p_obs * gv, "same p × empirical curve": r.p_obs * emp,
                             "base rate × empirical curve": base.get(r.level, tr["y"].mean()) * emp})
        print(f"[10] fold {k + 1}/{FOLDS} ({time.time() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(rows).dropna()
    models = ["timing model", "same p × empirical curve", "base rate × empirical curve"]
    table = []
    for (lv, h), d in cv.groupby(["level", "h"]):
        obs = d["observed"].mean()
        se = np.sqrt(obs * (1 - obs) / len(d)) if len(d) else np.nan
        rec = {"level": lv, "h": h, "n": len(d), "observed": obs, "obs_lo": max(0, obs - 1.96 * se),
               "obs_hi": min(1, obs + 1.96 * se)}
        for m in models:
            rec[f"pred: {m}"] = d[m].mean()
            rec[f"brier: {m}"] = float(np.mean((d[m] - d["observed"]) ** 2))
        rec["calibrated"] = bool(rec["obs_lo"] <= rec["pred: timing model"] <= rec["obs_hi"])
        rec["beats_benchmarks"] = bool(rec["brier: timing model"] < min(rec[f"brier: {m}"] for m in models[1:]))
        table.append(rec)
    table = pd.DataFrame(table)

    a = pop[pop["y"] == 1]
    final = tm.TimingModel.fit(a["level_idx"].to_numpy(), a["cov"].tolist(), a["t_months"].to_numpy(),
                               a["horizon"].to_numpy(), a["drug_id"].to_numpy(), n_boot=50)
    final.meta.update({"fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                       "validation": table.to_dict("records")})
    final.save(lmk.MODEL_DIR / "timing-v1")
    examples = []
    for lv, cov in (("L1", {}), ("L3", {"priority_review": True}), ("L3", {}),
                    ("L5", {"months_to_pdufa": 0.5}), ("L5", {})):
        g = final.g(lv, cov, np.array(HORIZONS, dtype=float))
        examples.append({"level": lv, "inputs": json.dumps(cov), **{f"G({h})": float(g[0, j]) for j, h in enumerate(HORIZONS)}})
    examples = pd.DataFrame(examples)
    (PROCESSED / "timing_validation.json").write_text(json.dumps({"cv": table.to_dict("records"),
                                                                  "examples": examples.to_dict("records")},
                                                                 indent=1, default=float), encoding="utf-8")
    write_report(table, examples, len(pop))
    pd.set_option("display.width", 250)
    cols = ["level", "h", "n", "observed", "pred: timing model", "brier: timing model",
            "brier: same p × empirical curve", "brier: base rate × empirical curve", "calibrated", "beats_benchmarks"]
    print(table[cols].round(3).to_string(index=False))
    print(examples.round(3).to_string(index=False))
    print(f"[10] done in {time.time() - t0:.0f}s -> {REPORTS / 'timing.html'}")


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell(v))}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f"<div class=tbl><table><tr>{head}</tr>{body}</table></div>"


def write_report(table: pd.DataFrame, examples: pd.DataFrame, n: int) -> None:
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#17140f;background:#f5f0e8}"
           "h1,h2{font-family:Georgia,serif}table{border-collapse:collapse;font-size:13px;background:#fff;margin:8px 0}"
           "th,td{border:1px solid #ddd6c8;padding:4px 8px;text-align:left}th{background:#faf7f2}.tbl{overflow-x:auto}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "timing.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Approval timing</title><style>{css}</style>"
        f"<h1>Approval timing model (timing-v1)</h1><p>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. {n:,} training rows. "
        f"P(approved by month h) = P(approved within the horizon) × G(h | approved). Cross-validated (folds hashed by "
        f"drug), scored against observed approvals by h.</p><h2>Calibration and accuracy by level and horizon</h2>"
        f"{_table(table)}<h2>G(h | approved): example curves</h2>{_table(examples)}", encoding="utf-8")


if __name__ == "__main__":
    main()
