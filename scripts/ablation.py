"""Paired ablation: does a data change (a new source, feature or label rule) change the fitted model's accuracy?

Single cross-validation runs are not comparable across data rebuilds: re-dealt folds alone moved a level's Brier
score by up to 0.017 (PLAN.md, P5/P6b). This compares two landmark builds fairly:
  * identical rows: programme × level present in both builds with the same Y_LEVEL (label changes are left out, so
    only the features differ);
  * identical folds: hashed from the drug id, repeated with several seeds;
  * the same model and sampler settings as script 09 (`bayes`).
Reports the Brier score per level for each build and the difference B − A (negative = B better). A difference
that keeps its sign across seeds and exceeds about 0.001 is a real effect; anything smaller is noise.

Typical use: copy data/processed/landmarks.parquet aside, make the change, rerun scripts 03–05, then
  uv run python scripts/ablation.py <landmarks_before.parquet> data/processed/landmarks.parquet [--seeds 3]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.model import landmark as lmk

_spec = importlib.util.spec_from_file_location("m09", Path(__file__).with_name("09_production_model.py"))
m09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m09)

LEVELS = ["L1", "L2", "L3", "L5", "all"]


def prep(path: Path, keep: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    pop = lmk.training_population(pd.read_parquet(path)).merge(keep, on=["programme_id", "level"])
    pop = pop.sort_values(["programme_id", "level"]).reset_index(drop=True)
    fcols = [c for c in pop.columns if c.startswith("f_")]
    feats = [{c[2:]: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for c, v in zip(fcols, row)}
             for row in pop[fcols].itertuples(index=False)]
    raw = lmk.raw_matrix(pop["level"].tolist(), feats)
    pop["level_idx"] = pop["level"].map(lmk.FIT_LEVELS.index)
    pop["ta_idx"] = pop["f_therapeutic_area"].map(lmk.ta_index)
    pop["y"] = pop["y_level"].astype(int)
    return pop, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_a", type=Path, help="landmarks parquet before the change")
    ap.add_argument("build_b", type=Path, help="landmarks parquet after the change")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cols = ["programme_id", "level", "y_level"]
    ka = lmk.training_population(pd.read_parquet(args.build_a))[cols]
    kb = lmk.training_population(pd.read_parquet(args.build_b))[cols]
    keep = ka.merge(kb, on=["programme_id", "level"])
    keep = keep[(keep["y_level_x"] == keep["y_level_y"]) & keep["level"].isin(lmk.FIT_LEVELS)][["programme_id", "level"]]
    priors = lmk.label_priors(json.loads((config.DATA_DIR / "processed" / "label_error.json").read_text(encoding="utf-8")))
    data = {"A": prep(args.build_a, keep), "B": prep(args.build_b, keep)}
    print(f"[ablation] {len(keep):,} shared rows: {data['A'][0]['level'].value_counts().sort_index().to_dict()}")

    res = []
    for seed in range(args.seeds):
        for name, (pop, raw) in data.items():
            fold = pop["drug_id"].map(lambda d: int(hashlib.md5(f"{d}-{seed}".encode()).hexdigest(), 16) % args.folds)
            fold = fold.to_numpy()
            q = np.zeros(len(pop))
            for k in range(args.folds):
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                q[te] = m09.bayes(tr, te, pop, raw, priors, args.draws, seed=10 * seed + k)[0]
            y = pop["y"].to_numpy()
            for lv in LEVELS:
                mk = np.ones(len(pop), bool) if lv == "all" else (pop["level"] == lv).to_numpy()
                res.append({"seed": seed, "build": name, "level": lv, "brier": float(np.mean((q[mk] - y[mk]) ** 2))})
        print(f"[ablation] seed {seed + 1}/{args.seeds} done", flush=True)

    r = pd.DataFrame(res).pivot_table(index=["seed", "level"], columns="build", values="brier")
    r["B-A"] = r["B"] - r["A"]
    pd.set_option("display.width", 200)
    print(r.round(4).to_string())
    print(r.groupby("level")["B-A"].agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
