"""09 — Production landmark model (PLAN.md §7.5–7.7, phase P5): fit, validate, calibrate, save.

The model (`fdaengine/model/landmark.py`) is a Bayesian logistic model across levels L1–L5 with level-specific
effects shrunk towards shared ones, a therapeutic-area random effect, and the training label treated as a noisy
measurement of the truth (sensitivity/specificity priors from `data/processed/label_error.json`, script 08).

Validation, all on the training population (novel drugs, final outcomes; script 08):
  * 5-fold cross-validation grouped by drug (a drug's programmes never straddle train and test);
  * a temporal holdout: fitted on rows with cut-offs before the 80th percentile date, scored on the later ones;
  * per level: Brier score (and skill vs the level base rate), log loss, AUROC, calibration intercept and slope,
    reliability by decile. Scores compare the model's probability of the *observed* label (true probability passed
    through the label-error model) with the observed labels, so they are on the same footing as the benchmarks;
  * benchmarks fitted on the same folds: level base rate, literature rates, TA prior rate, "endpoint met" alone,
    the provisional model the app uses today, a plain logistic regression on the same features, and a LightGBM
    challenger (§7.6).
P5 is met at a level when the calibration slope is 0.8–1.2 and the model's Brier score beats every benchmark.

  uv run python scripts/09_production_model.py [--folds 5] [--draws 1000]
Outputs: models/landmark-v1/ (posterior.npz, model.json), data/processed/oof_predictions.parquet,
         data/processed/validation.json, reports/validation.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.model import landmark as lmk
from fdaengine.model import provisional

PROCESSED = config.DATA_DIR / "processed"
REPORTS = config.PROJECT_ROOT / "reports"
VERSION = "landmark-v1"
JUDGED = ["L1", "L2", "L3", "L5"]          # L4 (16 rows) is reported, not judged
SLOPE_OK = (0.8, 1.2)


# ------------------------------------------------------------------------------------------------ data
def load() -> tuple[pd.DataFrame, np.ndarray]:
    lm = pd.read_parquet(PROCESSED / "landmarks.parquet")
    pop = lmk.training_population(lm)
    pop = pop[pop["level"].isin(lmk.FIT_LEVELS)].reset_index(drop=True)
    fcols = [c for c in pop.columns if c.startswith("f_")]
    feats = [{c[2:]: (None if pd.isna(v) else v) if not isinstance(v, (list, dict)) else v for c, v in zip(fcols, row)}
             for row in pop[fcols].itertuples(index=False)]
    raw = lmk.raw_matrix(pop["level"].tolist(), feats)
    pop["level_idx"] = pop["level"].map(lmk.FIT_LEVELS.index)
    pop["ta_idx"] = pop["f_therapeutic_area"].map(lmk.ta_index)
    pop["y"] = pop["y_level"].astype(int)
    return pop, raw


# ------------------------------------------------------------------------------------------------ models
def bayes(train: np.ndarray, test: np.ndarray, pop: pd.DataFrame, raw: np.ndarray, priors: dict, draws: int,
          seed: int) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    """(observed-scale p, true-scale p) for the test rows, plus the posterior, scaler and diagnostics."""
    scale = lmk.standardiser(raw[train])
    X = lmk.design(raw, scale)
    tr = pop.iloc[train]
    post = lmk.fit(X[train], tr["level_idx"].to_numpy(), tr["ta_idx"].to_numpy(), tr["y"].to_numpy(), priors,
                   draws=draws, tune=draws, seed=seed)
    te = pop.iloc[test]
    li, ti = te["level_idx"].to_numpy(), te["ta_idx"].to_numpy()
    p = 1 / (1 + np.exp(-lmk.eta_draws(post, X[test], li, ti)))
    q = lmk.observed_scale(post, p, li)
    return q.mean(0), p.mean(0), post, scale, post.pop("_diagnostics")


def _onehot(idx: np.ndarray, k: int) -> np.ndarray:
    return np.eye(k)[idx]


def benchmarks(train: np.ndarray, test: np.ndarray, pop: pd.DataFrame, raw: np.ndarray) -> dict[str, np.ndarray]:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression

    y = pop["y"].to_numpy()
    lv = _onehot(pop["level_idx"].to_numpy(), len(lmk.FIT_LEVELS))
    out = {}
    rate = pop.iloc[train].groupby("level")["y"].mean()
    out["level base rate"] = pop.iloc[test]["level"].map(rate).fillna(pop.iloc[train]["y"].mean()).to_numpy()
    out["literature rate"] = pop.iloc[test]["level"].map(lmk.LITERATURE_RATE).to_numpy()
    ta = pop["f_ta_prior_rate"].astype(float).fillna(pop["f_ta_prior_rate"].astype(float).mean()).clip(0.02, 0.98)
    Z = np.column_stack([lv, np.log(ta / (1 - ta))])
    out["TA prior rate"] = LogisticRegression(C=100, max_iter=2000).fit(Z[train], y[train]).predict_proba(Z[test])[:, 1]
    j2, j3 = lmk.KEYS.index("p2_met"), lmk.KEYS.index("p3_met")
    met = np.nan_to_num(raw[:, [j2, j3]])
    Z = np.column_stack([lv, met * lv[:, [0]], met * lv[:, [1]], met * lv[:, 2:].sum(1, keepdims=True)])
    out["endpoint met alone"] = LogisticRegression(C=100, max_iter=2000).fit(Z[train], y[train]).predict_proba(Z[test])[:, 1]
    out["provisional (app today)"] = pop.iloc[test]["provisional_p"].to_numpy()
    scale = lmk.standardiser(raw[train])
    X = lmk.design(raw, scale)
    Z = np.column_stack([lv, X, _onehot(pop["ta_idx"].to_numpy(), len(lmk.TAS))])
    out["logistic, same features"] = LogisticRegression(C=1.0, max_iter=5000).fit(Z[train], y[train]).predict_proba(Z[test])[:, 1]
    G = np.column_stack([raw, pop["level_idx"].to_numpy(), pop["ta_idx"].to_numpy()])
    gbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=3)
    gbm.fit(G[train], y[train], categorical_feature=[G.shape[1] - 1])
    out["LightGBM challenger"] = gbm.predict_proba(G[test])[:, 1]
    return out


# ------------------------------------------------------------------------------------------------ scoring
def scores(p: np.ndarray, y: np.ndarray) -> dict:
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(pc / (1 - pc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slope = float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])
            icpt = float(sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])
        except Exception:                                                   # noqa: BLE001 (tiny / separated levels)
            slope = icpt = float("nan")
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return {"n": int(len(y)), "observed": float(base), "mean_predicted": float(p.mean()), "brier": brier,
            "brier_skill": float(1 - brier / (base * (1 - base))) if 0 < base < 1 else float("nan"),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "cal_intercept": icpt, "cal_slope": slope}


def score_table(df: pd.DataFrame, cols: list[str], split: str) -> pd.DataFrame:
    rows = []
    for lv in lmk.FIT_LEVELS + ["all"]:
        d = df if lv == "all" else df[df["level"] == lv]
        if len(d) < 5:
            continue
        for c in cols:
            rows.append({"split": split, "level": lv, "model": c, **scores(d[c].to_numpy(), d["y"].to_numpy())})
    return pd.DataFrame(rows)


def reliability(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for lv, d in df.groupby("level"):
        if len(d) < 30:
            continue
        bins = pd.qcut(d[col].rank(method="first"), q=min(10, len(d) // 25), labels=False)
        for b, g in d.groupby(bins):
            rows.append({"level": lv, "bin": int(b) + 1, "n": len(g), "mean_predicted": g[col].mean(),
                         "observed": g["y"].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()

    pop, raw = load()
    priors = lmk.label_priors(json.loads((PROCESSED / "label_error.json").read_text(encoding="utf-8")))
    pop["provisional_p"] = [provisional.predict(lv, json.loads(f))["p_approval"]["mean"]
                            for lv, f in zip(pop["level"], pop["features_json"])]
    print(f"[09] {len(pop):,} rows: {pop['level'].value_counts().sort_index().to_dict()}")
    print(f"[09] label priors: {json.dumps(priors)}")

    # Cross-validation, grouped by drug.
    oof = pop[["programme_id", "drug_id", "level", "cutoff", "y"]].copy()
    diag = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(pop, groups=pop["drug_id"])):
        q, p, _post, _scale, d = bayes(tr, te, pop, raw, priors, max(500, args.draws // 2), seed=100 + k)
        oof.loc[te, "bayes (observed label)"] = q
        oof.loc[te, "bayes (true outcome)"] = p
        for name, v in benchmarks(tr, te, pop, raw).items():
            oof.loc[te, name] = v
        diag.append({"fold": k, **d})
        print(f"[09] fold {k + 1}/{args.folds} done ({time.time() - t0:.0f}s) {d}")
    model_cols = ["bayes (observed label)", "level base rate", "literature rate", "TA prior rate", "endpoint met alone",
                  "provisional (app today)", "logistic, same features", "LightGBM challenger"]
    cv = score_table(oof, model_cols, "cv")

    # Temporal holdout.
    t_split = sorted(pop["cutoff"])[int(0.8 * len(pop))]                  # ISO date strings
    tr = np.flatnonzero(pop["cutoff"] < t_split)
    te = np.flatnonzero(pop["cutoff"] >= t_split)
    q, p, _post, _scale, d = bayes(tr, te, pop, raw, priors, max(500, args.draws // 2), seed=200)
    hold = pop.iloc[te][["programme_id", "level", "cutoff", "y"]].copy()
    hold["bayes (observed label)"] = q
    for name, v in benchmarks(tr, te, pop, raw).items():
        hold[name] = v
    temporal = score_table(hold, model_cols, f"temporal (test: cut-off ≥ {t_split})")
    diag.append({"fold": "temporal", **d})
    print(f"[09] temporal holdout done: train {len(tr):,}, test {len(te):,} ({time.time() - t0:.0f}s)")

    # Final fit on everything.
    every = np.arange(len(pop))
    _q, p_all, post, scale, d = bayes(every, every, pop, raw, priors, args.draws, seed=300)
    diag.append({"fold": "final", **d})
    model = lmk.FittedModel(VERSION, post, scale, {
        "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"), "n_rows": int(len(pop)),
        "rows_by_level": pop["level"].value_counts().to_dict(), "label_priors": priors, "diagnostics": d,
        "level_mean_p": pd.Series(p_all).groupby(pop["level"].to_numpy()).mean().round(4).to_dict(),
        "target": "Y_LEVEL (L1-L2: approved within 7 years of the first Phase 3 start; L3+: within 3 years of the cut-off)"})
    model.save(lmk.MODEL_DIR / VERSION)
    print(f"[09] final model saved to {lmk.MODEL_DIR / VERSION} ({time.time() - t0:.0f}s) {d}")

    # Effects (odds ratios per stage) and the corrected rate check.
    effects = []
    for j, s in enumerate(lmk.SPECS):
        for stage, name in enumerate(["L1", "L2", "L3–L5"]):
            if stage < lmk.STAGE[s.first_level] or (stage == 2 and not s.late):
                continue
            c = post["beta"][:, j] + post["delta"][:, stage, j]
            effects.append({"feature": s.label, "stage": name, "per": "1 SD" if s.kind == "cont" else "yes vs no",
                            "or_median": float(np.exp(np.median(c))), "or_lo": float(np.exp(np.quantile(c, 0.025))),
                            "or_hi": float(np.exp(np.quantile(c, 0.975))), "p_positive": float((c > 0).mean())})
    effects = pd.DataFrame(effects)
    err = json.loads((PROCESSED / "label_error.json").read_text(encoding="utf-8"))
    pop["p_true"] = p_all
    rate_check = [{"level": lv, "mean_true_p": float(g["p_true"].mean()), "observed_label_rate": float(g["y"].mean()),
                   "gold_corrected_rate": err.get(lv, {}).get("corrected_rate")} for lv, g in pop.groupby("level")]

    allsc = pd.concat([cv, temporal], ignore_index=True)
    verdict = []
    for lv in JUDGED:
        d_ = cv[cv["level"] == lv].set_index("model")
        b = d_.loc["bayes (observed label)"]
        others = d_.drop(index="bayes (observed label)")
        best_other = others["brier"].idxmin()
        verdict.append({"level": lv, "n": int(b["n"]), "cal_slope": b["cal_slope"], "cal_intercept": b["cal_intercept"],
                        "brier": b["brier"], "best_benchmark": best_other, "best_benchmark_brier": others["brier"].min(),
                        "auroc": b["auroc"], "slope_ok": bool(SLOPE_OK[0] <= b["cal_slope"] <= SLOPE_OK[1]),
                        "beats_all": bool(b["brier"] < others["brier"].min())})
    verdict = pd.DataFrame(verdict)

    oof.to_parquet(PROCESSED / "oof_predictions.parquet", index=False)
    rel = reliability(oof, "bayes (observed label)")
    (PROCESSED / "validation.json").write_text(json.dumps({
        "version": VERSION, "verdict": verdict.to_dict("records"), "scores": allsc.to_dict("records"),
        "diagnostics": diag, "rate_check": rate_check, "effects": effects.to_dict("records"),
        "reliability": rel.to_dict("records"), "temporal_split": str(t_split)}, indent=1, default=float), encoding="utf-8")
    write_report(verdict, allsc, effects, rel, diag, rate_check, priors, len(pop), str(t_split))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(verdict.round(3).to_string(index=False))
    print(cv[cv["level"].isin(JUDGED + ["all"])].pivot(index="model", columns="level", values="brier").round(4).to_string())
    print(pd.DataFrame(rate_check).round(3).to_string(index=False))
    print(f"[09] done in {time.time() - t0:.0f}s -> {REPORTS / 'validation.html'}")


# ------------------------------------------------------------------------------------------------ report
def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return f.format(v) if f else str(v)


def _table(df: pd.DataFrame, cols: list[tuple[str, str, str | None]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h, _c, _f in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(r[c], f))}</td>" for _h, c, f in cols) + "</tr>"
                   for _, r in df.iterrows())
    return f"<div class=tbl><table><tr>{head}</tr>{body}</table></div>"


def write_report(verdict, allsc, effects, rel, diag, rate_check, priors, n, t_split) -> None:
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#17140f;background:#f5f0e8}"
           "h1,h2{font-family:Georgia,serif}table{border-collapse:collapse;font-size:13px;background:#fff;margin:8px 0}"
           "th,td{border:1px solid #ddd6c8;padding:4px 8px;text-align:left}th{background:#faf7f2}.tbl{overflow-x:auto}")
    sc = [("Split", "split", None), ("Level", "level", None), ("Model", "model", None), ("n", "n", None),
          ("Observed", "observed", "{:.3f}"), ("Mean predicted", "mean_predicted", "{:.3f}"), ("Brier", "brier", "{:.4f}"),
          ("Brier skill", "brier_skill", "{:.3f}"), ("Log loss", "log_loss", "{:.4f}"), ("AUROC", "auroc", "{:.3f}"),
          ("Cal. intercept", "cal_intercept", "{:.2f}"), ("Cal. slope", "cal_slope", "{:.2f}")]
    parts = [
        f"<!doctype html><meta charset=utf-8><title>Model validation</title><style>{css}</style>",
        f"<h1>Production landmark model: validation ({VERSION})</h1>",
        f"<p>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. {n:,} training rows (novel drugs, final outcomes). Scores use "
        f"the model's probability of the observed label, so they compare like with like against the benchmarks. "
        f"P5 target at each level: calibration slope {SLOPE_OK[0]}–{SLOPE_OK[1]} and a lower Brier score than every "
        f"benchmark (5-fold cross-validation grouped by drug). L4 has too few rows to judge.</p>",
        "<h2>Verdict</h2>",
        _table(verdict, [("Level", "level", None), ("n", "n", None), ("Cal. slope", "cal_slope", "{:.2f}"),
                         ("Cal. intercept", "cal_intercept", "{:.2f}"), ("Brier", "brier", "{:.4f}"),
                         ("Best benchmark", "best_benchmark", None), ("Its Brier", "best_benchmark_brier", "{:.4f}"),
                         ("AUROC", "auroc", "{:.3f}"), ("Slope OK", "slope_ok", None), ("Beats all", "beats_all", None)]),
        "<h2>Scores</h2>", _table(allsc, sc),
        f"<p>Temporal holdout: fitted on rows with cut-offs before {html.escape(t_split)}, scored on the rest.</p>",
        "<h2>Reliability (cross-validated, by decile)</h2>",
        _table(rel, [("Level", "level", None), ("Bin", "bin", None), ("n", "n", None),
                     ("Mean predicted", "mean_predicted", "{:.3f}"), ("Observed", "observed", "{:.3f}")]),
        "<h2>Effects (final model, odds ratios)</h2>",
        _table(effects, [("Feature", "feature", None), ("Stage", "stage", None), ("Per", "per", None),
                         ("OR", "or_median", "{:.2f}"), ("95% low", "or_lo", "{:.2f}"), ("95% high", "or_hi", "{:.2f}"),
                         ("P(OR > 1)", "p_positive", "{:.2f}")]),
        "<h2>Approval rate check</h2><p>Mean predicted probability of the true outcome against the observed label rate "
        "and the gold-set corrected rate (script 08).</p>",
        _table(pd.DataFrame(rate_check), [("Level", "level", None), ("Mean true p", "mean_true_p", "{:.3f}"),
                                          ("Observed label rate", "observed_label_rate", "{:.3f}"),
                                          ("Gold corrected rate", "gold_corrected_rate", "{:.3f}")]),
        "<h2>Sampler diagnostics</h2>", _table(pd.DataFrame(diag), [(c, c, None) for c in pd.DataFrame(diag).columns]),
        f"<h2>Label-error priors</h2><pre>{html.escape(json.dumps(priors, indent=1))}</pre>",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation.html").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
