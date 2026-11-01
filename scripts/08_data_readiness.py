"""08 — Is the training data ready for the fitted model (PLAN.md §7.5–7.7)? Evaluation only (D6).

Per level, on the modelling population (novel drugs, final outcomes; L2 = rows with a press-release-dated readout):
  - rows, approvals and the number of parameters they support (10–20 events per parameter);
  - the smallest odds ratio detectable with 80% power at α = 0.05 for a binary feature (prevalence 30%) and per SD of
    a continuous one;
  - label error from the gold set (data/gold/gold_report.json): PPV / NPV of the pipeline label, re-weighted to
    sensitivity, specificity and the corrected approval rate, with Beta priors the Bayesian model uses to treat the
    label as a noisy measurement of the truth (L1–L2 from the main sample's novel-drug Y_PROG; L3/L5 from `y_level`);
  - features with enough coverage to enter the model (≥ 70% of rows, or a designation flag);
  - a verdict.

  uv run python scripts/08_data_readiness.py
Outputs: data/processed/label_error.json, reports/data_readiness.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.model.landmark import training_population

PROCESSED = config.DATA_DIR / "processed"
REPORTS = config.PROJECT_ROOT / "reports"
GOLD_REPORT = config.DATA_DIR / "gold" / "gold_report.json"
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
Z = 1.96 + 0.8416                      # two-sided α = 0.05, power 0.8
MIN_COVERAGE = 0.7
FLAGS = {"orphan", "fast_track", "btd", "spa", "priority_review", "accelerated_pathway", "pdufa_extended"}
SKIP = {"anchor_year", "ta_prior_rate", "therapeutic_area", "sponsor_size", "sponsor_prior_approvals", "n_sponsors"}


population = training_population


def mde(n: int, p: float, q: float | None) -> float | None:
    """Minimum detectable odds ratio (binary feature with prevalence q, or per SD when q is None)."""
    if n == 0 or p in (0, 1):
        return None
    var = 1 / (n * p * (1 - p) * (q * (1 - q) if q else 1))
    return float(np.exp(Z * np.sqrt(var)))


def beta_from(x: float, n: float) -> list[float]:
    """Beta(x + 1, n − x + 1): the posterior for a rate after x successes in n (uniform prior)."""
    return [round(x + 1, 2), round(n - x + 1, 2)]


def label_error(gold: dict, lm_pop: pd.DataFrame) -> dict:
    out = {}
    # L1–L2: programme-level Y_PROG on the main sample's novel drugs (Y_LEVEL = Y_PROG there).
    yp = gold["y_prog"]
    nov = yp["by_cohort"].get("novel", {})
    pa, pf = yp["novel_pipeline_approved"], yp["novel_pipeline_failed"]
    n_app, n_fail = pa["n"], pf["n"]
    ppv, fpv = pa["truth_approved"] / n_app, pf["truth_failed"] / n_fail
    for lv in ("L1", "L2"):
        pi = float(lm_pop.loc[lm_pop["level"] == lv, "y_level"].mean())
        true_rate = pi * ppv + (1 - pi) * (1 - fpv)
        out[lv] = {"source": f"main gold sample, programme-level Y_PROG (novel accuracy {nov.get('accuracy')}, "
                             f"n={nov.get('n')})",
                   "ppv": round(ppv, 3), "npv": round(fpv, 3), "n_pipeline_approved": n_app, "n_pipeline_failed": n_fail,
                   "pipeline_rate": round(pi, 3), "corrected_rate": round(true_rate, 3),
                   "sensitivity": round(pi * ppv / true_rate, 3), "specificity": round((1 - pi) * fpv / (1 - true_rate), 3),
                   "ppv_beta": beta_from(pa["truth_approved"], n_app), "npv_beta": beta_from(pf["truth_failed"], n_fail)}
    for lv in ("L3", "L5"):
        g = gold.get("y_level", {}).get(lv)
        if not g or "sensitivity" not in g:
            continue
        p1, n0 = g["ppv_pipeline_approved"], g["npv_pipeline_not_approved"]
        out[lv] = {"source": f"gold Y_LEVEL at the pipeline's cut-off (n={g['n_both_known']}, both samples)",
                   "ppv": p1["rate"], "npv": n0["rate"], "n_pipeline_approved": p1["n"],
                   "n_pipeline_failed": n0["n"], "pipeline_rate": g["training_share_pipeline_approved"],
                   "corrected_rate": g["corrected_approval_rate"], "sensitivity": g["sensitivity"],
                   "specificity": g["specificity"], "ppv_beta": beta_from(p1["x"], p1["n"]),
                   "npv_beta": beta_from(n0["x"], n0["n"]),
                   "post_decision_landmarks": g["post_decision_landmarks"]["rate"],
                   "filing_event_right": (g.get("filing_event_right") or {}).get("rate")}
    for v in out.values():           # corrected-rate interval from the Beta priors (Monte Carlo)
        rng = np.random.default_rng(7)
        a = rng.beta(*v["ppv_beta"], 20000)
        b = rng.beta(*v["npv_beta"], 20000)
        r = v["pipeline_rate"] * a + (1 - v["pipeline_rate"]) * (1 - b)
        v["corrected_rate_ci95"] = [round(float(np.quantile(r, 0.025)), 3), round(float(np.quantile(r, 0.975)), 3)]
    out["L4"] = {"source": "pooled with L3/L5 (too few rows to estimate)"}
    return out


def coverage(df: pd.DataFrame) -> list[str]:
    feats = []
    for c in df.columns:
        if not c.startswith("f_"):
            continue
        name = c[2:]
        if name in FLAGS:
            if (df[c] == True).sum() >= 10:                                        # noqa: E712
                feats.append(name)
        elif name not in SKIP and df[c].notna().mean() >= MIN_COVERAGE and df[c].nunique(dropna=True) > 1:
            feats.append(name)
    return feats


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lm = pd.read_parquet(PROCESSED / "landmarks.parquet")
    gold = json.loads(GOLD_REPORT.read_text(encoding="utf-8"))
    pop = population(lm)
    err = label_error(gold, pop)
    rows = []
    for lv in LEVELS:
        d = pop[pop["level"] == lv]
        n, k = len(d), int(d["y_level"].sum())
        p = k / n if n else 0.0
        minority = min(k, n - k)
        e = err.get(lv, {})
        sens, spec = e.get("sensitivity"), e.get("specificity")
        youden = round(sens + spec - 1, 3) if sens is not None and spec is not None else None
        params = minority // 10
        if n < 60:
            verdict = "not modelled alone: pool with neighbouring levels"
        elif youden is not None and youden < 0.5:
            verdict = "labels too noisy"
        elif params < 5:
            verdict = "ready for a small model (≤ %d parameters) with label-error correction" % params
        else:
            verdict = "ready (with label-error correction)" if youden is not None else "ready; label error not measured"
        rows.append({"level": lv, "rows": n, "approved": k, "pipeline_rate": round(p, 3),
                     "corrected_rate": e.get("corrected_rate"), "corrected_rate_ci95": e.get("corrected_rate_ci95"),
                     "sensitivity": sens, "specificity": spec, "youden_J": youden,
                     "params_epv10": minority // 10, "params_epv20": minority // 20,
                     "mde_or_binary_30pct": round(mde(n, p, 0.3), 2) if mde(n, p, 0.3) else None,
                     "mde_or_per_sd": round(mde(n, p, None), 2) if mde(n, p, None) else None,
                     "usable_features": ", ".join(coverage(d)), "verdict": verdict})
    table = pd.DataFrame(rows)
    (PROCESSED / "label_error.json").write_text(json.dumps(err, indent=1), encoding="utf-8")
    REPORTS.mkdir(exist_ok=True)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in table.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                   for r in table.itertuples(index=False))
    css = ("body{font-family:system-ui,sans-serif;margin:24px;color:#17140f;background:#f5f0e8}h1{font-family:Georgia,"
           "serif}table{border-collapse:collapse;font-size:13px;background:#fff}th,td{border:1px solid #ddd6c8;"
           "padding:4px 8px;text-align:left;vertical-align:top}th{background:#faf7f2}.tbl{overflow-x:auto}")
    (REPORTS / "data_readiness.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Data readiness</title><style>{css}</style><h1>Training data "
        f"readiness</h1><p>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. Novel drugs, final outcomes; L2 rows have a "
        f"dated readout. Label error from the gold set ({gold['n_programmes']} main-sample programmes plus the "
        f"late-level supplement).</p><div class=tbl><table><tr>{head}</tr>{body}</table></div>"
        f"<h2>Label-error priors (data/processed/label_error.json)</h2><pre>{html.escape(json.dumps(err, indent=1))}"
        f"</pre>", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table.drop(columns=["usable_features"]).to_string(index=False))
    for r in rows:
        print(f"  {r['level']} usable features: {r['usable_features']}")
    print(f"[08] -> {REPORTS / 'data_readiness.html'}, {PROCESSED / 'label_error.json'}")


if __name__ == "__main__":
    main()
