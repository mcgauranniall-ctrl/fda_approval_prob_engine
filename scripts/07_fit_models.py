"""07 — Frequentist inference (PLAN.md §7.1, 7.3, 7.4; phase P4). The Bayesian production model (P5) builds on this.

Target and population as in 06: Y_LEVEL (Y_PROG at L1-L2, approval within 3 years of the cut-off from L3 on), novel
drugs, final outcomes only. L2 uses rows whose Phase 3 readout is
dated from a press release, the situation a live case is in. Outputs:

  hypotheses.csv   H1–H19 per level: adjusted OR (core covariates), 95% CI, p, verdict. GLM with standard errors
                   clustered by drug (one drug in several indications is not independent); Firth penalised
                   likelihood when a cell is sparse (< 5), with model-based SEs.
  stability.csv    L1-penalised logistic regression on 500 half-subsamples per level → selection probability
                   (keep ≥ 0.7). Candidates: features with ≥ 70% coverage at the level.
  cuminc.csv       Aalen–Johansen cumulative incidence of approval (failure competing) from each level's cut-off,
                   all rows including still-open programmes: P(approved within t).
  benchmark.csv    The provisional prior-only model (`model/provisional.py`, what the app uses today) scored on the
                   same rows: Brier, log loss, AUROC, calibration intercept/slope, mean predicted vs observed. Scored
                   against Y_LEVEL and, for reference, D1's Y_PROG.
  reports/inference.html

  uv run python scripts/07_fit_models.py
"""

from __future__ import annotations

import html
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.model import provisional

PROCESSED = config.DATA_DIR / "processed"
REPORTS = config.PROJECT_ROOT / "reports"
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
SEED = 20261101
MIN_N, MIN_CELL = 40, 10           # below these a hypothesis is "insufficient data"
N_SUBSAMPLES, KEEP = 500, 0.7

# (id, level, feature, encoding, expected sign, statement). Encodings: bin | yes_no | per_sd | log_sd | ge2.
HYPOTHESES = [
    ("H1", "L1", "p2_primary_endpoint_met", "yes_no", +1, "Phase 2 meeting its endpoint predicts approval"),
    ("H1", "L1", "p2_z", "per_sd", +1, "Phase 2 z-statistic adds information"),
    ("H3", "L1", "p2_randomized", "bin", +1, "Randomised Phase 2 beats non-randomised"),
    ("H3", "L1", "p2_single_arm", "bin", -1, "Single-arm Phase 2 lowers success"),
    ("H3", "L1", "p2_placebo_controlled", "bin", +1, "Placebo-controlled Phase 2 raises success"),
    ("H4", "L1", "assurance", "per_sd", +1, "Bayesian assurance is predictive"),
    ("H5", "L1", "moiety_approved", "bin", +1, "Repurposed (approved elsewhere) raises success"),
    ("H5", "L2", "moiety_approved", "bin", +1, "Repurposed (approved elsewhere) raises success"),
    ("—", "L2", "p3_primary_endpoint_met", "yes_no", +1, "Phase 3 meeting its endpoint predicts approval"),
    ("H6", "L2", "p3_z", "per_sd", +1, "Phase 3 z-statistic adds information"),
    ("H7", "L2", "p3_n_pivotal_trials", "ge2", +1, "Two or more pivotal trials beat one"),
    ("H7", "L2", "p3_n_positive_pivotal", "ge2", +1, "Two positive pivotal trials beat one"),
    ("H8", "L2", "p3_sae_rr", "log_sd", -1, "SAE imbalance lowers success"),
    ("H8", "L2", "p3_death_rd", "per_sd", -1, "Death imbalance lowers success"),
    ("H9", "L3", "priority_review", "bin", +1, "Priority review predicts approval"),
    ("H9", "L3", "btd", "bin", +1, "Breakthrough designation predicts approval"),
    ("H9", "L3", "fast_track", "bin", +1, "Fast Track predicts approval"),
    ("H10", "L3", "months_readout_to_filing", "log_sd", -1, "A long readout-to-filing delay signals problems"),
    ("H11", "L4", "adcom_yes_fraction", "per_sd", +1, "AdCom vote share predicts approval"),
    ("H11", "L3", "adcom_scheduled", "bin", -1, "Scheduling an AdCom is itself negative"),
    ("H12", "L5", "pdufa_extended", "bin", 0, "A PDUFA extension changes the odds (direction tested)"),
    ("H15", "L1", "sponsor_size", "large_vs_small", +1, "Large sponsors beat small/private ones"),
    ("H15", "L2", "sponsor_size", "large_vs_small", +1, "Large sponsors beat small/private ones"),
    ("H15", "L1", "sponsor_prior_approvals", "log_sd", +1, "Sponsor experience raises success"),
    ("H15", "L2", "sponsor_prior_approvals", "log_sd", +1, "Sponsor experience raises success"),
    ("H16", "L1", "anchor_year", "per_sd", 0, "Era effect (direction tested)"),
    ("H16", "L2", "anchor_year", "per_sd", 0, "Era effect (direction tested)"),
    ("H17", "L1", "p2_results_posted", "bin", +1, "Phase 2 results not posted signals risk"),
    ("H17", "L2", "p3_results_posted", "bin", +1, "Phase 3 results not posted signals risk"),
    ("H17", "L1", "has_phase2", "bin", +1, "No registered Phase 2 signals risk"),
]
NOT_AVAILABLE = [("H2", "L1", "Winner's curse shrinkage needs paired Phase 2/3 effect sizes (not extracted yet)"),
                 ("H5", "All", "Genetic support and first-in-class need Open Targets / mechanism data (not loaded)"),
                 ("H13", "L5", "483/OAI inspection data not loaded"),
                 ("H14", "L6", "CRL reason categories need L6 rows (not built)"),
                 ("H18", "L2+", "Label-outcome models come with P5")]
CORE = ["c_large", "c_public", "c_prior", "c_ta", "c_year", "c_pivotal", "c_partners"]
SAME_AS = {"sponsor_size": {"c_large", "c_public"}, "sponsor_prior_approvals": {"c_prior"}, "anchor_year": {"c_year"},
           "p3_n_pivotal_trials": {"c_pivotal"}}   # a tested feature replaces the core covariate it duplicates


# ------------------------------------------------------------------------------------------------ data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(PROCESSED / "landmarks.parquet")
    d = d[d["in_cohort"] & d["in_primary"]].copy()
    d["f_has_phase2"] = d["f_p2_randomized"].notna()
    z = lambda s: (s - s.mean()) / s.std()
    d["c_large"] = (d["f_sponsor_size"] == "large").astype(float)
    d["c_public"] = (d["f_sponsor_size"] == "public").astype(float)
    d["c_prior"] = z(np.log1p(d["f_sponsor_prior_approvals"]))
    d["c_ta"] = z(d["f_ta_prior_rate"].fillna(d["f_ta_prior_rate"].median()))
    d["c_year"] = (d["f_anchor_year"] - 2010) / 5
    d["c_pivotal"] = np.log1p(pd.to_numeric(d["f_p3_n_pivotal_trials"], errors="coerce").fillna(1))
    d["c_partners"] = np.log1p(d["f_n_sponsors"])
    d["y"] = d["y_level"]
    final = d["window_closed"] | d["level"].isin(["L3", "L4", "L5", "L6"])
    closed = d[final & d["y"].notna()]
    closed = closed[(closed["level"] != "L2") | ~closed["cutoff_estimated"]]
    return d, closed


def encode(df: pd.DataFrame, feature: str, enc: str) -> pd.Series:
    x = df[f"f_{feature}"]
    if enc == "bin":
        flags = {"priority_review", "btd", "fast_track", "pdufa_extended", "orphan", "spa", "accelerated_pathway"}
        if feature in flags:
            return x.map(lambda v: 1.0 if v is True else 0.0)
        return x.map(lambda v: np.nan if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
                     else float(bool(v)))
    if enc == "yes_no":
        return x.map({"yes": 1.0, "no": 0.0})
    if enc == "large_vs_small":
        return x.map({"large": 1.0, "private/unknown": 0.0})
    v = pd.to_numeric(x, errors="coerce")
    if enc == "ge2":
        return (v >= 2).astype(float).where(v.notna())
    if enc == "log_sd":
        v = np.log(v.clip(lower=1e-3)) if feature == "p3_sae_rr" else np.log1p(v.clip(lower=0))
    return (v - v.mean()) / v.std()


# ------------------------------------------------------------------------------------------------ models
def firth(y: np.ndarray, X: np.ndarray, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Firth penalised logistic regression (Heinze & Schemper 2002). Returns (beta, se)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        info = X.T @ (X * W[:, None])
        inv = np.linalg.pinv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], inv, X * np.sqrt(W)[:, None])
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        step = np.clip(step, -5, 5)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b, np.sqrt(np.diag(inv))


def adjusted(df: pd.DataFrame, x: pd.Series, feature: str) -> dict:
    keep = x.notna()
    sub, xv = df[keep], x[keep]
    y = sub["y"].astype(int).values
    core = [c for c in CORE if c not in SAME_AS.get(feature, set()) and sub[c].std() > 0]
    X = np.column_stack([np.ones(len(sub)), xv.values, sub[core].values])
    cells = pd.crosstab(xv, y).values if xv.nunique() <= 2 else np.array([[np.inf]])
    sparse = bool(cells.size and cells.min() < 5)
    if not sparse:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["drug_id"])[0]})
            beta, s, method = fit.params[1], fit.bse[1], "GLM, clustered by drug"
        except np.linalg.LinAlgError:
            sparse = True
    if sparse:
        b, se = firth(y, X)
        beta, s, method = b[1], se[1], "Firth"
    p = 2 * stats.norm.sf(abs(beta / s))
    return {"n": int(keep.sum()), "events": int(y.sum()), "or": np.exp(beta), "lo": np.exp(beta - 1.96 * s),
            "hi": np.exp(beta + 1.96 * s), "p": p, "method": method}


def hypotheses(closed: pd.DataFrame) -> pd.DataFrame:
    out = []
    for hid, level, feature, enc, sign, text in HYPOTHESES:
        df = closed[closed["level"] == level]
        x = encode(df, feature, enc)
        row = {"id": hid, "level": level, "feature": feature, "statement": text, "encoding": enc}
        k = x.notna()
        y = df.loc[k, "y"]
        minority = min(y.sum(), (1 - y).sum())
        xside = min((x[k] == x[k].max()).sum(), (x[k] == x[k].min()).sum()) if enc in ("bin", "yes_no", "ge2",
                                                                                     "large_vs_small") else k.sum()
        if k.sum() < MIN_N or minority < MIN_CELL or xside < 5:
            out.append({**row, "n": int(k.sum()), "verdict": "insufficient data"})
            continue
        r = adjusted(df, x, feature)
        if r["p"] >= 0.05:
            verdict = "not supported"
        elif sign == 0:
            verdict = "supported (" + ("raises" if r["or"] > 1 else "lowers") + " odds)"
        else:
            verdict = "supported" if (r["or"] > 1) == (sign > 0) else "contradicted"
        out.append({**row, **r, "verdict": verdict})
    out += [{"id": h, "level": lv, "feature": "", "statement": s, "verdict": "not testable yet"}
            for h, lv, s in NOT_AVAILABLE]
    return pd.DataFrame(out)


def stability(closed: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    out = []
    candidates = {"p2_randomized": "bin", "p2_single_arm": "bin", "p2_placebo_controlled": "bin",
                  "p2_results_posted": "bin", "has_phase2": "bin", "moiety_approved": "bin", "is_combination": "bin",
                  "orphan": "bin", "fast_track": "bin", "btd": "bin", "spa": "bin", "priority_review": "bin",
                  "p3_results_posted": "bin", "p3_primary_endpoint_met": "yes_no", "p3_planned_n": "log_sd",
                  "n_p3_trials": "log_sd", "accelerated_pathway": "bin"}
    for level in ["L1", "L2", "L3", "L5"]:
        df = closed[closed["level"] == level]
        cols = {c: df[c] for c in CORE}
        for f, enc in candidates.items():
            x = encode(df, f, enc)
            if x.notna().mean() >= 0.7 and x.nunique() > 1:
                cols[f] = x
        X = pd.DataFrame(cols)
        ok = X.notna().all(axis=1)
        X, y = X[ok], df.loc[ok, "y"].astype(int)
        if len(y) < 100:
            continue
        Xs = StandardScaler().fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C = LogisticRegressionCV(Cs=20, penalty="l1", solver="liblinear", cv=5, scoring="neg_log_loss",
                                     random_state=SEED).fit(Xs, y).C_[0]
            hits = np.zeros(X.shape[1])
            for _ in range(N_SUBSAMPLES):
                idx = rng.choice(len(y), len(y) // 2, replace=False)
                m = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xs[idx], y.values[idx])
                hits += m.coef_[0] != 0
            full = LogisticRegression(penalty="l1", solver="liblinear", C=C).fit(Xs, y)
        for name, h, coef in zip(X.columns, hits / N_SUBSAMPLES, full.coef_[0]):
            out.append({"level": level, "feature": name, "n": len(y), "selection_prob": h,
                        "coef_std": coef, "kept": h >= KEEP})
    return pd.DataFrame(out)


def cumulative_incidence(d: pd.DataFrame) -> pd.DataFrame:
    from lifelines import AalenJohansenFitter

    out = []
    rows = d[d["event_level"].notna() & d["t_level_days"].notna()]
    rows = rows[(rows["level"] != "L2") | ~rows["cutoff_estimated"]]
    for level in ["L1", "L2", "L3", "L5"]:
        df = rows[rows["level"] == level]
        t = (df["t_level_days"].clip(lower=0) / 365.25).values + 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ajf = AalenJohansenFitter(calculate_variance=False).fit(t, df["event_level"].astype(int).values,
                                                                  event_of_interest=1)
        ci = ajf.cumulative_density_
        for yrs in (0.5, 1, 2, 3, 5, 7):
            v = ci[ci.index <= yrs].iloc[-1, 0] if (ci.index <= yrs).any() else 0.0
            out.append({"level": level, "years": yrs, "p_approved_by": float(v), "n": len(df),
                        "censored": int((df["event_level"] == 0).sum())})
    return pd.DataFrame(out)


def benchmark(closed: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    out = []
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        df = closed[closed["level"] == level]
        if len(df) < 15:
            continue
        p = np.array([provisional.predict(level, json.loads(f))["p_approval"]["mean"] for f in df["features_json"]])
        for target, yv in (("Y_LEVEL", df["y"]), ("Y_PROG (D1, 7 years)", df["y_prog"])):
            yv = yv.dropna()
            out.append({"level": level, "target": target, **_scores(p[df.index.get_indexer(yv.index)],
                                                                    yv.astype(int).values, roc_auc_score)})
    return pd.DataFrame(out)


def _scores(p: np.ndarray, y: np.ndarray, roc_auc_score) -> dict:
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(pc / (1 - pc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    base = y.mean()
    return {"n": len(y), "observed": base, "mean_predicted": p.mean(),
            "brier": np.mean((p - y) ** 2), "brier_base_rate": base * (1 - base),
            "log_loss": -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
            "auroc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "cal_intercept": intercept, "cal_slope": slope}


# ------------------------------------------------------------------------------------------------ report
def _cell(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f.format(v) if f else str(v)


def _table(df: pd.DataFrame, cols: list[tuple[str, str, str | None]], hl=None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h, _c, _f in cols)
    body = "".join(
        f"<tr{' class=sig' if hl is not None and hl(r) else ''}>"
        + "".join(f"<td>{html.escape(_cell(r.get(c), f))}</td>" for _h, c, f in cols) + "</tr>"
        for _, r in df.iterrows())
    return f"<div class=tbl><table><tr>{head}</tr>{body}</table></div>"


def report(hyp, stab, cum, bench, path: Path) -> None:
    parts = [f"<h1>Statistical inference (P4)</h1><p class=meta>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
             "Target Y_LEVEL; novel drugs with final outcomes; L2 rows have a dated Phase 3 readout.</p>",
             "<h2>Benchmark: the provisional model on historical rows</h2>"
             "<p>Calibration slope 1 and intercept 0 are perfect; slope &lt; 1 means over-confident, intercept &lt; 0 "
             "means predictions too high. Brier below the base-rate Brier means skill.</p>",
             _table(bench, [("Level", "level", None), ("Target", "target", None), ("n", "n", "{:.0f}"), ("Observed", "observed", "{:.1%}"),
                            ("Mean predicted", "mean_predicted", "{:.1%}"), ("Brier", "brier", "{:.3f}"),
                            ("Brier (base rate)", "brier_base_rate", "{:.3f}"), ("Log loss", "log_loss", "{:.3f}"),
                            ("AUROC", "auroc", "{:.2f}"), ("Cal. intercept", "cal_intercept", "{:.2f}"),
                            ("Cal. slope", "cal_slope", "{:.2f}")]),
             "<h2>Hypothesis table</h2><p>Adjusted for sponsor size, sponsor prior approvals, TA prior rate, era, "
             "number of pivotal trials and number of sponsors.</p>",
             _table(hyp, [("ID", "id", None), ("Level", "level", None), ("Hypothesis", "statement", None),
                          ("Feature", "feature", None), ("n", "n", "{:.0f}"), ("Approved", "events", "{:.0f}"),
                          ("OR", "or", "{:.2f}"), ("95% CI lo", "lo", "{:.2f}"), ("hi", "hi", "{:.2f}"),
                          ("p", "p", "{:.2g}"), ("Method", "method", None), ("Verdict", "verdict", None)],
                    hl=lambda r: str(r.get("verdict", "")).startswith(("supported", "contradicted"))),
             "<h2>Stability selection</h2>",
             _table(stab.sort_values(["level", "selection_prob"], ascending=[True, False]),
                    [("Level", "level", None), ("Feature", "feature", None), ("n", "n", "{:.0f}"),
                     ("Selection prob.", "selection_prob", "{:.2f}"), ("Std. coef", "coef_std", "{:+.2f}")],
                    hl=lambda r: r["kept"]),
             "<h2>P(approved within t years of the cut-off)</h2><p>Aalen–Johansen, failure as a competing event, "
             "still-open programmes censored.</p>",
             cum.pivot(index="level", columns="years", values="p_approved_by").map(lambda v: f"{v:.1%}")
             .to_html(classes="tbl")]
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1300px;color:#17140f;background:#f5f0e8}"
           "h1{font-family:Georgia,serif}.meta{color:#6f6656}.tbl{overflow-x:auto}table{border-collapse:collapse;"
           "font-size:13px;margin:8px 0 16px;background:#fff}th,td{border:1px solid #ddd6c8;padding:4px 8px;"
           "text-align:left;vertical-align:top}th{background:#faf7f2}tr.sig td{background:#f7e3da}")
    path.write_text(f"<!doctype html><meta charset=utf-8><title>Inference</title><style>{css}</style>"
                    f"{''.join(parts)}", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    d, closed = load()
    hyp = hypotheses(closed)
    bench = benchmark(closed)
    stab = stability(closed)
    cum = cumulative_incidence(d)
    for name, df in (("hypotheses", hyp), ("stability", stab), ("cuminc", cum), ("benchmark", bench)):
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
    REPORTS.mkdir(exist_ok=True)
    report(hyp, stab, cum, bench, REPORTS / "inference.html")

    print("[07] benchmark (provisional model):")
    print(bench.round(3).to_string(index=False))
    print("[07] hypotheses:")
    print(hyp[["id", "level", "feature", "n", "or", "lo", "hi", "p", "method", "verdict"]]
          .round(3).to_string(index=False))
    print("[07] stability (kept):")
    print(stab[stab["kept"]].round(2).to_string(index=False))
    print("[07] P(approved by t):")
    print(cum.pivot(index="level", columns="years", values="p_approved_by").round(3).to_string())
    print(f"[07] -> {REPORTS / 'inference.html'}")


if __name__ == "__main__":
    main()
