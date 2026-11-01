"""06 — EDA and univariate screening per level (PLAN.md §7.2). Exploratory only: nothing here selects the model.

Target: Y_LEVEL (Y_PROG at L1-L2; approval within 3 years of the cut-off from L3 on; see labels.level_target).
Population: cohort novel-drug landmark rows (`in_primary`). The primary screen uses rows whose outcome is final (the
7-year window has closed at L1-L2; the 3-year horizon has passed or the drug is approved from L3 on), so no
censoring correction is needed; all labelled rows are a sensitivity column. L2 is also screened on rows whose Phase 3 readout is dated from a press release
(`cutoff_estimated` False), the situation a live case is in.

Tests: binary → Fisher exact OR; categorical → χ² across levels (+ one-vs-rest ORs for reading); continuous →
Mann–Whitney U + univariate logistic OR per SD (log1p for counts). Benjamini–Hochberg FDR (q = 0.05) per level and
population. Missingness of press-release-derived fields depends on the outcome (wins are announced more readily),
so it is reported as a diagnostic and never screened as a feature.

  uv run python scripts/06_eda_and_univariate.py
Outputs: data/processed/univariate.csv, data/processed/missingness.csv, reports/eda.html
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
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.features.stats_transforms import wilson_ci

PROCESSED = config.DATA_DIR / "processed"
REPORTS = config.PROJECT_ROOT / "reports"
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
MIN_N, MIN_MINORITY = 40, 8         # a feature is screened only with ≥ MIN_N known rows and ≥ MIN_MINORITY per side

# name → (kind, source, level it first applies at, hypothesis). Source matters: `edgar` fields exist only when a
# press release was found, so their coverage is outcome-dependent. Designations absent from the record count as
# "not reported" (False).
FEATURES = {
    "therapeutic_area": ("cat", "ctgov", "L1", "H15"),
    "sponsor_size": ("cat", "heuristic", "L1", "H15"),
    "modality": ("cat", "gsrs", "L1", "-"),
    "moiety_approved": ("bin", "drugsfda", "L1", "H5 repurposed"),
    "is_combination": ("bin", "ctgov", "L1", "-"),
    "sponsor_prior_approvals": ("count", "drugsfda", "L1", "H15"),
    "ta_prior_rate": ("cont", "derived", "L1", "H15"),
    "anchor_year": ("year", "ctgov", "L1", "H16"),
    "n_sponsors": ("count", "ctgov", "L1", "-"),
    "n_p3_trials": ("count", "ctgov", "L1", "-"),
    "p3_n_pivotal_trials": ("count", "ctgov", "L1", "H7"),
    "p3_planned_n": ("count", "ctgov", "L1", "-"),
    "p3_comparator": ("cat", "ctgov", "L1", "-"),
    "p3_double_blind": ("bin", "ctgov", "L1", "-"),
    "p3_dmc": ("bin", "ctgov", "L1", "-"),
    "p3_endpoint_class": ("cat", "ctgov", "L1", "-"),
    "sponsor_prior_crls": ("count", "edgar", "L1", "H15"),
    "p2_randomized": ("bin", "ctgov", "L1", "H3"),
    "p2_placebo_controlled": ("bin", "ctgov", "L1", "H3"),
    "p2_single_arm": ("bin", "ctgov", "L1", "H3"),
    "p2_n_total": ("count", "ctgov", "L1", "-"),
    "p2_results_posted": ("bin", "ctgov", "L1", "H17"),
    "p2_primary_endpoint_met": ("cat", "ctgov+edgar", "L1", "H1"),
    "p2_z": ("cont", "ctgov+edgar", "L1", "H1"),
    "assurance": ("cont", "derived", "L1", "H4"),
    "orphan": ("flag", "edgar", "L1", "-"),
    "fast_track": ("flag", "edgar", "L1", "-"),
    "btd": ("flag", "edgar", "L1", "H9"),
    "spa": ("flag", "edgar", "L1", "-"),
    "p3_results_posted": ("bin", "ctgov", "L2", "H17"),
    "p3_primary_endpoint_met": ("cat", "ctgov+edgar", "L2", "-"),
    "p3_z": ("cont", "ctgov+edgar", "L2", "H6"),
    "p3_n_positive_pivotal": ("count", "ctgov+edgar", "L2", "H7"),
    "p3_sae_rr": ("cont", "ctgov", "L2", "H8"),
    "p3_death_rd": ("cont", "ctgov", "L2", "H8"),
    "priority_review": ("flag", "edgar", "L3", "H9"),
    "accelerated_pathway": ("flag", "edgar", "L3", "-"),
    "months_readout_to_filing": ("count", "edgar", "L3", "H10"),
    "adcom_scheduled": ("bin", "edgar", "L3", "H11"),
    "adcom_yes_fraction": ("cont", "edgar", "L4", "H11"),
    "pdufa_extended": ("flag", "edgar", "L5", "H12"),
}


def _bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return float(bool(v))


def _prepare(d: pd.DataFrame) -> pd.DataFrame:
    out = d[["programme_id", "level", "cutoff_estimated"]].copy()
    out["y"] = d["y_level"]
    out["final"] = d["window_closed"] | d["level"].isin(["L3", "L4", "L5", "L6"])
    for name, (kind, *_rest) in FEATURES.items():
        col = d[f"f_{name}"]
        if kind == "flag":
            out[name] = col.map(lambda v: 1.0 if v is True else 0.0)
        elif kind == "bin":
            out[name] = col.map(_bool)
        elif kind == "cat":
            out[name] = col.where(col.notna(), None)
        else:
            out[name] = pd.to_numeric(col, errors="coerce")
    return out


def _or_ci(a, b, c, d):
    """Odds ratio with a Haldane-corrected Wald 95% CI. a,b = exposed (y=1, y=0); c,d = unexposed."""
    a, b, c, d = (x + 0.5 for x in (a, b, c, d))
    lor = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return np.exp(lor), np.exp(lor - 1.96 * se), np.exp(lor + 1.96 * se)


def _test(df: pd.DataFrame, name: str, kind: str) -> dict | None:
    x, y = df[name], df["y"]
    known = x.notna()
    x, y = x[known], y[known].astype(int)
    if len(x) < MIN_N:
        return None
    base = {"n": len(x), "rate_overall": y.mean()}
    if kind in ("bin", "flag"):
        x = x.astype(int)
        a, b = int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
        c, d = int(((x == 0) & (y == 1)).sum()), int(((x == 0) & (y == 0)).sum())
        if min(a + b, c + d) < MIN_MINORITY:
            return None
        o, lo, hi = _or_ci(a, b, c, d)
        p = stats.fisher_exact([[a, b], [c, d]])[1]
        return {**base, "test": "Fisher", "effect": o, "lo": lo, "hi": hi, "p": p,
                "detail": f"yes {a}/{a + b} ({a / (a + b):.0%}) vs no {c}/{c + d} ({c / (c + d):.0%})"}
    if kind == "cat":
        tab = pd.crosstab(x, y).reindex(columns=[0, 1], fill_value=0)
        tab = tab[tab.sum(axis=1) >= MIN_MINORITY]
        if len(tab) < 2 or (tab.sum(axis=0) == 0).any():
            return None
        p = stats.chi2_contingency(tab)[1]
        rate = tab[1] / tab.sum(axis=1)
        parts = [f"{lvl} {int(tab.loc[lvl, 1])}/{int(tab.loc[lvl].sum())} ({rate[lvl]:.0%})"
                 for lvl in rate.sort_values(ascending=False).index]
        return {**base, "n": int(tab.values.sum()), "test": "χ²", "effect": np.nan, "lo": np.nan, "hi": np.nan,
                "p": p, "detail": "; ".join(parts)}
    v = np.log1p(x.clip(lower=0)) if kind == "count" else x
    if v.std() == 0 or min(y.sum(), (1 - y).sum()) < MIN_MINORITY:
        return None
    z = (v - v.mean()) / v.std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.Logit(y.values, sm.add_constant(z.values)).fit(disp=0)
            b, se = fit.params[1], fit.bse[1]
        except Exception:
            return None
    p = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue
    unit = "per SD of log(1+x)" if kind == "count" else "per SD"
    return {**base, "test": "Mann–Whitney", "effect": np.exp(b), "lo": np.exp(b - 1.96 * se),
            "hi": np.exp(b + 1.96 * se), "p": p,
            "detail": f"median approved {x[y == 1].median():.3g} vs not {x[y == 0].median():.3g}; OR {unit}"}


def screen(rows: pd.DataFrame, population: str) -> pd.DataFrame:
    out = []
    for level in LEVELS:
        df = rows[rows["level"] == level]
        res = []
        for name, (kind, source, first, hyp) in FEATURES.items():
            if LEVELS.index(level) < LEVELS.index(first):
                continue
            r = _test(df, name, kind)
            if r:
                res.append({"population": population, "level": level, "feature": name, "kind": kind,
                            "source": source, "hypothesis": hyp, "coverage": df[name].notna().mean(), **r})
        if res:
            ps = [r["p"] for r in res]
            q = multipletests(ps, alpha=0.05, method="fdr_bh")[1]
            for r, qv in zip(res, q):
                r["q"] = qv
            out += res
    return pd.DataFrame(out)


def missingness(rows: pd.DataFrame) -> pd.DataFrame:
    """Approval rate with vs without each field present: shows which gaps are outcome-dependent."""
    out = []
    for level in LEVELS:
        df = rows[rows["level"] == level]
        for name, (kind, source, first, _h) in FEATURES.items():
            if kind == "flag" or LEVELS.index(level) < LEVELS.index(first):
                continue
            has = df[name].notna()
            if has.all() or not has.any() or len(df) < MIN_N:
                continue
            y = df["y"].astype(int)
            a, n1, c, n0 = int(y[has].sum()), int(has.sum()), int(y[~has].sum()), int((~has).sum())
            o, lo, hi = _or_ci(a, n1 - a, c, n0 - c)
            out.append({"level": level, "feature": name, "source": source, "coverage": has.mean(),
                        "rate_present": a / n1, "rate_missing": c / n0 if n0 else np.nan,
                        "or_present": o, "lo": lo, "hi": hi,
                        "p": stats.fisher_exact([[a, n1 - a], [c, n0 - c]])[1]})
    return pd.DataFrame(out)


def describe(d: pd.DataFrame) -> dict:
    out = {}
    for level in LEVELS:
        df = d[d["level"] == level]
        k = df["y"].notna()
        x = int(df.loc[k, "y"].sum())
        lo, hi = wilson_ci(x, int(k.sum())) if k.any() else (np.nan, np.nan)
        out[level] = {"rows": len(df), "known": int(k.sum()), "approved": x,
                      "rate": x / k.sum() if k.any() else np.nan, "ci": [lo, hi]}
    return out


# ------------------------------------------------------------------------------------------------ report
def _fmt(v, f="{:.2f}"):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h, _c, _f in cols)
    body = []
    for _, r in df.iterrows():
        cls = ' class="sig"' if r.get("q", 1) < 0.05 else ""
        cells = "".join(f"<td>{html.escape(_fmt(r[c], f) if f else str(r[c]))}</td>" for _h, c, f in cols)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="tbl"><table><tr>{head}</tr>{"".join(body)}</table></div>'


def report(desc: dict, uni: pd.DataFrame, miss: pd.DataFrame, path: Path) -> None:
    cols = [("Feature", "feature", None), ("Hyp.", "hypothesis", None), ("Source", "source", None),
            ("n", "n", "{:.0f}"), ("Coverage", "coverage", "{:.0%}"), ("Test", "test", None),
            ("OR", "effect", "{:.2f}"), ("95% CI lo", "lo", "{:.2f}"), ("hi", "hi", "{:.2f}"),
            ("p", "p", "{:.2g}"), ("q (BH)", "q", "{:.2g}"), ("Detail", "detail", None)]
    parts = [f"<h1>EDA and univariate screening</h1><p class=meta>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
             "Exploratory only (PLAN.md §7.2). Highlighted rows pass BH FDR q &lt; 0.05 within their level.</p>",
             "<h2>Rows and approval rates (novel drugs, final outcomes)</h2>"]
    dd = pd.DataFrame(desc).T.reset_index(names="level")
    dd["ci"] = dd["ci"].map(lambda c: f"{c[0]:.0%}–{c[1]:.0%}" if c[0] == c[0] else "")
    parts.append(_table(dd, [("Level", "level", None), ("Rows", "rows", "{:.0f}"), ("Known Y_PROG", "known", "{:.0f}"),
                             ("Approved", "approved", "{:.0f}"), ("Rate", "rate", "{:.1%}"), ("95% CI", "ci", None)]))
    for pop in uni["population"].unique():
        parts.append(f"<h2>Screen: {html.escape(pop)}</h2>")
        for level in LEVELS:
            s = uni[(uni["population"] == pop) & (uni["level"] == level)].sort_values("p")
            if len(s):
                parts.append(f"<h3>{level} ({int(s['n'].max())} rows max)</h3>" + _table(s, cols))
    parts.append("<h2>Missingness diagnostic (final outcomes)</h2><p>Approval rate when a field is present vs missing. "
                 "Large gaps on press-release fields mean coverage depends on the outcome: these gaps are never "
                 "features.</p>")
    parts.append(_table(miss.sort_values(["level", "p"]),
                        [("Level", "level", None), ("Feature", "feature", None), ("Source", "source", None),
                         ("Coverage", "coverage", "{:.0%}"), ("Rate present", "rate_present", "{:.0%}"),
                         ("Rate missing", "rate_missing", "{:.0%}"), ("OR present", "or_present", "{:.2f}"),
                         ("p", "p", "{:.2g}")]))
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1300px;color:#17140f;background:#f5f0e8}"
           "h1{font-family:Georgia,serif}.meta{color:#6f6656}.tbl{overflow-x:auto}table{border-collapse:collapse;"
           "font-size:13px;margin:8px 0 16px;background:#fff}th,td{border:1px solid #ddd6c8;padding:4px 8px;"
           "text-align:left;vertical-align:top}th{background:#faf7f2}tr.sig td{background:#f7e3da}")
    path.write_text(f"<!doctype html><meta charset=utf-8><title>EDA</title><style>{css}</style>{''.join(parts)}",
                    encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = pd.read_parquet(PROCESSED / "landmarks.parquet")
    d = d[d["in_cohort"] & d["in_primary"]]
    rows = _prepare(d)
    every = _prepare(d)
    rows = every[every["y"].notna()]
    closed = rows[rows["final"]]

    uni = pd.concat([screen(closed, "final outcomes"),
                     screen(closed[~closed["cutoff_estimated"]],
                            "final outcomes, readout dated").query("level == 'L2'"),
                     screen(rows, "all labelled (sensitivity)")], ignore_index=True)
    miss = missingness(closed)
    desc = describe(every[every["final"]])

    uni.to_csv(PROCESSED / "univariate.csv", index=False)
    miss.to_csv(PROCESSED / "missingness.csv", index=False)
    REPORTS.mkdir(exist_ok=True)
    report(desc, uni, miss, REPORTS / "eda.html")

    sig = uni[(uni["population"] == "final outcomes") & (uni["q"] < 0.05)]
    print(json.dumps({lv: v for lv, v in desc.items()}, indent=1, default=float))
    print(f"[06] {len(uni)} tests; {len(sig)} pass FDR in the primary screen")
    print(sig[["level", "feature", "n", "effect", "lo", "hi", "q", "detail"]].to_string(index=False, max_colwidth=70))
    print(f"[06] -> {REPORTS / 'eda.html'}")


if __name__ == "__main__":
    main()
