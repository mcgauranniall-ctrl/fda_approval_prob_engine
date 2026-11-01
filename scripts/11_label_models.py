"""11 — Label outcomes given approval (PLAN.md §3 "Label outcomes", phase P6c): labels, fit, validate, save.

1. Labels for every approved programme in the cohort, from documents the FDA released at approval (script 01e;
   post-decision, so labels only): accelerated approval (FDA's accelerated approval list; letter wording as a
   cross-check), REMS and elements to assure safe use (approval letter; approvals from 2008-03-25, when REMS began),
   boxed warning (label as approved; openFDA's current label as a cross-check) and a label narrower than the pivotal
   trials' registered population (`dataset/label_outcomes.narrower`; recorded but not modelled: too imprecise).
2. Training rows: landmark rows L1–L5 of approved novel drugs (approved within the level's horizon). Unlike
   P(approval), a row need not have a closed 7-year window: given approval the label outcome is fully observed. At L2
   only rows with a press-release-dated readout (as script 09).
3. Validation: 5-fold cross-validation with folds hashed by drug, 3 repeats; a temporal holdout (latest 20% of
   cut-offs). Benchmarks: the stage base rate and the therapeutic-area base rate. P6c target: beat the base rates
   on Brier score with calibration slope 0.8–1.2 (or say honestly where a label is not predictable).
4. Parser checks: precision of each outcome on the reviewed sample `data/gold/label_outcomes_review.csv` when
   present (columns: programme_id, outcome, parsed, verdict).

  uv run python scripts/11_label_models.py
Outputs: data/processed/label_outcomes.parquet, label_validation.json, models/label-v1/, reports/label_outcomes.html
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.dataset import label_outcomes as lo
from fdaengine.model import label_model as lbm
from fdaengine.model import landmark as lmk
from fdaengine.resolve.programme import indication_keywords

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"
GOLD = config.DATA_DIR / "gold"
REPORTS = config.PROJECT_ROOT / "reports"
FOLDS, REPEATS = 5, 3
# REMS are labelled from FDAAA (2008-03-25) but trained on approvals from 2012: in 2008–2011 a REMS was mostly a
# Medication Guide (39% of novel approvals had one, against 5–10% since 2012, nearly all now with elements to assure
# safe use), a policy regime live forecasts will not meet.
REMS_START = "2008-03-25"
REMS_TRAIN_START = "2012-01-01"
AA_START = "1992-12-11"              # accelerated approval regulations (subpart H)
AA_DAYS = 14                         # an FDA accelerated-approval date within this many days of the approval action
STAGES = {"L1": "L1", "L2": "L2", "L3": "L3–L5", "L4": "L3–L5", "L5": "L3–L5"}


# ------------------------------------------------------------------------------------------------ labels
def outcome_table() -> pd.DataFrame:
    lab = pd.read_parquet(PROCESSED / "labels.parquet")
    prog = pd.read_parquet(INTERIM / "approving_submissions.parquet")
    docs = pd.read_parquet(INTERIM / "approval_documents.parquet")
    aa = pd.read_parquet(INTERIM / "accelerated_approvals.parquet")
    el = pd.read_parquet(INTERIM / "p3_eligibility.parquet").set_index("nct_id")
    done = pd.read_parquet(INTERIM / "ctgov" / "studies.parquet", columns=["nct_id", "primary_completion_date",
                                                                         "completion_date"]).set_index("nct_id")
    done = done["primary_completion_date"].fillna(done["completion_date"]).to_dict()
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    seeds = pt[pt["relation"] == "p3_seed"].groupby("programme_id")["nct_id"].apply(list)
    ind_label = pd.read_parquet(PROCESSED / "programmes.parquet").set_index("programme_id")["indication_label"].to_dict()
    for c in ("SubmissionNo",):
        prog[c], docs[c] = prog[c].astype(str), docs[c].astype(str)
    d = (lab[lab["in_cohort"] & lab["approval_date"].notna()][["programme_id", "drug_id", "cohort", "approval_date",
                                                               "boxed_warning"]]
         .rename(columns={"boxed_warning": "boxed_warning_openfda"})
         .merge(prog, on="programme_id", how="left").merge(docs, on=lbm_key(), how="left"))
    aa_by = {k: g["aa_date"].tolist() for k, g in aa.groupby("ApplNo")}
    list_end = aa["aa_date"].max()                    # approvals after the list's latest entry: the letter decides
    rows = []
    for r in d.itertuples(index=False):
        letter = r.letter_text if isinstance(r.letter_text, str) else ""
        label = r.label_text if isinstance(r.label_text, str) else ""
        rec = {"programme_id": r.programme_id, "drug_id": r.drug_id, "cohort": r.cohort, "approval_date": r.approval_date,
               "ApplNo": r.ApplNo, "SubmissionType": r.SubmissionType, "SubmissionNo": r.SubmissionNo,
               "letter_url": r.letter_url, "label_url": r.label_url, "has_letter": len(letter) >= 500,
               "has_label": len(label) >= 500}
        on_list = isinstance(r.ApplNo, str) and any(
            abs((date.fromisoformat(a) - date.fromisoformat(r.approval_date)).days) <= AA_DAYS for a in aa_by.get(r.ApplNo, []))
        rec["accelerated_letter"] = lo.letter_accelerated(letter)
        if not isinstance(r.ApplNo, str) or r.approval_date < AA_START:
            rec["accelerated"] = np.nan
        elif r.approval_date > list_end:
            rec["accelerated"] = np.nan if rec["accelerated_letter"] is None else float(rec["accelerated_letter"])
        else:
            rec["accelerated"] = float(on_list)
        rems, etasu = lo.letter_rems(letter)
        late = r.approval_date >= REMS_START
        rec["rems"] = float(rems) if rems is not None and late else np.nan
        rec["etasu"] = float(etasu) if etasu is not None and late else np.nan
        bw = lo.label_boxed_warning(label)
        rec["boxed_warning"] = float(bw) if bw is not None else np.nan
        rec["boxed_warning_openfda"] = r.boxed_warning_openfda
        # The programme's own indication passage, compared with its seed Phase 3s completed before the approval.
        ncts = [n for n in seeds.get(r.programme_id, []) if n in el.index]
        kw = [indication_keywords(ind_label.get(r.programme_id) or "")] + [
            indication_keywords(str(el.loc[n, "conditions"]).split("; ")[0]) for n in ncts]
        ind = lo.indication_passage(lo.label_indications(label), kw)
        trials = []
        for n in ncts:
            if isinstance(done.get(n), str) and done[n] <= r.approval_date:
                e = el.loc[n]
                trials.append({"text": f"{e['official_title']} {e['conditions']} {lo.inclusion_text(e['eligibility'])}",
                               "min_age": lo.trial_min_age(e["minimum_age"])})
        nar, why = lo.narrower(ind, trials)
        rec["narrow"] = float(nar) if nar is not None else np.nan
        rec["narrow_reasons"] = "; ".join(f"{w.category}: {w.label_quote}" for w in why)
        rec["label_indication"] = ind[:1500]
        rows.append(rec)
    return pd.DataFrame(rows)


def lbm_key() -> list[str]:
    return ["ApplNo", "SubmissionType", "SubmissionNo"]


def parser_checks(out: pd.DataFrame) -> dict:
    """Agreement with independent sources, and precision on the reviewed sample when present."""
    nov = out[out["cohort"] == "novel"]
    chk = {}
    aa_end = pd.read_parquet(INTERIM / "accelerated_approvals.parquet")["aa_date"].max()
    a = nov.dropna(subset=["accelerated"])
    a = a[a["accelerated_letter"].notna() & (a["approval_date"] <= aa_end)]
    chk["accelerated: FDA list vs letter wording"] = pd.crosstab(a["accelerated"], a["accelerated_letter"].astype(bool)).to_dict()
    b = nov.dropna(subset=["boxed_warning"])
    b = b[b["boxed_warning_openfda"].notna()]
    chk["boxed warning: label as approved vs openFDA today"] = pd.crosstab(
        b["boxed_warning"], b["boxed_warning_openfda"].astype(bool)).to_dict()
    path = GOLD / "label_outcomes_review.csv"
    if path.exists():
        rv = pd.read_csv(path, dtype=str).fillna("")
        rv = rv[rv["verdict"].isin(["correct", "wrong"])]
        prec = {}
        for (o, parsed, s), g in rv.groupby(["outcome", "parsed", "sample"]):
            prec[f"{o} = {parsed} [{s}]"] = {"n": len(g), "correct": float((g["verdict"] == "correct").mean())}
        chk["reviewed sample (share correct)"] = prec
    return chk


# ------------------------------------------------------------------------------------------------ training rows
def load_rows(out: pd.DataFrame) -> pd.DataFrame:
    lm = pd.read_parquet(PROCESSED / "landmarks.parquet")
    d = lm[lm["in_cohort"] & lm["in_primary"] & (lm["y_level"] == 1) & lm["level"].isin(lmk.FIT_LEVELS)]
    d = d[(d["level"] != "L2") | ~d["cutoff_estimated"]].reset_index(drop=True)
    o = out[["programme_id", "approval_date"] + list(lbm.OUTCOMES)].copy()
    o.loc[o["approval_date"] < REMS_TRAIN_START, "rems"] = np.nan
    d = d.merge(o.drop(columns="approval_date"), on="programme_id", how="inner")
    fcols = [c for c in d.columns if c.startswith("f_")]
    d["features"] = [{c[2:]: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for c, v in zip(fcols, r)}
                     for r in d[fcols].itertuples(index=False)]
    d["stage"] = d["level"].map(STAGES)
    return d


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
        except Exception:                                                   # noqa: BLE001 (tiny / separated cells)
            slope = float("nan")
    return {"n": len(y), "events": int(y.sum()), "observed": float(y.mean()), "mean_predicted": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"), "cal_slope": slope}


def predict_fold(rows: pd.DataFrame, raw: np.ndarray, tr: np.ndarray, te: np.ndarray, outcome: str) -> dict:
    y = rows[outcome].to_numpy(dtype=float)
    trk = tr[~np.isnan(y[tr])]
    m = lbm.LabelModel.fit(rows["level"].iloc[trk].tolist(), raw[trk], {outcome: y[trk]},
                           rows["drug_id"].iloc[trk].to_numpy(), n_boot=0)
    out = {"label model": m.p(outcome, rows["level"].iloc[te].tolist(), raw[te])}
    t = rows.iloc[trk]
    stage_rate = t.groupby("stage")[outcome].mean()
    out["stage base rate"] = rows["stage"].iloc[te].map(stage_rate).fillna(t[outcome].mean()).to_numpy()
    # Therapeutic-area rate, shrunk towards the overall rate (10 pseudo-rows).
    g = t.groupby(t["f_therapeutic_area"].fillna("other"))[outcome].agg(["sum", "count"])
    prior = t[outcome].mean()
    ta_rate = (g["sum"] + 10 * prior) / (g["count"] + 10)
    out["TA base rate"] = rows["f_therapeutic_area"].iloc[te].fillna("other").map(ta_rate).fillna(prior).to_numpy()
    return out


def paired_brier_ci(oof: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 4) -> tuple[float, float, float]:
    """Brier(a) − Brier(b) with a 95% bootstrap interval over drugs."""
    d = (oof[a] - oof["y"]) ** 2 - (oof[b] - oof["y"]) ** 2
    by = d.groupby(oof["drug_id"].to_numpy()).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        boots.append(s[i].sum() / c[i].sum())
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


MODELS = ["label model", "stage base rate", "TA base rate"]


def evaluate(rows: pd.DataFrame, raw: np.ndarray, outcome: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    idx = np.flatnonzero(rows[outcome].notna().to_numpy())
    tabs, oof0 = [], None
    for rep in range(REPEATS):
        fold = rows["drug_id"].map(lambda d, r=rep: int(hashlib.md5(f"{d}-{r}".encode()).hexdigest(), 16) % FOLDS).to_numpy()
        run = rows.iloc[idx][["programme_id", "drug_id", "level", "stage", "cutoff"]].copy()
        run["y"] = rows[outcome].iloc[idx].to_numpy()
        for k in range(FOLDS):
            tr, te = np.flatnonzero(fold != k), idx[fold[idx] == k]
            for name, v in predict_fold(rows, raw, tr, te, outcome).items():
                run.loc[te, name] = v
        tabs.append(score_by_stage(run, "cv"))
        if rep == 0:
            oof0 = run
    cv = pd.concat(tabs).groupby(["split", "stage", "model"], sort=False).mean(numeric_only=True).reset_index()
    # Temporal holdout: train on the earliest 80% of cut-offs.
    split = sorted(rows["cutoff"].iloc[idx])[int(0.8 * len(idx))]
    tr = np.flatnonzero((rows["cutoff"] < split).to_numpy())
    te = idx[(rows["cutoff"].iloc[idx] >= split).to_numpy()]
    hold = rows.iloc[te][["programme_id", "drug_id", "level", "stage", "cutoff"]].copy()
    hold["y"] = rows[outcome].iloc[te].to_numpy()
    for name, v in predict_fold(rows, raw, tr, te, outcome).items():
        hold[name] = v
    temporal = score_by_stage(hold, f"temporal (test: cut-off ≥ {split})")
    diffs = {st: {b: paired_brier_ci(d, "label model", b) for b in MODELS[1:]}
             for st, d in [(s, oof0[oof0["stage"] == s]) for s in ("L1", "L2", "L3–L5")] + [("all", oof0)]}
    return pd.concat([cv, temporal], ignore_index=True), oof0, diffs


def score_by_stage(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for st in ["L1", "L2", "L3–L5", "all"]:
        d = df if st == "all" else df[df["stage"] == st]
        if len(d) < 10:
            continue
        for m in MODELS:
            rows.append({"split": split, "stage": st, "model": m, **scores(d[m].to_numpy(), d["y"].to_numpy())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    out = outcome_table()
    out.to_parquet(PROCESSED / "label_outcomes.parquet", index=False)
    nov = out[out["cohort"] == "novel"]
    print(f"[11] {len(out):,} approved programmes ({len(nov):,} novel); letter text {out['has_letter'].mean():.0%}, "
          f"label text {out['has_label'].mean():.0%}")
    for o in list(lbm.OUTCOMES) + ["etasu", "narrow"]:
        print(f"[11]   {o}: known {nov[o].notna().mean():.0%} of novel, rate {nov[o].mean():.1%}")
    checks = parser_checks(out)
    print(json.dumps(checks, indent=1, default=str))

    rows = load_rows(out)
    raw = lbm.raw_matrix(rows["level"].tolist(), rows["features"].tolist())
    print(f"[11] {len(rows):,} approved training rows ({rows['programme_id'].nunique():,} programmes, "
          f"{rows['drug_id'].nunique():,} drugs): {rows['level'].value_counts().sort_index().to_dict()}")
    results, verdict = [], []
    for o in lbm.OUTCOMES:
        tab, _oof, diffs = evaluate(rows, raw, o)
        tab.insert(0, "outcome", o)
        results.append(tab)
        for st, dd in diffs.items():
            cv_st = tab[(tab["split"] == "cv") & (tab["stage"] == st)].set_index("model")
            tmp = tab[tab["split"].str.startswith("temporal") & (tab["stage"] == st)].set_index("model")
            lm_ = cv_st.loc["label model"]
            best = cv_st.drop(index="label model")["brier"].idxmin()
            verdict.append({"outcome": o, "stage": st, "n": int(lm_["n"]), "events": int(lm_["events"]),
                            "brier": lm_["brier"], "auroc": lm_["auroc"], "cal_slope": lm_["cal_slope"],
                            "best_benchmark": best, "best_benchmark_brier": cv_st.loc[best, "brier"],
                            **{f"Δ vs {b} (95% CI)": f"{d[0]:+.4f} ({d[1]:+.4f} to {d[2]:+.4f})" for b, d in dd.items()},
                            # Shown in reports as "better than typical": the stage base rate is beaten in
                            # cross-validation (95% CI) and in the temporal holdout.
                            "beats_stage_rate": bool(dd["stage base rate"][2] < 0 and len(tmp) and
                                                     tmp.loc["label model", "brier"] <= tmp.loc["stage base rate", "brier"]),
                            "beats_base_rates": bool(all(d[2] < 0 for d in dd.values())),
                            "slope_ok": bool(0.8 <= lm_["cal_slope"] <= 1.2),
                            "temporal_brier": tmp.loc["label model", "brier"] if len(tmp) else np.nan,
                            "temporal_best_base_rate": tmp.drop(index="label model")["brier"].min() if len(tmp) else np.nan,
                            "temporal_mean_pred": tmp.loc["label model", "mean_predicted"] if len(tmp) else np.nan,
                            "temporal_observed": tmp.loc["label model", "observed"] if len(tmp) else np.nan})
        print(f"[11] {o}: validated ({time.time() - t0:.0f}s)", flush=True)
    results = pd.concat(results, ignore_index=True)
    verdict = pd.DataFrame(verdict)

    y = {o: rows[o].to_numpy(dtype=float) for o in lbm.OUTCOMES}
    final = lbm.LabelModel.fit(rows["level"].tolist(), raw, y, rows["drug_id"].to_numpy(), n_boot=50)
    final.meta.update({"fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                       "verdict": verdict.to_dict("records")})
    final.save(lmk.MODEL_DIR / "label-v1")
    effects = pd.concat([pd.DataFrame(final.coefficients(o)).assign(outcome=o) for o in lbm.OUTCOMES])
    (PROCESSED / "label_validation.json").write_text(json.dumps({
        "parser_checks": checks, "verdict": verdict.to_dict("records"), "scores": results.to_dict("records"),
        "effects": effects.to_dict("records"), "C": final.meta["C"]}, indent=1, default=float), encoding="utf-8")
    write_report(nov, checks, verdict, results, effects, len(rows))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(verdict.round(4).to_string(index=False))
    print(results[results["stage"] != "all"].pivot_table(index=["outcome", "split", "stage"], columns="model",
                                                         values="brier").round(4).to_string())
    big = effects[(effects["or_lo"] > 1) | (effects["or_hi"] < 1)]
    print(big.round(2).to_string(index=False))
    print(f"[11] done in {time.time() - t0:.0f}s -> {REPORTS / 'label_outcomes.html'}")


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell(v))}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f"<div class=tbl><table><tr>{head}</tr>{body}</table></div>"


def write_report(nov, checks, verdict, results, effects, n) -> None:
    css = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1200px;color:#17140f;background:#f5f0e8}"
           "h1,h2{font-family:Georgia,serif}table{border-collapse:collapse;font-size:13px;background:#fff;margin:8px 0}"
           "th,td{border:1px solid #ddd6c8;padding:4px 8px;text-align:left}th{background:#faf7f2}.tbl{overflow-x:auto}")
    rates = pd.DataFrame([{"outcome": o, "known": int(nov[o].notna().sum()), "rate": float(nov[o].mean())}
                          for o in lbm.OUTCOMES])
    cols = ["outcome", "split", "stage", "model", "n", "events", "observed", "mean_predicted", "brier", "auroc",
            "cal_slope"]
    eff = effects[(effects["or_lo"] > 1) | (effects["or_hi"] < 1)][["outcome", "feature", "or", "or_lo", "or_hi"]]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "label_outcomes.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>Label outcomes</title><style>{css}</style>"
        f"<h1>Label outcomes given approval (label-v1)</h1><p>Built {pd.Timestamp.now():%Y-%m-%d %H:%M}. {n:,} "
        f"approved training rows (novel drugs, L1–L5). Labels from FDA approval letters, labels as approved and FDA's "
        f"accelerated approval list (post-decision documents: labels only). Cross-validated with folds hashed by drug "
        f"(3 repeats) and a temporal holdout.</p><h2>Outcome rates among approved novel drugs</h2>{_table(rates)}"
        f"<h2>Parser checks</h2><pre>{html.escape(json.dumps(checks, indent=1, default=str))}</pre>"
        f"<h2>Verdict (cross-validation, all stages)</h2>{_table(verdict)}"
        f"<h2>Scores by stage</h2>{_table(results[cols])}"
        f"<h2>Effects whose 90% bootstrap interval excludes 1 (odds ratios; per SD for continuous inputs)</h2>"
        f"{_table(eff)}", encoding="utf-8")


if __name__ == "__main__":
    main()
