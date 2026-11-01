"""Score the pipeline against the gold set (PLAN.md §5). Evaluation only (D6).

Truth comes from data/gold/gold_set.csv `truth_*` columns where the reviewer filled them in, otherwise from
data/gold/verdicts_claude.csv (a review against primary sources; `verified` = yes when an FDA letter, Drugs@FDA /
Purple Book record, sponsor filing or ClinicalTrials.gov record was checked, `evidence` names it). Codes:
  grouping/drug ok: y | n | u      outcome: approved | failed | censored | unknown (approval within D1's window)
  L2/L3/L5 ok: y (right event, drug, indication; date within ~2 weeks of the announcement) | late (right event,
  dated after the first announcement) | n (wrong drug, indication, event or programme; or dated too early) | u
  truth_L2/L3/L5: true cut-off in the pipeline's convention (day after the first public Phase 3 topline / filing
  acceptance announcement; 7 days before the final PDUFA goal date), or "none" when the event never happened
  truth_next_L3/L5: FDA's next action on the drug x indication after the pipeline's cut-off:
  approved | crl | none_within_2y | unknown, with truth_next_L3_date / truth_next_L5_date

Samples: the main stratified sample (152 programmes) and the late-level supplement (2026-11-01: 60 filed novel
programmes drawn by the pipeline's Y_LEVEL). The Y_PROG blocks use the main sample only, so they stay comparable
across reviews. `y_level` scores the training target at L3/L5 on both samples: the truth at the pipeline's own
cut-off is approval for the indication within 3 years of it, or "invalid" when the true approval came first (a
post-decision landmark). PPV/NPV (truth given the pipeline label) are re-weighted with the training rows' share of
pipeline approvals to give sensitivity, specificity and the corrected approval rate the model uses.

  uv run python scripts/gold_score.py
Writes data/gold/gold_report.json.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.dataset.labels import LATE_HORIZON_YEARS
from fdaengine.features.stats_transforms import wilson_ci

GOLD = config.DATA_DIR / "gold"
FIELDS = ["truth_grouping_ok", "truth_drug_ok", "truth_outcome", "truth_approval_date", "truth_L2_ok", "truth_L3_ok",
          "truth_L5_ok", "truth_L2", "truth_L3", "truth_L5", "truth_next_L3", "truth_next_L3_date", "truth_next_L5",
          "truth_next_L5_date"]
DATE_TOLERANCE_DAYS = 30
OUTSIDE_TRIAL_TAG = "[outside-trial approval"      # approval credited by D1 from a trial outside the programme
PIPE_NEXT = {"approved": "approved", "crl": "crl", "no_action_found": "none_within_2y", "pending": "pending"}


def _truth(gold: pd.DataFrame, claude: pd.DataFrame) -> pd.DataFrame:
    """User-filled gold_set values win; the Claude review fills the rest."""
    c = claude.set_index("programme_id")
    out = gold.copy()
    for f in FIELDS:
        user = out[f].fillna("").astype(str).str.strip() if f in out else pd.Series("", index=out.index)
        out[f] = user.where(user != "", out["programme_id"].map(c[f] if f in c else {}).fillna("").astype(str))
    reviewer = gold["reviewer"].fillna("").astype(str) != ""
    out["truth_source"] = ["reviewer" if r else "claude" for r in reviewer]
    out["notes"] = out["programme_id"].map(c["notes"]).fillna("")
    verified = out["programme_id"].map(c["verified"] if "verified" in c else {}).fillna("")
    out["verified"] = verified.where(~reviewer, "yes")
    return out


def _rate(s: pd.Series, good=("y",), bad=("n",)) -> dict:
    s = s[s.isin(good + bad)]
    return {"n": int(len(s)), "correct": int(s.isin(good).sum()),
            "rate": round(float(s.isin(good).mean()), 3) if len(s) else None}


def _acc(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Y_PROG accuracy where both truth and pipeline are approved/failed."""
    g = df[mask & df["both"]]
    return {"n": int(len(g)), "accuracy": round(float((g["pipe_y"] == g["truth_y"]).mean()), 3) if len(g) else None}


def _by(df: pd.DataFrame, col: str) -> dict:
    return {str(k): _acc(df, df[col] == k) for k in sorted(df[col].dropna().unique(), key=str)}


REVIEWED = GOLD / "reviewed_pipeline_v1.csv"     # pipeline values the level verdicts were made against


def _levels(df: pd.DataFrame) -> dict:
    """Level-evidence precision. A verdict applies while the level's cut-off is the one reviewed; a level the
    pipeline no longer produces counts as fixed (if it was wrong) or lost (if it was right)."""
    old = (pd.read_csv(REVIEWED, dtype=str, keep_default_na=False).set_index("programme_id")
           if REVIEWED.exists() else None)
    out = {}
    for lv in ("L2", "L3", "L5"):
        v = df[f"truth_{lv}_ok"]
        new = df[f"pipeline_{lv}"].fillna("").astype(str)
        prev = df["programme_id"].map(old[f"pipeline_{lv}"]).fillna("") if old is not None else new
        same = new == prev
        cur = v.where(same & (new != ""), "")
        out[lv] = {**_rate(cur, good=("y", "late"), bad=("n",)), "late": int((cur == "late").sum()),
                   "removed_wrong": int(((new == "") & (prev != "") & (v == "n")).sum()),
                   "removed_right": int(((new == "") & (prev != "") & v.isin(["y", "late"])).sum()),
                   "changed_unreviewed": int(((new != "") & (prev != "") & ~same).sum()),
                   "new_unreviewed": int(((new != "") & (prev == "")).sum())}
    return out


def _is_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def _level_dates(df: pd.DataFrame) -> dict:
    """Pipeline cut-off minus true cut-off, in days. Negative = dated before the event was public (leakage)."""
    out = {}
    for lv in ("L2", "L3", "L5"):
        pipe, truth = df[f"pipeline_{lv}"].astype(str), df[f"truth_{lv}"].astype(str)
        both = pipe.map(_is_date) & truth.map(_is_date)
        err = pd.Series([(date.fromisoformat(a) - date.fromisoformat(b)).days
                         for a, b in zip(pipe[both], truth[both])], dtype=float)
        a = err.abs()
        out[lv] = {"n": int(len(err)),
                   "median_abs_days": float(a.median()) if len(err) else None,
                   "within_14_days": round(float((a <= 14).mean()), 3) if len(err) else None,
                   "within_30_days": round(float((a <= 30).mean()), 3) if len(err) else None,
                   "share_before_truth": round(float((err < 0).mean()), 3) if len(err) else None,
                   "n_before_truth": int((err < 0).sum()),
                   "before_truth": sorted(df.loc[both, "programme_id"][(err < 0).values].tolist()),
                   "pipeline_level_but_event_never_happened": int(((pipe != "") & (truth == "none")).sum())}
    return out


def _y_next(df: pd.DataFrame) -> dict:
    """Pipeline next-action status (landmark_labels.parquet) against the truth at each L3/L5 cut-off."""
    path = config.DATA_DIR / "processed" / "landmark_labels.parquet"
    if not path.exists():
        return {}
    ll = pd.read_parquet(path)
    out = {}
    for lv in ("L3", "L5"):
        p = ll[ll["level"] == lv].set_index("programme_id")
        pipe = df["programme_id"].map(p["next_action_status"]).map(PIPE_NEXT).fillna("")
        pdate = df["programme_id"].map(p["next_action_date"]).fillna("").astype(str)
        truth, tdate = df[f"truth_next_{lv}"], df[f"truth_next_{lv}_date"].astype(str)
        known = truth.isin(["approved", "crl", "none_within_2y"]) & pipe.isin(["approved", "crl", "none_within_2y"])
        binary = truth.isin(["approved", "crl"]) & pipe.isin(["approved", "crl"])
        same_date = known & (pipe == truth) & ((truth == "none_within_2y") | (pdate.str[:10] == tdate))
        out[lv] = {"n_truth": int(truth.isin(["approved", "crl", "none_within_2y"]).sum()),
                   "n_both_known": int(known.sum()),
                   "status_accuracy": round(float((pipe[known] == truth[known]).mean()), 3) if known.any() else None,
                   "status_and_date_exact": int(same_date.sum()),
                   "y_next_binary_n": int(binary.sum()),
                   "y_next_binary_accuracy": round(float((pipe[binary] == truth[binary]).mean()), 3)
                   if binary.any() else None,
                   "pipeline_pending": int((pipe == "pending").sum()),
                   "confusion_truth_by_pipeline": pd.crosstab(truth[truth != ""], pipe[truth != ""]).to_dict()}
    return out


def _plus_years(d: str, years: int) -> str:
    return f"{int(d[:4]) + years}{d[4:]}" if d[5:10] != "02-29" else f"{int(d[:4]) + years}-02-28"


def _share(x: int, n: int) -> dict:
    lo, hi = wilson_ci(x, n) if n else (None, None)
    return {"x": x, "n": n, "rate": round(x / n, 3) if n else None,
            "ci95": [round(lo, 3), round(hi, 3)] if n else None}


def _y_level(df: pd.DataFrame) -> dict:
    """Training target Y_LEVEL at L3/L5: pipeline label vs truth at the pipeline's own cut-off."""
    proc = config.DATA_DIR / "processed"
    if not (proc / "landmark_labels.parquet").exists():
        return {}
    ll = pd.read_parquet(proc / "landmark_labels.parquet")
    lm = pd.read_parquet(proc / "landmarks.parquet")
    as_of = json.loads((proc / "labels_summary.json").read_text())["as_of"]
    old = (pd.read_csv(REVIEWED, dtype=str, keep_default_na=False).set_index("programme_id")
           if REVIEWED.exists() else None)
    reviewed_l3 = df["programme_id"].map(old["pipeline_L3"]).fillna("") if old is not None else df["pipeline_L3"]
    df = df.assign(l3_verdict=df["truth_L3_ok"].where(df["pipeline_L3"].fillna("") == reviewed_l3, ""))
    out = {}
    for lv in ("L3", "L5"):
        p = ll[(ll["level"] == lv) & ll["valid"]].set_index("programme_id")
        rows = []
        for r in df[df["programme_id"].isin(p.index)].itertuples(index=False):
            c, a, o = p.at[r.programme_id, "cutoff"], r.truth_approval_date, r.truth_outcome
            horizon = _plus_years(c, LATE_HORIZON_YEARS)
            if o == "approved" and _is_date(a) and a <= c:
                t = "invalid"
            elif o == "approved" and _is_date(a) and a <= horizon:
                t = 1
            elif o in ("approved", "failed", "censored") and horizon <= as_of:
                t = 0
            else:
                t = None
            y = p.at[r.programme_id, "y_level"]
            rows.append({"sample": r.sample, "truth": t, "pl": None if pd.isna(y) else int(y),
                         "l3_ok": r.l3_verdict})
        g = pd.DataFrame(rows)
        if g.empty:
            continue
        valid = g[g["truth"] != "invalid"]
        both = valid[valid["truth"].notna() & valid["pl"].notna()]
        both = both.assign(truth=both["truth"].astype(int))
        ppv = _share(int(((both.pl == 1) & (both.truth == 1)).sum()), int((both.pl == 1).sum()))
        npv = _share(int(((both.pl == 0) & (both.truth == 0)).sum()), int((both.pl == 0).sum()))
        prim = lm[(lm["level"] == lv) & lm["in_primary"] & lm["y_level"].notna()]
        pi = float(prim["y_level"].mean())
        res = {"n_rows": len(g), "post_decision_landmarks": _share(int((g["truth"] == "invalid").sum()), len(g)),
               "n_both_known": len(both),
               "agreement": round(float((both.pl == both.truth).mean()), 3) if len(both) else None,
               "by_sample": {k: {"n": len(v), "agreement": round(float((v.pl == v.truth).mean()), 3)}
                             for k, v in both.groupby("sample")},
               "confusion_truth_by_pipeline": pd.crosstab(both["truth"], both["pl"]).to_dict(),
               "ppv_pipeline_approved": ppv, "npv_pipeline_not_approved": npv,
               "training_share_pipeline_approved": round(pi, 3),
               "filing_event_right": _rate(g["l3_ok"]) if lv == "L3" else None}
        if ppv["n"] and npv["n"]:
            p1, n0 = ppv["rate"], npv["rate"]
            true_rate = pi * p1 + (1 - pi) * (1 - n0)
            res.update(corrected_approval_rate=round(true_rate, 3),
                       sensitivity=round(pi * p1 / true_rate, 3) if true_rate else None,
                       specificity=round((1 - pi) * n0 / (1 - true_rate), 3) if true_rate < 1 else None)
        out[lv] = res
    return out


def main() -> None:
    gold = pd.read_csv(GOLD / "gold_set.csv", dtype=str, keep_default_na=False)
    claude = pd.read_csv(GOLD / "verdicts_claude.csv", dtype=str, keep_default_na=False)
    detached = gold["detached"].eq("yes") if "detached" in gold else pd.Series(False, index=gold.index)
    everything = _truth(gold[~detached], claude)
    everything["sample"] = everything["why"].map(lambda w: "late" if w.startswith("late-level") else "main")
    y_level = _y_level(everything)
    df = everything[everything["sample"] == "main"].copy()

    df["truth_y"] = df["truth_outcome"].map({"approved": 1.0, "failed": 0.0})
    df["pipe_y"] = pd.to_numeric(df["pipeline_y_prog"], errors="coerce")
    df["both"] = df["truth_y"].notna() & df["pipe_y"].notna()
    truth_y, pipe_y, both = df["truth_y"], df["pipe_y"], df["both"]
    pipe_status = df["pipeline_event"].map(lambda e: {"1.0": "approved", "2.0": "failed", "0.0": "censored"}
                                           .get(e, "excluded"))
    confusion = pd.crosstab(df["truth_outcome"], pipe_status).to_dict()
    df["cohort_group"] = df["cohort"].map(lambda c: "novel" if c == "novel" else "lifecycle")
    df["outside_trial"] = df["notes"].str.contains(OUTSIDE_TRIAL_TAG, regex=False)

    # Approval dates where both say approved.
    dates = df[(df["truth_outcome"] == "approved") & (pipe_y == 1) & (df["truth_approval_date"] != "")]
    gaps = [abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
            for a, b in zip(dates["pipeline_approval_date"], dates["truth_approval_date"])]
    gaps_v = [g for g, v in zip(gaps, dates["verified"]) if v == "yes"]

    suz = df[df["programme_id"] == "PGa01355e3c7"].iloc[0]
    suz_check = {lv: {"pipeline": suz[f"pipeline_{lv}"], "truth": suz[f"truth_{lv}"],
                      "exact": suz[f"pipeline_{lv}"] == suz[f"truth_{lv}"]} for lv in ("L1", "L2", "L3", "L5")}
    suz_check["approval"] = {"pipeline": suz["pipeline_approval_date"], "truth": suz["truth_approval_date"],
                             "exact": suz["pipeline_approval_date"] == suz["truth_approval_date"]}

    ver = df["verified"] == "yes"
    flagged = df["resolution_flags"].fillna("").astype(str) != "" if "resolution_flags" in df else \
        pd.Series(False, index=df.index)
    novel = df["cohort_group"] == "novel"

    def approved_share(mask: pd.Series) -> dict:
        g = df[mask & df["both"]]
        return {"n": int(len(g)), "pipeline": round(float(g["pipe_y"].mean()), 3) if len(g) else None,
                "truth": round(float(g["truth_y"].mean()), 3) if len(g) else None}
    report = {
        "n_programmes": len(df),
        "n_detached": int(detached.sum()),
        "truth_source": df["truth_source"].value_counts().to_dict(),
        "verified": df["verified"].replace("", "blank").value_counts().to_dict(),
        "grouping_ok": _rate(df["truth_grouping_ok"]),
        "drug_resolution_ok": _rate(df["truth_drug_ok"]),
        "y_prog": {
            "n_both_known": int(both.sum()),
            "accuracy": round(float((pipe_y[both] == truth_y[both]).mean()), 3),
            "approved_precision": round(float((truth_y[both & (pipe_y == 1)] == 1).mean()), 3),
            "approved_recall": round(float((pipe_y[both & (truth_y == 1)] == 1).mean()), 3),
            "failed_precision": round(float((truth_y[both & (pipe_y == 0)] == 0).mean()), 3),
            "truth_known_but_pipeline_excluded_or_censored": int((truth_y.notna() & pipe_y.isna()).sum()),
            "verified_only": _acc(df, ver),
            "unverified_only": _acc(df, ~ver),
            "novel_verified_only": _acc(df, ver & (df["cohort_group"] == "novel")),
            "novel_pipeline_approved": {"n": int((both & novel & ~flagged & (pipe_y == 1)).sum()),
                                        "truth_approved": int((both & novel & ~flagged & (pipe_y == 1) & (truth_y == 1)).sum())},
            "novel_pipeline_failed": {"n": int((both & novel & ~flagged & (pipe_y == 0)).sum()),
                                      "truth_failed": int((both & novel & ~flagged & (pipe_y == 0) & (truth_y == 0)).sum())},
            "excluding_outside_trial_approvals": _acc(df, ~df["outside_trial"]),
            "n_outside_trial_approvals": int(df["outside_trial"].sum()),
            "by_cohort": _by(df, "cohort_group"),
            "by_era": _by(df, "era"),
            "by_sample_stratum": _by(df, "outcome"),
            "by_truth_outcome": _by(df, "truth_outcome"),
            "by_reached_L3": _by(df, "reached_L3"),
            "by_label_source": _by(df, "label_source"),
            "by_label_confidence": _by(df, "label_confidence"),
        },
        "resolution_flags": {      # scripts/04 drops `resolution_suspect` programmes from the cohort
            "n_flagged": int(flagged.sum()),
            "truth_drug_ok_by_flag": pd.crosstab(df["truth_drug_ok"], flagged.map({True: "flagged", False: "clean"}))
            .to_dict(),
            "y_prog_unflagged": _acc(df, ~flagged),
            "y_prog_novel_unflagged": _acc(df, novel & ~flagged),
            "y_prog_flagged": _acc(df, flagged),
            "approved_share_novel_all": approved_share(novel),
            "approved_share_novel_unflagged": approved_share(novel & ~flagged),
        },
        "outcome_confusion_truth_by_pipeline": confusion,
        "approval_date": {"n": len(gaps), f"within_{DATE_TOLERANCE_DAYS}_days": sum(g <= DATE_TOLERANCE_DAYS for g in gaps),
                          "median_abs_days": float(pd.Series(gaps).median()) if gaps else None,
                          "verified_n": len(gaps_v),
                          f"verified_within_{DATE_TOLERANCE_DAYS}_days": sum(g <= DATE_TOLERANCE_DAYS for g in gaps_v)},
        "level_events": _levels(df),
        "level_date_error": _level_dates(df),
        "y_next": _y_next(df),
        "y_level": y_level,
        "late_supplement": {"n": int((everything["sample"] == "late").sum()),
                            "filing_event_right": _rate(everything.loc[everything["sample"] == "late", "truth_L3_ok"]),
                            "drug_resolution_ok": _rate(everything.loc[everything["sample"] == "late", "truth_drug_ok"])},
        "suzetrigine_acute_pain": suz_check,
    }
    (GOLD / "gold_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
