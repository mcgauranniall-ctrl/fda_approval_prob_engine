"""Gold set (PLAN.md §5): ~150 cohort programmes, stratified by outcome × era × whether L3 was reached, with
suzetrigine (acute pain and DPN) always included. Evaluation only: nothing here feeds training (D6).

Writes, under data/gold/ (small, committed):
  gold_set.csv       one row per programme: pipeline values + empty `truth_*` columns for the reviewer
  review_sheet.md    the evidence behind each pipeline value (trials, FDA label text, press-release quotes)
The sample is drawn once (fixed seed) and kept; re-running refreshes pipeline values and evidence but never
overwrites filled-in `truth_*` columns.

  uv run python scripts/gold_build.py
  uv run python scripts/gold_build.py --add-late 60    # append a late-level supplement (once)

Late-level supplement (2026-11-01): novel-drug programmes that reached L3 or L5 with a final Y_LEVEL, drawn with a
fixed seed and weighted towards pipeline failures (where missed approvals hide). It measures label error at the
levels the main sample covers thinly; `sample_stratum` records the draw so rates can be re-weighted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.dataset import bulk, drugsfda, purplebook

PROCESSED = config.DATA_DIR / "processed"
INTERIM = config.DATA_DIR / "interim"
GOLD = config.DATA_DIR / "gold"
SEED = 20261030
TARGET = {1: 55, 2: 55, 0: 20, "excluded": 20}           # approved, failed, censored, excluded labels
FORCED = {"PGa01355e3c7": "suzetrigine / acute pain (known dates)", "PG707eb13f4b": "suzetrigine / DPN"}
# Known dates from the brief (suzetrigine, acute pain): the only externally fixed truth in the set.
KNOWN = {"PGa01355e3c7": {"truth_outcome": "approved", "truth_approval_date": "2025-01-30",
                          "truth_L1": "2023-08-24", "truth_L2": "2024-02-06", "truth_L3": "2024-08-02",
                          "truth_L5": "2025-01-23", "reviewer": "brief"}}
TRUTH_COLS = ["truth_grouping_ok", "truth_drug_ok", "truth_outcome", "truth_approval_date", "truth_L2_ok",
              "truth_L3_ok", "truth_L5_ok", "truth_L1", "truth_L2", "truth_L3", "truth_L5", "truth_next_L3",
              "truth_next_L3_date", "truth_next_L5", "truth_next_L5_date", "notes", "reviewer", "user_override"]
LEVELS = ["L1", "L2", "L3", "L4", "L5"]
LATE_SEED = 20261101
LATE_SHARE_FAILED = 0.6                  # of the supplement, rows the pipeline labels not approved


def _era(anchor: str) -> str:
    y = int(anchor[:4])
    return "<2009" if y < 2009 else "2009-2015" if y <= 2015 else "2016+"


def sample(labels: pd.DataFrame, lv: pd.DataFrame) -> pd.DataFrame:
    coh = labels[labels["in_cohort"]].copy()
    coh["outcome"] = coh["event"].map(lambda e: "excluded" if pd.isna(e) else int(e))
    coh["era"] = coh["anchor_p3_start"].map(_era)
    l3 = set(lv[(lv["level"] == "L3") & lv["reached"]]["programme_id"])
    coh["reached_L3"] = coh["programme_id"].isin(l3)
    picks = []
    for outcome, n in TARGET.items():
        pool = coh[(coh["outcome"] == outcome) & ~coh["programme_id"].isin(FORCED)]
        cells = list(pool.groupby(["era", "reached_L3"]))
        per = max(1, n // len(cells))
        got = pd.concat([g.sample(min(per, len(g)), random_state=SEED) for _, g in cells])
        if len(got) < n:                                        # top up from what is left
            rest = pool[~pool["programme_id"].isin(got["programme_id"])]
            got = pd.concat([got, rest.sample(min(n - len(got), len(rest)), random_state=SEED)])
        picks.append(got.head(n))
    forced = coh[coh["programme_id"].isin(FORCED)]
    out = pd.concat([forced, *picks]).drop_duplicates("programme_id")
    out["why"] = out["programme_id"].map(FORCED).fillna("stratified sample")
    return out[["programme_id", "outcome", "era", "reached_L3", "why"]]


def late_sample(labels: pd.DataFrame, lm: pd.DataFrame, exclude: set[str], n: int) -> pd.DataFrame:
    """Novel programmes with a final Y_LEVEL at L3 (else L5), stratified by the pipeline's Y_LEVEL."""
    prim = set(labels.loc[labels["in_primary"], "programme_id"])
    rows = lm[lm["level"].isin(["L3", "L5"]) & lm["valid"] & lm["y_level"].notna() & lm["programme_id"].isin(prim)
              & ~lm["programme_id"].isin(exclude)].sort_values("level").drop_duplicates("programme_id")
    picks = []
    for y, share in ((0, LATE_SHARE_FAILED), (1, 1 - LATE_SHARE_FAILED)):
        pool = rows[rows["y_level"] == y]
        picks.append(pool.sample(min(round(n * share), len(pool)), random_state=LATE_SEED))
    out = pd.concat(picks)
    coh = labels.set_index("programme_id").loc[out["programme_id"]]
    return pd.DataFrame({"programme_id": out["programme_id"].values,
                         "outcome": coh["event"].map(lambda e: "excluded" if pd.isna(e) else int(e)).values,
                         "era": coh["anchor_p3_start"].map(_era).values, "reached_L3": True,
                         "why": "late-level supplement: " + out["level"].values + ", pipeline Y_LEVEL "
                                + out["y_level"].astype(int).astype(str).values})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-late", type=int, default=0, help="append N late-level programmes to the gold set")
    args = ap.parse_args()
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    lv = pd.read_parquet(PROCESSED / "levels.parquet")
    pg = pd.read_parquet(PROCESSED / "programmes.parquet").set_index("programme_id")
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    st = pd.read_parquet(INTERIM / "ctgov" / "studies.parquet").set_index("nct_id")
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    label_idx = pd.read_parquet(INTERIM / "labels_nda_bla.parquet").set_index("ApplNo")["indications"].to_dict()
    fda = drugsfda.load(bulk.raw_files("drugsfda")[0])
    appl_name = fda["Products"].groupby("ApplNo")["DrugName"].first().to_dict()
    bla_name = {b.bla: f"{b.proprietary_name} ({b.proper_name})"
                for b in purplebook.cber_blas(purplebook.load(bulk.raw_files("purplebook")[0]))}

    GOLD.mkdir(parents=True, exist_ok=True)
    path = GOLD / "gold_set.csv"
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        base = old[["programme_id", "outcome", "era", "reached_L3", "why"]]
        if args.add_late:
            if old["why"].str.startswith("late-level supplement").any():
                raise SystemExit("[gold] the late-level supplement is already in the gold set")
            lm = pd.read_parquet(PROCESSED / "landmark_labels.parquet")
            add = late_sample(labels, lm, set(base["programme_id"]), args.add_late)
            base = pd.concat([base, add.astype(str)], ignore_index=True)
            print(f"[gold] added {len(add)} late-level programmes")
    else:
        old, base = None, sample(labels, lv)

    L = labels.set_index("programme_id")
    rows, sheet = [], ["# Gold set review sheet", "",
                       "Evidence behind each pipeline value. Verdicts go in `gold_set.csv` (`truth_*` columns).", ""]
    old_rows = old.set_index("programme_id") if old is not None else None
    for b in base.itertuples(index=False):
        pid = b.programme_id
        if pid not in pg.index or pid not in L.index:
            # A rebuild changed the programme's drug, indication or anchor (e.g. a drug-resolution fix), so its id
            # changed. Keep the row and its verdicts for the record; the scorer leaves detached rows out.
            rows.append({**old_rows.loc[pid].to_dict(), "programme_id": pid, "detached": "yes"})
            continue
        p, lab = pg.loc[pid], L.loc[pid]
        lvp = lv[(lv["programme_id"] == pid) & lv["reached"]].set_index("level")
        row = {"programme_id": pid, "outcome": b.outcome, "era": b.era, "reached_L3": b.reached_L3, "why": b.why,
               "drug_name": p.drug_name, "indication": p.indication_label, "therapeutic_area": p.therapeutic_area,
               "sponsor": p.lead_sponsor, "anchor_p3_start": p.anchor_p3_start, "cohort": lab["cohort"],
               "pipeline_y_prog": lab["y_prog"], "pipeline_event": lab["event"],
               "pipeline_approval_date": lab["approval_date"], "label_source": lab["label_source"],
               "label_confidence": lab["label_confidence"], "failure_basis": lab["failure_basis"],
               "resolution_flags": lab.get("resolution_flags", "") if lab.get("resolution_suspect") else "",
               **{f"pipeline_{lv_}": (lvp.loc[lv_, "cutoff"] if lv_ in lvp.index else None) for lv_ in LEVELS}}
        rows.append(row)

        trials = pt[(pt["programme_id"] == pid) & pt["relation"].isin(["p3_seed", "p3_linked"])]["nct_id"]
        sheet += [f"## {pid}: {p.drug_name} / {p.indication_label}", "",
                  f"- Sponsor {p.lead_sponsor}; TA {p.therapeutic_area}; cohort {lab['cohort']}; first P3 "
                  f"{p.anchor_p3_start}; moiety first approved {p.moiety_first_approval}; sample: {b.why}",
                  f"- **Label:** event {lab['event']}, y_prog {lab['y_prog']}, approval {lab['approval_date']} "
                  f"(appl {lab['approval_appl']}), source {lab['label_source']} / {lab['label_confidence']}, "
                  f"failure basis {lab['failure_basis']}, CRLs {lab['crl_dates'] or '-'}"]
        appl = lab["approval_appl"]
        if isinstance(appl, str):
            name = bla_name.get(appl) or appl_name.get(appl, "")
            sheet.append(f"- **Approved product** {name} (appl {appl}); label indications: "
                         f"\"{' '.join(str(label_idx.get(appl, '(no label on file)')).split())[:400]}\"")
        sheet.append("- **Phase 3 trials:**")
        for n in trials:
            if n in st.index:
                t = st.loc[n]
                sheet.append(f"  - {n} [{t['status']}] {t['start_date']} -> {t['primary_completion_date']}: "
                             f"{str(t['title'])[:130]}")
        sheet.append("- **Levels:**")
        e = ev[ev["programme_id"] == pid]
        for lv_ in LEVELS:
            if lv_ not in lvp.index:
                continue
            r = lvp.loc[lv_]
            sheet.append(f"  - {lv_} {r['cutoff']} ({r['basis']}{', estimated' if r['estimated'] else ''})")
            types = {"L2": ["PHASE3_POSITIVE", "PHASE3_NEGATIVE"],
                     "L3": ["NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_DATE"],
                     "L4": ["ADCOM_VOTE"], "L5": ["PDUFA_DATE"]}.get(lv_, [])
            src = e[e["event_type"].isin(types) & (e["source"] == "press_release")].sort_values("date")
            if lv_ == "L5" and len(src):
                src = src.iloc[[-1]]
            for q in src.head(1).itertuples():
                sheet.append(f"    - {q.event_type} {q.date} [{q.date_basis}] {q.source_id}: "
                             f"\"{' '.join(str(q.quote).split())[:300]}\"")
        sheet.append("")

    out = pd.DataFrame(rows)
    out["detached"] = out["detached"].fillna("") if "detached" in out else ""
    for c in TRUTH_COLS:
        out[c] = ""
    if old is not None:                                    # keep the reviewer's work
        keep = old.set_index("programme_id")[[c for c in TRUTH_COLS if c in old.columns]]
        for c in keep.columns:
            out[c] = out["programme_id"].map(keep[c]).fillna("")
    for pid, truth in KNOWN.items():
        for c, v in truth.items():
            out.loc[out["programme_id"] == pid, c] = v
    out.to_csv(path, index=False)
    (GOLD / "review_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    print(f"[gold] {len(out)} programmes -> {path}")
    print(out.groupby(["outcome", "era"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
