"""05 — Landmark rows: one per programme per level it reached, with features built strictly from information public
on or before that level's cut-off (PLAN.md §5). Features use the live-case `features.build.compute` plus
structured context (`dataset.landmarks`); Claude-only fields stay empty (D6). The build fails if any row used
information dated after its cut-off (`landmarks.assert_no_leak`).

  uv run python scripts/05_build_landmark_rows.py [--workers 7]
Outputs: data/processed/landmarks.parquet, landmarks_summary.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from collections import defaultdict
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.dataset import bulk, drugs, edgar_bulk, purplebook
from fdaengine.dataset import landmarks as lm
from fdaengine.features import context as ctx_mod

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"
TA_PRIOR_WEIGHT = 20             # pseudo-programmes pulling a TA's rate towards the overall rate

# ------------------------------------------------------------------------------------------------ trials
_NEED: dict = {}
_PUBS: dict = {}


def _init(need, pubs):
    _NEED.update(need)
    _PUBS.update(pubs)


def _shard(path: str) -> dict[int, list[tuple[dict, str]]]:
    """Visible trial records for every (row, cut-off) that needs a trial in this shard."""
    out: dict[int, list] = defaultdict(list)
    for study in bulk.iter_jsonl_gz(Path(path)):
        nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        for row, cutoff, names in _NEED.get(nct, ()):
            vis = lm.visible_trial(study, names, date.fromisoformat(cutoff), _PUBS.get(nct))
            if vis:
                out[row].append(vis)
    return out


# ------------------------------------------------------------------------------------------------ as-of lookups
# Shared with live cases (`features.context`), so training and case context use one definition.
CompanyDates = ctx_mod.CompanyDates


def _modality(pt_p3: dict, iv: pd.DataFrame, names_by_prog: dict) -> dict[str, str]:
    """'biologic' if the drug's registered intervention type in its Phase 3 trials is BIOLOGICAL/GENETIC."""
    by_nct = {n: g.to_dict("records") for n, g in iv.groupby("nct_id")}
    out = {}
    for pid, ncts in pt_p3.items():
        keys = {drugs.key(n) for n in names_by_prog[pid]}
        types = set().union(*(ctx_mod.trial_types(by_nct[n], keys) for n in ncts if n in by_nct))
        if m := ctx_mod.modality(types):
            out[pid] = m
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)
    t0 = time.time()

    pg = pd.read_parquet(PROCESSED / "programmes.parquet").set_index("programme_id")
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet").set_index("programme_id")
    ll = pd.read_parquet(PROCESSED / "landmark_labels.parquet")
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    iv = pd.read_parquet(INTERIM / "ctgov" / "interventions.parquet")
    ctgov_names = pd.read_parquet(PROCESSED / "drug_names.parquet").set_index("drug_id")["ctgov_names"].to_dict()
    approvals = pd.read_parquet(PROCESSED / "fda_original_approvals.parquet")
    blas = purplebook.cber_blas(purplebook.load(bulk.raw_files("purplebook")[0]))
    bla_date = {b.bla: b.first_licensed for b in blas}

    # Rows: valid landmarks of labelled programmes (novel or sponsor-held lifecycle; `in_cohort` marks the default
    # population, the rest stay for sensitivity analyses).
    keep = labels[labels["cohort"].isin(["novel", "lifecycle_holder"]) & ~labels["healthy_volunteers"]
                  & ~labels["cber_unresolved"] & labels["anchor_p3_start"].notna()].index
    rows = ll[ll["valid"] & ll["programme_id"].isin(keep)].reset_index(drop=True)
    print(f"[05] {len(rows):,} landmark rows for {rows['programme_id'].nunique():,} programmes")

    rel = pt[pt["relation"].isin(lm.STRUCTURED_RELATIONS) & ((pt["relation"] != "p2") | (pt["match"] == "exact"))]
    trials_by_prog = rel.groupby("programme_id")["nct_id"].apply(list).to_dict()
    p3_by_prog = pt[pt["relation"].isin(["p3_seed", "p3_linked"])].groupby("programme_id")["nct_id"].apply(list).to_dict()
    names_by_prog = {pid: [r.drug_name, *(ctgov_names.get(r.drug_id) or "").split("|")] for pid, r in pg.iterrows()}
    names_by_prog = {pid: [n for n in ns if n] for pid, ns in names_by_prog.items()}
    need: dict[str, list] = defaultdict(list)
    for i, r in rows.iterrows():
        for n in trials_by_prog.get(r.programme_id, []):
            need[n].append((i, r.cutoff, names_by_prog[r.programme_id]))
    # PubMed main-paper results (script 01b), attached by `visible_trial` only once public by the cut-off.
    pub_path = INTERIM / "pubmed" / "trial_results.parquet"
    pubs = ({r["nct_id"]: {k: v for k, v in r.items() if k != "nct_id"}
             for r in pd.read_parquet(pub_path).to_dict("records") if r["nct_id"] in need} if pub_path.exists() else {})
    print(f"[05] PubMed main-paper results for {len(pubs):,} linked trials")
    with Pool(args.workers, initializer=_init, initargs=(dict(need), pubs)) as pool:
        vis_by_row: dict[int, list] = defaultdict(list)
        for part in pool.imap_unordered(_shard, [str(p) for p in bulk.raw_files("ctgov")]):
            for k, v in part.items():
                vis_by_row[k] += v
    print(f"[05] trial records as of each cut-off ({time.time() - t0:.0f}s)")

    # As-of context.
    modality = _modality(p3_by_prog, iv[iv["nct_id"].isin({n for ns in p3_by_prog.values() for n in ns})],
                         names_by_prog)
    sponsor_approvals = CompanyDates(ctx_mod.sponsor_approval_pairs(approvals, blas))
    # CRLs announced in press releases (one per drug and date), never the 2025 FDA CRL database (label source only).
    sponsor_crls = CompanyDates(ctx_mod.sponsor_crl_pairs(ev, pg))
    con = edgar_bulk.connect()
    filers = CompanyDates(con.execute("SELECT company, MIN(file_date) FROM hits GROUP BY cik, company").fetchall())
    # A programme's outcome counts towards its TA's prior rate from the date it became knowable: the approval, the
    # window closing, the last Phase 3 termination, or two quiet years after a failed readout.
    resolved = labels[labels["y_prog"].notna() & labels["event_date"].notna() & labels["in_cohort"]]
    resolved = resolved.join(pg[["therapeutic_area"]])
    known = resolved["event_date"].where(resolved["failure_basis"] != "p3_negative_no_progress",
                                         resolved["event_date"].map(lambda d: f"{int(d[:4]) + 2}{d[4:]}"))
    resolved = resolved.assign(event_date=known)
    res_sorted = resolved.sort_values("event_date")
    all_dates = res_sorted["event_date"].tolist()
    all_cum = res_sorted["y_prog"].cumsum().tolist()
    ta_series = {ta: (g["event_date"].tolist(), g["y_prog"].cumsum().tolist())
                 for ta, g in res_sorted.groupby("therapeutic_area")}
    ev_by_prog = {k: g.to_dict("records") for k, g in ev.groupby("programme_id")}

    def _prior_rate(ta: str, cutoff: str, self_pid: str) -> tuple[float | None, str | None]:
        n_all = bisect.bisect_left(all_dates, cutoff)
        if n_all < 50:
            return None, None
        m = all_cum[n_all - 1] / n_all
        ds, cum = ta_series.get(ta, ([], []))
        n = bisect.bisect_left(ds, cutoff)
        k = cum[n - 1] if n else 0.0
        own = resolved.loc[self_pid] if self_pid in resolved.index else None
        if own is not None and own["event_date"] < cutoff and own["therapeutic_area"] == ta:   # never count itself
            n, k = n - 1, k - own["y_prog"]
        return round((k + TA_PRIOR_WEIGHT * m) / (n + TA_PRIOR_WEIGHT), 4), all_dates[n_all - 1]

    out = []
    for i, r in rows.iterrows():
        p, L = pg.loc[r.programme_id], labels.loc[r.programme_id]
        cut = r.cutoff
        first_appr = p.moiety_first_approval if isinstance(p.moiety_first_approval, str) else None
        if isinstance(L["cber_bla"], str):
            first_appr = min(bla_date[b] for b in L["cber_bla"].split("|"))
        rate, rate_date = _prior_rate(p.therapeutic_area, cut, r.programme_id)
        ctx = lm.Context(therapeutic_area=p.therapeutic_area, modality=modality.get(r.programme_id),
                         moiety_approved=bool(first_appr and first_appr <= cut),
                         sponsor_prior_approvals=sponsor_approvals.count_before(p.lead_sponsor, cut),
                         sponsor_prior_crls=sponsor_crls.count_before(p.lead_sponsor, cut),
                         ta_prior_rate=rate, anchor_year=int(p.anchor_p3_start[:4]),
                         is_combination=bool(p.is_combination), n_sponsors=int(p.n_sponsors),
                         dates=[d for d in (first_appr if first_appr and first_appr <= cut else None, rate_date,
                                            p.anchor_p3_start if p.anchor_p3_start <= cut else None) if d])
        filer_since = filers.first(p.lead_sponsor)
        meta = {"sponsor": p.lead_sponsor, "cik": "edgar" if filer_since and filer_since <= cut else None}
        feats, inputs_max = lm.build_row(r.level, cut, meta, vis_by_row.get(i, []), ev_by_prog.get(r.programme_id, []),
                                         ctx)
        row = {"programme_id": r.programme_id, "level": r.level, "cutoff": cut, "cutoff_estimated": r.cutoff_estimated,
               "drug_id": p.drug_id, "drug_name": p.drug_name, "indication_label": p.indication_label,
               "lead_sponsor": p.lead_sponsor, "anchor_p3_start": p.anchor_p3_start,
               "y_prog": r.y_prog, "event": r.event, "t_event_from_cutoff_days": r.t_event_from_cutoff_days,
               "y_level": r.y_level, "event_level": r.event_level, "t_level_days": r.t_level_days,
               "y_next": r.get("y_next"), "next_action_status": r.get("next_action_status"),
               "next_action_date": r.get("next_action_date"),
               **{k: L[k] for k in ("approval_date", "label_source", "label_confidence", "resolved", "window_closed",
                                    "cohort", "in_cohort", "in_primary", "label_tier", "us_relevant", "is_rx", "cber_product", "event_date",
                                    "failure_basis")},
               "inputs_max_date": inputs_max, "n_trials_visible": len(vis_by_row.get(i, [])),
               **{f"f_{k}": v["value"] for k, v in feats.items()},
               "features_json": json.dumps(feats, default=str)}
        out.append(row)
    df = pd.DataFrame(out)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].where(df[c].notna(), None)
    lm.assert_no_leak(df.to_dict("records"))
    print(f"[05] leakage check passed on {len(df):,} rows ({time.time() - t0:.0f}s)")

    fcols = sorted(c for c in df.columns if c.startswith("f_"))
    for c in fcols:      # mixed bool/str columns → strings so parquet keeps one type per column
        if df[c].map(type).nunique() > 2:
            df[c] = df[c].map(lambda v: None if v is None else str(v))
    df.to_parquet(PROCESSED / "landmarks.parquet", index=False)
    coh = df[df["in_cohort"]]
    summary = {
        "built_at": bulk.now_iso(), "rows": len(df), "rows_in_cohort": len(coh),
        "rows_by_level_in_cohort": coh.groupby("level").size().to_dict(),
        "y_prog_known_by_level_in_cohort": coh[coh["y_prog"].notna()].groupby("level").size().to_dict(),
        "y_prog_rate_by_level_in_cohort": coh.groupby("level")["y_prog"].mean().round(3).to_dict(),
        "y_next_known_by_level_in_cohort": coh[coh["y_next"].notna()].groupby("level").size().to_dict()
        if "y_next" in coh else {},
        "feature_coverage_in_cohort": {c[2:]: {lv: round(float(g[c].notna().mean()), 3) for lv, g in coh.groupby("level")}
                                       for c in fcols},
    }
    (PROCESSED / "landmarks_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_coverage_in_cohort"}, indent=1, default=str))
    print(f"[05] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
