"""04 — Outcome labels per programme and per level. Labels only (never model inputs); no LLM calls (D6).

  Y_PROG   approved for the programme's indication within 7 years of the first Phase 3 start (D1), with the
           competing-risks coding used for IPCW / Aalen-Johansen / Fine-Gray: event 1 approval, 2 failure,
           0 censored (window still open), and a resolved-only flag for the sensitivity analysis.
  Y_LEVEL  the per-level training target: Y_PROG at L1-L2; from L3 on, approval within 3 years of the level's
           cut-off (same event coding), since the 7-year clock mostly penalises late filers (user decision
           2026-11-01).
  Y_NEXT   at each reached L3-L6 cut-off: approval (1) or CRL (0) at the FDA's next action.
  Y_P3     the pivotal Phase 3 succeeded (component at L1).
Each label carries `label_source` (which evidence decided it) and `label_confidence` (high / medium / low).

Cohorts (§3, D4): `novel` (moiety not approved before the programme's first Phase 3), `lifecycle_holder` (approved
moiety; a programme sponsor holds an NDA/BLA for it, so a supplement can approve the new use), `lifecycle_other`
(approved moiety, sponsor holds no application: generics, 505(b)(2) and biosimilar programmes; flagged),
`lifecycle_cber` (licensed CBER product: the Purple Book has no indication supplements, so not labelled).
CBER products (vaccines, blood products, cell and gene therapies) get their licences from the Purple Book.
`in_cohort` (the default training population) = novel or lifecycle_holder, US-relevant (a Phase 3 with a US site or
flagged FDA-regulated), not healthy-volunteer, with a Phase 3 start date, and not `resolution_suspect` (the resolved
drug is not what the trials test; reasons in `resolution_flags`, see drugs.resolution_flags).

  uv run python scripts/04_label_outcomes.py [--as-of YYYY-MM-DD] [--rebuild-labels]
Outputs in data/processed/: labels.parquet, landmark_labels.parquet, labels_summary.json
"""

from __future__ import annotations

import argparse
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
from fdaengine.dataset import bulk, drugs, drugsfda
from fdaengine.dataset import labels as lab
from fdaengine.dataset import purplebook
from fdaengine.dataset import timelines as tlb
from fdaengine.resolve.programme import indication_keywords

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"
LABEL_INDEX = INTERIM / "labels_nda_bla.parquet"
PROGRESS_EVENTS = {"P3_START", "PHASE3_POSITIVE", "NDA_SUBMITTED", "NDA_ACCEPTED", "PRIORITY_REVIEW",
                   "STANDARD_REVIEW", "PDUFA_DATE"}
NEXT_ACTION_LEVELS = {"L3", "L4", "L5", "L6"}


def _label_file_rows(path: str) -> list[dict]:
    return list(lab.label_rows(Path(path)))


def stage_label_index(rebuild: bool) -> pd.DataFrame:
    """ApplNo → indications text (all current label versions joined), boxed warning, product type."""
    key = json.dumps(sorted((rel, e["sha256"]) for rel, e in bulk.load_manifest()["files"].items()
                            if e["source"] == "openfda_label"))
    stamp = LABEL_INDEX.with_suffix(".source.json")
    if not rebuild and LABEL_INDEX.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key:
        return pd.read_parquet(LABEL_INDEX)
    t0 = time.time()
    files = [str(p) for p in bulk.raw_files("openfda_label")]
    with Pool(min(7, len(files))) as pool:
        rows = [r for chunk in pool.imap_unordered(_label_file_rows, files) for r in chunk]
    df = pd.DataFrame(rows)
    out = (df.groupby("ApplNo")
             .agg(appl_type=("appl_type", "first"), n_labels=("set_id", "nunique"),
                  indications=("indications", lambda s: " ".join(dict.fromkeys(x for x in s if x))[:60_000]),
                  boxed_warning=("boxed_warning", "any"),
                  product_type=("product_type", lambda s: "|".join(sorted(set("|".join(s).split("|")) - {""}))))
             .reset_index())
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LABEL_INDEX, index=False)
    stamp.write_text(key, encoding="utf-8")
    print(f"[labels] indexed {len(df):,} NDA/BLA label records -> {len(out):,} applications ({time.time() - t0:.0f}s)")
    return out


def _fda_tables(idx):
    t = drugsfda.load(bulk.raw_files("drugsfda")[0])
    pm = drugsfda.product_moieties(t["Products"], idx).dropna(subset=["drug_id"])
    apps = t["Applications"][t["Applications"]["ApplType"].isin(["NDA", "BLA"])].set_index("ApplNo")
    subs = t["Submissions"].merge(t["SubmissionClass_Lookup"], on="SubmissionClassCodeID", how="left").fillna("")
    subs = subs[(subs["SubmissionStatus"] == "AP") & (subs["status_date"] != "")]
    ms = t["MarketingStatus"].merge(t["MarketingStatus_Lookup"], on="MarketingStatusID")
    rx = ms.groupby("ApplNo")["MarketingStatusDescription"].agg(lambda s: "Prescription" in set(s))
    appl_drug = {}
    for appl, did in pm[["ApplNo", "drug_id"]].drop_duplicates().itertuples(index=False):
        if appl in apps.index:
            appl_drug.setdefault(appl, set()).add(did)
    # Applications whose products are all the same drug (a kit of two drugs is neither).
    appl_drug = {a: next(iter(ds)) for a, ds in appl_drug.items() if len(ds) == 1}
    acts_by_drug: dict[str, list[tuple]] = defaultdict(list)
    for r in subs[subs["ApplNo"].isin(appl_drug)].itertuples(index=False):
        acts_by_drug[appl_drug[r.ApplNo]].append((r.status_date, r.ApplNo, r.SubmissionType, r.SubmissionClassCode))
    return apps["SponsorName"].to_dict(), apps["ApplType"].to_dict(), acts_by_drug, rx.to_dict(), appl_drug


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", help="snapshot date (default: the Drugs@FDA download date)")
    ap.add_argument("--rebuild-labels", action="store_true", help="re-index the openFDA label files")
    args = ap.parse_args(argv)
    t0 = time.time()
    files = bulk.load_manifest()["files"]
    as_of = args.as_of or min(e["downloaded_at"][:10] for e in files.values() if e["source"] == "drugsfda")

    label_idx = stage_label_index(args.rebuild_labels)
    label_text = dict(zip(label_idx["ApplNo"], label_idx["indications"]))
    label_ptype = dict(zip(label_idx["ApplNo"], label_idx["product_type"]))
    label_boxed = dict(zip(label_idx["ApplNo"], label_idx["boxed_warning"]))

    pg = pd.read_parquet(PROCESSED / "programmes.parquet")
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    st = pd.read_parquet(INTERIM / "ctgov" / "studies.parquet")
    cond = pd.read_parquet(INTERIM / "ctgov" / "conditions.parquet")
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    lv = pd.read_parquet(PROCESSED / "levels.parquet")
    idx = drugs.build_index(pd.read_parquet(INTERIM / "gsrs_names.parquet").itertuples(index=False, name=None))
    holder, appl_type, acts_by_drug, appl_rx, appl_drug = _fda_tables(idx)
    infos = {p.programme_id: p for ps in tlb.programme_infos(pg, pt, st, cond).values() for p in ps}
    blas = purplebook.cber_blas(purplebook.load(bulk.raw_files("purplebook")[0]))
    bla_drug = purplebook.resolved_drug_ids(blas, idx)
    cber_drug_ids = set(bla_drug.values())
    ctgov_names = pd.read_parquet(PROCESSED / "drug_names.parquet").set_index("drug_id")["ctgov_names"].to_dict()
    print(f"[04] inputs loaded ({time.time() - t0:.0f}s); as_of {as_of}")

    stc = st.set_index("nct_id")
    p3 = pt[pt["relation"].isin(["p3_seed", "p3_linked"])]
    p3_by_prog = p3.groupby("programme_id")["nct_id"].apply(list).to_dict()
    ev_by_prog = {k: g for k, g in ev.groupby("programme_id")}
    first_orig = {d: min(a[0] for a in acts if a[2] == "ORIG") for d, acts in acts_by_drug.items()
                  if any(a[2] == "ORIG" for a in acts)}
    anchors_by_drug = pg.groupby("drug_id")["anchor_p3_start"].apply(lambda s: sorted(x for x in s if isinstance(x, str)))
    p3_end = {}                                     # programme → earliest Phase 3 primary completion (or start)
    for pid, ncts in p3_by_prog.items():
        ends = [t["primary_completion_date"] if isinstance(t["primary_completion_date"], str) else t["start_date"]
                for t in (stc.loc[n] for n in ncts if n in stc.index)]
        ends = [d for d in ends if isinstance(d, str)]
        if ends:
            p3_end[pid] = min(ends)
    progs_by_drug = pg.groupby("drug_id")["programme_id"].apply(list).to_dict()

    first_cond = cond[cond["pos"] == 0].set_index("nct_id")["condition"].to_dict()
    td = pd.read_parquet(PROCESSED / "trial_drugs.parquet")
    inv_moieties: dict[str, set[str]] = defaultdict(set)
    for n, did in td[td["investigational"] & td["drug_id"].str.startswith("UNII:")][["nct_id", "drug_id"]].itertuples(
            index=False):
        inv_moieties[n] |= set(did.removeprefix("UNII:").split("+"))
    combo_index: dict[str, list[str]] = defaultdict(list)          # single moiety → combination drug ids
    for did in acts_by_drug:
        if did.startswith("UNII:") and "+" in did:
            for m in did.removeprefix("UNII:").split("+"):
                combo_index[f"UNII:{m}"].append(did)
    rows, cov_cache = [], {}
    for r in pg.itertuples(index=False):
        info = infos[r.programme_id]
        anchor = r.anchor_p3_start if isinstance(r.anchor_p3_start, str) else None
        mfa = r.moiety_first_approval if isinstance(r.moiety_first_approval, str) else None
        lifecycle = bool(anchor and mfa and mfa < anchor)
        e = ev_by_prog.get(r.programme_id, pd.DataFrame(columns=ev.columns))
        kw_sets = lab.keyword_sets(info.keywords, [indication_keywords(first_cond[n])
                                                   for n in p3_by_prog.get(r.programme_id, []) if n in first_cond])
        # A co-formulated product counts when all its other moieties are investigational in this programme's
        # Phase 3 trials (glecaprevir's trials test glecaprevir/pibrentasvir: Mavyret is its approval).
        inv = set().union(*(inv_moieties.get(n, set()) for n in p3_by_prog.get(r.programme_id, [])))
        own = set(r.drug_id.removeprefix("UNII:").split("+")) if r.drug_id.startswith("UNII:") else set()
        combos = [c for c in combo_index.get(r.drug_id, []) if own and set(c.removeprefix("UNII:").split("+")) <= inv | own]
        acts = []
        for d, appl, kind, cls in [a for did in [r.drug_id, *combos] for a in acts_by_drug.get(did, [])]:
            ck = (appl, tuple(map(tuple, kw_sets)))
            if ck not in cov_cache:
                cov_cache[ck] = lab.best_coverage(kw_sets, label_text[appl]) if label_text.get(appl) else None
            acts.append(lab.Action(d, appl, kind, cls, any(tlb.same_company(holder.get(appl, ""), s)
                                                           for s in info.sponsors), cov_cache[ck]))
        orig_d = first_orig.get(r.drug_id)
        # CBER products: licences from the Purple Book replace the (absent) Drugs@FDA record.
        cber_like = lab.is_cber_product(r.drug_name) or r.drug_id in cber_drug_ids
        matched = purplebook.match(
            r.drug_id, [r.drug_name, *(ctgov_names.get(r.drug_id) or "").split("|")],
            lambda applicant: any(tlb.same_company(applicant, s) for s in info.sponsors), blas, bla_drug) \
            if cber_like else []
        if matched:
            acts = [lab.Action(b.first_licensed, b.bla, "ORIG", "CBER",
                               any(tlb.same_company(b.applicant, s) for s in info.sponsors),
                               lab.best_coverage(kw_sets, label_text[b.bla]) if label_text.get(b.bla) else None)
                    for b in matched]
            orig_d = min(b.first_licensed for b in matched)
            lifecycle = bool(anchor and orig_d < anchor)
        trials = [stc.loc[n] for n in p3_by_prog.get(r.programme_id, []) if n in stc.index]
        rivals = (sum(a <= orig_d for a in anchors_by_drug.get(r.drug_id, [])) - 1) if orig_d and anchor else 0
        neg = e[e["event_type"] == "PHASE3_NEGATIVE"]["date"]
        f = lab.Facts(
            r.programme_id, anchor, lifecycle, max(rivals, 0), acts, cber=bool(matched),
            rival_p3_ends=[p3_end[q] for q in progs_by_drug.get(r.drug_id, []) if q != r.programme_id and q in p3_end],
            press_approvals=sorted(e[(e["event_type"] == "APPROVED")]["date"]),
            p3_statuses=[t["status"] for t in trials],
            p3_end_dates=[(t["primary_completion_date"] if isinstance(t["primary_completion_date"], str)
                           else t["start_date"] if isinstance(t["start_date"], str) else None) for t in trials],
            p3_negative=neg.min() if len(neg) else None,
            progress_dates=sorted(e[e["event_type"].isin(PROGRESS_EVENTS)]["date"]))
        out = lab.programme_label(f, as_of)
        crls = sorted(d for d in e[e["event_type"] == "CRL"]["date"] if anchor and d >= anchor)
        pos = e[e["event_type"] == "PHASE3_POSITIVE"]["date"]
        y_p3, y_p3_basis = (1, "approved") if out["y_prog"] == 1 else \
            (1, "phase3_positive") if len(pos) else (0, "phase3_negative") if len(neg) else (None, None)
        holds = any(tlb.same_company(holder.get(a, ""), s) for a in {x[1] for x in acts_by_drug.get(r.drug_id, [])}
                    for s in info.sponsors)
        appl = out["approval_appl"]
        out.update(
            drug_id=r.drug_id, anchor_p3_start=anchor, cohort="novel" if not lifecycle else
            # A licensed CBER product's new uses can't be labelled: the Purple Book has no indication supplements.
            "lifecycle_cber" if matched else "lifecycle_holder" if holds else "lifecycle_other",
            healthy_volunteers=str(r.indication_label).lower().startswith("healthy"),
            cber_product=cber_like, cber_bla="|".join(b.bla for b in matched) or None,
            drug_resolution=r.drug_resolution,
            us_site=any(bool(t["has_us_site"]) for t in trials),
            fda_regulated=any(t["fda_regulated_drug"] is True for t in trials),
            crl_dates="|".join(crls), n_crl=len(crls), y_p3=y_p3, y_p3_basis=y_p3_basis,
            approval_appl_type=("BLA" if matched else appl_type.get(appl)) if appl else None,
            approval_class=next((a.class_code for a in acts if a.appl == appl and a.date == out["approval_date"]), None),
            is_rx=(bool(matched) or bool(appl_rx.get(appl)) or "PRESCRIPTION" in (label_ptype.get(appl) or ""))
            if appl else None,
            boxed_warning=bool(label_boxed.get(appl)) if appl and appl in label_boxed else None,
            label_cov=next((a.label_cov for a in acts if a.appl == appl), None) if appl else None)
        out["resolved"] = out["y_prog"] is not None and (out["window_closed"] or out["event"] in (1, 2))
        rows.append(out)
    labels = pd.DataFrame(rows)
    # Default population (user decision, 2026-10-30): US-relevant programmes only, i.e. a Phase 3 with a US site or
    # flagged FDA-regulated. Site lists are the registry's current ones: a cohort filter, not a model feature.
    labels["us_relevant"] = labels["us_site"] | labels["fda_regulated"]
    # A CBER product with no resolved name and no matching licence ('MENACWYCRMCONJUGATEVACCINE') may well be
    # licensed: its label cannot be trusted either way.
    labels["cber_unresolved"] = labels["cber_product"] & labels["cber_bla"].isna() \
        & (labels["drug_resolution"] == "name_only")
    # A resolved drug that is not what the trials test (code-name prefix, vehicle, class term, target protein) can
    # never match its approval, so its label is almost always 'failed': flagged and kept out of the cohort.
    # Names and trial text only (no approval status, no brand names), so the flag is independent of the outcome.
    moiety_keys, brand_keys = drugs.name_keys(idx), drugs.name_keys(idx, brands=True)
    before = dict(zip(pg["programme_id"], [isinstance(m, str) and isinstance(a, str) and m < a
                                           for m, a in zip(pg["moiety_first_approval"], pg["anchor_p3_start"])]))
    seed = (pt[pt["relation"] == "p3_seed"][["programme_id", "nct_id"]]
            .merge(pg[["programme_id", "drug_id"]], on="programme_id")
            .merge(td[["nct_id", "drug_id", "names", "resolved_by"]], on=["nct_id", "drug_id"], how="left"))
    seed_rows = {pid: list(zip(g["names"].fillna(""), g["resolved_by"].fillna(""))) for pid, g in seed.groupby("programme_id")}
    labels["resolution_flags"] = ["|".join(drugs.resolution_flags(d, seed_rows.get(p, []), moiety_keys, idx.display,
                                                                  brand_keys, before.get(p, False)))
                                  for p, d in zip(labels["programme_id"], labels["drug_id"])]
    # CBER products (vaccines, antigens, plasma proteins) are often named like targets: not suspect on names alone.
    # weak_match alone is about half right on review (real drugs named by abbreviation or brand): recorded for a
    # sensitivity analysis, not excluded.
    labels["resolution_suspect"] = labels["resolution_flags"].str.contains(
        "not_a_drug|vehicle|descriptive_name") & ~labels["cber_product"]
    labels["in_cohort"] = labels["cohort"].isin(["novel", "lifecycle_holder"]) & ~labels["healthy_volunteers"] \
        & labels["us_relevant"] & ~labels["cber_unresolved"] & ~labels["resolution_suspect"] \
        & labels["anchor_p3_start"].notna()
    # Gold set (2026-10-30): approvals attributed through a supplement whose *current* label mentions the disease
    # were right 9 times in 17 (line of therapy, other manifestations, negated text). Those labels are a
    # sensitivity tier, and lifecycle programmes, whose approvals mostly come that way, a secondary analysis.
    labels["label_tier"] = labels["label_source"].map(lambda s: "sensitivity" if s == "drugsfda_suppl+label"
                                                      else "primary")
    labels["in_primary"] = labels["in_cohort"] & (labels["cohort"] == "novel")
    print(f"[04] programme labels done ({time.time() - t0:.0f}s)")

    # Per-level labels: Y_PROG is the programme's; Y_NEXT from L3 on. A level whose cut-off is on/after the
    # approval date is not a pre-decision landmark under these labels and is marked invalid.
    lab_by = {r["programme_id"]: r for r in rows}
    lrows = []
    for r in lv[lv["reached"]].itertuples(index=False):
        L = lab_by[r.programme_id]
        crls = [d for d in L["crl_dates"].split("|") if d]
        app_date = L["approval_date"]
        row = {"programme_id": r.programme_id, "level": r.level, "cutoff": r.cutoff, "cutoff_estimated": r.estimated,
               "valid": not (app_date and r.cutoff >= app_date), "y_prog": L["y_prog"], "event": L["event"],
               "t_event_from_cutoff_days": (date.fromisoformat(L["event_date"]) - date.fromisoformat(r.cutoff)).days
               if L["event_date"] else None}
        row.update(lab.level_target(r.level, r.cutoff, app_date, row["y_prog"], row["event"],
                                    row["t_event_from_cutoff_days"], as_of))
        if r.level in NEXT_ACTION_LEVELS:
            row.update(lab.next_action(r.cutoff, app_date, crls, as_of))
        lrows.append(row)
    ll = pd.DataFrame(lrows)

    labels.to_parquet(PROCESSED / "labels.parquet", index=False)
    ll.to_parquet(PROCESSED / "landmark_labels.parquet", index=False)
    coh = labels[labels["in_cohort"]]
    summary = {
        "built_at": bulk.now_iso(), "as_of": as_of, "n_programmes": len(labels),
        "cohort": labels["cohort"].value_counts().to_dict(),
        "in_cohort": int(labels["in_cohort"].sum()),
        "in_primary": int(labels["in_primary"].sum()),
        "resolution_suspect_excluded": int((labels["resolution_suspect"] & labels["cohort"].isin(
            ["novel", "lifecycle_holder"]) & labels["us_relevant"]).sum()),
        "resolution_flags": labels["resolution_flags"].str.split("|").explode().replace("", None).dropna()
        .value_counts().to_dict(),
        "y_prog_rate_resolved_primary": round(float(labels[labels["in_primary"] & labels["resolved"]]["y_prog"].mean()), 3),
        "event_in_cohort": coh["event"].map({1: "approved", 2: "failed", 0: "censored"}).fillna("excluded")
        .value_counts().to_dict(),
        "label_source_approved": coh[coh["event"] == 1]["label_source"].value_counts().to_dict(),
        "label_confidence_approved": coh[coh["event"] == 1]["label_confidence"].value_counts().to_dict(),
        "failure_basis": coh["failure_basis"].value_counts().to_dict(),
        "resolved_in_cohort": int(coh["resolved"].sum()),
        "y_prog_rate_resolved_by_cohort": coh[coh["resolved"]].groupby("cohort")["y_prog"].mean().round(3).to_dict(),
        "y_prog_rate_resolved_by_us_relevance": {
            f"{c}:{'us' if u else 'non_us'}": [round(float(g.mean()), 3), int(len(g))]
            for (c, u), g in labels[labels["resolved"] & labels["cohort"].isin(["novel", "lifecycle_holder"])]
            .groupby(["cohort", "us_relevant"])["y_prog"]},
        "cber": {"programmes": int(labels["cber_product"].sum()),
                 "matched_to_licence": int(labels["cber_bla"].notna().sum()),
                 "in_cohort": int(coh["cber_product"].sum()),
                 "y_prog_rate_resolved_in_cohort": {
                     ("matched" if m else "unmatched"): [round(float(g.mean()), 3), int(len(g))]
                     for m, g in coh[coh["resolved"] & coh["cber_product"]].groupby(coh["cber_bla"].notna())["y_prog"]}},
        "rx_of_approved": coh[coh["event"] == 1]["is_rx"].value_counts(dropna=False).rename(str).to_dict(),
        "landmark_rows": ll.groupby("level").size().to_dict(),
        "landmark_rows_invalid": ll[~ll["valid"]].groupby("level").size().to_dict(),
        "y_level_rate_valid_cohort": {lv_: [round(float(g.mean()), 3), int(g.notna().sum())] for lv_, g in
                                      ll[ll["valid"] & ll["programme_id"].isin(coh["programme_id"])]
                                      .groupby("level")["y_level"]},
        "y_next_status": {f"{lv_}:{s}": int(n) for (lv_, s), n in ll[ll["valid"]].groupby(
            ["level", "next_action_status"]).size().items()} if "next_action_status" in ll else {},
    }
    (PROCESSED / "labels_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"[04] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
