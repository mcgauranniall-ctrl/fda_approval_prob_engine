"""02 — Group industry Phase 3 trials into drug × indication programmes; link earlier Phase 1/2 trials.

Stages (each cached; rerun with --rebuild to redo all):
  A. flatten  ClinicalTrials.gov JSON → data/interim/ctgov/<table>.parquet
  B. drugs    resolve each trial's investigational drug(s) via GSRS names + ClinicalTrials.gov aliases
  C. group    industry-led Phase 3 trials → programmes (drug × indication episode); link Phase 1/2 trials

Outputs in data/processed/: programmes.parquet, programme_trials.parquet, trial_drugs.parquet, drug_names.parquet
No LLM calls (D6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.dataset import bulk, ctgov_flat, drugs, drugsfda
from fdaengine.dataset import programmes as prog
from fdaengine.features.indication import therapeutic_area

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"
CTGOV_FLAT = INTERIM / "ctgov"


def _source_key(source: str) -> str:
    """Fingerprint of the raw files a stage depends on, so caches rebuild when the download changes."""
    m = bulk.load_manifest()["files"]
    return json.dumps(sorted((rel, e["sha256"]) for rel, e in m.items() if e["source"] == source))


def _fresh(stamp: Path, key: str) -> bool:
    return stamp.exists() and stamp.read_text(encoding="utf-8") == key


def stage_flatten(rebuild: bool) -> None:
    key = _source_key("ctgov") + f"|schema{ctgov_flat.SCHEMA_VERSION}"
    stamp = CTGOV_FLAT / "_source.json"
    if not rebuild and _fresh(stamp, key):
        print("[A] flattened tables are current")
        return
    t0 = time.time()
    rows: dict[str, list[dict]] = {t: [] for t in ctgov_flat.TABLES}
    n = 0
    for path in bulk.raw_files("ctgov"):
        for study in bulk.iter_jsonl_gz(path):
            for table, recs in ctgov_flat.flatten(study).items():
                rows[table].extend(recs)
            n += 1
    CTGOV_FLAT.mkdir(parents=True, exist_ok=True)
    for table, recs in rows.items():
        pd.DataFrame(recs).to_parquet(CTGOV_FLAT / f"{table}.parquet", index=False)
    stamp.write_text(key, encoding="utf-8")
    print(f"[A] flattened {n:,} studies in {time.time() - t0:.0f}s -> {CTGOV_FLAT}")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for table in ctgov_flat.TABLES:
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{(CTGOV_FLAT / table).as_posix()}.parquet')")
    return con


# --------------------------------------------------------------------------------------------- B. drugs
DRUG_IV_TYPES = {"DRUG", "BIOLOGICAL", "GENETIC", "COMBINATION_PRODUCT"}
COMPARATOR_ARMS = {"PLACEBO_COMPARATOR", "SHAM_COMPARATOR", "ACTIVE_COMPARATOR", "NO_INTERVENTION"}


def stage_gsrs(rebuild: bool) -> drugs.GsrsIndex:
    path = INTERIM / "gsrs_names.parquet"
    key = _source_key("openfda_substance")
    stamp = INTERIM / "gsrs_names._source.json"
    if rebuild or not _fresh(stamp, key):
        t0 = time.time()
        src = bulk.raw_files("openfda_substance")[0]
        df = pd.DataFrame(drugs.gsrs_rows(src), columns=["unii", "moiety", "substance_class", "name", "name_type",
                                                          "is_display"])
        df.to_parquet(path, index=False)
        stamp.write_text(key, encoding="utf-8")
        print(f"[B] GSRS: {len(df):,} names for {df['unii'].nunique():,} substances in {time.time() - t0:.0f}s")
    df = pd.read_parquet(path)
    return drugs.build_index(df.itertuples(index=False, name=None))


def _arm_roles(con) -> dict[tuple[str, int], set[str]]:
    """(nct, intervention index) → arm types, using links from both the intervention and the arm side."""
    ivs = con.sql("SELECT nct_id, iv_idx, type, name, arm_types FROM interventions").fetchall()
    arms = con.sql("SELECT nct_id, type, intervention_names FROM arms").fetchall()
    by_arm_name: dict[tuple[str, str], set[str]] = {}
    for nct, atype, names in arms:
        for n in (names or "").split("|"):
            if n:
                by_arm_name.setdefault((nct, n.split(": ", 1)[-1].strip().lower()), set()).add(atype or "")
    out = {}
    for nct, i, _itype, name, types in ivs:
        s = {t for t in (types or "").split("|") if t}
        s |= by_arm_name.get((nct, (name or "").strip().lower()), set())
        out[(nct, i)] = s - {""}
    return out


def _iv_role(types: set[str], trial_has_comparator: bool) -> str:
    if not types:
        return "unknown" if trial_has_comparator else "experimental"
    if types <= {"EXPERIMENTAL", "OTHER"}:
        return "experimental"
    if types <= COMPARATOR_ARMS:
        return "comparator"
    return "both"


def stage_trial_drugs(con, idx: drugs.GsrsIndex, first_approval: dict[str, str]) -> pd.DataFrame:
    t0 = time.time()
    roles = _arm_roles(con)
    has_comp = {n for (n,) in con.sql(
        "SELECT DISTINCT nct_id FROM arms WHERE type IN "
        "('PLACEBO_COMPARATOR','SHAM_COMPARATOR','ACTIVE_COMPARATOR','NO_INTERVENTION')").fetchall()}
    ivs = con.sql("SELECT nct_id, iv_idx, type, name, other_names FROM interventions").df()
    ivs = ivs[ivs["type"].isin(DRUG_IV_TYPES)]

    cache: dict[str, tuple[frozenset[str], str]] = {}

    def resolve(text: str) -> tuple[frozenset[str], str]:
        if text not in cache:
            cache[text] = drugs.resolve_string(text, idx)
        return cache[text]

    union = drugs.AliasUnion()
    rows = []
    for nct, i, _t, name, other in ivs.itertuples(index=False):
        others = [o for o in (other or "").split("|") if o]
        mo, how = resolve(name)
        if not mo:
            for o in others:
                mo, how = resolve(o)
                if mo:
                    how = f"other_name_{how}"
                    break
        base = drugs.drug_id(mo) if mo else drugs.name_id(name) or next(
            (x for x in map(drugs.name_id, others) if x), None)
        if base is None:
            continue
        for s in [name, *others]:
            nid = drugs.name_id(s)
            if nid and (not mo or not (hit := resolve(s)[0]) or hit == mo):
                union.union(base, nid)
        rows.append({"nct_id": nct, "iv_idx": i, "name": name, "base_id": base, "resolved_by": how or "name_only",
                     "role": _iv_role(roles.get((nct, i), set()), nct in has_comp)})
    df = pd.DataFrame(rows)
    df["drug_id"] = [union.find(b) for b in df["base_id"]]
    df.loc[df["resolved_by"].eq("name_only") & df["drug_id"].str.startswith("UNII:"), "resolved_by"] = "alias"

    # One row per (trial, drug): experimental only if never also in a comparator arm.
    def agg_role(rs: set[str]) -> str:
        if "both" in rs or {"experimental", "comparator"} <= rs or {"unknown", "comparator"} <= rs:
            return "background"
        return "experimental" if "experimental" in rs else "unknown" if "unknown" in rs else "comparator"

    agg: dict[tuple[str, str], list] = {}
    for nct, did, name, how, role in df[["nct_id", "drug_id", "name", "resolved_by", "role"]].itertuples(index=False):
        a = agg.setdefault((nct, did), [set(), set(), how])
        a[0].add(role)
        a[1].add(name)
    starts = dict(con.sql("SELECT nct_id, start_date FROM studies").fetchall())
    recs = []
    for (nct, did), (rs, names, how) in agg.items():
        s = starts.get(nct)
        recs.append({"nct_id": nct, "drug_id": did, "names": "|".join(sorted(names)), "resolved_by": how,
                     "role": agg_role(rs), "start_date": s,
                     "approved_before_start": isinstance(s, str) and did in first_approval
                     and first_approval[did] < s})

    # Investigational drug(s) per trial, first tier that yields any:
    #   1 experimental-arm drugs not yet approved at trial start (else approved ones: a new-indication trial)
    #   2 unapproved drugs in any arm (investigational arm mislabelled ACTIVE_COMPARATOR, or given in all arms
    #     by a delayed-start / crossover design)
    #   3 approved drugs given in every arm (new formulation / indication with placebo add-on designs)
    by_trial: dict[str, list[dict]] = {}
    for r in recs:
        r["investigational"], r["inv_tier"] = False, None
        by_trial.setdefault(r["nct_id"], []).append(r)
    for rows_t in by_trial.values():
        exp = [r for r in rows_t if r["role"] == "experimental"] or [r for r in rows_t if r["role"] == "unknown"]
        tiers = [
            (1, [r for r in exp if not r["approved_before_start"]] or exp),
            (2, [r for r in rows_t if not r["approved_before_start"]]),
            (3, [r for r in rows_t if r["role"] == "background"]),
        ]
        for tier, picked in tiers:
            if picked:
                for r in picked:
                    r["investigational"], r["inv_tier"] = True, tier
                break
    td = pd.DataFrame(recs)
    print(f"[B] trial drugs: {len(td):,} trial-drug rows, {td['drug_id'].nunique():,} drugs, "
          f"{td['investigational'].sum():,} investigational, in {time.time() - t0:.0f}s")
    return td


# --------------------------------------------------------------------------------------------- C. group
SEED_PURPOSES = {"TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", None}


def _trial_conds(con) -> dict[str, prog.TrialCond]:
    leaves: dict[str, list[str]] = {}
    for nct, mid in con.sql("SELECT nct_id, mesh_id FROM condition_mesh ORDER BY nct_id, pos").fetchall():
        leaves.setdefault(nct, []).append(mid)
    texts: dict[str, list[str]] = {}
    for nct, c in con.sql("SELECT nct_id, condition FROM conditions ORDER BY nct_id, pos").fetchall():
        texts.setdefault(nct, []).append(prog.norm_text(c))
    anc: dict[str, set[str]] = {}
    for nct, mid in con.sql("SELECT nct_id, mesh_id FROM condition_ancestors").fetchall():
        anc.setdefault(nct, set()).add(mid)
    starts = dict(con.sql("SELECT nct_id, start_date FROM studies").fetchall())
    return {n: prog.TrialCond(n, starts.get(n), leaves.get(n, []), texts.get(n, []), anc.get(n, set()))
            for n in starts}


def stage_group(con, td: pd.DataFrame, idx: drugs.GsrsIndex, first_approval: dict[str, str]):
    t0 = time.time()
    conds = _trial_conds(con)
    trial_leaves = {n: c.leaves for n, c in conds.items() if c.leaves}
    anc_map = prog.leaf_ancestors(trial_leaves, {n: c.ancestors for n, c in conds.items()})
    generic = prog.generic_terms(anc_map)
    mesh_term = dict(con.sql("SELECT DISTINCT mesh_id, term FROM condition_mesh UNION "
                             "SELECT DISTINCT mesh_id, term FROM condition_ancestors").fetchall())
    raw_text = {}
    for nct, c in con.sql("SELECT nct_id, condition FROM conditions WHERE pos = 0").fetchall():
        raw_text[nct] = c
    st = con.sql("SELECT * FROM studies").df().set_index("nct_id")

    seeds_mask = (st["is_phase3"] & (st["lead_sponsor_class"] == "INDUSTRY")
                  & st["primary_purpose"].isin(SEED_PURPOSES))
    seed_ids = set(st.index[seeds_mask])
    inv = td[td["investigational"]]
    seeds = inv[inv["nct_id"].isin(seed_ids)]

    programmes, links = [], []
    seen_ids: set[str] = set()
    exp_rows = td[td["role"].isin(["experimental", "unknown"])]
    trials_by_drug = exp_rows.groupby("drug_id")["nct_id"].apply(set).to_dict()
    for drug_id, g in seeds.groupby("drug_id"):
        drug_seeds = set(g["nct_id"])
        tcs = [conds[n] for n in g["nct_id"]]
        for grp in prog.group_trials(tcs, generic):
            for ep_i, members in enumerate(prog.split_episodes({n: conds[n].start for n in grp.trials})):
                p_leaves = [m for n in members for m in conds[n].leaves]
                spec_leaves = [m for m in p_leaves if m not in generic]
                p_anc = set().union(*(conds[n].ancestors for n in members))
                p_texts = {conds[n].texts[0] for n in members if conds[n].texts and conds[n].texts[0]}
                prim = Counter(next((m for m in conds[n].leaves if m not in generic), None) for n in members)
                prim.pop(None, None)
                if prim:
                    ind_key, ind_label = prim.most_common(1)[0][0], mesh_term.get(prim.most_common(1)[0][0], "")
                else:
                    t = Counter(raw_text.get(n, "") for n in members).most_common(1)[0][0]
                    ind_key, ind_label = "TEXT:" + prog.norm_text(t), t
                starts = sorted(s for n in members if (s := conds[n].start))
                anchor = starts[0] if starts else None
                pid = prog.programme_id(drug_id, ind_key, anchor)
                if pid in seen_ids:     # separate groups with the same indication text and anchor (e.g. 'Healthy')
                    pid = prog.programme_id(drug_id, f"{ind_key}|{min(members)}", anchor)
                seen_ids.add(pid)
                # TA from each trial's primary condition only (an obesity co-condition must not make a psoriasis
                # programme metabolic); ancestors per leaf come from the learned leaf → ancestor map.
                prim_leaves = [next((m for m in conds[n].leaves if m not in generic), conds[n].leaves[0])
                               for n in members if conds[n].leaves]
                leaf_terms = [mesh_term.get(m, "") for m in prim_leaves]
                anc_terms = [mesh_term.get(a, "") for m in prim_leaves
                             for a in (anc_map.get(m) or p_anc)]
                sponsors = Counter(st.loc[n, "lead_sponsor"] for n in members)
                moieties = drug_id.removeprefix("UNII:").split("+") if drug_id.startswith("UNII:") else []
                programmes.append({
                    "programme_id": pid, "drug_id": drug_id,
                    "drug_name": " + ".join(idx.display.get(m, m) for m in moieties) if moieties
                    else drug_id.removeprefix("NAME:"),
                    "drug_resolution": "gsrs" if moieties else "name_only",
                    "is_combination": len(moieties) > 1,
                    "substance_class": idx.substance_class.get(moieties[0], "") if len(moieties) == 1 else "",
                    "indication_key": ind_key, "indication_label": ind_label,
                    "indication_mesh_ids": sorted(set(spec_leaves)) or sorted(set(p_leaves)),
                    "indication_specificity": grp.specificity,
                    "therapeutic_area": therapeutic_area(leaf_terms, anc_terms,
                                                         [raw_text.get(n, "") for n in members],
                                                         list(g.loc[g["nct_id"].isin(members), "names"])),
                    "inv_tier": int(g.loc[g["nct_id"].isin(members), "inv_tier"].max()),
                    "episode": ep_i, "anchor_p3_start": anchor, "n_p3_seed": len(members),
                    "lead_sponsor": sponsors.most_common(1)[0][0], "n_sponsors": len(sponsors),
                    "moiety_first_approval": first_approval.get(drug_id),
                })
                for n in members:
                    links.append({"programme_id": pid, "nct_id": n, "relation": "p3_seed",
                                  "match": grp.attached.get(n, "own")})
                # Link other trials of the drug: non-seed Phase 3s (exact indication), Phase 1/2 (by indication;
                # Phase 1 also drug-level), all starting before the episode's 7-year window closes.
                horizon = (str(int(anchor[:4]) + prog.EPISODE_YEARS) + anchor[4:]) if anchor else "9999"
                for n in trials_by_drug.get(drug_id, set()) - drug_seeds:
                    c = conds.get(n)
                    if c is None or (c.start and c.start > horizon):
                        continue
                    phase = st.loc[n, "max_phase"]
                    match = prog.indication_link(c, set(p_leaves), p_anc, p_texts, generic)
                    if st.loc[n, "is_phase3"]:
                        if match == "exact":
                            links.append({"programme_id": pid, "nct_id": n, "relation": "p3_linked", "match": match})
                    elif phase >= 2:
                        if match:
                            links.append({"programme_id": pid, "nct_id": n, "relation": "p2", "match": match})
                    elif phase > 0:
                        links.append({"programme_id": pid, "nct_id": n, "relation": "p1",
                                      "match": match or "drug_level"})
    pg = pd.DataFrame(programmes)
    pt = pd.DataFrame(links)
    pt = pt.merge(st[["start_date", "lead_sponsor", "max_phase", "status", "results_first_post_date"]],
                  left_on="nct_id", right_index=True, how="left")
    pt = pt.merge(pg[["programme_id", "lead_sponsor"]].rename(columns={"lead_sponsor": "prog_sponsor"}),
                  on="programme_id")
    pt["same_sponsor"] = pt["lead_sponsor"] == pt["prog_sponsor"]
    pt = pt.drop(columns="prog_sponsor")
    print(f"[C] {len(pg):,} programmes from {len(seeds['nct_id'].unique()):,} seed Phase 3 trials; "
          f"{len(pt):,} trial links; {len(generic):,} generic MeSH terms; {time.time() - t0:.0f}s")
    return pg, pt, generic, mesh_term


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    stage_flatten(args.rebuild)
    con = connect()
    idx = stage_gsrs(args.rebuild)
    fda = drugsfda.load(bulk.raw_files("drugsfda")[0])
    pm = drugsfda.product_moieties(fda["Products"], idx)
    approvals = drugsfda.original_approvals(fda, pm)
    first_approval = drugsfda.first_approval_by_drug(approvals)
    print(f"[B] Drugs@FDA: {len(approvals):,} NDA/BLA original approvals resolved to {approvals['drug_id'].nunique():,} "
          f"drug ids ({pm['drug_id'].notna().mean():.0%} of products resolved)")
    td = stage_trial_drugs(con, idx, first_approval)
    pg, pt, generic, _ = stage_group(con, td, idx, first_approval)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    td.to_parquet(PROCESSED / "trial_drugs.parquet", index=False)
    pg.to_parquet(PROCESSED / "programmes.parquet", index=False)
    pt.to_parquet(PROCESSED / "programme_trials.parquet", index=False)
    approvals.to_parquet(PROCESSED / "fda_original_approvals.parquet", index=False)
    names = (td.groupby("drug_id")["names"].apply(lambda s: "|".join(sorted(set("|".join(s).split("|")))))
               .rename("ctgov_names").reset_index())
    names.to_parquet(PROCESSED / "drug_names.parquet", index=False)
    summary = {
        "built_at": bulk.now_iso(), "n_programmes": len(pg), "n_links": len(pt),
        "by_specificity": pg["indication_specificity"].value_counts().to_dict(),
        "by_resolution": pg["drug_resolution"].value_counts().to_dict(),
        "by_ta": pg["therapeutic_area"].value_counts().to_dict(),
        "links_by_relation": pt["relation"].value_counts().to_dict(),
        "n_generic_mesh": len(generic),
    }
    (PROCESSED / "programmes_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
