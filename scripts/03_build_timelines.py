"""03 — Event timelines per programme: EDGAR press releases (rule-based parser) + ClinicalTrials.gov dates +
Drugs@FDA actions + FDA CRL database. No LLM calls (D6).

Stages:
  index   EDGAR full-text search by pre-registered event phrase × year (all 8-K/6-K filers) → hit list
  fetch   download each hit's document text (resumable, ~9 req/s)
  build   match documents to programmes, parse events, assemble timelines, levels and decisions

  uv run python scripts/03_build_timelines.py index            # ~5 min
  uv run python scripts/03_build_timelines.py fetch            # hours; safe to interrupt and resume
  uv run python scripts/03_build_timelines.py build
Outputs: data/processed/events.parquet, levels.parquet, timeline_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.acquire import fetcher as fetcher_mod
from fdaengine.acquire.fetcher import Fetcher
from fdaengine.dataset import bulk, drugs, drugsfda, edgar_bulk
from fdaengine.dataset import timelines as tlb
from fdaengine.timeline import events as tl_events
from fdaengine.timeline import levels as tl_levels
from fdaengine.timeline import press_release_parser
from fdaengine.timeline.press_release_parser import ParsedEvent, _keyword_coverage

PROCESSED = config.DATA_DIR / "processed"


def stage_index(last_year: int) -> None:
    con = edgar_bulk.connect()
    fts = Fetcher(use_cache=False)
    t0 = time.time()
    n_q = 0
    for phrase, start, end in edgar_bulk.plan_index(con, fts, edgar_bulk.PHRASES, last_year):
        n = edgar_bulk.index_query(con, fts, phrase, start, end)
        n_q += 1
        if start.endswith("01-01") or n_q % 25 == 0:
            print(f"[index] {phrase!r} {start}..{end}: {n} hits", flush=True)
    (hits,) = con.execute("SELECT COUNT(*) FROM hits").fetchone()
    (todo,) = con.execute("SELECT COUNT(*) FROM hits h LEFT JOIN docs d USING (doc_id) "
                          "WHERE d.doc_id IS NULL").fetchone()
    print(f"[index] {n_q} queries in {time.time() - t0:.0f}s; {hits:,} unique documents, {todo:,} not yet fetched")


def stage_fetch(retry_errors: bool, limit: int | None) -> None:
    fetcher_mod._HOST_INTERVAL["www.sec.gov"] = 0.11          # ~9 req/s, under SEC's 10 req/s fair-access limit
    con = edgar_bulk.connect()
    if retry_errors:
        with con:
            con.execute("DELETE FROM docs WHERE error IS NOT NULL")
    todo = edgar_bulk.pending_docs(con)[:limit]
    t0 = time.time()
    last = [0.0]

    def progress(msg):
        if time.time() - last[0] > 60:
            rate = int(msg.split()[1].split("/")[0].replace(",", "")) / max(time.time() - t0, 1)
            print(f"{msg}  ({rate:.1f} docs/s)", flush=True)
            last[0] = time.time()

    edgar_bulk.fetch_docs(con, Fetcher(use_cache=False), todo, workers=10, progress=progress)
    (n_err,) = con.execute("SELECT COUNT(*) FROM docs WHERE error IS NOT NULL").fetchone()
    print(f"[fetch] {len(todo):,} documents in {time.time() - t0:.0f}s; {n_err:,} errors in total")
    _register_store()


def _register_store() -> None:
    """Record the SQLite store in the raw-data manifest (sha256 of the file as it stands)."""
    m = bulk.load_manifest()
    rel = edgar_bulk.DB_PATH.relative_to(bulk.RAW_DIR).as_posix()
    con = edgar_bulk.connect()
    (n_docs,) = con.execute("SELECT COUNT(*) FROM docs WHERE error IS NULL").fetchone()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    bulk.record(m, rel, "edgar_fts", edgar_bulk.FTS, bulk.sha256_file(edgar_bulk.DB_PATH),
                version=date.today().isoformat(), records=n_docs, phrases=edgar_bulk.PHRASES, forms=edgar_bulk.FORMS)
    bulk.save_manifest(m)


# ------------------------------------------------------------------------------------------------ build
INTERIM = config.DATA_DIR / "interim"
TIMELINE_TRIALS = {"p3_seed", "p3_linked", "p2"}
POST_ANCHOR_EVENTS = {"PHASE3_POSITIVE", "PHASE3_NEGATIVE"} | press_release_parser.REGULATORY
FILING_EVENTS = {"NDA_SUBMITTED", "NDA_ACCEPTED", "PRIORITY_REVIEW", "STANDARD_REVIEW", "PDUFA_DATE", "PDUFA_EXTENDED"}


def _load_inputs():
    pg = pd.read_parquet(PROCESSED / "programmes.parquet")
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    td = pd.read_parquet(PROCESSED / "trial_drugs.parquet")
    st = pd.read_parquet(INTERIM / "ctgov" / "studies.parquet")
    cond = pd.read_parquet(INTERIM / "ctgov" / "conditions.parquet")
    gs = pd.read_parquet(INTERIM / "gsrs_names.parquet")
    idx = drugs.build_index(gs.itertuples(index=False, name=None))
    return pg, pt, td, st, cond, idx


def _iter_docs(con, batch: int = 2000):
    cur = con.execute("SELECT h.doc_id, h.file_date, h.cik, h.company, d.text_z FROM hits h JOIN docs d "
                      f"USING (doc_id) WHERE d.error IS NULL AND {edgar_bulk.PRESS_RELEASE_TYPES} ORDER BY h.file_date")
    while rows := cur.fetchmany(batch):
        for doc_id, fd, cik, company, z in rows:
            yield doc_id, fd, cik, company, edgar_bulk.doc_text(z)


def _crl_events(key_to_drug, progs_by_drug, appl_drug: dict[str, str]) -> list[dict]:
    """FDA CRL database letters → CRL events (letter date = FDA action date; a label, never an input).

    A letter mentions many names (comparators, prior products), so it attaches to a programme only when the
    letter's application belongs to that drug in Drugs@FDA or its addressee is one of the programme's sponsors,
    and the letter text covers the programme's indication keywords.
    """
    src = bulk.raw_files("openfda_crl")[0]
    with zipfile.ZipFile(src) as z:
        letters = json.loads(z.read(z.namelist()[0]))["results"]
    out = []
    for r in letters:
        text = r.get("text") or ""
        try:
            d = datetime.strptime(r.get("letter_date", ""), "%m/%d/%Y").date().isoformat()
        except ValueError:
            continue
        appl = r.get("application_number") or ""
        appls = appl if isinstance(appl, list) else [appl]
        appl_drugs = {appl_drug.get(re.sub(r"\D", "", a).zfill(6)) for a in appls} - {None}
        for did in tlb.drugs_mentioned(text, key_to_drug):
            for p in progs_by_drug.get(did, []):
                owner = did in appl_drugs or any(tlb.same_company(r.get("company_name") or "", s) for s in p.sponsors)
                if not owner or _keyword_coverage(text, p.keywords) < 0.5:
                    continue
                out.append({"programme_id": p.programme_id, "date": d, "detail": ", ".join(map(str, appls)),
                            "source_id": r.get("file_name")})
    return out


def _fda_apps(idx) -> dict[str, list[tuple[str, str, list[dict]]]]:
    """drug id → [(ApplNo, ApplType, submissions)] for NDA/BLA applications whose products are exactly that drug."""
    t = drugsfda.load(bulk.raw_files("drugsfda")[0])
    pm = drugsfda.product_moieties(t["Products"], idx).dropna(subset=["drug_id"])
    apps = t["Applications"]
    apps = apps[apps["ApplType"].isin(["NDA", "BLA"])].set_index("ApplNo")["ApplType"]
    subs = t["Submissions"].merge(t["SubmissionClass_Lookup"], on="SubmissionClassCodeID", how="left")
    subs = subs.rename(columns={"SubmissionClassCode": "class_code", "SubmissionClassCodeDescription": "class_desc"})
    subs = subs.fillna("")
    by_app = {a: g.to_dict("records") for a, g in subs.groupby("ApplNo")}
    out: dict[str, list] = defaultdict(list)
    for appl, did in pm[["ApplNo", "drug_id"]].drop_duplicates().itertuples(index=False):
        if appl in apps.index:
            out[did].append((appl, apps[appl], by_app.get(appl, [])))
    return out


def _parse_all(con, key_to_drug, drug_names, progs_by_drug, workers: int) -> tuple[pd.DataFrame, int]:
    (n_docs,) = con.execute(f"SELECT COUNT(*) FROM hits h JOIN docs d USING (doc_id) WHERE d.error IS NULL "
                            f"AND {edgar_bulk.PRESS_RELEASE_TYPES}").fetchone()
    events, mentions = [], []
    with Pool(workers, initializer=tlb.init_worker, initargs=(key_to_drug, drug_names, progs_by_drug)) as pool:
        for i, (ev, me) in enumerate(pool.imap_unordered(tlb.process_doc, _iter_docs(con), chunksize=64), 1):
            events += ev
            mentions += me
            if i % 10_000 == 0:
                print(f"[build] parsed {i:,}/{n_docs:,} documents, {len(events):,} events", flush=True)
    ev = pd.DataFrame(events)
    me = pd.DataFrame(mentions, columns=["drug_id", "doc_id", "cik", "company", "file_date"])
    me["adsh"] = me["doc_id"].str.split("_").str[0]
    repeat = me.groupby(["drug_id", "cik"])["adsh"].nunique()
    if len(ev):
        ev["doc_drugs"] = ev["source_id"].map(me.groupby("doc_id")["drug_id"].nunique()).fillna(1).astype(int)
        ev["n_filings_by_filer"] = [repeat.get((d, c), 0) for d, c in ev[["drug_id", "cik"]].itertuples(index=False)]
        ev["filer_basis"] = ["sponsor" if s else "repeat" if n >= tlb.REPEAT_FILINGS else "other"
                             for s, n in ev[["filer_is_sponsor", "n_filings_by_filer"]].itertuples(index=False)]
    print(f"[build] {len(ev):,} parsed events; {me['doc_id'].nunique():,} of {n_docs:,} documents mention a "
          f"programme drug", flush=True)
    return ev, n_docs


def stage_build(as_of: date, workers: int) -> None:
    t0 = time.time()
    pg, pt, td, st, cond, idx = _load_inputs()
    ctgov_names = td.groupby("drug_id")["names"].apply(lambda s: sorted(set("|".join(s).split("|")))).to_dict()
    key_to_drug, _ = tlb.build_aliases(set(pg["drug_id"]), ctgov_names, idx)
    # Second pass: drop aliases that are ordinary words in this corpus (document frequency on a random sample).
    con = edgar_bulk.connect()
    sample = (edgar_bulk.doc_text(z) for (z,) in con.execute(
        f"SELECT d.text_z FROM hits h JOIN docs d USING (doc_id) WHERE d.error IS NULL "
        f"AND {edgar_bulk.PRESS_RELEASE_TYPES} ORDER BY random() LIMIT 5000"))
    common = tlb.common_alias_keys(sample, key_to_drug)
    key_to_drug, drug_names = tlb.build_aliases(set(pg["drug_id"]), ctgov_names, idx, common)
    print(f"[build] dropped {len(common):,} aliases that are common words in the corpus, e.g. "
          f"{sorted(common)[:12]}", flush=True)
    progs_by_drug = tlb.programme_infos(pg, pt, st, cond)
    print(f"[build] {len(key_to_drug):,} aliases for {len(drug_names):,} drugs ({time.time() - t0:.0f}s)", flush=True)

    ev, n_docs = _parse_all(edgar_bulk.connect(), key_to_drug, drug_names, progs_by_drug, workers)
    apps_by_drug = _fda_apps(idx)
    appl_drug = {appl: did for did, apps in apps_by_drug.items() for appl, _t, _s in apps}
    crl = pd.DataFrame(_crl_events(key_to_drug, progs_by_drug, appl_drug))

    # Assemble each programme's timeline with the live-case code. Phase 2 trials join only on an exact
    # indication match (they set the L1 cut-off).
    stc = st.set_index("nct_id")
    links = pt[pt["relation"].isin(TIMELINE_TRIALS) & ((pt["relation"] != "p2") | (pt["match"] == "exact"))]
    trials_by_prog = links.groupby("programme_id")["nct_id"].apply(list).to_dict()
    accepted = ev[ev["filer_basis"].isin(["sponsor", "repeat"])] if len(ev) else ev
    if len(accepted):
        # Partner and licensor releases (Ligand, Adamas) discuss many drugs: a Phase 3 or regulatory event from a
        # filer that is not the programme's sponsor needs the drug and the indication in the event sentence itself.
        strict = accepted["event_type"].isin(POST_ANCHOR_EVENTS) & (accepted["filer_basis"] != "sponsor")
        precise = (accepted["name_match"] == "sentence") & (accepted["indication_match"] == "sentence")
        accepted = accepted[~strict | precise]
        # Filing milestones set L3 and L5. In a document about several programme drugs (a pipeline round-up), a
        # milestone whose sentence does not name the drug is usually another product's ('The PDUFA date for the
        # submission is September 3, 2019'): gold set 2026-11-01, 20 of 33 such L3s wrong vs 18 of 80 with the
        # drug in the sentence. Requiring the name everywhere cost 44% of L5 rows, so single-drug releases keep
        # their context matches.
        accepted = accepted[~accepted["event_type"].isin(FILING_EVENTS) | (accepted["name_match"] == "sentence")
                            | (accepted["doc_drugs"] <= 1)]
        kw = {pi.programme_id: pi.keywords for ps in progs_by_drug.values() for pi in ps}
        kept = tlb.best_indication(accepted.to_dict("records"), kw, POST_ANCHOR_EVENTS)
        print(f"[build] {len(accepted) - len(kept):,} Phase 3 / regulatory events dropped from programmes whose "
              f"indication their sentence matches less well than another programme of the drug", flush=True)
        accepted = pd.DataFrame(kept, columns=accepted.columns)
    ev_by_prog = dict(iter(accepted.groupby("programme_id"))) if len(accepted) else {}
    crl_by_prog = dict(iter(crl.groupby("programme_id"))) if len(crl) else {}
    pe_fields = list(ParsedEvent.__dataclass_fields__)
    tl_rows, lv_rows, dec_rows = [], [], []
    for r in pg.itertuples(index=False):
        trials = [tlb.trial_summary({"nct_id": n, **stc.loc[n].to_dict()})
                  for n in trials_by_prog.get(r.programme_id, []) if n in stc.index]
        anchor = r.anchor_p3_start if isinstance(r.anchor_p3_start, str) else None
        # Phase 3 readouts and regulatory events dated before this programme's first Phase 3 belong to an earlier
        # programme of the drug (e.g. the previous episode); earlier-phase news and designations are kept.
        parsed = [ParsedEvent(**{k: e[k] for k in pe_fields})
                  for e in (ev_by_prog[r.programme_id].to_dict("records") if r.programme_id in ev_by_prog else [])
                  if not (anchor and e["event_date"] < anchor and e["event_type"] in POST_ANCHOR_EVENTS)]
        prior = (isinstance(r.moiety_first_approval, str) and isinstance(r.anchor_p3_start, str)
                 and r.moiety_first_approval < r.anchor_p3_start)
        apps = []
        for appl, atype, subs in apps_by_drug.get(r.drug_id, []):
            # For an already-approved moiety only a supplement can approve this programme's indication.
            keep = [s for s in subs if not (prior and s["SubmissionType"] == "ORIG")]
            apps.append(tlb.openfda_app(appl, atype, keep))
        tl = tl_events.build(trials, parsed, apps)
        if r.programme_id in crl_by_prog:
            for c in crl_by_prog[r.programme_id].to_dict("records"):
                if anchor and c["date"] < anchor:
                    continue
                tl.events.append(tl_events.Event("CRL", c["date"], "fda_crl", detail=c["detail"] or "",
                                                 source_id=c["source_id"]))
        tl_rows += [{"programme_id": r.programme_id, **e} for e in tl.to_list()]
        lv_rows += [{"programme_id": r.programme_id, **lv.to_dict()} for lv in tl_levels.compute(tl, as_of)]
        outcome, decided = tl_levels.decision(tl)
        dec_rows.append({"programme_id": r.programme_id, "first_decision": outcome,
                         "first_decision_date": decided.isoformat() if decided else None})

    tlf, lv, dec = pd.DataFrame(tl_rows), pd.DataFrame(lv_rows), pd.DataFrame(dec_rows)
    ev.to_parquet(PROCESSED / "parsed_events_all.parquet", index=False)
    tlf.to_parquet(PROCESSED / "events.parquet", index=False)
    lv.to_parquet(PROCESSED / "levels.parquet", index=False)
    dec.to_parquet(PROCESSED / "decisions.parquet", index=False)
    summary = {
        "built_at": bulk.now_iso(), "as_of": as_of.isoformat(), "n_programmes": len(pg), "edgar_docs": int(n_docs),
        "parsed_events": len(ev),
        "parsed_events_by_filer_basis": ev["filer_basis"].value_counts().to_dict() if len(ev) else {},
        "crl_db_events": len(crl),
        "timeline_events_by_type": tlf["event_type"].value_counts().to_dict(),
        "levels_reached": lv[lv["reached"]].groupby("level").size().to_dict(),
        "levels_estimated": lv[lv["reached"] & lv["estimated"]].groupby("level").size().to_dict(),
        "first_decision": dec["first_decision"].fillna("none").value_counts().to_dict(),
    }
    (PROCESSED / "timeline_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"[build] done in {time.time() - t0:.0f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["index", "fetch", "build"])
    ap.add_argument("--last-year", type=int, default=date.today().year)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--limit", type=int, help="fetch at most N documents (testing)")
    ap.add_argument("--as-of", default=date.today().isoformat(), help="snapshot date for level reach / censoring")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    if args.stage == "index":
        stage_index(args.last_year)
    elif args.stage == "fetch":
        stage_fetch(args.retry_errors, args.limit)
    else:
        stage_build(date.fromisoformat(args.as_of), args.workers)


if __name__ == "__main__":
    main()
