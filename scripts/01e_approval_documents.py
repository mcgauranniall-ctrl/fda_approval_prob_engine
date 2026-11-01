"""01e — The FDA approval letter and the label as approved, for the action that approved each programme (P6c).

For every approved programme in the cohort (script 04's `approval_appl` and `approval_date`), the approving Drugs@FDA
submission is found and its ApplicationDocs letter (type 1) and label (type 2) are downloaded from
accessdata.fda.gov (robots.txt allows /drugsatfda_docs/appletter/ and /label/; /drugsatfda_docs/rems/ is disallowed
and never fetched). Script 11 reads accelerated approval and REMS from the letter and the boxed warning and
indication wording from the label as approved (the openFDA label is today's version). Also saved: FDA's list of
CDER accelerated approvals (fda.gov/media/151146), and the registered eligibility criteria of the approved
programmes' seed Phase 3 trials (from the raw ClinicalTrials.gov store), for the "narrower label" comparison.

These are post-decision documents: **training labels only, never model inputs** (CLAUDE.md, D2).

  uv run python scripts/01e_approval_documents.py
Outputs: data/interim/approval_documents.parquet (one row per approving submission: letter and label text),
approving_submissions.parquet, accelerated_approvals.parquet, p3_eligibility.parquet
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.acquire.fetcher import FetchError, default_fetcher
from fdaengine.dataset import bulk, drugsfda
from fdaengine.dataset import label_outcomes as lo

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"
AA_LIST_URL = "https://www.fda.gov/media/151146/download?attachment"
LETTER_CHARS = 80_000
LABEL_CHARS = 150_000
KEY = ["ApplNo", "SubmissionType", "SubmissionNo"]


def pdf_text(content: bytes) -> str:
    import pymupdf
    try:
        with pymupdf.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            return " ".join(" ".join(p.get_text() for p in doc).split())
    except Exception:                                                     # noqa: BLE001 (corrupt / non-PDF)
        return ""


def approving_submissions(lab: pd.DataFrame, subs: pd.DataFrame) -> pd.DataFrame:
    """Programme → the approved submission on its approval application dated on its approval date."""
    ap = lab[lab["in_cohort"] & lab["approval_date"].notna() & lab["approval_appl"].notna()]
    s = subs[subs["SubmissionStatus"] == "AP"].assign(approval_date=lambda d: d["status_date"].astype(str).str[:10])
    m = ap[["programme_id", "approval_appl", "approval_date"]].merge(
        s[KEY + ["approval_date"]], left_on=["approval_appl", "approval_date"], right_on=["ApplNo", "approval_date"])
    # Two submissions approved on one day (Mylotarg's Orig1/Orig2): the original wins, then the lowest number.
    m = m.assign(_o=(m["SubmissionType"] != "ORIG").astype(int), _n=pd.to_numeric(m["SubmissionNo"], errors="coerce"))
    return m.sort_values(["_o", "_n"]).drop_duplicates("programme_id")[["programme_id"] + KEY]


def pick(docs: pd.DataFrame, type_id: str, prefer: str) -> pd.DataFrame:
    d = docs[docs["ApplicationDocsTypeID"].astype(str) == type_id].copy()
    d["_p"] = ~d["ApplicationDocsURL"].str.contains(prefer, case=False, na=False)
    d = d.sort_values(["_p", "ApplicationDocsDate"]).drop_duplicates(KEY)
    return d[KEY + ["ApplicationDocsURL", "ApplicationDocsDate"]]


def accelerated_list() -> pd.DataFrame:
    """FDA's list of CDER accelerated approvals (fda.gov/drugs/nda-and-bla-approvals/accelerated-approvals)."""
    import pymupdf
    resp = default_fetcher().get(AA_LIST_URL)
    with pymupdf.open(stream=io.BytesIO(resp.content), filetype="pdf") as doc:
        text = "\n".join(p.get_text() for p in doc)
    df = pd.DataFrame(lo.parse_accelerated_list(text))
    df.to_parquet(INTERIM / "accelerated_approvals.parquet", index=False)
    return df


def p3_eligibility(lab: pd.DataFrame) -> pd.DataFrame:
    """Registered title, conditions, eligibility criteria and minimum age of the seed Phase 3 trials of approved
    programmes (the trial population a label is compared with). Registry versions are today's."""
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    ids = set(lab.loc[lab["in_cohort"] & lab["approval_date"].notna(), "programme_id"])
    want = set(pt.loc[pt["programme_id"].isin(ids) & (pt["relation"] == "p3_seed"), "nct_id"])
    rows = []
    for path in bulk.raw_files("ctgov"):
        for s in bulk.iter_jsonl_gz(path):
            p = s.get("protocolSection", {})
            nct = p.get("identificationModule", {}).get("nctId")
            if nct not in want:
                continue
            e = p.get("eligibilityModule", {})
            rows.append({"nct_id": nct, "official_title": p["identificationModule"].get("officialTitle") or
                         p["identificationModule"].get("briefTitle", ""),
                         "conditions": "; ".join(p.get("conditionsModule", {}).get("conditions", [])),
                         "eligibility": (e.get("eligibilityCriteria") or "")[:20000],
                         "minimum_age": e.get("minimumAge")})
    df = pd.DataFrame(rows).drop_duplicates("nct_id")
    df.to_parquet(INTERIM / "p3_eligibility.parquet", index=False)
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    t = drugsfda.load(bulk.raw_files("drugsfda")[0])
    lab = pd.read_parquet(PROCESSED / "labels.parquet")
    aa = accelerated_list()
    el = p3_eligibility(lab)
    print(f"[01e] {len(aa)} accelerated approvals on FDA's list; eligibility for {len(el):,} Phase 3 trials", flush=True)
    prog = approving_submissions(lab, t["Submissions"])
    acts = prog[KEY].drop_duplicates()
    letters = pick(t["ApplicationDocs"], "1", "ltr").rename(columns={"ApplicationDocsURL": "letter_url",
                                                                      "ApplicationDocsDate": "letter_date"})
    labels = pick(t["ApplicationDocs"], "2", "lbl").rename(columns={"ApplicationDocsURL": "label_url",
                                                                     "ApplicationDocsDate": "label_date"})
    acts = acts.merge(letters, on=KEY, how="left").merge(labels, on=KEY, how="left")
    print(f"[01e] {len(prog):,} approved programmes → {len(acts):,} approving submissions; letter on file "
          f"{acts['letter_url'].notna().mean():.0%}, label {acts['label_url'].notna().mean():.0%}", flush=True)
    f = default_fetcher()
    rows, failed = [], 0
    for i, r in enumerate(acts.itertuples(index=False), 1):
        rec = {k: getattr(r, k) for k in KEY}
        for kind, url, n in (("letter", r.letter_url, LETTER_CHARS), ("label", r.label_url, LABEL_CHARS)):
            rec[f"{kind}_url"], rec[f"{kind}_date"], rec[f"{kind}_text"] = None, None, ""
            if not isinstance(url, str):
                continue
            url = re.sub(r"^http://", "https://", url)
            try:
                resp = f.get(url, allow_404=True)
            except FetchError:
                failed += 1
                continue
            rec[f"{kind}_url"] = url
            rec[f"{kind}_date"] = str(getattr(r, f"{kind}_date"))[:10]
            rec[f"{kind}_text"] = pdf_text(resp.content)[:n] if resp.status == 200 else ""
        rows.append(rec)
        if i % 100 == 0:
            print(f"[01e] {i:,}/{len(acts):,} submissions ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(INTERIM / "approval_documents.parquet", index=False)
    prog.to_parquet(INTERIM / "approving_submissions.parquet", index=False)
    print(f"[01e] {len(df):,} submissions saved; letter text {(df['letter_text'].str.len() > 500).mean():.0%}, "
          f"label text {(df['label_text'].str.len() > 500).mean():.0%}; {failed} failed ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
