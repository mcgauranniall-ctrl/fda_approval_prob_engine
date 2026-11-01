"""01d — FDA approval letters for efficacy supplements and new-use original applications (Drugs@FDA
ApplicationDocs, accessdata.fda.gov).

An efficacy supplement's letter says what it "provides for" (a new indication, population or dosing), which ties
a supplement to the indication it added. Script 04 uses this to date and attribute approvals of new uses (P6b).
Letters are post-decision documents: **labels only, never model inputs** (CLAUDE.md, D2).

  uv run python scripts/01d_approval_letters.py
Output: data/interim/approval_letters.parquet (ApplNo, SubmissionNo, letter date, URL, text, provides-for text)
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
from fdaengine.dataset import bulk, drugs, drugsfda
from fdaengine.dataset import labels as lab

INTERIM = config.DATA_DIR / "interim"
PROCESSED = config.DATA_DIR / "processed"


def pdf_text(content: bytes) -> str:
    import pymupdf
    try:
        with pymupdf.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            return " ".join(" ".join(p.get_text() for p in doc).split())
    except Exception:                                                     # noqa: BLE001 (corrupt / non-PDF)
        return ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    t = drugsfda.load(bulk.raw_files("drugsfda")[0])
    idx = drugs.build_index(pd.read_parquet(INTERIM / "gsrs_names.parquet").itertuples(index=False, name=None))
    pm = drugsfda.product_moieties(t["Products"], idx).dropna(subset=["drug_id"])
    prog_drugs = set(pd.read_parquet(PROCESSED / "programmes.parquet")["drug_id"])
    apps = set(pm[pm["drug_id"].isin(prog_drugs)]["ApplNo"])
    subs = t["Submissions"].merge(t["SubmissionClass_Lookup"], on="SubmissionClassCodeID", how="left")
    # Efficacy supplements, and original applications of the classes that bring an approved moiety to a new use
    # (new formulation, combination, indication: Mirvaso, Qnasl, Cymbalta's NDA 022516).
    eff = subs[subs["ApplNo"].isin(apps) & (subs["SubmissionStatus"] == "AP") & (
        ((subs["SubmissionType"] == "SUPPL") & (subs["SubmissionClassCode"] == "EFFICACY"))
        | ((subs["SubmissionType"] == "ORIG") & subs["SubmissionClassCode"].isin(lab.LIFECYCLE_ORIG_CLASSES)))]
    docs = t["ApplicationDocs"]
    letters = docs[docs["ApplicationDocsTypeID"].astype(str) == "1"]
    todo = eff.merge(letters, on=["ApplNo", "SubmissionType", "SubmissionNo"]).drop_duplicates("ApplicationDocsURL")
    print(f"[01d] {len(eff):,} approved efficacy supplements and new-use originals of programme drugs; "
          f"{len(todo):,} letters", flush=True)
    f = default_fetcher()
    rows, failed = [], 0
    for i, r in enumerate(todo.itertuples(index=False), 1):
        url = re.sub(r"^http://", "https://", r.ApplicationDocsURL)
        try:
            resp = f.get(url, allow_404=True)
        except FetchError:
            failed += 1
            continue
        text = pdf_text(resp.content) if resp.status == 200 else ""
        rows.append({"ApplNo": r.ApplNo, "SubmissionNo": str(r.SubmissionNo), "letter_date": str(r.ApplicationDocsDate)[:10],
                     "url": url, "status": resp.status, "n_chars": len(text), "text": text[:12000],
                     "provides_for": lab.letter_indication(text)})
        if i % 250 == 0:
            print(f"[01d] {i:,}/{len(todo):,} letters ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(INTERIM / "approval_letters.parquet", index=False)
    print(f"[01d] {len(df):,} letters saved; {int((df['n_chars'] > 200).sum()):,} with text, "
          f"{int((df['provides_for'] != '').sum()):,} with a provides-for passage; {failed} failed "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
