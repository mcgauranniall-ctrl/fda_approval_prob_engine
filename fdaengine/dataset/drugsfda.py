"""Drugs@FDA bulk tables (https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files).

Used for outcome labels and decision dates, and for "was this moiety already approved before date d" (an input only
when d ≤ the cut-off). Never an input about the decision being predicted.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from fdaengine.dataset import drugs

_BIOSIMILAR_SUFFIX = re.compile(r"-[A-Z]{4}$", re.IGNORECASE)    # FDA's four-letter biologic suffix
TABLES = ("Applications", "Products", "Submissions", "SubmissionClass_Lookup", "MarketingStatus",
          "MarketingStatus_Lookup", "ApplicationDocs", "ApplicationsDocsType_Lookup")


def load(zip_path: Path) -> dict[str, pd.DataFrame]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for t in TABLES:
            raw = z.read(f"{t}.txt").decode("latin-1")
            ncol = raw.split("\n", 1)[0].count("\t") + 1
            # A few free-text fields contain stray tabs; fold the overflow back into the last column.
            out[t] = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str, keep_default_na=False, quoting=3,
                                 engine="python",
                                 on_bad_lines=lambda f, n=ncol: f[:n - 1] + [" ".join(f[n - 1:])])
    for df in out.values():
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].str.strip()
    subs = out["Submissions"]
    subs["status_date"] = subs["SubmissionStatusDate"].str[:10].replace("", None)
    return out


def product_moieties(products: pd.DataFrame, idx: drugs.GsrsIndex) -> pd.DataFrame:
    """One row per product: ApplNo, ProductNo, moiety set as a drug id (None if any ingredient is unresolved)."""
    rows = []
    for appl, prod, ingr, name in products[["ApplNo", "ProductNo", "ActiveIngredient", "DrugName"]].itertuples(
            index=False):
        # ';' separates ingredients; co-packaged products also use ',' ('ELEXACAFTOR,TEZACAFTOR,IVACAFTOR; IVACAFTOR').
        parts = sorted({p.strip() for p in re.split(r"[;,]", ingr) if p.strip()})
        found = []
        for p in parts:
            p = _BIOSIMILAR_SUFFIX.sub("", p)                   # 'ADUCANUMAB-AVWA' → 'ADUCANUMAB'
            hit = idx.lookup(p)
            if len(hit) != 1:
                hit = idx.lookup(drugs.strip_dose(p))
            if len(hit) == 1:
                found.append(next(iter(hit)))
        did = drugs.drug_id(frozenset(found)) if found and len(found) == len(parts) else None
        rows.append({"ApplNo": appl, "ProductNo": prod, "drug_id": did, "ingredients": ingr, "drug_name": name,
                     "n_ingredients": len(parts)})
    return pd.DataFrame(rows)


def original_approvals(t: dict[str, pd.DataFrame], prod_moieties: pd.DataFrame) -> pd.DataFrame:
    """NDA/BLA original approvals: ApplNo, ApplType, drug_id, approval date, Rx flag, review priority."""
    apps = t["Applications"]
    subs = t["Submissions"]
    orig = subs[(subs["SubmissionType"] == "ORIG") & (subs["SubmissionStatus"] == "AP")]
    orig = orig.groupby("ApplNo", as_index=False).agg(approval_date=("status_date", "min"),
                                                       review_priority=("ReviewPriority", "first"))
    ms = t["MarketingStatus"].merge(t["MarketingStatus_Lookup"], on="MarketingStatusID")
    rx = ms.groupby("ApplNo")["MarketingStatusDescription"].agg(lambda s: "Prescription" in set(s)).rename("is_rx")
    pm = prod_moieties.dropna(subset=["drug_id"]).drop_duplicates(["ApplNo", "drug_id"])
    out = (apps[apps["ApplType"].isin(["NDA", "BLA"])][["ApplNo", "ApplType", "SponsorName"]]
           .merge(orig, on="ApplNo").merge(pm[["ApplNo", "drug_id"]], on="ApplNo")
           .merge(rx, left_on="ApplNo", right_index=True, how="left"))
    out["is_rx"] = out["is_rx"].fillna(False).astype(bool)
    return out


def first_approval_by_drug(approvals: pd.DataFrame) -> dict[str, str]:
    """drug id (single moiety or exact combination) → earliest NDA/BLA original approval date ('' if undated).

    Single moieties also inherit the date of any combination product that contains them.
    """
    first: dict[str, str] = {}

    def put(k, d):
        d = d or "1900-01-01"                   # pre-1982 approvals often have no date in Drugs@FDA
        if k not in first or d < first[k]:
            first[k] = d

    for did, d in approvals[["drug_id", "approval_date"]].itertuples(index=False):
        put(did, d)
        for m in did.removeprefix("UNII:").split("+"):
            put(f"UNII:{m}", d)
    return first
