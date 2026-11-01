"""As-of context features (PLAN.md P6d), one definition for training rows (script 05) and live cases.

  therapeutic_area     each Phase 3 trial's primary MeSH condition (first non-generic leaf) and its ancestors, as in
                       script 02, through `indication.therapeutic_area`
  modality             'biologic' if the drug's registered intervention type is BIOLOGICAL/GENETIC, else
                       'small_molecule' for DRUG
  moiety_approved      the active ingredient (GSRS-resolved) had an NDA/BLA original approval, or a CBER licence, on
                       or before the cut-off
  sponsor_prior_approvals / sponsor_prior_crls
                       the sponsor's NDA/BLA original approvals and CBER licences before the cut-off; complete response
                       letters it announced in press releases before the cut-off (never the 2025 FDA CRL database,
                       which is a label source only)
  is_combination       more than one resolved active ingredient

Live cases read the training data's reference tables (data/processed, data/interim) once per process; without them
the context is simply missing (the models treat missing inputs as unknown).
"""

from __future__ import annotations

import bisect
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from fdaengine import config
from fdaengine.dataset import drugs
from fdaengine.dataset import timelines as tlb

PROCESSED = config.DATA_DIR / "processed"
INTERIM = config.DATA_DIR / "interim"
LABEL_ONLY_SOURCES = {"fda_crl"}               # published after the decisions they describe


# ------------------------------------------------------------------------------------------------ shared definitions
class CompanyDates:
    """Company name → sorted dates (filings or approvals), matched with `timelines.same_company`."""

    def __init__(self, pairs):
        self.by_first: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
        grouped: dict[str, list[str]] = defaultdict(list)
        for name, d in pairs:
            if name and isinstance(d, str) and d:
                grouped[name].append(d)
        for name, ds in grouped.items():
            k = tlb.company_key(name)
            if k:
                self.by_first[k.split()[0]].append((name, sorted(ds)))
        self._cache: dict[str, list[str]] = {}

    def dates(self, sponsor: str) -> list[str]:
        if sponsor not in self._cache:
            k = tlb.company_key(sponsor or "")
            cands = self.by_first.get(k.split()[0], []) if k else []
            self._cache[sponsor] = sorted(d for name, ds in cands if tlb.same_company(name, sponsor) for d in ds)
        return self._cache[sponsor]

    def count_before(self, sponsor: str, cutoff: str) -> int:
        return bisect.bisect_left(self.dates(sponsor), cutoff)

    def first(self, sponsor: str) -> str | None:
        ds = self.dates(sponsor)
        return ds[0] if ds else None


def sponsor_approval_pairs(approvals, blas) -> list[tuple[str, str]]:
    """(sponsor, date) of every NDA/BLA original approval (Drugs@FDA) and CBER licence (Purple Book)."""
    return [*approvals[["SponsorName", "approval_date"]].itertuples(index=False),
            *((b.applicant, b.first_licensed) for b in blas)]


def sponsor_crl_pairs(events, programmes) -> list[tuple[str, str]]:
    """(lead sponsor, date) of press-release CRLs, one per drug and date."""
    crl = events[(events["event_type"] == "CRL") & ~events["source"].isin(LABEL_ONLY_SOURCES)]
    crl = crl.join(programmes[["drug_id", "lead_sponsor"]], on="programme_id").drop_duplicates(["drug_id", "date"])
    return list(crl[["lead_sponsor", "date"]].itertuples(index=False))


def trial_types(interventions, keys: set[str]) -> set[str]:
    """Registered intervention types of the drug in one trial: rows naming it, else the experimental arms' rows.
    `interventions`: dicts with 'name', 'type', 'arm_types'."""
    rows = list(interventions)
    hit = [r for r in rows if drugs.key(r["name"]) in keys]
    exp = [r for r in rows if isinstance(r.get("arm_types"), str) and "EXPERIMENTAL" in r["arm_types"]]
    return {r["type"] for r in (hit or exp)}


def modality(types: set[str]) -> str | None:
    if types & {"BIOLOGICAL", "GENETIC"}:
        return "biologic"
    return "small_molecule" if "DRUG" in types else None


def primary_leaf(leaves: list[str], generic: set[str]) -> str | None:
    """A trial's primary condition: its first MeSH leaf that is not a generic term."""
    return next((m for m in leaves if m not in generic), leaves[0]) if leaves else None


# ------------------------------------------------------------------------------------------------ live cases
def approvals_by_drug(approvals) -> dict[str, list[tuple[str, str]]]:
    """drug id → [(approval date, application)], single moieties also listing combinations that contain them (as
    `drugsfda.first_approval_by_drug`); undated pre-1982 approvals count as 1900-01-01."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for did, d, appl in approvals[["drug_id", "approval_date", "ApplNo"]].itertuples(index=False):
        if not isinstance(did, str):
            continue
        keys = {did} | {f"UNII:{m}" for m in did.removeprefix("UNII:").split("+")}
        for k in keys:
            out[k].append((d or "1900-01-01", str(appl).zfill(6)))
    return dict(out)


def own_applications(meta: dict) -> set[str]:
    """The case's own NDA/BLA numbers ('NDA221276' → '221276')."""
    return {"".join(c for c in a if c.isdigit()).zfill(6) for a in meta.get("applications") or [] if any(
        c.isdigit() for c in a)}


@dataclass
class Reference:
    idx: drugs.GsrsIndex
    drug_approvals: dict[str, list[tuple[str, str]]]
    blas: list
    bla_drug: dict[str, str]
    approvals: CompanyDates
    crls: CompanyDates
    anc_map: dict[str, set[str]]
    generic: set[str]
    mesh_term: dict[str, str]


def _mesh_maps() -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """Leaf → ancestors and generic terms learned from every flattened trial (script 02's definitions), cached."""
    import pandas as pd

    from fdaengine.dataset import programmes as prog
    cache = INTERIM / "ctgov" / "mesh_maps.pkl"
    cm, ca = INTERIM / "ctgov" / "condition_mesh.parquet", INTERIM / "ctgov" / "condition_ancestors.parquet"
    if cache.exists() and cache.stat().st_mtime >= max(cm.stat().st_mtime, ca.stat().st_mtime):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    mesh = pd.read_parquet(cm).sort_values(["nct_id", "pos"])
    anc = pd.read_parquet(ca)
    leaves = mesh.groupby("nct_id")["mesh_id"].apply(list).to_dict()
    ancestors = anc.groupby("nct_id")["mesh_id"].apply(set).to_dict()
    anc_map = dict(prog.leaf_ancestors(leaves, ancestors))
    terms = dict(zip(mesh["mesh_id"], mesh["term"])) | dict(zip(anc["mesh_id"], anc["term"]))
    out = (anc_map, prog.generic_terms(anc_map), terms)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


@lru_cache(maxsize=1)
def reference() -> Reference | None:
    """Reference tables for live cases; None when the training data are not on this machine."""
    import pandas as pd

    from fdaengine.dataset import bulk, purplebook
    try:
        idx = drugs.build_index(pd.read_parquet(INTERIM / "gsrs_names.parquet").itertuples(index=False, name=None))
        approvals = pd.read_parquet(PROCESSED / "fda_original_approvals.parquet")
        blas = purplebook.cber_blas(purplebook.load(bulk.raw_files("purplebook")[0]))
        events = pd.read_parquet(PROCESSED / "events.parquet")
        pg = pd.read_parquet(PROCESSED / "programmes.parquet").set_index("programme_id")
        anc_map, generic, terms = _mesh_maps()
    except (OSError, IndexError, KeyError, ValueError):
        return None
    return Reference(idx, approvals_by_drug(approvals), blas, purplebook.resolved_drug_ids(blas, idx),
                     CompanyDates(sponsor_approval_pairs(approvals, blas)), CompanyDates(sponsor_crl_pairs(events, pg)),
                     anc_map, generic, terms)


def resolve_drug(names: list[str], idx: drugs.GsrsIndex) -> str | None:
    """The case's drug id: the first of its names (generic or code) that GSRS resolves."""
    for n in names:
        found, _ = drugs.resolve_string(n, idx)
        if found:
            return drugs.drug_id(found)
    return None


def live_context(meta: dict, studies: list[dict], cutoff: str, ref: Reference | None = None) -> dict:
    """{feature: (value, detail)} for a live case at `cutoff`. `studies`: raw ClinicalTrials.gov records of the
    case's trials registered by the cut-off (conditions and intervention types are registration facts)."""
    from fdaengine.dataset import ctgov_flat, purplebook
    from fdaengine.features.indication import therapeutic_area
    ref = ref or reference()
    if ref is None:
        return {}
    out: dict[str, tuple] = {}
    names = [n for n in meta.get("names") or [meta.get("drug", "")] if n]
    sponsor = meta.get("sponsor") or ""
    did = resolve_drug(names, ref.idx)
    own = own_applications(meta)
    if did:
        out["is_combination"] = ("+" in did, f"resolved to {did}")
        cber = purplebook.match(did, names, lambda a: tlb.same_company(a, sponsor), ref.blas, ref.bla_drug)
        acts = [*ref.drug_approvals.get(did, []), *((b.first_licensed, b.bla) for b in cber)]
        # Another use: the case's own applications are its outcome, never "already approved" (training cut-offs
        # always precede the programme's own decision).
        prior = min((d for d, appl in acts if d <= cutoff and appl not in own), default=None)
        out["moiety_approved"] = (prior is not None, f"first FDA approval of {did} for another use {prior}" if prior
                                  else f"no FDA approval of {did} for another use before the cut-off")
        mine = min((d for d, appl in acts if d <= cutoff and appl in own), default=None)
        if mine:
            # Not a model input: tells the report that the outcome was already public at the cut-off.
            out["own_application_approved"] = (mine, f"the case's own application was approved on {mine}")
    if sponsor:
        out["sponsor_prior_approvals"] = (ref.approvals.count_before(sponsor, cutoff),
                                          f"{sponsor}: NDA/BLA original approvals and CBER licences before {cutoff}")
        out["sponsor_prior_crls"] = (ref.crls.count_before(sponsor, cutoff),
                                     f"{sponsor}: complete response letters announced before {cutoff}")
    flat = [ctgov_flat.flatten(s) for s in studies]
    p3 = [f for f in flat if f["studies"][0]["is_phase3"]]
    keys = {drugs.key(n) for n in names}
    mod = modality(set().union(*(trial_types(f["interventions"], keys) for f in p3))) if p3 else None
    if mod:
        out["modality"] = (mod, "registered intervention type in the Phase 3 trials")
    leaves, ancs, texts = [], [], []
    for f in p3:
        leaf = primary_leaf([m["mesh_id"] for m in sorted(f["condition_mesh"], key=lambda m: m["pos"])], ref.generic)
        if leaf:
            leaves.append(ref.mesh_term.get(leaf, ""))
            ancs += [ref.mesh_term.get(a, "") for a in (ref.anc_map.get(leaf) or
                                                        {a["mesh_id"] for a in f["condition_ancestors"]})]
        texts += [c["condition"] for c in f["conditions"][:1]]
    if p3:
        ta = therapeutic_area(leaves, ancs, texts, [i["name"] for f in p3 for i in f["interventions"]])
        out["therapeutic_area"] = (ta, "primary MeSH condition of the Phase 3 trials: " + "; ".join(sorted(set(leaves)))[:200])
    return out
