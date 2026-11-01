"""Drug identity for the training set: ClinicalTrials.gov intervention strings → active-moiety UNII.

Names come from GSRS (openFDA `other/substance`: INNs, USANs, development codes, salts → ACTIVE MOIETY) and from
ClinicalTrials.gov itself (an intervention's `otherNames` are aliases of the same product). Unresolved names keep a
`NAME:<key>` id so code-named drugs that GSRS doesn't know still group across trials.

GSRS also lists brand names. They are used only to *recognise* an intervention string, never as features, and
the programme's display name prefers INN/code names (brand names only exist after approval).
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_TAG = re.compile(r"\s*\[[^\]]*\]\s*")
_NONALNUM = re.compile(r"[^A-Z0-9]")

NON_DRUG = re.compile(
    r"placebo|\bsham\b|vehicle|\bsaline\b|standard (of )?care|\bsoc\b|best supportive|supportive care|usual care|"
    r"no (treatment|intervention|drug)|observation|watchful|\bcontrol\b|matching|dummy|\bdiet\b|exercise|counsel|"
    r"physician'?s choice|investigator'?s choice|treatment of choice|background therapy|standard therapy|"
    r"\bsurgery\b|radiotherapy|radiation|\bbsc\b|"
    r"study (drug|medication|treatment)|investigational|active (drug|treatment)|test (drug|product|article)|"
    r"\bimp\b|experimental|comparator|reference product",
    re.IGNORECASE)
_DOSE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|iu|units?|u|%|mg/kg|mg/m2|mg/ml|μg|nmol|mmol|ng|kg)\b(/\s*(day|kg|m2|dose))?|"
    r"\b(tablets?|capsules?|injections?|injectable|oral|solution|suspension|infusion|intravenous|subcutaneous|"
    r"intramuscular|topical|ophthalmic|cream|gel|ointment|patch|inhalation|inhaler|nasal|spray|powder|"
    r"extended[- ]release|modified[- ]release|delayed[- ]release|prolonged[- ]release|sustained[- ]release|"
    r"film[- ]coated|xr|er|sr|ir|iv|sc|im|po|dose|doses|dosing|low|high|medium|mid|fixed|once|twice|daily|"
    r"weekly|monthly|qd|bid|tid|q\d+w|regimen|arm|group|cohort|formulation|treatment|therapy|drug|study|"
    r"single|multiple|ascending|loading|maintenance|for|of|and|with|the|in|alone|only|plus)\b|"
    # Standalone numbers only: digits joined to letters or a hyphen are a code name ('TAK-861', 'AR101'), and
    # stripping them collapsed codes to a sponsor prefix that resolved to another substance ('TAK' -> CDK9).
    r"(?<![A-Za-z0-9-])\d+(\.\d+)?(?![A-Za-z0-9-])",
    re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:\+|/|,|;|\band\b|\bplus\b|\bwith\b|&|\bin combination with\b|\bfollowed by\b)\s*",
                    re.IGNORECASE)
_PAREN = re.compile(r"\(([^()]*)\)")
GENERIC_WORDS = {"INSULIN", "VACCINE", "VITAMIN", "ANTIBODY", "PROTEIN", "HORMONE", "CHEMOTHERAPY", "ANTIBIOTIC",
                 "STEROID", "OPIOID", "STATIN", "ANALGESIC", "CORTICOSTEROID", "IMMUNOGLOBULIN", "INTERFERON"}
# Counter-ions and hydrate words: never the drug on their own ('ulipristal acetate' is ulipristal, not acetate).
SALT_WORDS = {"ACETATE", "HYDROCHLORIDE", "HCL", "SODIUM", "POTASSIUM", "PHOSPHATE", "SULFATE", "SULPHATE",
              "MALEATE", "MESYLATE", "MESILATE", "TARTRATE", "BITARTRATE", "CITRATE", "BESYLATE", "BESILATE",
              "FUMARATE", "SUCCINATE", "BROMIDE", "CHLORIDE", "CALCIUM", "MAGNESIUM", "HYDROBROMIDE", "TOSYLATE",
              "MONOHYDRATE", "DIHYDRATE", "HYDRATE", "HYCLATE", "DIPROPIONATE", "PROPIONATE", "FUROATE",
              "VALERATE", "DECANOATE", "PALMITATE", "LACTATE", "GLUCONATE", "NITRATE", "OXALATE", "MALATE",
              "DISODIUM", "ANHYDROUS", "BASE", "FREE"}
VEHICLES = ("SODIUM CHLORIDE", "ISOTONIC SODIUM CHLORIDE SOLUTION", "WATER", "STERILE WATER", "WATER FOR INJECTION",
            "DEXTROSE", "LACTATED RINGER'S", "RINGER'S SOLUTION", "CHLORIDE ION", "SODIUM CATION")


def key(name: str) -> str:
    """Compact match key: tags like '[INN]' removed, then only A-Z/0-9 (so 'VX-548' == 'vx 548' == 'VX548')."""
    return _NONALNUM.sub("", _TAG.sub(" ", name).upper())


def strip_dose(name: str) -> str:
    return re.sub(r"\s+", " ", _DOSE.sub(" ", _PAREN.sub(" ", name))).strip(" -,.")


def candidate_strings(name: str) -> list[list[str]]:
    """Ways to read an intervention string, most literal first. Each entry is a list of parts (≥2 = combination)."""
    out = [[name]]
    if m := re.match(r"^(.*[a-z]{3,})-[a-z]{4}$", name.strip(), re.IGNORECASE):   # biologic suffix '-atto'
        out.append([m.group(1)])
    stripped = strip_dose(name)
    if stripped and stripped != name:
        out.append([stripped])
    out += [[p] for p in _PAREN.findall(name) if p.strip()]
    parts = [strip_dose(p) for p in _SPLIT.split(_PAREN.sub(" ", name)) if p.strip()]
    parts = [p for p in parts if p and not NON_DRUG.search(p)]
    if len(parts) >= 2:
        out.append(parts)
    words = (stripped or name).split()
    for k in range(len(words) - 1, 0, -1):                # 'Ulipristal acetate Esmya' → 'Ulipristal acetate'
        prefix = " ".join(words[:k])
        if len(key(prefix)) >= 5:
            out.append([prefix])
    return out


@dataclass
class GsrsIndex:
    by_key: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))   # name key → moiety UNIIs
    display: dict[str, str] = field(default_factory=dict)                          # moiety → INN-style name
    names: dict[str, set[tuple[str, str]]] = field(default_factory=lambda: defaultdict(set))  # moiety → (name, type)
    substance_class: dict[str, str] = field(default_factory=dict)
    vehicles: set[str] = field(default_factory=set)                                # saline, water, ...

    def lookup(self, text: str) -> set[str]:
        """Moieties named by `text`. Trailing salt words are tried off first ('GEMCITABINE HYDROCHLORIDE' →
        'GEMCITABINE'): GSRS doesn't link every salt to its parent, and this keeps Drugs@FDA ingredient strings and
        trial intervention names on the same id."""
        words = _TAG.sub(" ", text).upper().replace(",", " ").split()
        core = list(words)
        while core and _NONALNUM.sub("", core[-1]) in SALT_WORDS:
            core.pop()
        if core and len(core) < len(words):
            hit = self._lookup_key(key(" ".join(core)))
            if len(hit) == 1:
                return hit
        return self._lookup_key(key(text))

    def _lookup_key(self, k: str) -> set[str]:
        if len(k) < 3 or k in GENERIC_WORDS or k in SALT_WORDS:
            return set()
        hits = self.by_key.get(k, set())
        if len(hits) <= 1:
            return hits
        # Stereoisomer/variant records share the main name ('ESTRADIOL' also names ent-estradiol). Prefer the record
        # whose display name *is* this name, else a much better documented record; otherwise stay ambiguous.
        exact = {m for m in hits if key(self.display.get(m, "")) == k}
        pool = exact or hits
        ranked = sorted(pool, key=lambda m: len(self.names.get(m, ())), reverse=True)
        if len(ranked) == 1 or len(self.names.get(ranked[0], ())) >= 2 * len(self.names.get(ranked[1], ())):
            return {ranked[0]}
        return hits


def gsrs_rows(zip_path: Path):
    """Stream GSRS substances → (unii, moiety_unii, substance_class, name, name_type, is_display)."""
    import ijson

    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as fh:
        for rec in ijson.items(fh, "results.item"):
            unii = rec.get("unii")
            if not unii:
                continue
            moiety = unii
            for rel in rec.get("relationships", []) or []:
                if rel.get("type") == "ACTIVE MOIETY" and (rel.get("related_substance") or {}).get("unii"):
                    moiety = rel["related_substance"]["unii"]
                    break
            for n in rec.get("names", []) or []:
                if n.get("type") == "sys" or not n.get("name"):
                    continue
                yield (unii, moiety, rec.get("substance_class", ""), n["name"], n.get("type", ""),
                       bool(n.get("display_name")))


def build_index(rows) -> GsrsIndex:
    """rows: iterable of (unii, moiety_unii, substance_class, name, name_type, is_display)."""
    idx = GsrsIndex()
    best: dict[str, tuple] = {}
    for unii, moiety, sclass, name, ntype, is_display in rows:
        k = key(name)
        if len(k) < 3:
            continue
        idx.by_key[k].add(moiety)
        idx.names[moiety].add((name, ntype))
        if unii == moiety:
            idx.substance_class[moiety] = sclass
        # Display name: INN/USAN-tagged > GSRS display flag > other non-brand > brand.
        tag = name.upper()
        rank = (0 if ("[INN]" in tag or "[USAN]" in tag) else 1 if is_display and ntype != "bn"
                else 3 if ntype == "bn" else 2)
        clean = _TAG.sub(" ", name).strip()
        score = (rank, unii != moiety, len(clean))      # within a tier, the moiety's own record names it
        if moiety not in best or score < best[moiety]:
            best[moiety] = score
            idx.display[moiety] = clean

    idx.vehicles = set().union(*(idx.by_key.get(key(v), set()) for v in VEHICLES))
    return idx


def resolve_string(name: str, idx: GsrsIndex) -> tuple[frozenset[str], str]:
    """Moiety UNIIs for one intervention string and how they were found ('' if unresolved)."""
    found, how = _resolve(name, idx)
    if found and found <= idx.vehicles:
        return frozenset(), ""
    return found, how


def _resolve(name: str, idx: GsrsIndex) -> tuple[frozenset[str], str]:
    if NON_DRUG.search(name) and not idx.lookup(name):
        return frozenset(), ""
    for i, parts in enumerate(candidate_strings(name)):
        found = []
        for part in parts:
            hit = idx.lookup(part)
            if len(hit) == 1:
                found.append(next(iter(hit)))
        if len(parts) == 1 and found:
            return frozenset(found), ("exact" if i == 0 else "normalised")
        if len(parts) >= 2 and len(found) == len(parts):
            return frozenset(found), "combination"
    # Last resort: a single distinctive word that GSRS knows ('Suzetrigine 50 mg q12h tablets' already handled).
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{4,}", strip_dose(name))
             if not NON_DRUG.search(w) and key(w) not in SALT_WORDS]
    hits = {next(iter(h)) for w in words if len(h := idx.lookup(w)) == 1}
    if len(hits) == 1:
        return frozenset(hits), "word"
    return frozenset(), ""


# Resolution checks (gold set, 2026-10-30): a resolved drug that is not what the trials test almost always ends up
# labelled failed (no approval can match it), so these programmes bias the approval rate down. The checks use only
# names and trial text, never approval status or later names (brand names), so dropping flagged programmes cannot
# select on the outcome.
_CODE_NAME = re.compile(r"^[A-Za-z]{1,6}[- ]?[A-Za-z]{0,2}[- ]?\d{2,}[A-Za-z]?$")         # 'TAK-861', 'CT-P53'
_DESCRIPTION = re.compile(r"STANDARD|THERAPY|TREATMENT|ADJUVANT|ENDOCRINE|REGIMEN|LEVEL|MONTH|CHEMO|BEST|CARE|"
                          r"CONTROL|PROTOCOL|ARM|ONLY|ALONE|COMBINATION|DOSE")
# A target or gene-product name, not a drug ('CYCLIN-DEPENDENT KINASE 9', 'ATP-BINDING CASSETTE SUB-FAMILY D').
_TARGET = re.compile(r"KINASE|RECEPTOR|SUB-FAMILY|FAMILY MEMBER|PHOSPHODIESTERASE|PYROPHOSPHATASE|ANHYDRASE|"
                     r"TRANSPORTER|\bCHANNEL\b|\bPROTEIN\b")
CLASS_TERMS = {"CARBAPENEM", "BOTULINUM TOXIN", "CEPHALOSPORIN", "PENICILLIN", "BENZODIAZEPINE", "TETRACYCLINE",
               "FLUOROQUINOLONE", "SULFONAMIDE", "AMINOGLYCOSIDE", "MACROLIDE"}
_EXACT_HOW = {"exact", "other_name_exact", "combination", "other_name_combination"}


def name_keys(idx: GsrsIndex, brands: bool = False) -> dict[str, set[str]]:
    """Name keys per moiety: non-brand names, or (brands=True) brand names only. Brand names exist only after an
    approval, so they may be used for a programme only when the moiety was approved before it started."""
    return {m: {key(n) for n, t in names if (t == "bn") == brands} for m, names in idx.names.items()}


def not_a_drug(display: str, acronym: bool = True) -> bool:
    """A target protein, a drug class, or (acronym=True) a bare acronym ('CTP') rather than a drug substance."""
    d = display.upper().strip()
    return bool(_TARGET.search(d)) or d in CLASS_TERMS or (acronym and 2 <= len(key(d)) <= 4)


def resolution_flags(drug_id: str, rows: list[tuple[str, str]], moiety_keys: dict[str, set[str]],
                     display: dict[str, str], brand_keys: dict[str, set[str]] | None = None,
                     approved_before: bool = False) -> list[str]:
    """Reasons to doubt that `drug_id` is the product a programme's Phase 3 trials test. `rows`: (intervention names
    joined by '|', resolved_by) for the drug in each seed trial; `display`: moiety -> display name; brand names
    (`brand_keys`) count only when the moiety was `approved_before` the programme's first Phase 3.
      not_a_drug        the substance is a target protein, a class term or a bare acronym ('CYCLIN-DEPENDENT KINASE 9')
      code_prefix       ...reached by stripping a pure code name's digits ('TAK-861' -> 'TAK'); reported with not_a_drug
      vehicle           every name is a placebo/vehicle ('Castor Oil (Placebo)')
      weak_match        fuzzy resolutions where no 5+ character name of the substance appears in the trial's text
      descriptive_name  an unresolved name that describes a regimen ('Standard Adjuvant Endocrine Therapy')"""
    moieties = drug_id.removeprefix("UNII:").split("+") if drug_id.startswith("UNII:") else []
    rows = [(n or "", h) for n, h in rows if h]
    own_keys = set().union(*(moiety_keys.get(m, set()) for m in moieties)) if moieties else set()
    if approved_before and brand_keys:
        own_keys |= set().union(*(brand_keys.get(m, set()) for m in moieties))
    out = []
    # In a combination an acronym part is usually an abbreviation of a real drug ('ADAPALENE + BPO').
    if moieties and any(not_a_drug(display.get(m, ""), acronym=len(moieties) == 1) for m in moieties):
        out.append("not_a_drug")
        prefix = 0
        for n, h in rows:
            codes = [s.strip() for s in n.split("|") if _CODE_NAME.match(s.strip())]
            prefix += "normalised" in h and bool(codes) and not any(key(c) in own_keys for c in codes)
        if rows and prefix * 2 >= len(rows):
            out.append("code_prefix")
    if rows:
        if all(re.search(r"placebo|vehicle", s, re.IGNORECASE) for n, _ in rows for s in n.split("|") if s.strip()):
            out.append("vehicle")
        # Brand aliases ('Stribild'), abbreviations ('BD MDI', 'SOF/VEL') and combinations defeat a text check.
        fuzzy = [n for n, h in rows if h not in _EXACT_HOW and h != "alias"
                 and any(len(w) > 4 for w in re.findall(r"[A-Za-z]+", strip_dose(n)))]
        if len(moieties) == 1 and fuzzy:
            long_keys = [k for k in own_keys if len(k) >= 5]
            weak = sum(not any(k in key(n) for k in long_keys) for n in fuzzy)
            if weak * 2 > len(rows):
                out.append("weak_match")
    if drug_id.startswith("NAME:") and _DESCRIPTION.search(drug_id[5:]):
        out.append("descriptive_name")
    return out


def name_id(name: str) -> str | None:
    """Fallback id for names GSRS doesn't know, e.g. an unpublished code name."""
    k = key(strip_dose(name) or name)
    if len(k) < 4 or k in GENERIC_WORDS or NON_DRUG.search(name):
        return None
    return f"NAME:{k}"


def drug_id(moieties: frozenset[str]) -> str:
    return "UNII:" + "+".join(sorted(moieties))


class AliasUnion:
    """Union-find over drug ids, refusing merges that would join two different GSRS moiety sets."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if ra.startswith("UNII:") and rb.startswith("UNII:"):
            return False                                       # conflict: two known, different drugs
        if rb.startswith("UNII:"):
            ra, rb = rb, ra
        self.parent[rb] = ra                                   # a known UNII always becomes the root
        return True
