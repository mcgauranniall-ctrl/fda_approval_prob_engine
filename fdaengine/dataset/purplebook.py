"""FDA Purple Book (licensed biologics) as the approval source for CBER products: vaccines, blood and plasma
products, allergenics, cell and gene therapies. Drugs@FDA covers CDER only. Labels only, never model inputs.

The monthly CSV holds a changes report followed by the full product list; each row is a product presentation of a
BLA with its licensure date. There is no indication or efficacy-supplement information.

Vaccines are named differently in trials ('Hepatitis A virus strain HM175 antigen') and in the Purple Book
('Hepatitis A Vaccine'), so a BLA matches a programme when the BLA's core words (disease / antigen, without words
like 'vaccine', 'live', 'recombinant') all appear in the programme's names and the license holder is one of the
programme's sponsors; the most specific matching BLAs win ('Influenza Vaccine, H5N1' over 'Influenza Vaccine').
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from fdaengine.dataset import drugs

_NOISE = {"vaccine", "vaccines", "live", "recombinant", "adjuvanted", "adsorbed", "inactivated", "antigen", "antigens",
          "strain", "conjugate", "conjugated", "human", "injection", "suspension", "for", "and", "the", "of", "with",
          "subunit", "protein", "surface", "formaldehyde", "propiolactone", "purified", "oral", "intranasal",
          "polyvalent", "valent", "type", "virus", "viruses", "attenuated", "reduced", "solution", "intravenous",
          "subcutaneous", "products", "product", "biological", "in", "derived", "cell", "based"}
_WORD = re.compile(r"[a-z0-9]+")


def core_tokens(name: str) -> frozenset[str]:
    """Distinctive words of a biologic's name ('Zoster Vaccine Recombinant, Adjuvanted' → {'zoster'}). Single
    letters are kept: 'Hepatitis A' and 'Hepatitis B' differ only by one."""
    name = re.sub(r"-[a-z]{4}\b", "", (name or "").lower())             # FDA four-letter suffix
    return frozenset(w for w in _WORD.findall(name) if w not in _NOISE and not w.isdigit())


@dataclass(frozen=True)
class Bla:
    bla: str                    # six-digit, as Drugs@FDA application numbers
    applicant: str
    proper_name: str
    proprietary_name: str
    first_licensed: str         # earliest original licensure date of any product under the BLA
    tokens: frozenset[str]


def load(csv_path: Path) -> pd.DataFrame:
    """The full product list (the second table in the file)."""
    lines = csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    start = [i for i, line in enumerate(lines) if line.startswith("N/R/U,")][-1]
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _iso(d: str) -> str | None:
    try:
        return datetime.strptime(d.strip(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def cber_blas(df: pd.DataFrame) -> list[Bla]:
    out = []
    for bla, g in df[df["Center"] == "CBER"].groupby("BLA Number"):
        dates = [d for d in map(_iso, g.loc[g["Submission Type"] == "Original", "Approval Date"]) if d] \
            or [d for d in map(_iso, g["Approval Date"]) if d]
        if not dates:
            continue
        r = g.iloc[0]
        out.append(Bla(bla.zfill(6), r["Applicant"], r["Proper Name"], r["Proprietary Name"], min(dates),
                       core_tokens(r["Proper Name"])))
    return out


def resolved_drug_ids(blas: list[Bla], idx: drugs.GsrsIndex) -> dict[str, str]:
    """BLA → programme-style drug id when GSRS resolves the proper name (cell and gene therapies, factors)."""
    out = {}
    for b in blas:
        found, _ = drugs.resolve_string(re.sub(r"-[a-z]{4}$", "", b.proper_name.strip()), idx)
        if found:
            out[b.bla] = drugs.drug_id(found)
    return out


def match(drug_id: str, names: list[str], holder_ok, blas: list[Bla], bla_drug: dict[str, str]) -> list[Bla]:
    """CBER BLAs for a programme: same resolved drug id, or (holder is a sponsor and) all core words of the BLA's
    proper name appear in the programme's names, or the brand name appears in them; most specific matches only."""
    prog_tokens = frozenset().union(*(core_tokens(n) for n in names)) if names else frozenset()
    name_keys = {drugs.key(n) for n in names}
    exact = [b for b in blas if bla_drug.get(b.bla) == drug_id]
    if exact:
        return exact
    scored = []
    for b in blas:
        if not holder_ok(b.applicant):
            continue
        if b.proprietary_name and drugs.key(b.proprietary_name) in name_keys:
            scored.append((99, b))
        elif b.tokens and b.tokens <= prog_tokens:
            scored.append((len(b.tokens), b))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    return [b for s, b in scored if s == best]
