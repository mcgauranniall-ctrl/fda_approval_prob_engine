"""Verify that every quote Claude returns actually appears in the cited case file."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=256)
def _file_text(path: str) -> str:
    try:
        return normalise(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""


def find_quote(case_root: Path, quote: str, source_file: str | None) -> str | None:
    """Return the relative path of the file containing `quote` (the cited one first), or None."""
    q = normalise(quote)
    if len(q) < 8:
        return None
    candidates = []
    if source_file:
        candidates.append(case_root / source_file)
    candidates += sorted((case_root / "text").glob("*.txt")) + [case_root / "extracted" / "structured.json"]
    candidates += sorted((case_root / "raw" / "ctgov").glob("*.json"))
    for p in candidates:
        text = _file_text(str(p))
        if not text:
            continue
        if q in text or (len(q) > 30 and fuzz.partial_ratio(q, text) >= 95):
            return p.relative_to(case_root).as_posix()
    return None
