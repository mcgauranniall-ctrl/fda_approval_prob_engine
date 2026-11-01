"""NLM RxNorm (RxNav) name normalisation: brand <-> ingredient names."""

from __future__ import annotations

from fdaengine.acquire.fetcher import Fetcher, default_fetcher

BASE = "https://rxnav.nlm.nih.gov/REST"


def related_names(name: str, fetcher: Fetcher | None = None) -> list[str]:
    """Ingredient (IN) and brand (BN) names related to `name`. Empty for unapproved drugs, which RxNorm lacks."""
    fetcher = fetcher or default_fetcher()
    ids = fetcher.get(f"{BASE}/rxcui.json", params={"name": name, "search": 2}).json()
    rxcuis = ids.get("idGroup", {}).get("rxnormId", [])
    names: list[str] = []
    for rxcui in rxcuis[:2]:
        rel = fetcher.get(f"{BASE}/rxcui/{rxcui}/related.json", params={"tty": "IN BN"}).json()
        for group in rel.get("relatedGroup", {}).get("conceptGroup", []):
            names.extend(c["name"] for c in group.get("conceptProperties", []) or [])
    return names
