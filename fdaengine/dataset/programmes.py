"""Drug × indication programme grouping (PLAN.md §5 script 02). Pure functions over plain data, so they are testable.

A programme is one drug's Phase 3 effort in one indication. Within a drug, industry-led Phase 3 trials are joined
when their *primary* condition agrees: the same specific MeSH descriptor, or near-identical condition text when
MeSH is missing. Only the primary (first-listed) condition creates links, so a psoriatic-arthritis trial that also
lists psoriasis does not merge the two programmes. A trial coded only with a broad term ("Pain", "Breast
Neoplasms") joins the drug's one more specific group under that term; if there are several, it stays on its own
and is flagged as generic. Trials starting more than `EPISODE_YEARS` after a group's first Phase 3 open a new
programme episode (the Y_PROG window is 7 years).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

from rapidfuzz import fuzz

EPISODE_YEARS = 7
GENERIC_MIN_DESCENDANTS = 3
_TEXT_STOP = {"patients", "patient", "subjects", "adults", "adult", "with", "the", "of", "and", "in", "for", "a",
              "an", "treatment", "participants", "volunteers", "to", "or", "disease", "disorder",
              # severity and line-of-therapy words: one FDA indication, many trial phrasings
              "mild", "moderate", "severe", "moderately", "severely", "advanced", "metastatic", "locally",
              "recurrent", "refractory", "relapsed", "unresectable", "previously", "treated", "untreated", "newly",
              "diagnosed", "stage", "first", "second", "line", "frown"}
GENERIC_TEXTS = {"cancer", "solid tumor", "solid tumors", "advanced cancer", "tumor", "tumors", "neoplasms",
                 "pain", "infection", "infections", "advanced solid tumors", "metastatic cancer", "healthy"}


def norm_text(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(w for w in words if w not in _TEXT_STOP)


def texts_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or fuzz.token_sort_ratio(a, b) >= 92:
        return True
    # 'glabellar lines' within 'temporary improvement glabellar lines': the shorter, multi-word text is contained.
    short, long_ = sorted((a.split(), b.split()), key=len)
    return len(short) >= 2 and " ".join(short) not in GENERIC_TEXTS and set(short) <= set(long_)


def leaf_ancestors(trial_leaves: dict[str, list[str]], trial_ancestors: dict[str, set[str]]) -> dict[str, set[str]]:
    """MeSH leaf → ancestor ids, learned from trials with exactly one leaf (ancestors aren't given per leaf)."""
    out: dict[str, set[str]] = defaultdict(set)
    for nct, leaves in trial_leaves.items():
        if len(set(leaves)) == 1:
            out[leaves[0]] |= trial_ancestors.get(nct, set())
    return out


def generic_terms(anc: dict[str, set[str]]) -> set[str]:
    """MeSH ids that sit above at least GENERIC_MIN_DESCENDANTS distinct leaves seen in the data."""
    desc = Counter(a for leaf, ancestors in anc.items() for a in ancestors if a != leaf)
    return {m for m, n in desc.items() if n >= GENERIC_MIN_DESCENDANTS}


@dataclass
class TrialCond:
    nct_id: str
    start: str | None                    # ISO date
    leaves: list[str]                    # MeSH ids in listed order
    texts: list[str]                     # normalised condition texts in listed order
    ancestors: set[str] = field(default_factory=set)


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


@dataclass
class Group:
    trials: list[str]
    specificity: str                     # 'specific' | 'text' | 'generic' | 'none'
    attached: dict[str, str]             # nct → how it joined ('primary_mesh', 'text', 'generic_attached', ...)


def group_trials(trials: list[TrialCond], generic: set[str]) -> list[Group]:
    """Group one drug's seed Phase 3 trials by indication."""
    by_id = {t.nct_id: t for t in trials}
    prim_leaf = {t.nct_id: next((m for m in t.leaves if m not in generic), None) for t in trials}
    prim_text = {t.nct_id: next((x for x in t.texts if x and x not in GENERIC_TEXTS), None) for t in trials}
    uf = _UF(by_id)
    how: dict[str, str] = {}
    ids = list(by_id)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if prim_leaf[a] and prim_leaf[a] == prim_leaf[b]:
                uf.union(a, b)
                how.setdefault(a, "primary_mesh"); how.setdefault(b, "primary_mesh")
            elif texts_match(prim_text[a], prim_text[b]) and not (prim_leaf[a] and prim_leaf[b]):
                # Text only links trials where at least one lacks a specific MeSH term (MeSH outranks text).
                uf.union(a, b)
                how.setdefault(a, "text"); how.setdefault(b, "text")

    # Trials with neither a specific leaf nor specific text: attach under their broad term if unambiguous.
    loose = [n for n in ids if not prim_leaf[n] and not prim_text[n]]
    anchored = [n for n in ids if n not in loose]
    for n in loose:
        broad = set(by_id[n].leaves)
        roots = {uf.find(m) for m in anchored
                 if broad & (by_id[m].ancestors | set(by_id[m].leaves))}
        if len(roots) == 1:
            uf.union(n, next(iter(roots)))
            how[n] = "generic_attached"
    for i, a in enumerate(loose):                                # remaining loose trials: share a broad term
        for b in loose[i + 1:]:
            if uf.find(a) != uf.find(b) and how.get(a) != "generic_attached" and how.get(b) != "generic_attached" \
                    and set(by_id[a].leaves) & set(by_id[b].leaves):
                uf.union(a, b)

    comps: dict[str, list[str]] = defaultdict(list)
    for n in ids:
        comps[uf.find(n)].append(n)
    groups = []
    for members in comps.values():
        if any(prim_leaf[m] for m in members):
            spec = "specific"
        elif any(prim_text[m] for m in members):
            spec = "text"
        elif any(by_id[m].leaves for m in members):
            spec = "generic"
        else:
            spec = "none"
        for m in members:
            how.setdefault(m, "own" if len(members) == 1 else "broad_term")
        groups.append(Group(sorted(members), spec, {m: how[m] for m in members}))
    return groups


def split_episodes(starts: dict[str, str | None], years: int = EPISODE_YEARS) -> list[list[str]]:
    """Split a group's trials into episodes: a trial starting > `years` after the episode anchor opens a new one."""
    dated = sorted((s, n) for n, s in starts.items() if s)
    undated = [n for n, s in starts.items() if not s]
    episodes: list[list[str]] = []
    anchor: date | None = None
    for s, n in dated:
        d = date.fromisoformat(s)
        if anchor is None or (d - anchor).days > years * 365.25:
            episodes.append([])
            anchor = d
        episodes[-1].append(n)
    if not episodes:
        episodes.append([])
    episodes[0].extend(undated)
    return [e for e in episodes if e]


def programme_id(drug_id: str, indication_key: str, anchor: str | None) -> str:
    raw = f"{drug_id}|{indication_key}|{anchor or ''}"
    return "PG" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def indication_link(trial: TrialCond, prog_leaves: set[str], prog_ancestors: set[str], prog_texts: set[str],
                    generic: set[str]) -> str | None:
    """How an earlier-phase trial relates to a programme's indication: 'exact', 'related' or None."""
    leaves = set(trial.leaves)
    if leaves & (prog_leaves - generic):
        return "exact"
    texts = [t for t in trial.texts if t and t not in GENERIC_TEXTS]
    if any(texts_match(t, p) for t in texts for p in prog_texts):
        return "exact"
    if (leaves - generic) & prog_ancestors or (prog_leaves - generic) & trial.ancestors:
        return "related"
    if leaves & prog_leaves or leaves & prog_ancestors:
        return "related"
    return None
