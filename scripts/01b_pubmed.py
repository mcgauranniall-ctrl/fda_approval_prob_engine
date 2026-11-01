"""01b — PubMed publications of every Phase 2/3 trial linked to a programme, and each trial's main-paper result.

Searches PubMed by ClinicalTrials.gov ID (`NCT…[si]`, 100 IDs per request), fetches the records (200 per request,
cached), and extracts the primary-endpoint result rule-based (`sources/pubmed.py`, D6). Each publication carries
its availability date; script 05 only uses a result once the main paper was public by the cut-off.

  uv run python scripts/01b_pubmed.py
Outputs: data/interim/pubmed/publications.parquet (one row per article × trial), trial_results.parquet
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.acquire.fetcher import default_fetcher
from fdaengine.sources import pubmed

OUT = config.DATA_DIR / "interim" / "pubmed"
PROCESSED = config.DATA_DIR / "processed"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    pt = pd.read_parquet(PROCESSED / "programme_trials.parquet")
    ncts = sorted(pt[pt["relation"].isin(["p2", "p3_seed", "p3_linked"])]["nct_id"].unique())
    f = default_fetcher()
    pmids: list[str] = []
    for i in range(0, len(ncts), 2000):
        pmids += pubmed.search_by_nct(f, ncts[i:i + 2000])
        print(f"[01b] searched {min(i + 2000, len(ncts)):,}/{len(ncts):,} trials, {len(set(pmids)):,} articles "
              f"({time.time() - t0:.0f}s)", flush=True)
    pmids = sorted(set(pmids), key=int)
    pubs: list[pubmed.Publication] = []
    for i in range(0, len(pmids), 2000):
        pubs += pubmed.fetch(f, pmids[i:i + 2000])
        print(f"[01b] fetched {min(i + 2000, len(pmids)):,}/{len(pmids):,} articles ({time.time() - t0:.0f}s)", flush=True)

    wanted = set(ncts)
    by_trial: dict[str, list[pubmed.Publication]] = defaultdict(list)
    rows = []
    for p in pubs:
        for n in p.ncts:
            if n in wanted:
                by_trial[n].append(p)
                rows.append({"nct_id": n, **p.to_dict(), "ncts": "|".join(p.ncts), "pub_types": "|".join(p.pub_types)})
    results = []
    for n, ps in by_trial.items():
        r = pubmed.trial_result(ps)
        if r:
            results.append({"nct_id": n, **r})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "publications.parquet", index=False)
    res = pd.DataFrame(results)
    res.to_parquet(OUT / "trial_results.parquet", index=False)
    summary = {"trials_searched": len(ncts), "articles": len(pubs), "trials_with_articles": len(by_trial),
               "trials_with_result": len(res), "result_basis": res["basis"].value_counts().to_dict() if len(res) else {},
               "built_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"[01b] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
