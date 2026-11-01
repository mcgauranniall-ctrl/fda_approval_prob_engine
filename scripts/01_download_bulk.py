"""01 — Download the bulk public sources for the historical dataset (PLAN.md §5). No LLM calls (D6).

Sources (all free, no login):
  ctgov       ClinicalTrials.gov API v2 full study records (same JSON as live cases → `structure_trial()`),
              Phase 1–3 drug/biologic interventional trials with an industry lead sponsor or collaborator,
              sharded by first-posted year into data/raw/ctgov/studies_<year>.jsonl.gz
  drugsfda    Drugs@FDA bulk zip (applications, submissions, action dates, marketing status, document links)
  purplebook  Purple Book monthly CSV: licensed biologics incl. CBER (vaccines, blood, cell/gene therapy)
  openfda     openFDA bulk JSON: drug/drugsfda, transparency/crl (FDA CRL database), other/unii,
              other/substance (GSRS: code names ↔ INN), drug/label (SPL labels; label outcomes only)

Idempotent: a file already in data/raw/manifest.json with the same source version (openFDA export date,
Drugs@FDA file name, ClinicalTrials.gov shard count) is skipped. `--refresh` re-downloads everything.

  uv run python scripts/01_download_bulk.py --dry-run          # show what would be downloaded, with sizes
  uv run python scripts/01_download_bulk.py                    # everything
  uv run python scripts/01_download_bulk.py --only ctgov openfda_crl
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine.dataset import bulk

CTGOV = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA_INDEX = "https://api.fda.gov/download.json"
DRUGSFDA_ZIP = "https://www.fda.gov/media/89850/download"
PURPLEBOOK_DOWNLOADS = "https://purplebooksearch.fda.gov/downloads"
FIRST_YEAR = 1999                       # ClinicalTrials.gov went live in 2000; a few records predate it

_PHASES = "(AREA[Phase]EARLY_PHASE1 OR AREA[Phase]PHASE1 OR AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"
_DRUG = "(AREA[InterventionType]DRUG OR AREA[InterventionType]BIOLOGICAL)"
CTGOV_SCOPES = {
    "industry": f"(AREA[LeadSponsorClass]INDUSTRY OR AREA[CollaboratorClass]INDUSTRY) "
                f"AND AREA[StudyType]INTERVENTIONAL AND {_PHASES} AND {_DRUG}",
    "all": f"AREA[StudyType]INTERVENTIONAL AND {_PHASES} AND {_DRUG}",
}
# openFDA endpoints: (category, endpoint) → manifest source name
OPENFDA_SETS = {
    ("drug", "drugsfda"): "openfda_drugsfda",
    ("transparency", "crl"): "openfda_crl",
    ("other", "unii"): "openfda_unii",
    ("other", "substance"): "openfda_substance",
    ("drug", "label"): "openfda_label",
}
ALL_SOURCES = ["ctgov", "drugsfda", "purplebook", *OPENFDA_SETS.values()]


# ---------------------------------------------------------------------------------------------------- ctgov
def _ctgov_get(client: httpx.Client, params: dict) -> dict:
    for attempt in range(5):
        try:
            r = client.get(CTGOV, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def ctgov_shards(scope: str) -> list[tuple[str, str]]:
    """(year label, advanced filter) per first-posted year; the first shard catches everything earlier."""
    base = CTGOV_SCOPES[scope]
    shards = [(str(FIRST_YEAR), f"{base} AND AREA[StudyFirstPostDate]RANGE[MIN,{FIRST_YEAR}-12-31]")]
    for y in range(FIRST_YEAR + 1, date.today().year + 1):
        shards.append((str(y), f"{base} AND AREA[StudyFirstPostDate]RANGE[{y}-01-01,{y}-12-31]"))
    return shards


def ctgov_count(client: httpx.Client, flt: str) -> int:
    return _ctgov_get(client, {"filter.advanced": flt, "countTotal": "true", "pageSize": 1,
                               "fields": "NCTId"})["totalCount"]


def download_ctgov(manifest: dict, scope: str, refresh: bool, dry_run: bool) -> None:
    throttle = bulk.Throttle(0.5)
    with httpx.Client(timeout=180, headers={"User-Agent": bulk._UA, "Accept-Encoding": "gzip"}) as client:
        total = ctgov_count(client, CTGOV_SCOPES[scope])
        print(f"[ctgov] scope={scope}: {total:,} studies in total")
        if dry_run:
            print(f"[ctgov] would page through ~{total // 1000 + len(ctgov_shards(scope))} requests; "
                  f"est. {total * 6.2 / 1000:,.0f} MB gzipped on the wire and on disk")
            return
        got = 0
        current_year = str(date.today().year)
        for label, flt in ctgov_shards(scope):
            rel = f"ctgov/{scope}/studies_{label}.jsonl.gz"
            throttle.wait()
            expected = ctgov_count(client, flt)
            # Past years can still change (results get posted later), so a shard is re-fetched when its count
            # moves, on --refresh, and always for the current year.
            version = f"{scope}:{expected}"
            if not refresh and label != current_year and bulk.is_current(manifest, rel, version):
                got += manifest["files"][rel]["records"]
                continue
            writer = bulk.JsonlGzWriter(bulk.RAW_DIR / rel)
            token = None
            try:
                while True:
                    params = {"filter.advanced": flt, "pageSize": 1000, "format": "json"}
                    if token:
                        params["pageToken"] = token
                    throttle.wait()
                    data = _ctgov_get(client, params)
                    for s in data.get("studies", []):
                        writer.write(s)
                    token = data.get("nextPageToken")
                    if not token:
                        break
            except BaseException:
                writer.abort()
                raise
            n = writer.n
            sha = writer.close()
            if n != expected:
                print(f"[ctgov] WARNING {label}: wrote {n} studies, API count said {expected}")
            bulk.record(manifest, rel, "ctgov", CTGOV, sha, version, records=n, scope=scope,
                        snapshot_date=date.today().isoformat(), filter=flt)
            bulk.save_manifest(manifest)
            got += n
            print(f"[ctgov] {label}: {n:,} studies ({(bulk.RAW_DIR / rel).stat().st_size / 1e6:.1f} MB)")
        print(f"[ctgov] done: {got:,} studies (API total {total:,})")


# -------------------------------------------------------------------------------------------------- openFDA
def openfda_partitions(selected: set[str]) -> list[dict]:
    index = httpx.get(OPENFDA_INDEX, timeout=60, headers={"User-Agent": bulk._UA}).json()["results"]
    out = []
    for (cat, ep), source in OPENFDA_SETS.items():
        if source not in selected:
            continue
        meta = index[cat][ep]
        for p in meta["partitions"]:
            name = p["file"].rsplit("/", 1)[-1]
            out.append({"source": source, "url": p["file"], "rel": f"openfda/{cat}_{ep}/{name}",
                        "size_mb": float(p.get("size_mb", 0)), "records": p.get("records"),
                        "version": meta.get("export_date")})
    return out


def download_files(manifest: dict, jobs: list[dict], refresh: bool, dry_run: bool) -> None:
    todo = [j for j in jobs if refresh or not bulk.is_current(manifest, j["rel"], j["version"])]
    for source in sorted({j["source"] for j in jobs}):
        mine = [j for j in jobs if j["source"] == source]
        pending = [j for j in todo if j["source"] == source]
        mb = sum(j["size_mb"] for j in pending)
        print(f"[{source}] {len(mine)} file(s), export {mine[0]['version']}: "
              f"{len(pending)} to download ({mb:,.0f} MB), {len(mine) - len(pending)} already current")
    if dry_run or not todo:
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(bulk.download, j["url"], bulk.RAW_DIR / j["rel"]): j for j in todo}
        for fut in as_completed(futures):
            j = futures[fut]
            sha, _ = fut.result()
            bulk.record(manifest, j["rel"], j["source"], j["url"], sha, j["version"], records=j["records"])
            bulk.save_manifest(manifest)
            print(f"[{j['source']}] {j['rel']} ({(bulk.RAW_DIR / j['rel']).stat().st_size / 1e6:.1f} MB)")


def drugsfda_job() -> dict:
    """Drugs@FDA's zip is renamed daily (drugsfda_YYYYMMDD.zip); that name is its version."""
    r = httpx.head(DRUGSFDA_ZIP, timeout=60, follow_redirects=True, headers={"User-Agent": bulk._UA})
    r.raise_for_status()
    disp = r.headers.get("content-disposition", "")
    name = disp.split("filename=")[-1].strip('"; ') or "drugsfda.zip"
    return {"source": "drugsfda", "url": DRUGSFDA_ZIP, "rel": "drugsfda/drugsfda.zip", "version": name,
            "size_mb": int(r.headers.get("content-length", 0)) / 1e6, "records": None}


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]


def purplebook_job() -> dict:
    """The Purple Book's latest monthly full-data CSV (all licensed biologics, CDER and CBER); its name is the
    version. Needed because Drugs@FDA has no CBER products (vaccines, blood products, cell and gene therapies)."""
    page = httpx.get(PURPLEBOOK_DOWNLOADS, timeout=60, headers={"User-Agent": bulk._UA}).text
    links = set(re.findall(r"https://www\.accessdata\.fda\.gov/drugsatfda_docs/PurpleBook/(\d{4})/"
                           r"purplebook-search-(\w+)-data-download\.csv", page))
    year, month = max(((int(y), m) for y, m in links if m.lower() in _MONTHS),
                      key=lambda ym: (ym[0], _MONTHS.index(ym[1].lower())))
    url = f"https://www.accessdata.fda.gov/drugsatfda_docs/PurpleBook/{year}/purplebook-search-{month}-data-download.csv"
    r = httpx.head(url, timeout=60, follow_redirects=True, headers={"User-Agent": bulk._UA})
    r.raise_for_status()
    return {"source": "purplebook", "url": url, "rel": "purplebook/purplebook.csv", "version": f"{year}-{month}",
            "size_mb": int(r.headers.get("content-length", 0)) / 1e6, "records": None}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=ALL_SOURCES, help="download only these sources")
    ap.add_argument("--skip", nargs="+", choices=ALL_SOURCES, default=[], help="skip these sources")
    ap.add_argument("--ctgov-scope", choices=list(CTGOV_SCOPES), default="industry")
    ap.add_argument("--refresh", action="store_true", help="re-download even if the manifest says current")
    ap.add_argument("--dry-run", action="store_true", help="report sizes without downloading")
    args = ap.parse_args(argv)
    selected = set(args.only or ALL_SOURCES) - set(args.skip)

    manifest = bulk.load_manifest()
    jobs = []
    if "drugsfda" in selected:
        jobs.append(drugsfda_job())
    if "purplebook" in selected:
        jobs.append(purplebook_job())
    jobs += openfda_partitions(selected)
    download_files(manifest, jobs, args.refresh, args.dry_run)
    if "ctgov" in selected:
        download_ctgov(manifest, args.ctgov_scope, args.refresh, args.dry_run)
    if not args.dry_run:
        bulk.save_manifest(manifest)
        n = len(manifest["files"])
        gb = sum(e["bytes"] for e in manifest["files"].values()) / 1e9
        print(f"manifest: {n} files, {gb:.2f} GB -> {bulk.MANIFEST}")


if __name__ == "__main__":
    main()
