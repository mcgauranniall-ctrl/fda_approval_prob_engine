"""01c — Federal Register notices of FDA advisory committee meetings, with their full text.

  uv run python scripts/01c_federal_register.py
Output: data/interim/federal_register/adcom_notices.parquet (document number, publication date, title, text)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fdaengine import config
from fdaengine.acquire.fetcher import FetchError, default_fetcher
from fdaengine.sources import federal_register as fr

OUT = config.DATA_DIR / "interim" / "federal_register"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    f = default_fetcher()
    notices = fr.list_notices(f)
    print(f"[01c] {len(notices):,} advisory committee meeting notices ({time.time() - t0:.0f}s)", flush=True)
    rows, failed = [], []
    for i, d in enumerate(notices, 1):
        if not d.get("raw_text_url"):
            continue
        try:
            r = f.get(d["raw_text_url"], allow_404=True)
        except FetchError as e:                     # rate limited after retries: skip; a rerun resumes from cache
            failed.append(d["document_number"])
            print(f"[01c] skipped {d['document_number']}: {e}", flush=True)
            continue
        if r.status != 200:
            continue
        rows.append({"document_number": d["document_number"], "publication_date": d["publication_date"],
                     "title": d["title"], "html_url": d.get("html_url"), "text": r.text})
        if i % 500 == 0:
            print(f"[01c] fetched {i:,}/{len(notices):,} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "adcom_notices.parquet", index=False)
    print(f"[01c] {len(df):,} notices saved, {df['publication_date'].min()} to {df['publication_date'].max()} "
          f"({time.time() - t0:.0f}s); {len(failed)} skipped")


if __name__ == "__main__":
    main()
