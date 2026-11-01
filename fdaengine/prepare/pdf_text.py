"""PDF -> page-tagged text. Page markers let extracted quotes cite a page number that can be verified."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

PAGE_MARKER = "=== PAGE {n} ==="


def pdf_to_text(pdf: Path) -> tuple[str, int, int]:
    """(text with page markers, page count, pages with almost no text — likely scans needing OCR)."""
    parts, empty = [], 0
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            t = page.get_text("text")
            if len(t.strip()) < 40:
                empty += 1
            parts.append(f"{PAGE_MARKER.format(n=i)}\n{t.strip()}\n")
        return "\n".join(parts), len(doc), empty
