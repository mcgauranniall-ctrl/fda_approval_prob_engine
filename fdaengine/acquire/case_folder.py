"""A case folder holds everything for one drug × indication × level analysis (PLAN.md §10.2)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from fdaengine import config


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


@dataclass
class Case:
    root: Path

    @classmethod
    def create(cls, drug: str, indication: str, level: str, run_date: date) -> Case:
        root = config.CASES_DIR / f"{slugify(drug)}_{slugify(indication)}_{level}_{run_date.isoformat()}"
        for sub in ("raw/ctgov", "raw/ctgov_docs", "raw/edgar", "text", "tables", "extracted", "analysis",
                    "report", "logs"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        case = cls(root)
        if not case.manifest_path.exists():
            case._write_json(case.manifest_path, {"files": []})
        return case

    @classmethod
    def open(cls, root: Path | str) -> Case:
        return cls(Path(root))

    # -- paths ---------------------------------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    # -- json helpers --------------------------------------------------------------------------------------
    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def write_json(self, rel: str, data) -> Path:
        p = self.path(rel)
        self._write_json(p, data)
        return p

    def read_json(self, rel: str, default=None):
        p = self.path(rel)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))

    # -- manifest ------------------------------------------------------------------------------------------
    def add_file(self, rel: str, content: bytes | str, *, url: str | None, source: str,
                 source_date: str | None, within_cutoff: bool, note: str = "") -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        p.write_bytes(data)
        manifest = self.read_json("manifest.json", {"files": []})
        manifest["files"] = [f for f in manifest["files"] if f["path"] != rel]
        manifest["files"].append({
            "path": rel, "url": url, "source": source, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            "source_date": source_date, "within_cutoff": within_cutoff, "note": note,
        })
        self._write_json(self.manifest_path, manifest)
        return p

    def log(self, message: str) -> None:
        with open(self.path("logs", "pipeline.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
