"""Case extraction stage: prompt → headless Claude run → quote verification → extracted/fields.json."""

from __future__ import annotations

from collections.abc import Callable

from fdaengine import config
from fdaengine.acquire.case_folder import Case
from fdaengine.agent import claude_runner
from fdaengine.schema import FIELDS, extraction_schema
from fdaengine.timeline.levels import ORDER
from fdaengine.validate.quotes import find_quote

# Context fields may legitimately come from general knowledge (prompt rule 2), so they need no quote.
_KNOWLEDGE_OK = {"therapeutic_area", "modality", "first_in_class", "mechanism"}


def build_prompt(case: Case) -> str:
    meta = case.read_json("case.json")
    template = (config.PROMPTS_DIR / "extract_case.md").read_text(encoding="utf-8")
    return template.format(drug=meta["drug"], names=", ".join(meta["names"]), indication=meta["indication"],
                           sponsor=meta.get("sponsor") or "unknown", level=meta["level"],
                           level_name=meta["level_name"], cutoff=meta["cutoff"])


def verify(case: Case, output: dict) -> tuple[dict, list[dict], list[str]]:
    """Return (verified fields, verified analyst flags, rejection notes)."""
    level_idx = ORDER.index(case.read_json("case.json")["level"])
    fields, notes = {}, []
    for name, raw in (output.get("fields") or {}).items():
        if name not in FIELDS or not isinstance(raw, dict) or raw.get("value") is None:
            continue
        min_level = FIELDS[name][2]
        if ORDER.index(min_level) > level_idx:
            notes.append(f"{name}: dropped (only knowable from {min_level}, case is {ORDER[level_idx]})")
            continue
        quote = raw.get("quote")
        if quote:
            found = find_quote(case.root, quote, raw.get("source_file"))
            if not found:
                notes.append(f"{name}: dropped (quote not found in case files)")
                continue
            raw = {**raw, "source_file": found, "quote_verified": True}
        elif name not in _KNOWLEDGE_OK:
            notes.append(f"{name}: dropped (no supporting quote)")
            continue
        else:
            raw = {**raw, "quote_verified": False, "confidence": "low"}
        fields[name] = raw
    flags = []
    for f in output.get("analyst_flags") or []:
        q = f.get("quote")
        if q and (found := find_quote(case.root, q, f.get("source_file"))):
            flags.append({**f, "source_file": found})
        else:
            notes.append(f"analyst flag dropped (unverifiable quote): {f.get('flag', '')[:60]}")
    return fields, flags, notes


def run_extraction(case: Case, model: str = "opus", on_progress: Callable[[str], None] | None = None) -> dict:
    result = claude_runner.run(build_prompt(case), cwd=case.root, schema=extraction_schema(), model=model,
                               on_progress=on_progress, log_path=case.path("logs", "claude_extract.jsonl"))
    fields, flags, notes = verify(case, result.structured)
    out = {"model": model, "cost_usd": result.cost_usd, "turns": result.turns,
           "duration_s": round(result.duration_s, 1), "fields": fields,
           "summary": (result.structured or {}).get("summary", ""), "rejected": notes}
    case.write_json("extracted/fields.json", out)
    case.write_json("extracted/analyst_flags.json", flags)
    case.log(f"claude extraction: {len(fields)} fields kept, {len(notes)} rejected, cost≈${result.cost_usd}")
    return out
