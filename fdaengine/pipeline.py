"""Case pipeline orchestration (PLAN.md §1): resolve → timeline → acquire → prepare → extract → predict → report.

Each stage writes into the case folder, so a case can be re-run from any stage.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from fdaengine.acquire.case_folder import Case
from fdaengine.acquire.fetcher import FetchError, default_fetcher
from fdaengine.prepare import pdf_text
from fdaengine.prepare.ctgov_structured import structure_trial
from fdaengine.resolve.programme import Resolution, indication_keywords
from fdaengine.sources import ctgov, edgar, pubmed
from fdaengine.timeline import events as tl_events
from fdaengine.timeline import levels as tl_levels
from fdaengine.timeline.press_release_parser import parse_document, parse_filing_date

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _attach_publications(case: Case, structured: list[dict], cutoff: date, progress: Progress) -> None:
    """PubMed main-paper results (as in training, script 01b), for trials whose main paper was public by the
    cut-off. The cited sentence is saved as a source file."""
    fetcher = default_fetcher()
    progress("Searching PubMed for the trials' publications…")
    try:
        pubs = pubmed.fetch(fetcher, pubmed.search_by_nct(fetcher, [t["nct_id"] for t in structured]))
    except FetchError:
        return
    for t in structured:
        r = pubmed.trial_result([p for p in pubs if t["nct_id"] in p.ncts])
        if not r or r["available"] > cutoff.isoformat():
            continue
        t["publication_result"] = r
        case.add_file(f"raw/pubmed/{r['pmid']}.txt", f"{r['title']}\n\nResults: {r['quote']}\n",
                      url=f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/", source="pubmed",
                      source_date=r["available"], within_cutoff=True, note=f"main paper of {t['nct_id']}")


@dataclass
class Programme:
    resolution: Resolution
    selected: list[str]                                  # confirmed NCT IDs
    filings: list[edgar.Filing] = field(default_factory=list)
    parsed_events: list = field(default_factory=list)
    timeline: tl_events.Timeline | None = None

    @property
    def trials(self) -> list[ctgov.TrialSummary]:
        by_id = {c.trial.nct_id: c.trial for c in self.resolution.candidates}
        return [by_id[n] for n in self.selected if n in by_id]

    def levels(self, as_of: date) -> list[tl_levels.LevelStatus]:
        return tl_levels.compute(self.timeline, as_of)


def discover(resolution: Resolution, selected: list[str], progress: Progress = _noop) -> Programme:
    """Find the sponsor's press releases, parse milestones and build the full programme timeline."""
    prog = Programme(resolution, selected)
    names = resolution.names
    keywords = indication_keywords(resolution.indication)
    if resolution.cik:
        progress("Searching SEC EDGAR for the sponsor's press releases…")
        try:
            filings = [f for f in edgar.search(names) if f.cik == resolution.cik]
        except FetchError as e:
            filings = []
            resolution.warnings.append(f"EDGAR search failed: {e}")
        for i, f in enumerate(filings, 1):
            progress(f"Reading press release {i}/{len(filings)} ({f.file_date})…")
            _, text = edgar.fetch_text(f)
            prog.parsed_events += parse_document(text, parse_filing_date(f.file_date), f.doc_id, names,
                                                 resolution.indication, keywords)
        prog.filings = filings
    prog.timeline = tl_events.build(prog.trials, prog.parsed_events, resolution.applications)
    return prog


def choose_cutoff(prog: Programme, level: str, as_of: date) -> tuple[str, date, tl_levels.LevelStatus | None]:
    """Resolve 'Auto' or an explicit level to (level, cutoff)."""
    levels = prog.levels(as_of)
    if level == "Auto":
        lv = tl_levels.auto_level(levels)
        if lv is None:
            raise ValueError("No forecast level has been reached yet for this programme.")
        return lv.level, as_of if lv.level == tl_levels.ORDER[-1] else min(as_of, _next_boundary(levels, lv, as_of)), lv
    lv = next(x for x in levels if x.level == level)
    if not lv.reached:
        raise ValueError(f"{level} ({lv.name}) has not been reached: {lv.basis}")
    return level, date.fromisoformat(lv.cutoff), lv


def _next_boundary(levels, current, as_of: date) -> date:
    """For Auto: use information up to as_of, but never past the next level's cut-off."""
    later = [date.fromisoformat(x.cutoff) for x in levels
             if x.cutoff and tl_levels.ORDER.index(x.level) > tl_levels.ORDER.index(current.level)]
    later = [d for d in later if d > date.fromisoformat(current.cutoff)]
    return min(later + [as_of])


def acquire(prog: Programme, level: str, as_of: date, mode: str, progress: Progress = _noop) -> Case:
    level, cutoff, status = choose_cutoff(prog, level, as_of)
    res = prog.resolution
    case = Case.create(res.drug, res.indication, level, as_of)
    fetcher = default_fetcher()
    case.log(f"acquire level={level} cutoff={cutoff} mode={mode}")

    structured = []
    for t in prog.trials:
        progress(f"Downloading {t.nct_id}…")
        study = copy.deepcopy(res.studies.get(t.nct_id) or ctgov.get_study(t.nct_id))
        posted = t.results_first_post_date
        note = ""
        if study.get("resultsSection") and posted and posted > cutoff:
            study.pop("resultsSection")
            for key in ("resultsFirstSubmitDate", "resultsFirstSubmitQcDate", "resultsFirstPostDateStruct",
                        "lastUpdatePostDateStruct", "lastUpdateSubmitDate"):
                study["protocolSection"]["statusModule"].pop(key, None)
            note = "results posted after cut-off removed"
        docs = study.get("documentSection", {}).get("largeDocumentModule", {})
        if docs.get("largeDocs"):
            docs["largeDocs"] = [d for d in docs["largeDocs"] if (d.get("uploadDate") or "")[:10] <= cutoff.isoformat()]
        registered = t.first_post_date
        case.add_file(f"raw/ctgov/{t.nct_id}.json", _json(study), url=f"{ctgov.BASE}/studies/{t.nct_id}",
                      source="ctgov", source_date=registered.isoformat() if registered else None,
                      within_cutoff=bool(registered and registered <= cutoff),
                      note=note or "registration fields are the current version (history API unavailable)")
        if not (registered and registered <= cutoff):
            continue
        structured.append(structure_trial(res.studies.get(t.nct_id) or study, res.names, cutoff))
        for doc in ctgov.large_documents(study):
            up = doc["upload_date"]
            if not up or up > cutoff.isoformat():
                continue
            pdf = fetcher.get(doc["url"]).content
            rel = f"raw/ctgov_docs/{t.nct_id}_{doc['filename']}"
            case.add_file(rel, pdf, url=doc["url"], source="ctgov_document", source_date=up,
                          within_cutoff=True, note=doc["label"])
            text, pages, empty = pdf_text.pdf_to_text(case.path(rel))
            case.add_file(f"text/{t.nct_id}_{doc['filename']}.txt", text, url=doc["url"], source="ctgov_document",
                          source_date=up, within_cutoff=True,
                          note=f"{pages} pages" + (f"; {empty} pages without text (scan?)" if empty else ""))

    for f in prog.filings:
        if f.file_date > cutoff.isoformat():
            continue
        progress(f"Saving press release {f.file_date}…")
        raw, text = edgar.fetch_text(f)
        case.add_file(f"raw/edgar/{f.file_date}_{f.doc_id}", raw, url=f.url, source="sec_edgar",
                      source_date=f.file_date, within_cutoff=True, note=f"{f.form} {f.file_type}")
        case.add_file(f"text/edgar_{f.file_date}_{f.doc_id}.txt", text, url=f.url, source="sec_edgar",
                      source_date=f.file_date, within_cutoff=True)

    _attach_publications(case, structured, cutoff, progress)
    timeline = prog.timeline.until(cutoff)
    case.write_json("timeline.json", timeline.to_list())
    case.write_json("extracted/structured.json", {"cutoff": cutoff.isoformat(), "trials": structured})
    li = tl_levels.ORDER.index(level)
    # Everything Claude can read must be knowable at the cut-off: no future level dates, no brand names coined
    # at approval, no application numbers before filing was announced.
    case.write_json("case.json", {
        "drug": res.drug, "indication": res.indication, "names": names_known_at(case, res), "sponsor": res.sponsor,
        "ticker": res.ticker, "cik": res.cik, "level": level, "level_name": tl_levels.LEVELS[level],
        "cutoff": cutoff.isoformat(), "mode": mode, "selected_trials": prog.selected,
        "applications": ([a["application_number"] for a in res.applications]
                         if li >= tl_levels.ORDER.index("L3") else []),
        "level_info": status.to_dict() if status else None,
        "warnings": res.warnings,
    })
    # UI/backtest metadata lives beside the case folder, outside Claude's working directory.
    outcome, decided = tl_levels.decision(prog.timeline)
    case.root.with_name(case.root.name + ".meta.json").write_text(json.dumps({
        "as_of": as_of.isoformat(),
        "levels": [lv.to_dict() for lv in prog.levels(as_of)],
        "actual_outcome": {"outcome": outcome, "date": decided.isoformat()} if decided and mode == "backtest" else None,
    }, indent=2), encoding="utf-8")
    return case


def analyse(case: Case, use_claude: bool = True, model: str = "opus", progress: Progress = _noop) -> dict:
    """Extraction (optional) → features → prediction → probability path → report. Returns a status dict."""
    from fdaengine.agent.claude_runner import ClaudeRunError
    from fdaengine.agent.extract import run_extraction
    from fdaengine.features import build as feature_build
    from fdaengine.predict.path import compute_path
    from fdaengine.predict.predictor import predict_case
    from fdaengine.report.render import render

    status = {"claude": "skipped", "claude_error": None}
    if use_claude:
        progress(f"Starting headless Claude extraction ({model})…")
        try:
            out = run_extraction(case, model=model, on_progress=lambda m: progress(f"Claude · {m}"))
            status["claude"] = f"{len(out['fields'])} fields, {len(out['rejected'])} discarded"
        except ClaudeRunError as e:
            status.update(claude="failed", claude_error=f"{e} {e.hint}")
            case.log(f"claude extraction failed: {e.kind}: {e}")
    progress("Building features…")
    feature_build.build(case)
    progress("Running the probability model…")
    predict_case(case)
    progress("Computing the probability path across levels…")
    compute_path(case)
    progress("Rendering the report…")
    status["report"] = str(render(case))
    return status


def names_known_at(case: Case, res: Resolution) -> list[str]:
    """The user's drug name plus aliases that appear in pre-cut-off case files."""
    corpus = " ".join(p.read_text(encoding="utf-8", errors="replace").lower()
                      for p in list(case.root.glob("text/*.txt")) + list(case.root.glob("raw/ctgov/*.json")))
    keep = {res.drug}
    keep.update(n for n in res.names if n.lower() in corpus)
    return sorted(keep, key=str.lower)


def _json(obj) -> str:
    return json.dumps(obj, indent=1)
