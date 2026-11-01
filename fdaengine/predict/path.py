"""Probability path: re-score one case at every earlier level it reached, each with that level's cut-off.

Claude-extracted fields are re-used only when their cited source file is dated on or before the earlier
cut-off (manifest `source_date`), so each point on the path sees only what was public at the time.
"""

from __future__ import annotations

import json
from datetime import date

from fdaengine.acquire.case_folder import Case
from fdaengine.features.build import compute
from fdaengine.model import provisional
from fdaengine.prepare.ctgov_structured import structure_trial
from fdaengine.schema import FIELDS
from fdaengine.timeline.levels import LEVELS, ORDER

# Registration/design facts are knowable whenever the trial is registered, so they carry to earlier levels.
_DESIGN_FIELDS = {"p2_randomized", "p2_placebo_controlled", "p2_n_total", "p3_n_pivotal_trials",
                  "p3_primary_endpoint", "p3_comparator", "p3_planned_n", "p3_same_endpoint_as_p2",
                  "p3_non_inferiority"}


def compute_path(case: Case) -> list[dict]:
    meta = case.read_json("case.json")
    side = case.root.with_name(case.root.name + ".meta.json")
    levels = json.loads(side.read_text(encoding="utf-8"))["levels"] if side.exists() else []
    top = ORDER.index(meta["level"])
    manifest = {f["path"]: f for f in case.read_json("manifest.json", {"files": []})["files"]}
    studies = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(case.root.glob("raw/ctgov/*.json"))]
    registered = {f["path"].split("/")[-1][:-5]: f for f in manifest.values() if f["source"] == "ctgov"}
    timeline = case.read_json("timeline.json", [])
    claude = (case.read_json("extracted/fields.json") or {}).get("fields", {})

    path = []
    for lv in levels:
        if not lv["reached"] or ORDER.index(lv["level"]) > top:
            continue
        cutoff = meta["cutoff"] if lv["level"] == meta["level"] else lv["cutoff"]
        cut = date.fromisoformat(cutoff)
        structured = []
        for s in studies:
            nct = s["protocolSection"]["identificationModule"]["nctId"]
            reg = registered.get(nct, {}).get("source_date")
            if reg and reg <= cutoff:
                structured.append(structure_trial(s, meta["names"], cut))
        if lv["level"] == meta["level"]:
            pred = case.read_json("prediction.json")      # the headline prediction is this level's point
        else:
            fields = {}
            for name, f in claude.items():
                if ORDER.index(FIELDS[name][2]) > ORDER.index(lv["level"]):
                    continue
                src_file = f.get("source_file") or ""
                if src_file == "extracted/structured.json":
                    if name in _DESIGN_FIELDS:
                        fields[name] = f
                    continue
                sdate = manifest.get(src_file, {}).get("source_date")
                if (sdate is None and f.get("quote_verified")) or (sdate and sdate > cutoff):
                    continue
                fields[name] = f
            feats, _ = compute(lv["level"], meta, structured, [e for e in timeline if e["date"] <= cutoff], fields)
            pred = provisional.predict(lv["level"], feats)
        path.append({"level": lv["level"], "name": LEVELS[lv["level"]], "cutoff": cutoff,
                     "p": pred["p_approval"]["median"], "ci95": pred["p_approval"]["ci95"],
                     "events": [e["event_type"] for e in timeline
                                if (not path or e["date"] > path[-1]["cutoff"]) and e["date"] <= cutoff
                                and e["source"] != "ctgov"]})
    case.write_json("analysis/path.json", path)
    return path
