"""Flatten ClinicalTrials.gov API v2 study records into relational rows (for DuckDB / parquet).

Only identity, design, date and condition/intervention fields are flattened here. Results stay in the raw JSON and
are read by `prepare.ctgov_structured.structure_trial()`, the same code live cases use.
"""

from __future__ import annotations

from fdaengine.dates import parse_partial
from fdaengine.sources.ctgov import PHASE_ORDER


def _d(struct: dict | None, end: bool = False) -> str | None:
    d = parse_partial((struct or {}).get("date"), end)
    return d.isoformat() if d else None


def flatten(study: dict) -> dict[str, list[dict]]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    spons = p.get("sponsorCollaboratorsModule", {})
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    arms_mod = p.get("armsInterventionsModule", {})
    derived = study.get("derivedSection", {})
    nct = ident.get("nctId", "")
    phases = design.get("phases", [])
    collab = spons.get("collaborators", [])
    countries = {loc.get("country") for loc in p.get("contactsLocationsModule", {}).get("locations", [])
                 if loc.get("country")}

    studies = [{
        "nct_id": nct,
        "title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "acronym": ident.get("acronym"),
        "phases": "|".join(phases),
        "max_phase": max((PHASE_ORDER.get(x, 0) for x in phases), default=0),
        "is_phase3": "PHASE3" in phases,
        "status": status.get("overallStatus", ""),
        "why_stopped": status.get("whyStopped"),
        "lead_sponsor": spons.get("leadSponsor", {}).get("name", ""),
        "lead_sponsor_class": spons.get("leadSponsor", {}).get("class", ""),
        "industry_collaborator": any(c.get("class") == "INDUSTRY" for c in collab),
        "collaborators": "|".join(c.get("name", "") for c in collab),
        "start_date": _d(status.get("startDateStruct")),
        "start_date_type": (status.get("startDateStruct") or {}).get("type"),
        "primary_completion_date": _d(status.get("primaryCompletionDateStruct"), True),
        "primary_completion_type": (status.get("primaryCompletionDateStruct") or {}).get("type"),
        "completion_date": _d(status.get("completionDateStruct"), True),
        "first_post_date": _d(status.get("studyFirstPostDateStruct")),
        "results_first_post_date": _d(status.get("resultsFirstPostDateStruct")),
        "last_update_post_date": _d(status.get("lastUpdatePostDateStruct")),
        "has_results": bool(study.get("hasResults")),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "enrollment_type": (design.get("enrollmentInfo") or {}).get("type"),
        "allocation": info.get("allocation"),
        "intervention_model": info.get("interventionModel"),
        "primary_purpose": info.get("primaryPurpose"),
        "masking": (info.get("maskingInfo") or {}).get("masking"),
        "n_arms": len(arms_mod.get("armGroups", [])),
        # Site countries and FDA oversight as the record stands today (sites are mostly listed during recruitment;
        # 'isFdaRegulatedDrug' exists only for records updated since 2017).
        "countries": "|".join(sorted(countries)),
        "has_us_site": "United States" in countries,
        "fda_regulated_drug": p.get("oversightModule", {}).get("isFdaRegulatedDrug"),
    }]

    arms = [{"nct_id": nct, "label": a.get("label", ""), "type": a.get("type") or "",
             "intervention_names": "|".join(a.get("interventionNames", []))}
            for a in arms_mod.get("armGroups", [])]
    arm_type = {a["label"]: a["type"] for a in arms}

    interventions = []
    for i, iv in enumerate(arms_mod.get("interventions", [])):
        labels = iv.get("armGroupLabels", [])
        types = sorted({arm_type.get(lbl, "") for lbl in labels})
        interventions.append({
            "nct_id": nct, "iv_idx": i, "type": iv.get("type", ""), "name": iv.get("name", ""),
            "other_names": "|".join(iv.get("otherNames", [])),
            "arm_labels": "|".join(labels), "arm_types": "|".join(types),
        })

    conditions = [{"nct_id": nct, "pos": i, "condition": c}
                  for i, c in enumerate(p.get("conditionsModule", {}).get("conditions", []))]
    cbm = derived.get("conditionBrowseModule", {})
    condition_mesh = [{"nct_id": nct, "pos": i, "mesh_id": m.get("id"), "term": m.get("term")}
                      for i, m in enumerate(cbm.get("meshes", []))]
    condition_ancestors = [{"nct_id": nct, "mesh_id": m.get("id"), "term": m.get("term")}
                           for m in cbm.get("ancestors", [])]
    ibm = derived.get("interventionBrowseModule", {})
    intervention_mesh = [{"nct_id": nct, "mesh_id": m.get("id"), "term": m.get("term")}
                         for m in ibm.get("meshes", [])]
    return {"studies": studies, "arms": arms, "interventions": interventions, "conditions": conditions,
            "condition_mesh": condition_mesh, "condition_ancestors": condition_ancestors,
            "intervention_mesh": intervention_mesh}


TABLES = ("studies", "arms", "interventions", "conditions", "condition_mesh", "condition_ancestors",
          "intervention_mesh")
SCHEMA_VERSION = 2                  # bump when `flatten` output changes, so cached tables are rebuilt
