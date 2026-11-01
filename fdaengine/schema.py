"""Extraction field definitions → the JSON Schema passed to `claude -p --json-schema`.

Each field is returned as {value, source_file, location, quote, confidence}. Python verifies every quote against
the case's text files before a value is used (validate/quotes.py).
"""

from __future__ import annotations

THERAPEUTIC_AREAS = ["oncology", "hematology", "rare_disease", "neurology", "psychiatry", "cardiovascular",
                     "infectious_disease", "metabolic_endocrine", "respiratory", "immunology", "dermatology",
                     "ophthalmology", "gastroenterology", "musculoskeletal", "pain", "nephrology", "womens_health",
                     "other"]
MODALITIES = ["small_molecule", "antibody", "adc", "protein_peptide", "oligonucleotide", "cell_therapy",
              "gene_therapy", "vaccine", "other"]
YES_NO = ["yes", "no", "mixed", "unknown"]

# name -> (json type, enum or None, minimum level at which it may be filled, description)
FIELDS: dict[str, tuple[str, list | None, str, str]] = {
    # Programme context
    "therapeutic_area": ("string", THERAPEUTIC_AREAS, "L1", "Therapeutic area of the indication"),
    "modality": ("string", MODALITIES, "L1", "Drug modality"),
    "first_in_class": ("string", YES_NO, "L1", "Is the mechanism first-in-class (no approved drug with it)?"),
    "approved_other_indication": ("boolean", None, "L1", "Was the drug already FDA-approved for another indication before the cut-off?"),
    "mechanism": ("string", None, "L1", "Mechanism of action in a few words"),
    # Phase 2
    "p2_primary_endpoint_met": ("string", YES_NO, "L1", "Did the Phase 2 trial(s) in this indication meet their primary endpoint?"),
    "p2_best_pvalue": ("number", None, "L1", "Primary-endpoint p-value of the best dose vs control in Phase 2"),
    "p2_effect_estimate": ("number", None, "L1", "Phase 2 primary effect estimate vs control (difference, HR, OR...)"),
    "p2_effect_measure": ("string", None, "L1", "What the effect estimate measures, e.g. 'LS mean difference SPID48'"),
    "p2_ci_lower": ("number", None, "L1", "Lower 95% CI of the Phase 2 effect"),
    "p2_ci_upper": ("number", None, "L1", "Upper 95% CI of the Phase 2 effect"),
    "p2_randomized": ("boolean", None, "L1", "Was Phase 2 randomized?"),
    "p2_placebo_controlled": ("boolean", None, "L1", "Was Phase 2 placebo-controlled?"),
    "p2_n_total": ("integer", None, "L1", "Total Phase 2 participants in this indication"),
    "p2_dose_response": ("string", ["yes", "no", "unclear"], "L1", "Did Phase 2 show a dose-response?"),
    # Phase 3 design (registered before results)
    "p3_n_pivotal_trials": ("integer", None, "L1", "Number of pivotal (controlled) Phase 3 trials"),
    "p3_primary_endpoint": ("string", None, "L1", "Phase 3 primary endpoint"),
    "p3_comparator": ("string", ["placebo", "active", "soc_addon", "single_arm", "mixed"], "L1", "Phase 3 comparator"),
    "p3_planned_n": ("integer", None, "L1", "Total planned Phase 3 enrolment across pivotal trials"),
    "p3_same_endpoint_as_p2": ("boolean", None, "L1", "Is the Phase 3 primary endpoint the same as Phase 2's?"),
    "p3_non_inferiority": ("boolean", None, "L1", "Is the pivotal Phase 3 a non-inferiority design?"),
    # Phase 3 results (only public from the readout)
    "p3_primary_endpoint_met": ("string", YES_NO, "L2", "Did the pivotal Phase 3 trial(s) meet the primary endpoint?"),
    "p3_n_positive_pivotal": ("integer", None, "L2", "Number of pivotal Phase 3 trials that met the primary endpoint"),
    "p3_best_pvalue": ("number", None, "L2", "Primary-endpoint p-value in Phase 3 (worst of the pivotal trials if several)"),
    "p3_effect_estimate": ("number", None, "L2", "Phase 3 primary effect estimate vs control"),
    "p3_ci_lower": ("number", None, "L2", "Lower 95% CI of the Phase 3 effect"),
    "p3_ci_upper": ("number", None, "L2", "Upper 95% CI of the Phase 3 effect"),
    "p3_key_secondaries_met": ("string", YES_NO, "L2", "Were key secondary endpoints met?"),
    "p3_sae_rate_drug": ("number", None, "L2", "Serious adverse event rate in drug arm(s), 0-1"),
    "p3_sae_rate_control": ("number", None, "L2", "Serious adverse event rate in control arm, 0-1"),
    "p3_deaths_drug": ("integer", None, "L2", "Deaths in drug arm(s)"),
    "p3_deaths_control": ("integer", None, "L2", "Deaths in control arm"),
    "p3_discontinuation_ae_drug": ("number", None, "L2", "Discontinuation due to AEs in drug arm(s), 0-1"),
    "safety_signal": ("string", ["none", "minor", "moderate", "serious"], "L1", "Most serious safety signal reported so far"),
    # Regulatory
    "btd": ("boolean", None, "L1", "Breakthrough Therapy designation granted for this indication?"),
    "fast_track": ("boolean", None, "L1", "Fast Track designation granted?"),
    "orphan": ("boolean", None, "L1", "Orphan Drug designation granted?"),
    "spa": ("boolean", None, "L1", "Special Protocol Assessment agreed?"),
    "priority_review": ("boolean", None, "L3", "Priority Review granted?"),
    "accelerated_pathway": ("boolean", None, "L3", "Filed under the accelerated approval pathway?"),
    "pdufa_date": ("string", None, "L3", "PDUFA goal date (YYYY-MM-DD)"),
    "pdufa_extended": ("boolean", None, "L3", "Was the PDUFA date extended?"),
    "adcom_held": ("boolean", None, "L3", "Was an FDA Advisory Committee meeting held?"),
    "adcom_votes_yes": ("integer", None, "L4", "AdCom votes in favour on the main question"),
    "adcom_votes_no": ("integer", None, "L4", "AdCom votes against on the main question"),
    "crl_received": ("boolean", None, "L3", "Has a Complete Response Letter been received?"),
    "crl_reason": ("string", ["efficacy", "safety", "cmc_manufacturing", "labeling", "other", "none"], "L3", "Main CRL reason"),
}

_FIELD_OBJ = {
    "type": "object",
    "properties": {
        "value": {},
        "source_file": {"type": ["string", "null"]},
        "location": {"type": ["string", "null"]},
        "quote": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["value", "source_file", "quote", "confidence"],
}


def extraction_schema() -> dict:
    props = {}
    for name, (jtype, enum, _, desc) in FIELDS.items():
        value = {"type": [jtype, "null"], "description": desc}
        if enum:
            value["enum"] = enum + [None]
        obj = {**_FIELD_OBJ, "properties": {**_FIELD_OBJ["properties"], "value": value}}
        props[name] = obj
    flag = {
        "type": "object",
        "properties": {
            "flag": {"type": "string"},
            "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
            "explanation": {"type": "string"},
            "source_file": {"type": ["string", "null"]},
            "quote": {"type": ["string", "null"]},
        },
        "required": ["flag", "severity", "explanation", "source_file", "quote"],
    }
    return {
        "type": "object",
        "properties": {
            "fields": {"type": "object", "properties": props, "required": list(FIELDS)},
            "analyst_flags": {"type": "array", "items": flag},
            "summary": {"type": "string", "description": "3-5 sentence factual summary of the evidence up to the cut-off"},
        },
        "required": ["fields", "analyst_flags", "summary"],
    }
