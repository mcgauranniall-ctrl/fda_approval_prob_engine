import pandas as pd

from fdaengine.dataset import drugs
from fdaengine.features import build as fb
from fdaengine.features import context as cx


def _study(nct, interventions, mesh, ancestors, conditions, phase="PHASE3"):
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": "t"},
        "designModule": {"phases": [phase]},
        "statusModule": {},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Pharma"}},
        "armsInterventionsModule": {"armGroups": [{"label": "A", "type": "EXPERIMENTAL", "interventionNames": []}],
                                    "interventions": [{"type": t, "name": n, "armGroupLabels": ["A"]}
                                                      for n, t in interventions]},
        "conditionsModule": {"conditions": conditions}},
        "derivedSection": {"conditionBrowseModule": {
            "meshes": [{"id": i, "term": t} for i, t in mesh],
            "ancestors": [{"id": i, "term": t} for i, t in ancestors]}}}


def _reference():
    idx = drugs.build_index([("UNII1", "UNII1", "chemical", "zelotinib", "cn", True),
                             ("UNII1", "UNII1", "chemical", "ACM-101", "cd", False),
                             ("UNII2", "UNII2", "protein", "fakumab", "cn", True)])
    approvals = pd.DataFrame({"ApplNo": ["111111", "222222", "333333"], "SponsorName": ["ACME PHARMA"] * 3,
                              "approval_date": ["2015-03-01", "2021-06-01", "2018-01-01"],
                              "drug_id": ["UNII:UNII1", "UNII:UNII1", "UNII:OTHER"]})
    crls = cx.CompanyDates([("Acme Pharma Inc.", "2019-05-01"), ("Acme Pharma", "2023-01-01")])
    return cx.Reference(idx, cx.approvals_by_drug(approvals), [], {},
                        cx.CompanyDates(cx.sponsor_approval_pairs(approvals, [])), crls,
                        {"D1": {"D1", "DV"}}, {"DV"}, {"D1": "Hepatitis B, Chronic", "DV": "Virus Diseases"})


def test_live_context_matches_the_training_definitions():
    ref = _reference()
    s = _study("NCT1", [("ACM-101", "DRUG")], [("D1", "Hepatitis B, Chronic")], [("DV", "Virus Diseases")],
               ["Chronic Hepatitis B"])
    meta = {"names": ["zelotinib", "ACM-101"], "sponsor": "Acme Pharma", "applications": ["NDA222222"]}
    c = cx.live_context(meta, [s], "2022-01-01", ref)
    assert c["therapeutic_area"][0] == "infectious_disease"
    assert c["modality"][0] == "small_molecule"
    assert c["is_combination"][0] is False
    assert c["sponsor_prior_approvals"][0] == 3 and c["sponsor_prior_crls"][0] == 1
    # The 2015 approval (another application) counts; the case's own 2021 approval is its outcome, not context.
    assert c["moiety_approved"][0] is True and "2015-03-01" in c["moiety_approved"][1]
    assert c["own_application_approved"][0] == "2021-06-01"
    early = cx.live_context(meta, [s], "2014-01-01", ref)
    assert early["moiety_approved"][0] is False and "own_application_approved" not in early
    assert early["sponsor_prior_approvals"][0] == 0


def test_live_context_biologic_and_unresolved_names():
    ref = _reference()
    s = _study("NCT2", [("fakumab", "BIOLOGICAL"), ("placebo", "DRUG")], [], [], ["Plaque psoriasis"])
    c = cx.live_context({"names": ["fakumab"], "sponsor": ""}, [s], "2022-01-01", ref)
    assert c["modality"][0] == "biologic"
    assert c["therapeutic_area"][0] == "immunology"           # text fallback ('psoria')
    assert "sponsor_prior_approvals" not in c
    unknown = cx.live_context({"names": ["XYZ-9"], "sponsor": ""}, [], "2022-01-01", ref)
    assert "moiety_approved" not in unknown and "therapeutic_area" not in unknown


def test_reference_context_outranks_claude_without_a_conflict():
    ctx = {"therapeutic_area": ("oncology", "MeSH"), "sponsor_prior_approvals": (4, "Drugs@FDA")}
    claude = {"therapeutic_area": {"value": "haematological malignancy", "quote": "a blood cancer"}}
    feats, conflicts = fb.compute("L1", {"sponsor": "Acme"}, [], [], claude, ctx)
    assert feats["therapeutic_area"] == {"value": "oncology", "source": "reference", "detail": "MeSH"}
    assert feats["sponsor_prior_approvals"]["value"] == 4
    assert not conflicts
    feats, _ = fb.compute("L1", {"sponsor": "Acme"}, [], [], claude)
    assert feats["therapeutic_area"]["source"] == "claude"


def test_own_approval_raises_a_data_flag():
    from fdaengine.predict import flags
    out = flags.data_flags("L3", {"own_application_approved": {"value": "2026-10-20"}}, [], [], {}, {})
    assert any("already known" in f["text"] for f in out)
