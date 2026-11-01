from datetime import date

from fdaengine.prepare.ctgov_structured import structure_trial
from fdaengine.timeline.events import Event, Timeline
from fdaengine.timeline.levels import auto_level, compute, decision


def _timeline():
    return Timeline([
        Event("P2_PRIMARY_COMPLETION", "2022-02-17", "ctgov"),
        Event("PHASE2_POSITIVE", "2022-03-01", "press_release"),
        Event("P3_PRIMARY_COMPLETION", "2023-08-25", "ctgov"),
        Event("PHASE3_POSITIVE", "2024-02-05", "press_release"),
        Event("NDA_ACCEPTED", "2024-08-01", "press_release"),
        Event("PDUFA_DATE", "2024-08-01", "press_release", value="2025-01-30"),
        Event("APPROVED", "2025-01-30", "press_release"),
        Event("FDA_APPROVAL_ACTION", "2025-01-30", "drugsfda"),
    ])


def test_level_cutoffs():
    levels = {lv.level: lv for lv in compute(_timeline(), date(2026, 1, 1))}
    assert levels["L1"].cutoff == "2023-08-24"           # day before first P3 primary completion
    assert levels["L2"].cutoff == "2024-02-06"
    assert levels["L3"].cutoff == "2024-08-02"
    assert levels["L5"].cutoff == "2025-01-23"           # PDUFA - 7 days
    assert not levels["L4"].reached and not levels["L6"].reached
    assert decision(_timeline()) == ("APPROVED", date(2025, 1, 30))


def test_levels_respect_as_of():
    levels = compute(_timeline(), date(2024, 12, 1))
    assert [lv.level for lv in levels if lv.reached] == ["L1", "L2", "L3"]
    assert auto_level(levels).level == "L3"


def _study(results_posted: str):
    return {
        "hasResults": True,
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Drug X vs placebo"},
            "statusModule": {"overallStatus": "COMPLETED", "resultsFirstPostDateStruct": {"date": results_posted}},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "S", "class": "INDUSTRY"}},
            "designModule": {"phases": ["PHASE3"], "designInfo": {"allocation": "RANDOMIZED",
                                                                   "maskingInfo": {"masking": "DOUBLE"}},
                             "enrollmentInfo": {"count": 200}},
            "armsInterventionsModule": {"armGroups": [{"label": "Drug X", "type": "EXPERIMENTAL"},
                                                      {"label": "Placebo", "type": "PLACEBO_COMPARATOR"}]},
        },
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [{
            "type": "PRIMARY", "title": "Primary", "groups": [{"id": "OG0", "title": "Placebo"},
                                                              {"id": "OG1", "title": "Drug X"}],
            "analyses": [{"groupIds": ["OG0", "OG1"], "pValue": "<0.001"}]}]}},
    }


def test_results_posted_after_cutoff_are_excluded():
    before = structure_trial(_study("2024-01-10"), ["Drug X"], date(2024, 6, 1))
    after = structure_trial(_study("2024-09-10"), ["Drug X"], date(2024, 6, 1))
    assert before["results_status"] == "available" and before["key_analysis"]["p_value"] == 0.001
    assert after["results_status"] == "posted_after_cutoff" and "key_analysis" not in after


def test_each_crl_review_cycle_is_kept():
    from fdaengine.timeline.events import build
    from fdaengine.timeline.press_release_parser import ParsedEvent

    def crl(d):
        return ParsedEvent("CRL", d, d, "filing_date", "received a complete response letter", f"doc{d}")

    tl = build([], [crl("2012-01-17"), crl("2012-01-26"), crl("2013-12-13")], [])
    assert [e.date for e in tl.all("CRL")] == ["2012-01-17", "2013-12-13"]
