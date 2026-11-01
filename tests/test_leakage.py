"""Leakage guards: nothing dated after a level's cut-off may reach its feature row (CLAUDE.md, PLAN.md §3)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fdaengine import config
from fdaengine.dataset import landmarks as lm

NAMES = ["X-101"]


def study(nct="NCT1", first_post="2019-01-15", results_post=None, pcd="2020-06-30", enrollment=(300, "ACTUAL")):
    status = {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2019-02-01"},
              "primaryCompletionDateStruct": {"date": pcd}, "studyFirstPostDateStruct": {"date": first_post}}
    if results_post:
        status["resultsFirstPostDateStruct"] = {"date": results_post}
    s = {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": "X-101 in pain"},
        "statusModule": status,
        "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": enrollment[0], "type": enrollment[1]},
                         "designInfo": {"allocation": "RANDOMIZED", "maskingInfo": {"masking": "DOUBLE"}}},
        "armsInterventionsModule": {
            "armGroups": [{"label": "X-101", "type": "EXPERIMENTAL", "interventionNames": ["Drug: X-101"]},
                          {"label": "Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Placebo"]}],
            "interventions": [{"name": "X-101", "type": "DRUG"}, {"name": "Placebo", "type": "DRUG"}]},
        "conditionsModule": {"conditions": ["Pain"]}},
        "hasResults": bool(results_post)}
    if results_post:
        s["resultsSection"] = {"outcomeMeasuresModule": {"outcomeMeasures": []}}
    return s


def test_trial_registered_after_cutoff_is_invisible():
    assert lm.visible_trial(study(first_post="2021-03-01"), NAMES, date(2021, 1, 1)) is None


def test_results_posted_after_cutoff_are_withheld():
    # Results are often posted after the approval: they must not be used at an earlier cut-off.
    rec, available = lm.visible_trial(study(results_post="2022-05-01"), NAMES, date(2021, 1, 1))
    assert rec["results_status"] == "posted_after_cutoff" and "primary_analyses" not in rec
    assert available == "2019-01-15"
    rec, available = lm.visible_trial(study(results_post="2020-11-01"), NAMES, date(2021, 1, 1))
    assert rec["results_status"] == "available" and available == "2020-11-01"


def test_actual_enrolment_of_a_running_trial_is_withheld():
    rec, _ = lm.visible_trial(study(pcd="2021-06-30"), NAMES, date(2021, 1, 1))
    assert rec["design"]["enrollment"] is None
    rec, _ = lm.visible_trial(study(pcd="2020-06-30"), NAMES, date(2021, 1, 1))
    assert rec["design"]["enrollment"] == 300


def test_post_cutoff_events_and_crl_database_never_reach_features():
    events = [
        {"event_type": "PHASE3_POSITIVE", "date": "2020-09-01", "source": "press_release", "quote": "met", "value": None},
        {"event_type": "PRIORITY_REVIEW", "date": "2021-02-01", "source": "press_release", "quote": "pr", "value": None},
        {"event_type": "CRL", "date": "2020-12-01", "source": "fda_crl", "quote": None, "value": None},
    ]
    trials = [lm.visible_trial(study(), NAMES, date(2021, 1, 1))]
    feats, inputs_max = lm.build_row("L3", "2021-01-01", {"sponsor": "Acme"}, trials, events, lm.Context())
    assert "priority_review" not in feats            # announced after the cut-off
    assert "crl_received" not in feats               # FDA CRL database: label source only
    assert feats["p3_primary_endpoint_met"]["value"] == "yes"
    assert inputs_max <= "2021-01-01"


def test_assert_no_leak_rejects_late_inputs_and_post_decision_rows():
    ok = {"programme_id": "PG1", "level": "L2", "cutoff": "2021-01-01", "inputs_max_date": "2020-12-31",
          "approval_date": "2022-01-01"}
    lm.assert_no_leak([ok])
    with pytest.raises(lm.LeakageError):
        lm.assert_no_leak([{**ok, "inputs_max_date": "2021-01-02"}])
    with pytest.raises(lm.LeakageError):
        lm.assert_no_leak([{**ok, "approval_date": "2021-01-01"}])


LANDMARKS = config.DATA_DIR / "processed" / "landmarks.parquet"


@pytest.mark.skipif(not Path(LANDMARKS).exists(), reason="landmarks.parquet not built")
def test_built_dataset_has_no_post_cutoff_inputs():
    df = pd.read_parquet(LANDMARKS, columns=["programme_id", "level", "cutoff", "inputs_max_date", "approval_date"])
    lm.assert_no_leak(df.where(df.notna(), None).to_dict("records"))
