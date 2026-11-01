from fdaengine.features.build import compute


def _ev(etype, value, d="2024-01-10"):
    return {"event_type": etype, "date": d, "value": value, "quote": f"{etype} quote"}


def test_press_release_p_value_fills_phase3_result_when_no_registry_results():
    feats, _ = compute("L2", {"sponsor": "X"}, [], [_ev("PHASE3_POSITIVE", "0.001"), _ev("PHASE3_POSITIVE", "0.02")], {})
    assert feats["p3_primary_endpoint_met"]["value"] == "yes"
    assert feats["p3_best_pvalue"]["value"] == 0.02           # worst readout, as for registry results
    assert feats["p3_best_pvalue"]["source"] == "timeline"
    assert feats["p3_z"]["value"] > 2


def test_negative_readout_without_p_value_never_borrows_a_positive_ones():
    feats, _ = compute("L2", {"sponsor": "X"}, [], [_ev("PHASE3_NEGATIVE", None), _ev("PHASE3_POSITIVE", "0.001")], {})
    assert feats["p3_primary_endpoint_met"]["value"] == "no"
    assert "p3_best_pvalue" not in feats


def test_empty_event_values_read_back_as_nan_are_ignored():
    feats, _ = compute("L2", {"sponsor": "X"}, [], [_ev("PHASE3_POSITIVE", float("nan"))], {})
    assert feats["p3_primary_endpoint_met"]["value"] == "yes"
    assert "p3_best_pvalue" not in feats and "p3_z" not in feats


def test_phase3_results_are_hidden_before_l2():
    feats, _ = compute("L1", {"sponsor": "X"}, [], [_ev("PHASE2_POSITIVE", "0.03"), _ev("PHASE3_POSITIVE", "0.001")], {})
    assert feats["p2_best_pvalue"]["value"] == 0.03
    assert "p3_best_pvalue" not in feats


def test_months_to_pdufa_uses_the_latest_announced_goal_date():
    tl = [{"event_type": "PDUFA_DATE", "date": "2024-02-01", "value": "2024-09-30", "quote": "goal date"},
          {"event_type": "PDUFA_DATE", "date": "2024-08-01", "value": "2024-12-30", "quote": "extended"}]
    feats, _ = compute("L5", {"sponsor": "X", "cutoff": "2024-09-01"}, [], tl, {})
    assert feats["months_to_pdufa"]["value"] == 3.9
    feats, _ = compute("L2", {"sponsor": "X", "cutoff": "2024-09-01"}, [], tl, {})
    assert "months_to_pdufa" not in feats
