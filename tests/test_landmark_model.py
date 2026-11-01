import numpy as np

from fdaengine.model import landmark as lmk


def test_phase3_results_do_not_enter_at_l1_and_missing_values_are_neutral():
    f = {"p3_primary_endpoint_met": "yes", "p3_z": 4.0, "p2_primary_endpoint_met": "no", "sponsor_size": "large"}
    l1 = dict(zip(lmk.KEYS, lmk.raw_row("L1", f)))
    l2 = dict(zip(lmk.KEYS, lmk.raw_row("L2", f)))
    assert l1["p3_met"] is None and l2["p3_met"] == 1.0 and l1["p2_met"] == -1.0
    raw = lmk.raw_matrix(["L2", "L2"], [f, {"p3_z": 2.0}])
    scale = lmk.standardiser(raw)
    x = lmk.design(raw, scale)
    assert x[1, lmk.KEYS.index("p2_z")] == 0.0              # missing → training mean → no contribution


def test_label_priors_are_moment_matched_betas():
    err = {lv: {"ppv_beta": [34, 3], "npv_beta": [34, 4], "pipeline_rate": 0.4} for lv in ("L1", "L2", "L3", "L5")}
    pri = lmk.label_priors(err)
    a, b = pri["L4"]["sens"]
    assert pri["L4"] == pri["L3"] and 0.8 < a / (a + b) < 1.0


def test_predict_matches_the_posterior_and_lists_contributions():
    J, D = len(lmk.KEYS), 200
    post = {"alpha": np.zeros((D, 5)), "a_ta": np.zeros((D, len(lmk.TAS))), "beta": np.zeros((D, J)),
            "delta": np.zeros((D, 3, J))}
    post["beta"][:, lmk.KEYS.index("p3_met")] = 2.0
    m = lmk.FittedModel("test", post, {s.key: [0.0, 1.0] for s in lmk.SPECS if s.kind == "cont"}, {})
    out = m.predict("L2", {"p3_primary_endpoint_met": {"value": "yes"}})
    assert abs(out["p_approval"]["mean"] - 1 / (1 + np.exp(-2))) < 1e-9
    assert [c["key"] for c in out["contributions"]] == ["p3_met"]
    assert m.predict("L1", {"p3_primary_endpoint_met": {"value": "yes"}})["p_approval"]["mean"] == 0.5
