import numpy as np

from fdaengine.model import landmark as lmk
from fdaengine.model import timing as tm


def test_curve_jumps_at_the_pdufa_date_and_is_monotone():
    rng = np.random.default_rng(0)
    n = 300
    mp = rng.uniform(2, 10, n)                          # months to the announced goal date
    t = np.where(rng.random(n) < 0.8, mp + 0.1, mp + 12)   # 80% approved at the goal date, the rest a cycle later
    cov = [tm.covariates("L5", {"months_to_pdufa": m}) for m in mp]
    model = tm.TimingModel.fit(np.full(n, lmk.FIT_LEVELS.index("L5")), cov, t, np.full(n, 36.0),
                               np.arange(n), n_boot=0)
    g = model.g("L5", {"months_to_pdufa": 5.0}, np.array([4, 6, 12, 24], dtype=float))[0]
    assert g[0] < 0.3 and g[1] > 0.6 and g[3] > 0.9
    assert np.all(np.diff(g) >= 0)


def test_approval_by_month_multiplies_probability_and_curve():
    p = np.full(1000, 0.5)
    g = np.array([[0.2, 0.6, 1.0]])
    out = tm.approval_by_month(p, g, np.array([6, 12, 24]))
    assert [round(o["p"], 3) for o in out] == [0.1, 0.3, 0.5]


def test_plan_year_probabilities_follow_the_calendar(monkeypatch):
    from fdaengine.model import registry
    monkeypatch.setenv("FDAENGINE_MODEL", "fitted")
    if registry._fitted() is None or registry._timing() is None:
        return                                               # models not built in this checkout
    t = registry.approval_timing("L3", {"priority_review": {"value": True}}, "2024-11-01")
    soon, later = t["by_plan_year"]
    assert soon["p"] <= t["by_months"][0]["p"] <= later["p"]   # 2 months ≤ 6 months ≤ 14 months
