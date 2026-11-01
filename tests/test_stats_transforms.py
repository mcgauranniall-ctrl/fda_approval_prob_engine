import math

import pytest

from fdaengine.features import stats_transforms as st


def test_parse_pvalue():
    assert st.parse_pvalue("<0.001") == (0.001, "<")
    assert st.parse_pvalue("0.0002") == (0.0002, "=")
    assert st.parse_pvalue(" ≤ 0.05") == (0.05, "<")
    assert st.parse_pvalue("NA") == (None, "")


def test_z_from_p_round_trip():
    assert st.z_from_p(0.05) == pytest.approx(1.95996, abs=1e-4)
    assert st.z_from_p(0.0002) == pytest.approx(3.719, abs=1e-3)


def test_z_from_ratio_ci_and_consistency():
    z = st.z_from_ratio_ci(0.72, 0.58, 0.89)
    assert z == pytest.approx(3.0, abs=0.2)
    assert st.p_consistent_with_ci(0.003, z)
    assert not st.p_consistent_with_ci(0.40, z)


def test_newcombe_matches_published_example():
    # Newcombe (1998) Table II example: 56/70 vs 48/80 -> 0.2000 (0.0524, 0.3339)
    d, lo, hi = st.newcombe_diff_ci(56, 70, 48, 80)
    assert d == pytest.approx(0.2)
    assert lo == pytest.approx(0.0524, abs=5e-4)
    assert hi == pytest.approx(0.3339, abs=5e-4)


def test_risk_ratio_zero_cell():
    rr, lo, hi = st.risk_ratio_ci(0, 50, 3, 50)
    assert rr < 1 and lo < rr < hi


def test_assurance_behaviour():
    weak = st.assurance(1.0, 100, 300)
    strong = st.assurance(3.5, 100, 300)
    assert 0 < weak < strong < 1
    # More shrinkage -> lower assurance
    assert st.assurance(3.0, 100, 300, shrinkage=0.5) < st.assurance(3.0, 100, 300, shrinkage=1.0)
    assert st.smd_to_log_or(1.0) == pytest.approx(math.pi / math.sqrt(3))
