from fdaengine.model import provisional
from fdaengine.validate.quotes import find_quote


def test_find_quote(tmp_path):
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "pr.txt").write_text("Both studies   met their primary endpoint and established proof\n"
                                              "of concept for VX-548.", encoding="utf-8")
    assert find_quote(tmp_path, "Both studies met their primary endpoint", "text/pr.txt") == "text/pr.txt"
    assert find_quote(tmp_path, "Both studies met their primary endpoint", "text/missing.txt") == "text/pr.txt"
    assert find_quote(tmp_path, "The trial failed its primary endpoint badly", None) is None


def _f(**kw):
    return {k: {"value": v, "source": "test"} for k, v in kw.items()}


def test_provisional_model_direction():
    base = provisional.predict("L2", _f(therapeutic_area="pain"))["p_approval"]["median"]
    win = provisional.predict("L2", _f(therapeutic_area="pain", p3_primary_endpoint_met="yes"))["p_approval"]["median"]
    loss = provisional.predict("L2", _f(therapeutic_area="pain", p3_primary_endpoint_met="no"))["p_approval"]["median"]
    assert loss < 0.2 < base < win
    p = provisional.predict("L3", _f(therapeutic_area="pain"))
    assert "p_next_action" in p and p["p_next_action"]["median"] < p["p_approval"]["median"]
    lo, hi = p["p_approval"]["ci95"]
    assert 0 < lo < p["p_approval"]["median"] < hi < 1
