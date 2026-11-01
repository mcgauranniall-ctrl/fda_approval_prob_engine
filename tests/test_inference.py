import importlib.util
from pathlib import Path

import numpy as np
import statsmodels.api as sm

_spec = importlib.util.spec_from_file_location(
    "fit_models", Path(__file__).resolve().parent.parent / "scripts" / "07_fit_models.py")
fit_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_models)


def test_firth_close_to_mle_on_large_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.3 + 0.8 * x)))).astype(int)
    X = sm.add_constant(x)
    b, se = fit_models.firth(y, X)
    mle = sm.Logit(y, X).fit(disp=0)
    assert np.allclose(b, mle.params, atol=0.02)
    assert np.allclose(se, mle.bse, rtol=0.05)


def test_firth_finite_under_separation():
    x = np.array([0] * 20 + [1] * 10, dtype=float)
    y = np.array([0] * 12 + [1] * 8 + [1] * 10)          # every exposed row is a success
    b, se = fit_models.firth(y, sm.add_constant(x))
    assert np.isfinite(b).all() and np.isfinite(se).all()
    assert b[1] > 1                                      # large but finite effect
