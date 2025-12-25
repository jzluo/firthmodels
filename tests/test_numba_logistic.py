import numpy as np
import pytest
from scipy.special import expit

from firthmodels import NUMBA_AVAILABLE
from firthmodels.logistic import _Workspace, compute_logistic_quantities

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")

if NUMBA_AVAILABLE:
    from firthmodels._numba.logistic import (
        compute_logistic_quantities as compute_logistic_quantities_numba,
    )
    from firthmodels._numba.logistic import expit, log1pexp, max_abs


@pytest.mark.parametrize(
    "x", [-np.inf, -710, -100, -10, -5, -1, 0, 1, 5, 10, 100, 710, np.inf]
)
def test_expit(x):
    assert expit(x) == pytest.approx(expit(x), rel=1e-14, abs=1e-300)


@pytest.mark.parametrize(
    "x", [-np.inf, -710, -100, -10, -5, -1, 0, 1, 5, 10, 100, 710, np.inf]
)
def test_log1pexp(x):
    assert log1pexp(x) == pytest.approx(np.logaddexp(0, x), rel=1e-14)


def test_max_abs():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(100)
    np.testing.assert_allclose(max_abs(arr), np.abs(arr).max())


def test_compute_logistic_quantities():
    rng = np.random.default_rng(0)
    n, k = 50, 5
    X = rng.standard_normal((n, k))
    y = rng.integers(0, 2, n).astype(np.float64)
    beta = rng.standard_normal(k)
    sample_weight = np.ones(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)

    ws = _Workspace(n, k)
    ref = compute_logistic_quantities(X, y, beta, sample_weight, offset, ws)

    loglik, info = compute_logistic_quantities_numba(
        X,
        y,
        beta,
        sample_weight,
        offset,
        ws.numba_buffers(),
    )

    assert info == 0
    np.testing.assert_allclose(loglik, ref.loglik, rtol=1e-14)
    np.testing.assert_allclose(ws.temp_k, ref.modified_score, rtol=1e-14)
    np.testing.assert_allclose(ws.fisher_info_aug, ref.fisher_info, rtol=1e-14)
