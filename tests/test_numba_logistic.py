import numpy as np
import pytest
from scipy.special import expit as scipy_expit

from firthmodels import NUMBA_AVAILABLE, FirthLogisticRegression
from firthmodels._solvers import newton_raphson
from firthmodels.logistic import _Workspace, compute_logistic_quantities

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")

if NUMBA_AVAILABLE:
    from firthmodels._numba.logistic import (
        _STATUS_CONVERGED,
        _STATUS_STEP_HALVING_FAILED,
        expit,
        log1pexp,
        max_abs,
        newton_raphson_logistic,
    )
    from firthmodels._numba.logistic import (
        compute_logistic_quantities as compute_logistic_quantities_numba,
    )


@pytest.mark.parametrize(
    "x", [-np.inf, -710, -100, -10, -5, -1, 0, 1, 5, 10, 100, 710, np.inf]
)
def test_expit(x):
    assert expit(x) == pytest.approx(scipy_expit(x), rel=1e-14, abs=1e-300)


@pytest.mark.parametrize(
    "x", [-np.inf, -710, -100, -10, -5, -1, 0, 1, 5, 10, 100, 710, np.inf]
)
def test_log1pexp(x):
    assert log1pexp(x) == pytest.approx(np.logaddexp(0, x), rel=1e-14)


def test_max_abs():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(100)
    np.testing.assert_allclose(max_abs(arr), np.abs(arr).max())


def test_max_abs_nan_returns_inf():
    arr = np.array([1e-6, np.nan, 1e-7])
    assert max_abs(arr) == np.inf


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

    loglik = compute_logistic_quantities_numba(
        X,
        y,
        beta,
        sample_weight,
        offset,
        ws.numba_buffers(),
    )

    np.testing.assert_allclose(loglik, ref.loglik, rtol=1e-14)
    np.testing.assert_allclose(ws.temp_k, ref.modified_score, rtol=1e-14)
    np.testing.assert_allclose(ws.fisher_info_aug, ref.fisher_info, rtol=1e-14)


def test_compute_logistic_quantities_non_pd_symmetrizes():
    rng = np.random.default_rng(0)
    n = 12
    x = rng.standard_normal(n)
    X = np.column_stack([x, np.zeros_like(x)])  # rank-deficient
    y = rng.integers(0, 2, n).astype(np.float64)
    beta = np.zeros(2, dtype=np.float64)
    sample_weight = np.ones(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)

    ws = _Workspace(n, 2)
    loglik = compute_logistic_quantities_numba(
        X,
        y,
        beta,
        sample_weight,
        offset,
        ws.numba_buffers(),
    )

    p = scipy_expit(np.zeros(n))
    w = sample_weight * p * (1.0 - p)
    XtW = X.T * np.sqrt(w)
    expected = XtW @ XtW.T
    np.testing.assert_allclose(ws.fisher_info, expected, rtol=1e-14)
    np.testing.assert_array_equal(ws.fisher_info, ws.fisher_info.T)
    assert np.isneginf(loglik)


# def test_numba_matches_numpy_on_rank_deficient():
#     """Numba and numpy backends produce same results on rank-deficient data."""
#     rng = np.random.default_rng(0)
#     n = 12
#     x = rng.standard_normal(n)
#     X = np.column_stack([x, np.zeros_like(x)])  # rank-deficient
#     y = rng.integers(0, 2, n).astype(np.float64)

#     model_numpy = FirthLogisticRegression(backend="numpy").fit(X, y)
#     model_numba = FirthLogisticRegression(backend="numba").fit(X, y)

#     np.testing.assert_allclose(model_numba.coef_, model_numpy.coef_, rtol=1e-14)
#     np.testing.assert_allclose(
#         model_numba.intercept_, model_numpy.intercept_, rtol=1e-14
#     )
#     assert model_numba.converged_ == model_numpy.converged_


def run_both_backends(X, y, sample_weight=None, offset=None, **solver_params):
    n, k = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)

    defaults = dict(max_iter=25, max_step=5.0, max_halfstep=25, gtol=1e-4, xtol=1e-4)
    defaults.update(solver_params)

    # numpy/scipy backend
    workspace = _Workspace(n, k)

    def compute_quantities(beta):
        return compute_logistic_quantities(X, y, beta, sample_weight, offset, workspace)

    ref = newton_raphson(compute_quantities, k, **defaults)

    # numba backend
    numba_result = newton_raphson_logistic(
        X, y, sample_weight, offset, workspace=workspace.numba_buffers(), **defaults
    )

    return ref, numba_result


class TestNewtonRaphsonNumba:
    def test_separated_data(self):
        X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        ref, numba = run_both_backends(X, y)

        assert numba[-1] == _STATUS_CONVERGED
        np.testing.assert_allclose(numba[0], ref.beta, rtol=1e-14)

    def test_step_halving_failure_returns_consistent_fisher_info(self):
        # dataset chosen to deterministically hit the step-halving failure path.
        X = np.array(
            [
                [2.04091912],
                [-2.55566503],
                [0.41809885],
                [-0.56776961],
                [-0.45264929],
                [-0.21559716],
            ],
            dtype=np.float64,
        )
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        sample_weight = np.ones(6, dtype=np.float64)
        offset = np.zeros(6, dtype=np.float64)

        workspace = _Workspace(6, 1)
        beta, loglik, fisher_info, _, status = newton_raphson_logistic(
            X,
            y,
            sample_weight,
            offset,
            max_iter=25,
            max_step=5.0,
            max_halfstep=1,
            gtol=1e-12,
            xtol=1e-12,
            workspace=workspace.numba_buffers(),
        )
        assert status == _STATUS_STEP_HALVING_FAILED
        fisher_info = fisher_info.copy()

        ref_workspace = _Workspace(6, 1)
        # Recompute quantities for the returned beta to make sure the returned fisher_info
        # corresponds to the accepted (rather than tried and rejected) beta and loglik
        loglik_ref = compute_logistic_quantities_numba(
            X,
            y,
            beta,
            sample_weight,
            offset,
            ref_workspace.numba_buffers(),
        )

        np.testing.assert_allclose(loglik_ref, loglik, rtol=1e-10)
        np.testing.assert_allclose(
            ref_workspace.fisher_info_aug, fisher_info, rtol=1e-10
        )


class TestFirthLogisticRegressionNumba:
    def test_numba_matches_logistf_with_separation(self, separation_data):
        """Matches logistf on quasi-separated data."""
        X, y = separation_data
        model = FirthLogisticRegression(backend="numba")
        model.fit(X, y)
        model.lrt()
        ci = model.conf_int(alpha=0.05, method="pl")

        # coefficients
        expected_intercept = -0.4434562830
        expected_coef = np.array(
            [3.6577140153, 0.6759781501, -0.8633119501, 0.3385788510]
        )
        np.testing.assert_allclose(model.intercept_, expected_intercept, rtol=1e-4)
        np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-4)
        assert model.converged_

        # Wald
        expected_wald_bse = np.array(
            [1.4822786334, 0.2687886213, 0.3874453554, 0.1370778814, 0.3452906671]
        )
        np.testing.assert_allclose(model.bse_, expected_wald_bse, rtol=1e-4)

        # LRT
        expected_lrt_pvalues = np.array(
            [
                0.0002084147149,
                0.0093173148959,
                0.0236385713206,
                0.0055887969164,
                0.1997147194,
            ]
        )
        expected_lrt_bse = np.array(
            [0.9862809793, 0.2599729631, 0.3814979166, 0.1221874315, 0.3458113448]
        )
        np.testing.assert_allclose(model.lrt_pvalues_, expected_lrt_pvalues, rtol=1e-4)
        np.testing.assert_allclose(model.lrt_bse_, expected_lrt_bse, rtol=1e-4)

        # profile CI
        expected_lower = np.array(
            [1.3901042899, 0.1602568036, -1.6677883758, 0.0893054012, -1.16089382506]
        )
        expected_upper = np.array(
            [8.5876062114, 1.2568999211, -0.1135186948, 0.6572681313, 0.2295139209]
        )

        np.testing.assert_allclose(ci[:, 0], expected_lower, rtol=1e-4)
        np.testing.assert_allclose(ci[:, 1], expected_upper, rtol=1e-4)

    def test_numba_fit_intercept_false(self, separation_data):
        """Fits without intercept."""
        X, y = separation_data
        n_features = X.shape[1]
        model = FirthLogisticRegression(fit_intercept=False, backend="numba")
        model.fit(X, y)

        assert model.intercept_ == 0.0
        assert len(model.coef_) == n_features

    def test_numba_classes_encoded_correctly(self):
        """Handles arbitrary binary labels."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))

        for labels in [(0, 1), (1, 2), (-1, 1)]:
            y = rng.choice(labels, 50)
            model = FirthLogisticRegression(backend="numba")
            model.fit(X, y)
            np.testing.assert_array_equal(model.classes_, sorted(labels))


def test_numba_symmetrizes_fisher_info_on_cholesky_fail():
    rng = np.random.default_rng(0)
    n = 8
    x = rng.standard_normal(n)
    X = np.column_stack([x, x])  # rank deficient
    y = rng.integers(0, 2, n).astype(np.float64)
    beta = np.zeros(2, dtype=np.float64)
    sample_weight = np.ones(n, dtype=np.float64)
    offset = np.zeros(n, dtype=np.float64)

    ws = _Workspace(n, 2)
    loglik = compute_logistic_quantities_numba(
        X, y, beta, sample_weight, offset, ws.numba_buffers()
    )

    sqrt_w = np.sqrt(sample_weight * 0.25)
    XtW = X.T * sqrt_w
    expected = XtW @ XtW.T
    np.testing.assert_allclose(ws.fisher_info, expected, rtol=1e-14)
    np.testing.assert_array_equal(ws.fisher_info, ws.fisher_info.T)
    assert np.isneginf(loglik)
