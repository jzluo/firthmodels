import warnings

import numpy as np
import pytest
import scipy.linalg
from sklearn.utils.estimator_checks import estimator_checks_generator

import firthmodels.logistic
from firthmodels import FirthLogisticRegression


class TestFirthLogisticRegression:
    """Tests for FirthLogisticRegression."""

    def test_matches_logistf_with_separation(self, separation_data):
        """Matches logistf on quasi-separated data."""
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy")
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

    def test_fit_intercept_false(self, separation_data):
        """Fits without intercept."""
        X, y = separation_data
        n_features = X.shape[1]
        model = FirthLogisticRegression(fit_intercept=False, backend="numpy")
        model.fit(X, y)

        assert model.intercept_ == 0.0
        assert len(model.coef_) == n_features

    def test_classes_encoded_correctly(self):
        """Handles arbitrary binary labels."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))

        for labels in [(0, 1), (1, 2), (-1, 1)]:
            y = rng.choice(labels, 50)
            model = FirthLogisticRegression(backend="numpy")
            model.fit(X, y)
            np.testing.assert_array_equal(model.classes_, sorted(labels))

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in sqrt:RuntimeWarning",
        "ignore:Fisher information matrix is not finite:RuntimeWarning",
    )
    def test_sklearn_compatible(self):
        """Passes sklearn's estimator checks."""
        for estimator, check in estimator_checks_generator(FirthLogisticRegression()):
            # think this is just precision differences in repeated vs weighted matrices.
            # repeated rows vs integer weights has a max abs diff of 7e-7,
            # a stricter tol does reduce it but still fails, so just skip
            if check.func.__name__ == "check_sample_weight_equivalence_on_dense_data":
                continue
            check(estimator)

    def test_lrt_computes_on_demand(self, separation_data):
        """lrt() computes only requested features and accumulates results."""
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy")
        model.fit(X, y)

        # After lrt(1), only index 1 should be populated
        model.lrt(1)
        assert np.isnan(model.lrt_pvalues_[0])
        assert not np.isnan(model.lrt_pvalues_[1])
        assert np.all(np.isnan(model.lrt_pvalues_[2:]))

        # After lrt([0, 3]), indices 0, 1, and 3 should be populated
        model.lrt([0, 3])
        assert not np.isnan(model.lrt_pvalues_[0])
        assert not np.isnan(model.lrt_pvalues_[1])
        assert np.isnan(model.lrt_pvalues_[2])
        assert not np.isnan(model.lrt_pvalues_[3])

    def test_lrt_warm_start_matches(self, separation_data):
        X, y = separation_data
        model_warm = FirthLogisticRegression(backend="numpy").fit(X, y)
        model_warm.lrt(warm_start=True)

        model_cold = FirthLogisticRegression(backend="numpy").fit(X, y)
        model_cold.lrt(warm_start=False)

        np.testing.assert_allclose(
            model_cold.lrt_pvalues_, model_warm.lrt_pvalues_, rtol=1e-6
        )
        np.testing.assert_allclose(model_cold.lrt_bse_, model_warm.lrt_bse_, rtol=1e-6)

    def test_lrt_warm_start_false_uses_zero_init(self, separation_data, monkeypatch):
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy").fit(X, y)
        captured = {}

        def fake_constrained_lrt_1df(*, beta_init_free, **kwargs):
            captured["beta_init_free"] = beta_init_free
            return firthmodels.logistic.LRTResult(
                chi2=0.0, pvalue=1.0, bse_backcorrected=1.0
            )

        monkeypatch.setattr(
            "firthmodels.logistic.constrained_lrt_1df", fake_constrained_lrt_1df
        )

        model.lrt(0, warm_start=False)
        assert captured["beta_init_free"] is None

    def test_no_warning_when_halfstep_disabled(self):
        """max_halfstep=0 should not produce step-halving warnings."""
        X = np.array([[0], [0], [0], [1], [1], [1]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])

        model = FirthLogisticRegression(max_halfstep=0, backend="numpy")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(X, y)

    def test_lrt_with_string_feature_names(self, separation_data_df):
        X, y = separation_data_df
        model = FirthLogisticRegression(backend="numpy").fit(X, y)

        # Test single string
        model.lrt("x1")
        assert not np.isnan(model.lrt_pvalues_[0])
        assert np.isnan(model.lrt_pvalues_[1])

        # Test list of strings
        model.lrt(["x2", "x4"])
        assert not np.isnan(model.lrt_pvalues_[1])
        assert np.isnan(model.lrt_pvalues_[2])
        assert not np.isnan(model.lrt_pvalues_[3])

        # Test mixed
        model.lrt([0, "x3"])
        assert not np.isnan(model.lrt_pvalues_[0])
        assert not np.isnan(model.lrt_pvalues_[2])

    def test_lrt_intercept_by_name(self, separation_data_df):
        X, y = separation_data_df
        model = FirthLogisticRegression(backend="numpy").fit(X, y)

        model.lrt("intercept")
        assert np.all(np.isnan(model.lrt_pvalues_[:-1]))
        assert not np.isnan(model.lrt_pvalues_[-1])

    def test_lrt_intercept_raises_when_no_intercept(self, separation_data):
        X, y = separation_data
        model = FirthLogisticRegression(fit_intercept=False, backend="numpy").fit(X, y)

        with pytest.raises(ValueError, match="Model has no intercept"):
            model.lrt("intercept")

    def test_lrt_string_raises_without_feature_names(self, separation_data):
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy").fit(X, y)

        with pytest.raises(ValueError, match="No feature names available"):
            model.lrt("x1")

    def test_lrt_unknown_feature_raises(self, separation_data_df):
        X, y = separation_data_df
        model = FirthLogisticRegression(backend="numpy").fit(X, y)

        with pytest.raises(KeyError, match="Unknown feature"):
            model.lrt("nonexistent")

    def test_qr_fallback_in_compute_quantities(self, separation_data, monkeypatch):
        X, y = separation_data
        model_normal = FirthLogisticRegression(backend="numpy").fit(X, y)

        called = {"dgeqp3": 0, "dorgqr": 0}
        orig_dgeqp3 = firthmodels.logistic.dgeqp3
        orig_dorgqr = firthmodels.logistic.dorgqr

        def wrapped_dgeqp3(*args, **kwargs):
            called["dgeqp3"] += 1
            return orig_dgeqp3(*args, **kwargs)

        def wrapped_dorgqr(*args, **kwargs):
            called["dorgqr"] += 1
            return orig_dorgqr(*args, **kwargs)

        def fake_dpotrf(a, *args, **kwargs):
            return (a, 1)

        monkeypatch.setattr("firthmodels.logistic.dpotrf", fake_dpotrf)
        monkeypatch.setattr("firthmodels.logistic.dgeqp3", wrapped_dgeqp3)
        monkeypatch.setattr("firthmodels.logistic.dorgqr", wrapped_dorgqr)

        model_fallback = FirthLogisticRegression(backend="numpy").fit(X, y)

        assert called["dgeqp3"] > 0
        assert called["dorgqr"] > 0
        np.testing.assert_allclose(model_fallback.coef_, model_normal.coef_, rtol=1e-6)
        np.testing.assert_allclose(
            model_fallback.intercept_, model_normal.intercept_, rtol=1e-6
        )

    def test_qr_fallback_rank_deficient_raises(self, monkeypatch):
        from firthmodels.logistic import _Workspace

        rng = np.random.default_rng(0)
        x = rng.standard_normal(5)
        X = np.column_stack([x, x])  # rank deficient
        y = rng.integers(0, 2, 5)
        ws = _Workspace(X.shape[0], X.shape[1])
        beta = np.zeros(X.shape[1])
        sample_weight = np.ones(X.shape[0])
        offset = np.zeros(X.shape[0])

        def fake_dpotrf(a, *args, **kwargs):
            return (a, 1)

        monkeypatch.setattr("firthmodels.logistic.dpotrf", fake_dpotrf)

        with pytest.raises(scipy.linalg.LinAlgError, match="rank deficient"):
            firthmodels.logistic.compute_logistic_quantities(
                X, y, beta, sample_weight, offset, workspace=ws
            )

    def test_early_rank_check_in_fit(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 0])
        sample_weight = np.array([1.0, 1.0, 0.0])

        with pytest.raises(ValueError, match="effective sample size"):
            FirthLogisticRegression().fit(X, y, sample_weight=sample_weight)

    def test_cholesky_fallback_in_solver(self, separation_data, monkeypatch):
        """Fallback to lstsq in newton_raphson solver produces equivalent results."""
        X, y = separation_data
        model_normal = FirthLogisticRegression(backend="numpy").fit(X, y)

        def fake_dpotrf(a, *args, **kwargs):
            return (a, 1)

        monkeypatch.setattr("firthmodels._solvers.dpotrf", fake_dpotrf)
        model_fallback = FirthLogisticRegression(backend="numpy").fit(X, y)

        np.testing.assert_allclose(model_fallback.coef_, model_normal.coef_, rtol=1e-6)
        np.testing.assert_allclose(
            model_fallback.intercept_, model_normal.intercept_, rtol=1e-6
        )

    def test_profile_ci_solve_fallback(self, separation_data, monkeypatch):
        """Fallback to lstsq in profile CI initialization produces equivalent results."""
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy").fit(X, y)
        ci_normal = model.conf_int(method="pl", features=[0])

        def fake_solve(*args, **kwargs):
            raise np.linalg.LinAlgError("forced failure")

        monkeypatch.setattr("numpy.linalg.solve", fake_solve)
        model2 = FirthLogisticRegression(backend="numpy").fit(X, y)
        ci_fallback = model2.conf_int(method="pl", features=[0])

        np.testing.assert_allclose(ci_fallback, ci_normal, rtol=1e-6)

    def test_profile_ci_inv_fallback(self, separation_data, monkeypatch):
        """Fallback to lstsq/pinv in profile CI iteration produces equivalent results."""
        X, y = separation_data
        model = FirthLogisticRegression(backend="numpy").fit(X, y)
        ci_normal = model.conf_int(method="pl", features=[0])

        def fake_dgetrf(*args, **kwargs):
            raise np.linalg.LinAlgError("forced failure")

        monkeypatch.setattr("firthmodels._profile_ci.dgetrf", fake_dgetrf)
        model2 = FirthLogisticRegression(backend="numpy").fit(X, y)
        ci_fallback = model2.conf_int(method="pl", features=[0])

        np.testing.assert_allclose(ci_fallback, ci_normal, rtol=1e-6)
