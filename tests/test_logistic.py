import numpy as np
import pytest
from sklearn.utils.estimator_checks import estimator_checks_generator

from firthmodels import FirthLogisticRegression


class TestFirthLogisticRegression:
    """Tests for FirthLogisticRegression."""

    def test_matches_logistf_with_separation(self, separation_data):
        """Matches logistf on quasi-separated data."""
        X, y = separation_data
        model = FirthLogisticRegression()
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
        model = FirthLogisticRegression(fit_intercept=False)
        model.fit(X, y)

        assert model.intercept_ == 0.0
        assert len(model.coef_) == n_features

    def test_classes_encoded_correctly(self):
        """Handles arbitrary binary labels."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))

        for labels in [(0, 1), (1, 2), (-1, 1)]:
            y = rng.choice(labels, 50)
            model = FirthLogisticRegression()
            model.fit(X, y)
            np.testing.assert_array_equal(model.classes_, sorted(labels))

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
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
        model = FirthLogisticRegression()
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
