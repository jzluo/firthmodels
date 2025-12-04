import numpy as np

from firthmodels import FirthLogisticRegression


class TestFirthLogisticRegression:
    """Tests for FirthLogisticRegression."""

    def test_matches_logistf_with_separation(self, separation_data):
        """Coefficients match logistf on quasi-separated data."""
        X, y = separation_data
        model = FirthLogisticRegression()
        model.fit(X, y)

        expected_intercept = -0.4434563
        expected_coef = np.array([3.6577140, 0.6759782, -0.8633120, 0.3385789])

        np.testing.assert_allclose(model.intercept_, expected_intercept, rtol=1e-3)
        np.testing.assert_allclose(model.coef_, expected_coef, rtol=1e-3)
        assert model.converged_

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
