import numpy as np
import pytest

from firthmodels import detect_separation


class TestDetectSeparation:
    def test_complete_separation_single_predictor(self):
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y)

        assert result.separation is True
        np.testing.assert_array_equal(
            result.directions, [1, -1]
        )  # X1=+Inf, intercept=-Inf
        np.testing.assert_array_equal(result.is_finite, [False, False])

        # summary() uses x0, x1, ... when feature names not available
        summary = result.summary()
        assert "x0" in summary
        assert "x1" in summary

    def test_overlap_no_separation(self):
        X = np.array([[1], [1], [2], [2], [3], [3]], dtype=float)
        y = np.array([0, 1, 0, 1, 0, 1])

        result = detect_separation(X, y)

        assert result.separation is False
        np.testing.assert_array_equal(result.directions, [0, 0])
        np.testing.assert_array_equal(result.is_finite, [True, True])

        summary = result.summary()
        assert "Separation: False" in summary
        assert "finite" in summary

    def test_multi_predictor_first_separates(self):
        X1 = np.array([0, 0, 0, 1, 1, 1], dtype=float)  # separator
        X2 = np.array([0.5, -0.2, 0.8, -0.5, 0.3, 0.1])
        X = np.column_stack([X1, X2])
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y)

        assert result.separation is True
        # X1=+Inf, X2=-Inf, intercept=-Inf
        np.testing.assert_array_equal(result.directions, [1, -1, -1])

    def test_no_intercept_complete_separation(self):
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y, fit_intercept=False)

        assert result.separation is False
        np.testing.assert_array_equal(result.directions, [0])

    def test_negative_separation_direction(self):
        X = np.array([[4], [5], [6], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y)

        assert result.separation is True
        # X=-Inf, intercept=+Inf
        np.testing.assert_array_equal(result.directions, [-1, 1])

    def test_non_01_labels(self):
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
        y = np.array([-1, -1, -1, 1, 1, 1])

        result = detect_separation(X, y)
        assert result.separation is True


class TestDetectSeparationDataframe:
    def test_summary_with_names(self):
        """summary() shows feature names when available."""
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame({"treatment": [0, 0, 0, 1, 1, 1], "age": [1, 2, 3, 1, 2, 3]})
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y)
        summary = result.summary()

        assert "Separation: True" in summary
        assert "treatment" in summary
        assert "+Inf" in summary
        assert "intercept" in summary
        assert result.feature_names == ("treatment", "age", "intercept")

    def test_dataframe_feature_names_no_intercept(self):
        """DataFrame feature names without intercept."""
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame({"x1": [0, 0, 0, 1, 1, 1], "x2": [1, 2, 3, 4, 5, 6]})
        y = np.array([0, 0, 0, 1, 1, 1])

        result = detect_separation(X, y, fit_intercept=False)

        assert result.feature_names == ("x1", "x2")
