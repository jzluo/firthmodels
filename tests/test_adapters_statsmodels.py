import numpy as np
import pytest

from firthmodels.adapters.statsmodels import FirthLogit, FirthLogitResults


@pytest.fixture
def toy_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y


class TestFirthLogit:
    def test_stores_endog_exog(self, toy_data):
        X, y = toy_data
        model = FirthLogit(y, X)
        assert isinstance(model.endog, np.ndarray)
        assert isinstance(model.exog, np.ndarray)
        np.testing.assert_array_equal(model.endog, y)
        np.testing.assert_array_equal(model.exog, X)

    def test_stores_offset(self, toy_data):
        X, y = toy_data
        offset = np.array([0.1, 0.2, 0.3, 0.4])
        model = FirthLogit(y, X, offset=offset)
        np.testing.assert_array_equal(model.offset, offset)

    def test_exog_names_from_array(self, toy_data):
        X, y = toy_data
        model = FirthLogit(y, X)
        expected_names = [f"x{i + 1}" for i in range(X.shape[1])]
        assert model.exog_names == expected_names

    def test_unknown_kwargs_raise_typeerror(self, toy_data):
        X, y = toy_data
        with pytest.raises(TypeError, match="myeyesaresodry"):
            FirthLogit(y, X, myeyesaresodry=123)

    def test_exog_names_from_dataframe(self):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        model = FirthLogit(data["A"], data[["B", "C"]])
        assert model.exog_names == ["B", "C"]

    def test_missing_raise_with_nan(self, toy_data):
        X, _ = toy_data
        y = np.array([0, 1, np.nan, 1])
        with pytest.raises(ValueError, match="NaN"):
            model = FirthLogit(y, X, missing="raise")

    def test_missing_drop_not_implemented(self, toy_data):
        X, y = toy_data
        with pytest.raises(NotImplementedError):
            model = FirthLogit(y, X, missing="drop")

    def test_fit_returns_results(self, toy_data):
        X, y = toy_data
        results = FirthLogit(y, X).fit()
        assert isinstance(results, FirthLogitResults)


class TestFirthLogitResults:
    @pytest.fixture
    def fitted_results(self, toy_data):
        X, y = toy_data
        return FirthLogit(y, X).fit()

    def test_predict_default_uses_training_data(self, fitted_results):
        pred = fitted_results.predict()
        assert pred.shape == (4,)
        assert np.all((pred >= 0) & (pred <= 1))  # probabilities

    def test_predict_new_data(self, fitted_results):
        X_new = np.array([[1, 3], [9, 4]])
        pred = fitted_results.predict(X_new)
        assert pred.shape == (2,)
        assert np.all((pred >= 0) & (pred <= 1))

    def test_conf_int_shape(self, fitted_results):
        ci = fitted_results.conf_int()
        assert ci.shape == (2, 2)  # 2 params, lower upper
