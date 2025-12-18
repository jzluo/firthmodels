from typing import Literal

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from firthmodels import FirthLogisticRegression
from firthmodels.logistic import compute_logistic_quantities


class FirthLogit:
    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        **kwargs,
    ):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self.offset = np.asarray(offset) if offset is not None else None

        missing = kwargs.pop("missing", "none")

        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        if missing == "drop":
            raise NotImplementedError("missing='drop' is not supported")
        elif missing == "raise":
            if np.isnan(self.endog).any() or np.isnan(self.exog).any():
                raise ValueError("Input contains NaN values")

        if hasattr(exog, "columns"):
            self.exog_names = list(exog.columns)
        else:
            self.exog_names = [f"x{i + 1}" for i in range(self.exog.shape[1])]

    def fit(
        self,
        start_params: ArrayLike | None = None,
        method: Literal["newton"] = "newton",
        maxiter: int = 25,
        pl: bool = True,
        **kwargs,  # gtol, xtol
    ):
        if start_params is not None:
            raise NotImplementedError("start_params is not currently supported.")
        if method != "newton":
            raise ValueError("Only 'newton' method is currently supported.")

        gtol = kwargs.pop("gtol", 1e-4)
        xtol = kwargs.pop("xtol", 1e-4)
        if kwargs:
            raise TypeError(
                f"fit() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        estimator = FirthLogisticRegression(
            fit_intercept=False,
            max_iter=maxiter,
            gtol=gtol,
            xtol=xtol,
        )
        estimator.fit(self.exog, self.endog, offset=self.offset)
        if pl:
            estimator.lrt()
        return FirthLogitResults(self, estimator, pl=pl)


class FirthLogitResults:
    def __init__(
        self, model: FirthLogit, estimator: FirthLogisticRegression, pl: bool = True
    ):
        self.model = model
        self.estimator = estimator
        self._pl = pl

    @property
    def params(self) -> NDArray[np.float64]:
        return self.estimator.coef_

    @property
    def bse(self) -> NDArray[np.float64]:
        return self.estimator.lrt_bse_ if self._pl else self.estimator.bse_

    @property
    def tvalues(self) -> NDArray[np.float64]:
        return self.params / self.bse

    @property
    def pvalues(self) -> NDArray[np.float64]:
        return self.estimator.lrt_pvalues_ if self._pl else self.estimator.pvalues_

    @property
    def llf(self) -> float:
        return self.estimator.loglik_

    @property
    def converged(self) -> bool:
        return self.estimator.converged_

    @property
    def nobs(self) -> int:
        return self.model.exog.shape[0]

    @property
    def df_model(self) -> int:
        return len(self.params) - 1

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.params)

    @property
    def mle_retvals(self) -> dict:
        return {"converged": self.converged, "iterations": self.estimator.n_iter_}

    def predict(
        self,
        exog: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        **kwargs,
    ) -> NDArray[np.float64]:
        if exog is None:
            exog = self.model.exog
            offset = self.model.offset

        if offset is None:
            offset = np.zeros(exog.shape[0])

        linear_pred = exog @ self.params + offset

        if kwargs.get("linear", False):
            return linear_pred

        return expit(linear_pred)

    def conf_int(
        self,
        alpha=0.05,
        method: Literal["wald", "pl"] | None = None,
        **kwargs,  # maxiter, tol
    ) -> NDArray[np.float64]:
        if method is None:
            method = "pl" if self._pl else "wald"

        max_iter = kwargs.pop("maxiter", 25)
        tol = kwargs.pop("tol", 1e-4)
        if kwargs:
            raise TypeError(
                f"conf_int() got unexpected keyword arguments: {list(kwargs.keys())}"
            )
        return self.estimator.conf_int(
            alpha=alpha, method=method, max_iter=max_iter, tol=tol
        )

    def cov_params(self) -> NDArray[np.float64]:
        X, y, sample_weight, offset = self.estimator._fit_data
        q = compute_logistic_quantities(X, y, self.params, sample_weight, offset)
        return np.linalg.inv(q.fisher_info)
