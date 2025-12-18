from typing import Literal

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from firthmodels import FirthLogisticRegression


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

        return FirthLogitResults(self, estimator)


class FirthLogitResults:
    def __init__(self, model: FirthLogit, estimator: FirthLogisticRegression):
        self.model = model
        self.estimator = estimator

    @property
    def params(self) -> NDArray[np.float64]:
        return self.estimator.coef_

    @property
    def bse(self) -> NDArray[np.float64]:
        return self.estimator.bse_

    @property
    def tvalues(self) -> NDArray[np.float64]:
        return self.estimator.coef_ / self.estimator.bse_

    @property
    def pvalues(self) -> NDArray[np.float64]:
        tvals = self.tvalues
        pvals = 2 * scipy.stats.norm.sf(np.abs(tvals))
        return pvals

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
