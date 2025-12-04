import numpy as np
import scipy

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Literal, Self


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    solver : {'newton-raphson', 'irls', 'lbfgs'}, default='newton-raphson'
    max_iter : int, default=25
        Maximum number of iterations
    max_step : float, default=5.0
        Maximum step size per coefficient (Newton-Raphson only)
    max_halfstep : int, default=25
        Maximum number of step-halvings per iteration (Newton-Raphson only)
    tol : float, default=1e-4
        Tolerance for stopping criteria
    fit_intercept : bool, default=True
        Whether to fit intercept

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        A list of the class labels.
    coef_ : ndarray of shape (n_features,)
        The coefficients of the features.
    intercept_ : float
        Fitted intercept. Set to 0.0 if `fit_intercept=False`.
    loglik_ : float
        Fitted penalized log-likelihood.
    n_iter_ : int
        Number of iterations the solver ran.
    converged_ : bool
        Whether the solver converged within `max_iter`.
    """

    def __init__(
        self,
        solver: Literal["newton-raphson", "irls", "lbfgs"] = "newton-raphson",
        max_iter: int = 25,
        max_step: float = 5.0,
        max_halfstep: int = 25,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        self.solver = solver
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray[np.int_]:
        pass

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        pass

    def predict_log_proba(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        pass

    def decision_function(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        pass


@dataclass
class LogisticQuantities:
    """Quantities needed for one Newton-Raphson iteration"""

    loglik: float
    modified_score: NDArray[
        np.float64
    ]  # (n_features,) U* = X'[weights*(y - p) + h*(0.5 - p)]
    fisher_info: NDArray[np.float64]  # (n_features, n_features) X'WX


def compute_logistic_quantities(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    beta: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None = None,
) -> LogisticQuantities:
    """Compute all quantities needed for one Newton-Raphson iteration."""
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=np.float64)

    eta = X @ beta
    p = expit(eta)

    # W = diag(weights * p * (1-p))
    w = sample_weight * p * (1 - p)

    # Fisher information: X'WX
    sqrt_w = np.sqrt(w)
    XtW = X.T * sqrt_w  # (k, n) broadcast so we don't materialize (n, n) diag matrix
    fisher_info = XtW @ XtW.T

    # hat diag
    try:
        cho = scipy.linalg.cho_factor(fisher_info)
        solved = scipy.linalg.cho_solve(cho, XtW)  # (k, n)
        L = cho[0]
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except scipy.linalg.LinAlgError:
        solved, *_ = np.linalg.lstsq(fisher_info, XtW, rcond=None)
        # fallback to slogdet
        sign, logdet = np.linalg.slogdet(fisher_info)
        if sign <= 0:
            # use -inf so loglik approaches -inf
            logdet = -np.inf
    h = np.sum(solved * XtW, axis=0)

    # L*(β) = Σ weight_i * [y_i*log(p_i) + (1-y_i)*log(1-p_i)] + 0.5*log|I(β)|
    loglik = (
        np.sum(sample_weight * (y * np.log(p) + (1 - y) * np.log(1 - p))) + 0.5 * logdet
    )

    # modified score U* = X'[weights*(y-p) + h*(0.5-p)]
    residual = sample_weight * (y - p) + h * (0.5 - p)
    modified_score = X.T @ residual

    return LogisticQuantities(
        loglik=loglik,
        modified_score=modified_score,
        fisher_info=fisher_info,
    )
