import numpy as np

from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Literal, Self


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    solver : {'newton-raphson', 'irls', 'lbfgs'}, default='newton-raphson'
    max_iter : int, default=25
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
        solver: Literal['newton-raphson', 'irls', 'lbfgs'] = 'newton-raphson',
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
