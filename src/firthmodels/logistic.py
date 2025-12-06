import numpy as np
import scipy

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._tags import Tags, ClassifierTags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing import Literal, Self, cast

from firthmodels._solvers import newton_raphson


class FirthLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Parameters
    ----------
    solver : {'newton-raphson'}, default='newton-raphson'
        Optimization algorithm. Only 'newton-raphson' is currently supported.
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
    bse_: ndarray of shape (n_features,)
        Wald standard errors for the coefficient estimates.
    intercept_bse_ : float
        Wald standard error for the intercept.
    pvalues_ : ndarray of shape (n_features,)
        Wald p-values for the coefficients.
    intercept_pvalue_ : float
        Wald p-value for the intercept.
    """

    def __init__(
        self,
        solver: Literal["newton-raphson"] = "newton-raphson",
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

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> Self:
        """
        Fit the Firth logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to linear predictor.

        Returns
        -------
        self : FirthLogisticRegression
            Fitted estimator.
        """
        # === Validate and prep inputs ===
        X, y = self._validate_input(X, y)
        sample_weight = (
            np.ones(X.shape[0], dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        offset = (
            np.zeros(X.shape[0], dtype=np.float64)
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        )

        if self.fit_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])

        n_features = X.shape[1]

        # === run solver ===
        def compute_quantities(beta):
            return compute_logistic_quantities(
                X=X,
                y=y,
                beta=beta,
                sample_weight=sample_weight,
                offset=offset,
            )

        result = newton_raphson(
            compute_quantities=compute_quantities,
            n_features=n_features,
            max_iter=self.max_iter,
            max_step=self.max_step,
            max_halfstep=self.max_halfstep,
            tol=self.tol,
        )

        # === Extract coefficients ===
        if self.fit_intercept:
            self.coef_ = result.beta[:-1]
            self.intercept_ = result.beta[-1]
        else:
            self.coef_ = result.beta
            self.intercept_ = 0.0

        self.loglik_ = result.loglik
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged

        # === Wald ===
        try:
            cov = np.linalg.inv(result.fisher_info)
            bse = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            bse = np.full_like(result.beta, np.nan)

        z = result.beta / bse
        pvalues = 2 * scipy.stats.norm.sf(np.abs(z))

        if self.fit_intercept:
            self.bse_, self.intercept_bse_ = bse[:-1], bse[-1]
            self.pvalues_, self.intercept_pvalue_ = pvalues[:-1], pvalues[-1]
        else:
            self.bse_ = bse
            self.pvalues_ = pvalues

        return self

    def conf_int(self, alpha: float = 0.05) -> NDArray[np.float64]:
        """
        Wald confidence intervals for the coefficients.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (default 0.05 for 95% CI)

        Returns
        -------
        ndarray, shape(n_features, 2)
            Column 0: lower bounds, Column 1: upper bounds
        """
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        lower = self.coef_ - z * self.bse_
        upper = self.coef_ + z * self.bse_
        return np.column_stack([lower, upper])

    def decision_function(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return linear predictor."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)  # for mypy
        return X @ self.coef_ + self.intercept_

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return class probabilities."""
        scores = self.decision_function(X)
        p1 = expit(scores)
        return np.column_stack([1 - p1, p1])

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray[np.int_]:
        """Return predicted class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_log_proba(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return log class probabilities"""
        return np.log(self.predict_proba(X))

    def _validate_input(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate parameters and inputs, encode y to 0/1"""
        if self.solver != "newton-raphson":
            raise ValueError(
                f"solver='{self.solver}' is not supported. "
                "Only 'newton-raphson' is currently implemented."
            )
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.max_halfstep < 0:
            raise ValueError(
                f"max_halfstep must be non-negative, got {self.max_halfstep}"
            )
        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}")
        X, y = validate_data(
            self, X, y, dtype=np.float64, y_numeric=False, ensure_min_samples=2
        )

        y_type = type_of_target(y)
        if y_type == "continuous":
            raise ValueError(
                "Unknown label type: continuous. Only binary classification is supported."
            )

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Got {len(self.classes_)} classes. Only binary classification is supported."
            )

        # encode y to 0/1
        y = (y == self.classes_[1]).astype(np.float64)

        X = cast(NDArray[np.float64], X)  # for mypy
        y = cast(NDArray[np.float64], y)
        return X, y


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
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
) -> LogisticQuantities:
    """Compute all quantities needed for one Newton-Raphson iteration."""
    eta = X @ beta + offset
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

    # augmented fisher information
    w_aug = (sample_weight + h) * p * (1 - p)
    sqrt_w_aug = np.sqrt(w_aug)
    XtW_aug = X.T * sqrt_w_aug
    fisher_info_aug = XtW_aug @ XtW_aug.T

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
        fisher_info=fisher_info_aug,
    )
