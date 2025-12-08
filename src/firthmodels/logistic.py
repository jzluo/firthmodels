import numpy as np
import scipy
import warnings

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._tags import Tags, ClassifierTags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing import Literal, Self, Sequence, cast

from firthmodels._solvers import newton_raphson


class FirthLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Logistic regression with Firth's bias reduction method.

    This estimator fits a logistic regression model with Firth's bias-reduction
    penalty, which helps to mitigate small-sample bias and the problems caused
    by (quasi-)complete separation. In such cases, standard maximum-likelihood
    logistic regression can produce infinite (or extremely large) coefficient
    estimates, whereas Firth logistic regression yields finite, well-behaved
    estimates.

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
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `fit`. Defined only when X has feature names that are all strings.

    References
    ----------
    Firth D (1993). Bias reduction of maximum likelihood estimates.
    Biometrika 80, 27–38.

    Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
    regression. Statistics in Medicine 21: 2409-2419.

    Mbatchou J et al. (2021). Computationally efficient whole-genome regression for
    quantitative and binary traits. Nature Genetics 53, 1097–1103.

    Examples
    --------
    >>> import numpy as np
    >>> from firthmodels import FirthLogisticRegression
    >>> # x=1 perfectly predicts y=1 (separated data)
    >>> X = np.array([[0], [0], [0], [1], [1], [1]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> model = FirthLogisticRegression().fit(X, y)
    >>> model.coef_
    array([3.89181893])
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

        self.bse_ = bse
        self.pvalues_ = pvalues

        # need these for LRT
        self._fit_data = (X, y, sample_weight, offset)  # X includes intercept column

        self.lrt_pvalues_ = np.full(len(result.beta), np.nan)
        self.lrt_bse_ = np.full(len(result.beta), np.nan)
        self._profile_ci_cache: dict[float, NDArray[np.float64]] = {}
        return self

    def conf_int(
        self,
        alpha: float = 0.05,
        method: Literal["wald", "pl"] = "wald",
        features: int | str | Sequence[int | str] | None = None,
        max_iter: int = 25,
        tol: float = 1e-4,
    ) -> NDArray[np.float64]:
        """
        Compute confidence intervals for the coefficients. If `method='pl'`, profile
        likelihood confidence intervals are computed using the Venzon-Moolgavkar method.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (default 0.05 for 95% CI)
        method : {'wald', 'pl'}, default='wald'
            Method to compute confidence intervals.
            - 'wald': Wald confidence intervals (fast)
            - 'pl': Profile likelihood confidence intervals (more accurate, slower)
        features: int, str, sequence of int, sequence of str, or None, default=None
            Features to compute CIs for (only used for `method='pl'`).
            If None, compute for all features.
            - int: single feature by index
            - str: single feature by name (requires `feature_names_in`)
            - Sequence[int]: multiple features by index
            - Sequence[str]: multiple features by name
            - None: all features (including intercept if `fit_intercept=True`)
        max_iter : int, default=25
            Maximum number of iterations per bound (only used for `method='pl'`)
        tol : float, default=1e-4
            Convergence tolerance (only used for `method='pl'`)

        Returns
        -------
        ndarray, shape(n_features, 2)
            Column 0: lower bounds, Column 1: upper bounds
            Includes intercept as last row if `fit_intercept=True`.

        Notes
        -----
        Profile-likelihood CIs are superior to Wald CIs when the likelihood is
        asymmetric, which can occur with small samples or separated data.
        For `method='pl'`, results are cached. Subsequent calls with the same alpha and
        method return cached values without recomputation.

        References
        ----------
        Venzon, D. J., & Moolgavkar, S. H. (1988). A Method for Computing
        Profile-Likelihood-Based Confidence Intervals. Applied Statistics.
        """
        check_is_fitted(self)
        n_params = len(self.bse_)

        if method == "wald":
            z = scipy.stats.norm.ppf(1 - alpha / 2)
            if self.fit_intercept:
                beta = np.concatenate([self.coef_, [self.intercept_]])
            else:
                beta = self.coef_
            lower = beta - z * self.bse_
            upper = beta + z * self.bse_
            return np.column_stack([lower, upper])

        elif method == "pl":
            # get or create cache for this alpha
            if alpha not in self._profile_ci_cache:
                self._profile_ci_cache[alpha] = np.full((n_params, 2), np.nan)
            ci = self._profile_ci_cache[alpha]

            # normalize features to indices
            n_coef = len(self.coef_)
            if features is None:
                indices = list(range(n_params))
            else:
                features_seq = (
                    [features]
                    if isinstance(features, (int, np.integer, str))
                    else features
                )
                indices = []
                for feat in features_seq:
                    if isinstance(feat, str):
                        if feat == "intercept":
                            if not self.fit_intercept:
                                raise ValueError(
                                    "Cannot compute CI for intercept when fit_intercept=False"
                                )
                            indices.append(n_coef)
                        else:
                            indices.append(self._feature_name_to_index(feat))
                    elif isinstance(feat, (int, np.integer)):
                        indices.append(int(feat))
                    else:
                        raise TypeError(
                            f"Elements of `features` must be int or str, got {type(feat)}"
                        )

            # compute profile CIs for features not already cached
            chi2_crit = scipy.stats.chi2.ppf(1 - alpha, 1)
            l_star = self.loglik_ - chi2_crit / 2

            for idx in indices:
                if np.isnan(ci[idx, 0]):  # not yet computed
                    for bound_idx, which in enumerate([-1, 1]):  # lower, upper
                        bound, converged, n_iter = self._compute_profile_ci_bound(
                            idx=idx,
                            l_star=l_star,
                            which=which,
                            max_iter=max_iter,
                            tol=tol,
                            chi2_crit=chi2_crit,
                        )

                        if converged:
                            ci[idx, bound_idx] = bound
                        else:
                            warnings.warn(
                                f"Profile-likelihood CI did not converge for parameter {idx} "
                                f"({'lower' if which == -1 else 'upper'} bound) after {n_iter} iterations.",
                                ConvergenceWarning,
                                stacklevel=2,
                            )
            return ci

        else:
            raise ValueError(f"method must be 'wald' or 'pl', got '{method}'")

    def lrt(
        self,
        features: int | str | Sequence[int | str] | None = None,
    ) -> Self:
        """
        Compute penalized likelihood ratio test p-values.
        Standard errors are also back-corrected using the effect size estimate and the
        LRT p-value, as in regenie. Useful for meta-analysis where studies are weighted
        by 1/SE².

        Parameters
        ----------
        features : int, str, sequence of int, sequence of str, or None, default=None
            Features to test. If None, test all features.
            - int: single feature by index
            - str: single feature by name (requires `feature_names_in`)
            - Sequence[int]: multiple features by index
            - Sequence[str]: multiple features by name
            - None: all features (including intercept if `fit_intercept=True`)

        Returns
        -------
        self : FirthLogisticRegression

        Examples
        --------
        >>> model.fit(X, y).lrt()  # compute LR for all features
        >>> model.lrt_pvalues_
        array([0.00020841, 0.00931731, 0.02363857, 0.0055888 ])
        >>> model.lrt_bse_
        array([0.98628022, 0.25997282, 0.38149783, 0.12218733])
        >>> model.fit(X, y).lrt(0)
        >>> model.lrt_pvalues_
        array([0.00020841,        nan,        nan,        nan])
        >>> model.fit(X, y).lrt(['snp', 'age'])  # by name (requires DataFrame input)
        >>> model.lrt_pvalues_
        array([0.00020841,        nan, 0.02363857,        nan])
        """
        check_is_fitted(self)

        # normalize input to a list of indices
        n_coef = len(self.coef_)
        if features is None:
            indices = list(range(n_coef))
            if self.fit_intercept:
                indices.append(n_coef)  # intercept index
        else:
            features_seq = (
                [features] if isinstance(features, (int, np.integer, str)) else features
            )
            indices = []
            for feat in features_seq:
                if isinstance(feat, str):
                    if feat == "intercept":
                        indices.append(n_coef)
                    else:
                        indices.append(self._feature_name_to_index(feat))
                elif isinstance(feat, (int, np.integer)):
                    indices.append(feat)
                else:
                    raise TypeError(
                        f"Elements of `features` must be int or str, but got {type(feat)}"
                    )

        if n_coef in indices and not self.fit_intercept:
            raise ValueError("Cannot test intercept when fit_intercept=False")

        # compute LRT
        for idx in indices:
            if np.isnan(self.lrt_pvalues_[idx]):
                self._compute_single_lrt(idx)
        return self

    def _feature_name_to_index(self, name: str) -> int:
        """Map feature name to its index"""
        if not hasattr(self, "feature_names_in_"):
            raise ValueError(
                "No feature names available. Pass a DataFrame to fit(), "
                "or use integer indices."
            )
        try:
            return list(self.feature_names_in_).index(name)
        except ValueError:
            raise KeyError(f"Unknown feature: '{name}'") from None

    def _compute_single_lrt(self, idx: int) -> None:
        """
        Fit constrained model with `beta[idx]=0` and compute LRT p-value and
        back-corrected standard error.

        Parameters
        ----------
        idx : int
            Index of the coefficient to test. Use len(coef_) for the intercept.
        """
        X, y, sample_weight, offset = self._fit_data

        # fit all indices except for the feature being tested
        free_indices = [i for i in range(X.shape[1]) if i != idx]

        # Wrapper so the solver only optimizes the k-1 "free" parameters.
        # Reconstructs full k-vector, computes quantities with full Fisher,
        # then slices score and Fisher to k-1 dimensions before returning to the solver.
        def constrained_quantities(beta_free):
            beta_full = np.insert(beta_free, idx, 0.0)
            q = compute_logistic_quantities(X, y, beta_full, sample_weight, offset)
            return LogisticQuantities(
                loglik=q.loglik,
                modified_score=q.modified_score[free_indices],
                fisher_info=q.fisher_info[np.ix_(free_indices, free_indices)],
            )

        result = newton_raphson(
            compute_quantities=constrained_quantities,
            n_features=X.shape[1] - 1,
            max_iter=self.max_iter,
            max_step=self.max_step,
            max_halfstep=self.max_halfstep,
            tol=self.tol,
        )

        chi_sq = max(0.0, 2.0 * (self.loglik_ - result.loglik))
        pval = scipy.stats.chi2.sf(chi_sq, df=1)

        # back-corrected SE: |β|/√χ² (ensures (β/SE)² = χ²)
        beta_val = self.intercept_ if idx == len(self.coef_) else self.coef_[idx]
        bse = np.abs(beta_val) / np.sqrt(chi_sq) if chi_sq > 0 else np.inf

        self.lrt_pvalues_[idx] = pval
        self.lrt_bse_[idx] = bse

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
