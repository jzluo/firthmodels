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
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)
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
    bse_ : ndarray of shape (n_params,)
        Wald standard errors. Includes intercept as last element if
        `fit_intercept=True`, where n_params = n_features + 1.
    pvalues_ : ndarray of shape (n_params,)
        Wald p-values. Includes intercept as last element if `fit_intercept=True`.
    lrt_pvalues_ : ndarray of shape (n_params,)
        Likelihood ratio test p-values. Computed by `lrt()`. Values are
        NaN until computed. Includes intercept as last element if `fit_intercept=True`.
    lrt_bse_ : ndarray of shape (n_params,)
        Back-corrected standard errors from LRT. Computed by `lrt()`.
        Values are NaN until computed. Includes intercept as last element if
        `fit_intercept=True`.
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `fit`. Defined only when X has feature names
        that are all strings.

    References
    ----------
    Firth D (1993). Bias reduction of maximum likelihood estimates.
    Biometrika 80, 27-38.

    Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
    regression. Statistics in Medicine 21: 2409-2419.

    Mbatchou J et al. (2021). Computationally efficient whole-genome regression for
    quantitative and binary traits. Nature Genetics 53, 1097-1103.

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
        sample_weight = cast(
            NDArray[np.float64],
            _check_sample_weight(
                sample_weight, X, dtype=np.float64, ensure_non_negative=True
            ),
        )
        if offset is None:
            offset = np.zeros(X.shape[0], dtype=np.float64)
        else:
            offset = cast(
                NDArray[np.float64],
                check_array(
                    offset, ensure_2d=False, dtype=np.float64, input_name="offset"
                ),
            )
            if offset.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Length of offset ({offset.shape[0]}) does not match "
                    f"number of samples ({X.shape[0]})"
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

        # _profile_ci_cache and _profile_ci_computed are keyed by (alpha, tol, max_iter)
        self._profile_ci_cache: dict[tuple[float, float, int], NDArray[np.float64]] = {}
        # tracks completed bound computations; False means never tried or interrupted
        self._profile_ci_computed: dict[
            tuple[float, float, int], NDArray[np.bool_]
        ] = {}
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
        For `method='pl'`, results are cached. Subsequent calls with the same `alpha`,
        `tol`, and `max_iter` return cached values without recomputation.

        References
        ----------
        Venzon, D.J. and Moolgavkar, S.H. (1988). "A Method for Computing
        Profile-Likelihood-Based Confidence Intervals." Applied Statistics,
        37(1), 87-94
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
            # get or create cache for this (alpha, tol, max_iter) combination
            cache_key = (alpha, tol, max_iter)
            if cache_key not in self._profile_ci_cache:
                self._profile_ci_cache[cache_key] = np.full((n_params, 2), np.nan)
                self._profile_ci_computed[cache_key] = np.zeros(
                    (n_params, 2), dtype=bool
                )
            ci = self._profile_ci_cache[cache_key]
            computed = self._profile_ci_computed[cache_key]

            indices = self._resolve_feature_indices(features)

            # compute profile CIs for bounds not already attempted
            chi2_crit = scipy.stats.chi2.ppf(1 - alpha, 1)
            l_star = self.loglik_ - chi2_crit / 2

            for idx in indices:
                for bound_idx, which in enumerate([-1, 1]):  # lower, upper
                    if not computed[idx, bound_idx]:
                        which = cast(Literal[-1, 1], which)  # mypy -_-
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
                        # mark AFTER completion (interrupt safety for jupyter people)
                        computed[idx, bound_idx] = True
            return ci

        else:
            raise ValueError(f"method must be 'wald' or 'pl', got '{method}'")

    def _compute_profile_ci_bound(
        self,
        idx: int,
        l_star: float,
        which: Literal[-1, 1],
        max_iter: int,
        tol: float,
        chi2_crit: float,
    ) -> tuple[float, bool, int]:
        """
        Compute one profile CI bound using Venzon-Moolgavkar (1988) algorithm.

        Solves F(theta) = [l(theta) - l*, dl/dw]' = 0 where beta = theta[idx]
        is the parameter of interest and w (omega) are nuisance parameters.

        References
        ----------
        Venzon, D.J. and Moolgavkar, S.H. (1988). "A Method for Computing
        Profile-Likelihood-Based Confidence Intervals." Applied Statistics,
        37(1), 87-94.
        """
        X, y, sample_weight, offset = self._fit_data
        k = X.shape[1]

        # Initialize at MLE
        if self.fit_intercept:
            theta = np.concatenate([self.coef_, [self.intercept_]])
        else:
            theta = self.coef_.copy()

        # Appendix step 1: compute and store D0 = d2l/dtheta2 at MLE
        q = compute_logistic_quantities(X, y, theta, sample_weight, offset)
        # negative Fisher info == Hessian of log-likelihood
        D0 = -q.fisher_info

        # beta = parameter of interest, omega = nuisance parameters
        other_idx = [i for i in range(k) if i != idx]

        # Appendix step 2: compute dw/dbeta and h
        if len(other_idx) > 0:
            D0_ww = D0[np.ix_(other_idx, other_idx)]  # d2l/dw2
            D0_bw = D0[idx, other_idx]  # d2l/dbeta*dw

            # dw/dbeta = -(d2l/dw2)^-1 @ (d2l/dbeta*dw)  (Eq. 5)
            try:
                dw_db = -np.linalg.solve(D0_ww, D0_bw)
            except np.linalg.LinAlgError:
                dw_db = -np.linalg.lstsq(D0_ww, D0_bw, rcond=None)[0]

            # d2l_bar/dbeta2 = D_bb - D_bw @ D_ww^-1 @ D_wb (eq. 6 denominator)
            d2l_db2 = D0[idx, idx] + D0_bw @ dw_db
        else:
            # Single parameter case
            dw_db = np.array([])
            d2l_db2 = D0[idx, idx]

        # step size h (Eq. 6)
        if d2l_db2 >= 0:
            # Hessian should be negative definite, fallback
            h = which * 0.5
        else:
            h = which * np.sqrt(chi2_crit / abs(d2l_db2)) / 2

        # Appendix step 3: theta(1) = theta_hat + h * [1, dw/dbeta]' (Eq. 4)
        tangent = np.zeros(k)
        tangent[idx] = 1.0
        if len(other_idx) > 0:
            tangent[other_idx] = dw_db
        theta = theta + h * tangent

        # Appendix steps 4-9: Modified Newton-Raphson
        for iteration in range(1, max_iter + 1):
            # Appendix step 4: compute score and Hessian at theta(i)
            q = compute_logistic_quantities(X, y, theta, sample_weight, offset)

            # Appendix step 5: F = [l - l*, dl/dw]' (eq. 2)
            F = q.modified_score.copy()
            F[idx] = q.loglik - l_star

            # Appendix step 9: check convergence
            if np.max(np.abs(F)) <= tol:
                return theta[idx], True, iteration

            # D = d2l/dtheta2 at current theta (Appendix step 4)
            D = -q.fisher_info
            G = D.copy()
            G[idx, :] = q.modified_score  # Jacobian (Eq. 3)

            # Appendix step 6: v = G^-1 F (direction to subtract)
            try:
                G_inv = np.linalg.inv(G)
                v = G_inv @ F
            except np.linalg.LinAlgError:
                v = cast(NDArray[np.float64], np.linalg.lstsq(G, F, rcond=None)[0])
                try:
                    G_inv = np.linalg.pinv(G)
                except np.linalg.LinAlgError:
                    # singular G; take damped step (pg 92)
                    theta = theta - 0.1 * v
                    continue

            # Appendix step 7: quadratic correction
            # g'Dg*s^2 + (2v'Dg - 2)*s + v'Dv = 0 (Eq. 8)
            g_j = G_inv[:, idx]
            a = g_j @ D @ g_j
            b = 2 * v @ D @ g_j - 2
            c = v @ D @ v

            discriminant = b * b - 4 * a * c

            if discriminant >= 0 and abs(a) > 1e-10:
                sqrt_disc = np.sqrt(discriminant)
                s1 = (-b + sqrt_disc) / (2 * a)
                s2 = (-b - sqrt_disc) / (2 * a)

                # Pick root giving smaller step (pg 90)
                step1 = -v - s1 * g_j
                step2 = -v - s2 * g_j
                norm1 = step1 @ (-D0) @ step1
                norm2 = step2 @ (-D0) @ step2

                s = s1 if norm1 < norm2 else s2
                delta = -v - s * g_j  # Eq. 9
            else:
                # No real roots: damped step (pg 92)
                delta = -0.1 * v

            # Appendix step 8: theta(i+1) = theta(i) + delta
            theta = theta + delta

        # failed to converge
        return theta[idx], False, max_iter

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
        indices = self._resolve_feature_indices(features)

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

    def _resolve_feature_indices(
        self,
        features: int | str | Sequence[int | str] | None,
    ) -> list[int]:
        """Convert feature names and/or indices to list of parameter indices."""
        n_coef = len(self.coef_)
        n_params = n_coef + 1 if self.fit_intercept else n_coef

        if features is None:
            return list(range(n_params))

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
                indices.append(int(feat))
            else:
                raise TypeError(
                    f"Elements of `features` must be int or str, got {type(feat)}"
                )

        if n_coef in indices and not self.fit_intercept:
            raise ValueError("Cannot specify intercept when fit_intercept=False")

        return indices

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

    # hat diagonal: h_i = v_i' Fisher^{-1} v_i where v_i = sqrt(w_i) * x_i
    try:
        k = fisher_info.shape[0]
        cho = scipy.linalg.cho_factor(fisher_info, check_finite=False)
        inv_fisher_info = scipy.linalg.cho_solve(
            cho, np.eye(k, dtype=np.float64), check_finite=False
        )
        L = cho[0]
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        solved = inv_fisher_info @ XtW
        h = np.einsum("ij,ij->j", solved, XtW)  # h_i = solved[:,i] · XtW[:,i]
    except (
        scipy.linalg.LinAlgError
    ):  # fisher info not positive definite - fall back to pinv
        solved, *_ = np.linalg.lstsq(fisher_info, XtW, rcond=None)
        sign, logdet = np.linalg.slogdet(fisher_info)
        if sign <= 0:
            # use -inf so loglik approaches -inf
            logdet = -np.inf
        h = np.einsum("ij,ij->j", solved, XtW)

    # augmented fisher information
    w_aug = (sample_weight + h) * p * (1 - p)
    sqrt_w_aug = np.sqrt(w_aug)
    XtW_aug = X.T * sqrt_w_aug
    fisher_info_aug = XtW_aug @ XtW_aug.T

    # L*(β) = Σ weight_i * [y_i*log(p_i) + (1-y_i)*log(1-p_i)] + 0.5*log|I(β)|
    # y*log(p) + (1-y)*log(1-p) = y*eta - log(1+exp(eta))
    # avoids log(0) when p>-0 or p->1
    loglik = sample_weight @ (y * eta - np.logaddexp(0, eta)) + 0.5 * logdet

    # modified score U* = X'[weights*(y-p) + h*(0.5-p)]
    residual = sample_weight * (y - p) + h * (0.5 - p)
    modified_score = X.T @ residual

    return LogisticQuantities(
        loglik=loglik,
        modified_score=modified_score,
        fisher_info=fisher_info_aug,
    )
