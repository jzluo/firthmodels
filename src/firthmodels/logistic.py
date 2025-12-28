import math
import warnings
from dataclasses import dataclass
from typing import Literal, Self, Sequence, cast

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import dpotrf, dpotrs
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._tags import ClassifierTags, Tags
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    validate_data,
)

from firthmodels import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from firthmodels._numba.logistic import (
        _STATUS_CONVERGED,
        _STATUS_MAX_ITER,
        _STATUS_STEP_HALVING_FAILED,
        constrained_lrt_1df_logistic,
        newton_raphson_logistic,
        profile_ci_bound_logistic,
    )

from firthmodels._lrt import LRTResult, constrained_lrt_1df
from firthmodels._profile_ci import ProfileCIBoundResult, profile_ci_bound
from firthmodels._solvers import newton_raphson
from firthmodels._utils import FirthResult, resolve_feature_indices


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
    backend : {'auto', 'numba', 'numpy'}, default='auto'
        Computational backend to use.
        - 'auto': uses the Numba implementation when available, otherwise falls back to
          the NumPy/SciPy path.
        - 'numba': forces the Numba backend (raises ImportError if Numba isn't installed).
        - 'numpy': forces the NumPy/SciPy implementation.
    solver : {'newton-raphson'}, default='newton-raphson'
        Optimization algorithm. Only 'newton-raphson' is currently supported.
    max_iter : int, default=25
        Maximum number of iterations
    max_step : float, default=5.0
        Maximum step size per coefficient (Newton-Raphson only)
    max_halfstep : int, default=25
        Maximum number of step-halvings per iteration (Newton-Raphson only)
    gtol : float, default=1e-4
        Gradient convergence criteria. Converged when max|gradient| < gtol.
    xtol : float, default=1e-4
        Parameter convergence criteria. Converged when max|delta| < xtol.
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
        backend: Literal["auto", "numba", "numpy"] = "auto",
        solver: Literal["newton-raphson"] = "newton-raphson",
        max_iter: int = 25,
        max_step: float = 5.0,
        max_halfstep: int = 25,
        gtol: float = 1e-4,
        xtol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        self.solver = solver
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.gtol = gtol
        self.xtol = xtol
        self.fit_intercept = fit_intercept
        self.backend = backend

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.classifier_tags = ClassifierTags()
        tags.classifier_tags.multi_class = False
        return tags

    def _resolve_backend(self) -> Literal["numba", "numpy"]:
        if self.backend == "auto":
            return "numba" if NUMBA_AVAILABLE else "numpy"
        if self.backend == "numba":
            if not NUMBA_AVAILABLE:
                raise ImportError("backend='numba' but numba is not installed.")
            return "numba"
        if self.backend == "numpy":
            return "numpy"
        raise ValueError(
            f"backend must be 'auto', 'numba', or 'numpy', got '{self.backend}'"
        )

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
        # pre-allocate workspace arrays to reduce allocations
        workspace = _Workspace(n=X.shape[0], k=n_features)

        if self._resolve_backend() == "numba":
            beta, loglik, fisher_info, n_iter, status = newton_raphson_logistic(
                X=X,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                workspace=workspace.numba_buffers(),
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
            )
            if status == _STATUS_STEP_HALVING_FAILED:
                warnings.warn(
                    "Step-halving failed to converge.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
            elif status == _STATUS_MAX_ITER:
                warnings.warn(
                    "Maximum number of iterations reached without convergence.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
            result = FirthResult(
                beta=beta,
                loglik=loglik,
                fisher_info=fisher_info,
                n_iter=n_iter,
                converged=(status == _STATUS_CONVERGED),
            )
        else:

            def compute_quantities(beta):
                return compute_logistic_quantities(
                    X=X,
                    y=y,
                    beta=beta,
                    sample_weight=sample_weight,
                    offset=offset,
                    workspace=workspace,
                )

            result = newton_raphson(
                compute_quantities=compute_quantities,
                n_features=n_features,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
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
        if not np.all(np.isfinite(result.fisher_info)):
            warnings.warn(
                "Fisher information matrix is not finite; "
                "standard errors and p-values cannot be computed.",
                RuntimeWarning,
                stacklevel=2,
            )
            bse = np.full_like(result.beta, np.nan)
        else:
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

        self._workspace = workspace
        return self

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

        if self.fit_intercept:
            beta_hat_full = np.concatenate([self.coef_, [self.intercept_]])
        else:
            beta_hat_full = self.coef_

        if self._resolve_backend() == "numba":
            loglik_constrained, n_iter, status = constrained_lrt_1df_logistic(
                X=X,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                idx=idx,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
                workspace=self._workspace.numba_buffers(),
            )
            if status == _STATUS_STEP_HALVING_FAILED:
                warnings.warn(
                    "Step-halving failed to converge.",
                    ConvergenceWarning,
                    stacklevel=3,  # caller -> lrt() -> _compute_single_lrt
                )
            elif status == _STATUS_MAX_ITER:
                warnings.warn(
                    "Maximum number of iterations reached without convergence.",
                    ConvergenceWarning,
                    stacklevel=3,
                )
            chi2 = max(0.0, 2 * (self.loglik_ - loglik_constrained))
            pvalue = scipy.stats.chi2.sf(chi2, df=1)
            #  back-corrected SE: |beta|/sqrt(chi2), ensures (beta/SE)^2 = chi2
            bse = abs(beta_hat_full[idx]) / math.sqrt(chi2) if chi2 > 0 else math.inf
            result = LRTResult(chi2=chi2, pvalue=pvalue, bse_backcorrected=bse)
        else:

            def compute_quantities_full(beta):
                return compute_logistic_quantities(
                    X=X,
                    y=y,
                    beta=beta,
                    sample_weight=sample_weight,
                    offset=offset,
                    workspace=self._workspace,
                )

            result = constrained_lrt_1df(
                idx=idx,
                beta_hat_full=beta_hat_full,
                loglik_full=self.loglik_,
                compute_quantities_full=compute_quantities_full,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
            )

        self.lrt_pvalues_[idx] = result.pvalue
        self.lrt_bse_[idx] = result.bse_backcorrected

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

            X, y, sample_weight, offset = self._fit_data

            def compute_quantities_full(beta: NDArray[np.float64]):
                return compute_logistic_quantities(
                    X=X,
                    y=y,
                    beta=beta,
                    sample_weight=sample_weight,
                    offset=offset,
                    workspace=self._workspace,
                )

            theta_hat = (
                np.concatenate([self.coef_, [self.intercept_]])
                if self.fit_intercept
                else self.coef_
            )
            D0 = -compute_quantities_full(theta_hat).fisher_info  # hessian at MLE
            for idx in indices:
                for bound_idx, which in enumerate([-1, 1]):  # lower, upper
                    if not computed[idx, bound_idx]:
                        which = cast(Literal[-1, 1], which)  # mypy -_-
                        if self._resolve_backend() == "numba":
                            bound, converged, iterations = profile_ci_bound_logistic(
                                X=X,
                                y=y,
                                sample_weight=sample_weight,
                                offset=offset,
                                idx=idx,
                                theta_hat=theta_hat,
                                l_star=l_star,
                                which=which,
                                chi2_crit=chi2_crit,
                                max_iter=max_iter,
                                tol=tol,
                                D0=D0,
                                workspace=self._workspace.numba_buffers(),
                            )
                            result = ProfileCIBoundResult(
                                bound=bound,
                                converged=converged,
                                n_iter=iterations,
                            )

                        else:
                            result = profile_ci_bound(
                                idx=idx,
                                theta_hat=theta_hat,
                                l_star=l_star,
                                which=which,
                                max_iter=max_iter,
                                tol=tol,
                                chi2_crit=chi2_crit,
                                compute_quantities_full=compute_quantities_full,
                                D0=D0,
                            )

                        if result.converged:
                            ci[idx, bound_idx] = result.bound
                        else:
                            warnings.warn(
                                f"Profile-likelihood CI did not converge for parameter {idx} "
                                f"({'lower' if which == -1 else 'upper'} bound) after {result.n_iter} iterations.",
                                ConvergenceWarning,
                                stacklevel=2,
                            )
                        # mark AFTER completion (interrupt safety for jupyter people)
                        computed[idx, bound_idx] = True
            return ci

        else:
            raise ValueError(f"method must be 'wald' or 'pl', got '{method}'")

    def _resolve_feature_indices(
        self,
        features: int | str | Sequence[int | str] | None,
    ) -> list[int]:
        """Convert feature names and/or indices to list of parameter indices."""
        n_coefs = len(self.coef_)
        n_params = n_coefs + 1 if self.fit_intercept else n_coefs

        return resolve_feature_indices(
            features,
            n_params=n_params,
            feature_names_in=getattr(self, "feature_names_in_", None),
            intercept_idx=n_coefs if self.fit_intercept else None,
        )

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
        if self.gtol < 0 or self.xtol < 0:
            raise ValueError("gtol and xtol must be non-negative.")
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            y_numeric=False,
            ensure_min_samples=2,
            order="C",
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


class _Workspace:
    """Pre-allocated arrays for compute_logistic_quantities"""

    __slots__ = (
        "eta",
        "p",
        "w",
        "sqrt_w",
        "XtW",
        "fisher_info",
        "eye_k",
        "solved",
        "h",
        "w_aug",
        "sqrt_w_aug",
        "XtW_aug",
        "fisher_info_aug",
        "residual",
        "temp_k",
    )

    def __init__(self, n: int, k: int) -> None:
        self.eta = np.empty(n, dtype=np.float64)
        self.p = np.empty(n, dtype=np.float64)
        self.w = np.empty(n, dtype=np.float64)
        self.sqrt_w = np.empty(n, dtype=np.float64)
        self.XtW = np.empty((k, n), dtype=np.float64, order="F")
        self.fisher_info = np.zeros((k, k), dtype=np.float64, order="F")
        self.eye_k = np.eye(k, dtype=np.float64, order="F")
        self.solved = np.empty((k, n), dtype=np.float64, order="F")
        self.h = np.empty(n, dtype=np.float64)
        self.w_aug = np.empty(n, dtype=np.float64)
        self.sqrt_w_aug = np.empty(n, dtype=np.float64)
        self.XtW_aug = np.empty((k, n), dtype=np.float64, order="F")
        self.fisher_info_aug = np.empty((k, k), dtype=np.float64, order="F")
        self.residual = np.empty(n, dtype=np.float64)
        self.temp_k = np.empty(k, dtype=np.float64)

    def numba_buffers(self):  # for numba
        return (
            self.eta,
            self.p,
            self.w,
            self.sqrt_w,
            self.XtW,
            self.fisher_info,
            self.solved,
            self.h,
            self.w_aug,
            self.sqrt_w_aug,
            self.XtW_aug,
            self.fisher_info_aug,
            self.residual,
            self.temp_k,  # used as modified_score
        )


def compute_logistic_quantities(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    beta: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
    workspace: _Workspace | None = None,
) -> LogisticQuantities:
    """Compute all quantities needed for one Newton-Raphson iteration."""
    n, k = X.shape

    if workspace is not None:
        # eta = X @ beta + offset
        # p = expit(eta)
        ws = workspace
        np.dot(X, beta, out=ws.eta)
        np.add(ws.eta, offset, out=ws.eta)
        expit(ws.eta, out=ws.p)

        # w = sample_weight * p * (1 - p)
        np.subtract(1.0, ws.p, out=ws.w)
        np.multiply(ws.p, ws.w, out=ws.w)
        np.multiply(sample_weight, ws.w, out=ws.w)

        # # Fisher information: X'WX
        # sqrt_w = np.sqrt(w)
        # XtW = X.T * sqrt_w  # (k, n) broadcast so we don't materialize (n, n) diag matrix
        # fisher_info = XtW @ XtW.T
        np.sqrt(ws.w, out=ws.sqrt_w)
        np.multiply(X.T, ws.sqrt_w, out=ws.XtW)
        np.matmul(ws.XtW, ws.XtW.T, out=ws.fisher_info)

        try:
            L, info = dpotrf(ws.fisher_info, lower=1, overwrite_a=0)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrf failed")

            np.log(L.diagonal(), out=ws.temp_k)
            logdet = 2.0 * ws.temp_k.sum()

            inv_fisher_info, info = dpotrs(L, ws.eye_k, lower=1)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrs failed")

            np.matmul(inv_fisher_info, ws.XtW, out=ws.solved)

        except scipy.linalg.LinAlgError:
            ws.solved[:] = np.linalg.lstsq(ws.fisher_info, ws.XtW, rcond=None)[0]
            sign, logdet = np.linalg.slogdet(ws.fisher_info)
            if sign <= 0:
                # use -inf so loglik approaches -inf
                logdet = -np.inf

        np.einsum(
            "ij,ij->j", ws.solved, ws.XtW, out=ws.h
        )  # h_i = solved[:,i] · XtW[:,i]

        # augmented fisher information
        # w_aug = (sample_weight + h) * p * (1 - p)
        np.add(sample_weight, ws.h, out=ws.w_aug)
        np.subtract(1.0, ws.p, out=ws.sqrt_w_aug)  # reuse
        np.multiply(ws.w_aug, ws.p, out=ws.w_aug)
        np.multiply(ws.w_aug, ws.sqrt_w_aug, out=ws.w_aug)

        np.sqrt(ws.w_aug, out=ws.sqrt_w_aug)
        np.multiply(X.T, ws.sqrt_w_aug, out=ws.XtW_aug)
        np.matmul(ws.XtW_aug, ws.XtW_aug.T, out=ws.fisher_info_aug)

        # loglik = sample_weight @ (y * eta - np.logaddexp(0, eta)) + 0.5 * logdet
        np.multiply(y, ws.eta, out=ws.w)  # reuse, w = y * eta
        np.logaddexp(
            0, ws.eta, out=ws.sqrt_w_aug
        )  # reuse, sqrt_w_aug = log(1 + exp(eta))
        np.subtract(ws.w, ws.sqrt_w_aug, out=ws.w)  # reuse, w = y*eta - log(1+exp(eta))
        loglik = sample_weight @ ws.w + 0.5 * logdet

        # modified score U* = X'[weights*(y-p) + h*(0.5-p)]
        # residual = sample_weight * (y - p) + h * (0.5 - p)
        np.subtract(y, ws.p, out=ws.residual)
        np.multiply(sample_weight, ws.residual, out=ws.residual)
        np.subtract(0.5, ws.p, out=ws.w)  # reuse, w = 0.5 - p
        np.multiply(ws.h, ws.w, out=ws.w)
        np.add(ws.residual, ws.w, out=ws.residual)
        modified_score = X.T @ ws.residual

    else:
        eta = X @ beta + offset
        p = expit(eta)

        # W = diag(weights * p * (1-p))
        w = sample_weight * p * (1 - p)

        # Fisher information: X'WX
        sqrt_w = np.sqrt(w)
        XtW = (
            X.T * sqrt_w
        )  # (k, n) broadcast so we don't materialize (n, n) diag matrix
        fisher_info = XtW @ XtW.T

        # hat diagonal: h_i = v_i' Fisher^{-1} v_i where v_i = sqrt(w_i) * x_i
        try:
            k = fisher_info.shape[0]

            L, info = dpotrf(fisher_info, lower=1, overwrite_a=0)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrf failed")
            logdet = 2.0 * np.log(L.diagonal()).sum()

            inv_fisher_info, info = dpotrs(
                L, np.eye(k, dtype=np.float64, order="F"), lower=1
            )
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrs failed")

            solved = inv_fisher_info @ XtW
        except (
            scipy.linalg.LinAlgError
        ):  # fisher info not positive definite - fall back to pinv
            solved, *_ = np.linalg.lstsq(fisher_info, XtW, rcond=None)
            sign, logdet = np.linalg.slogdet(fisher_info)
            if sign <= 0:
                # use -inf so loglik approaches -inf
                logdet = -np.inf

        h = np.einsum("ij,ij->j", solved, XtW)  # h_i = solved[:,i] · XtW[:,i]

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
        fisher_info=ws.fisher_info_aug if workspace else fisher_info_aug,
    )
