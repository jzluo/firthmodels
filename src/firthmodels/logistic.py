import math
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence, cast

try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import dgeqp3, dorgqr, dpotrf, dpotrs
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
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
        _STATUS_LINALG_FAIL,
        _STATUS_MAX_ITER,
        _STATUS_RANK_DEFICIENT,
        _STATUS_STEP_HALVING_FAILED,
        constrained_lrt_1df_logistic,
        newton_raphson_logistic,
        profile_ci_bound_logistic,
    )

from firthmodels._lrt import constrained_lrt_1df, lrt_result_from_loglik
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
    penalty_weight : float, default=0.5
        Weight of the Firth penalty term. The default 0.5 corresponds to the standard
        Firth bias reduction method (Firth, 1993), equivalent to using Jeffreys'
        invariant prior. Set to 0 for unpenalized maximum likelihood estimation.

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
        NaN until computed or if the constrained fit fails. Includes intercept as
        last element if `fit_intercept=True`.
    lrt_bse_ : ndarray of shape (n_params,)
        Back-corrected standard errors from LRT. Computed by `lrt()`.
        Values are NaN until computed or if the constrained fit fails. Includes
        intercept as last element if `fit_intercept=True`.
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
        penalty_weight: float = 0.5,
    ) -> None:
        self.solver = solver
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.gtol = gtol
        self.xtol = xtol
        self.fit_intercept = fit_intercept
        self.penalty_weight = penalty_weight
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
            Array of weights that are assigned to individual samples. If not provided,
            then each sample is given unit weight.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to linear predictor. Offsets aren't stored for
            prediction, and need to be passed to the prediction methods if needed.


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
        if not np.any(sample_weight[y == 0] > 0) or not np.any(
            sample_weight[y == 1] > 0
        ):
            raise ValueError("Need at least one positive-weight sample in each class.")

        offset = self._validate_offset(offset, X.shape[0])

        if self.fit_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])

        n_features = X.shape[1]

        # early rank check
        n_eff = np.count_nonzero(sample_weight > 0)
        if n_features > n_eff:
            raise ValueError(
                f"Number of parameters ({n_features}) exceeds the effective sample size "
                f"({n_eff})."
            )

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
                penalty_weight=self.penalty_weight,
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
            elif status == _STATUS_RANK_DEFICIENT:
                raise scipy.linalg.LinAlgError(
                    "Weighted design matrix is rank deficient."
                )
            elif status == _STATUS_LINALG_FAIL:
                raise scipy.linalg.LinAlgError(
                    "Weighted design QR factorization failed."
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
                    penalty_weight=self.penalty_weight,
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
        self._cov = None
        if not np.all(np.isfinite(result.fisher_info)):
            warnings.warn(
                "Fisher information matrix is not finite; "
                "standard errors and p-values cannot be computed.",
                RuntimeWarning,
                stacklevel=2,
            )
            bse = np.full_like(result.beta, np.nan)
        else:
            L, info = dpotrf(result.fisher_info, lower=1, overwrite_a=0)
            if info == 0:
                k = result.fisher_info.shape[0]
                eye_k = np.eye(k, dtype=np.float64, order="F")
                inv_fisher_info, info = dpotrs(L, eye_k, lower=1)
                if info == 0:
                    self._cov = inv_fisher_info
                    bse = np.sqrt(self._cov.diagonal())
            if info != 0:
                warnings.warn(
                    "Fisher information is not positive definite; "
                    "standard errors and p-values cannot be computed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
        warm_start: bool = True,
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
        warm_start : bool, default=True
            If True, warm-start constrained fits using the covariance from the full
            model (when available).

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

        if indices and (not self.converged_ or not math.isfinite(self.loglik_)):
            reason = (
                "the full-model fit did not converge"
                if not self.converged_
                else "the full-model penalized log-likelihood is non-finite"
            )
            warnings.warn(
                f"LRT inference was not computed because {reason}.",
                ConvergenceWarning,
                stacklevel=2,
            )
            return self

        # compute LRT
        for idx in indices:
            if np.isnan(self.lrt_pvalues_[idx]):
                self._compute_single_lrt(idx, warm_start=warm_start)
        return self

    def _compute_single_lrt(self, idx: int, *, warm_start: bool = True) -> None:
        """
        Fit constrained model with `beta[idx]=0` and compute LRT p-value and
        back-corrected standard error.

        Parameters
        ----------
        idx : int
            Index of the coefficient to test. Use len(coef_) for the intercept.
        warm_start : bool, default=True
            If True, warm-start constrained fits using the covariance from the full
            model (when available).
        """
        X, y, sample_weight, offset = self._fit_data

        if self.fit_intercept:
            beta_hat_full = np.concatenate([self.coef_, [self.intercept_]])
        else:
            beta_hat_full = self.coef_

        k = beta_hat_full.shape[0]
        free_idx = np.array([i for i in range(k) if i != idx], dtype=np.intp)
        beta_free = beta_hat_full[free_idx]
        beta_j = beta_hat_full[idx]
        beta_init_free = None

        # Warm start for constrained LRT (beta_j=0) using cov from the full fit
        # (Schur complement)
        if warm_start and self._cov is not None:
            denom = self._cov[idx, idx]
            if np.isfinite(denom) and denom > 0.0:
                col = self._cov[free_idx, idx]
                if np.all(np.isfinite(col)):
                    beta_init_free = beta_free - beta_j * (col / denom)

        if self._resolve_backend() == "numba":
            if beta_init_free is None:
                beta_init_free_numba = np.zeros(k - 1, dtype=np.float64)
            else:
                beta_init_free_numba = beta_init_free
            loglik_constrained, n_iter, status = constrained_lrt_1df_logistic(
                X=X,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                idx=idx,
                beta_init_free=beta_init_free_numba,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
                workspace=self._workspace.numba_buffers(),
                penalty_weight=self.penalty_weight,
            )
            if status == _STATUS_RANK_DEFICIENT:
                raise scipy.linalg.LinAlgError(
                    f"Weighted design matrix is rank deficient during LRT for parameter {idx}."
                )
            elif status == _STATUS_LINALG_FAIL:
                raise scipy.linalg.LinAlgError(
                    f"Weighted design QR factorization failed during LRT for parameter {idx}."
                )
            failure_reason: Literal["step_halving", "max_iter"] | None = None
            if status == _STATUS_STEP_HALVING_FAILED:
                failure_reason = "step_halving"
            elif status == _STATUS_MAX_ITER:
                failure_reason = "max_iter"

            result = lrt_result_from_loglik(
                beta=beta_hat_full[idx],
                loglik_full=self.loglik_,
                loglik_constrained=loglik_constrained,
                converged=(status == _STATUS_CONVERGED),
                n_iter=n_iter,
                failure_reason=failure_reason,
            )
        else:

            def compute_quantities_full(beta):
                return compute_logistic_quantities(
                    X=X,
                    y=y,
                    beta=beta,
                    sample_weight=sample_weight,
                    offset=offset,
                    penalty_weight=self.penalty_weight,
                    workspace=self._workspace,
                )

            result = constrained_lrt_1df(
                idx=idx,
                beta_hat_full=beta_hat_full,
                loglik_full=self.loglik_,
                compute_quantities_full=compute_quantities_full,
                beta_init_free=beta_init_free,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
            )

        if not result.valid:
            warnings.warn(
                f"LRT for parameter {idx} was not computed "
                f"({result.failure_reason}); p-value and back-corrected "
                "standard error remain NaN.",
                ConvergenceWarning,
                stacklevel=3,  # caller -> lrt() -> _compute_single_lrt
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
                    penalty_weight=self.penalty_weight,
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
                            bound, status, iterations = profile_ci_bound_logistic(
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
                                penalty_weight=self.penalty_weight,
                            )
                            if status == _STATUS_RANK_DEFICIENT:
                                raise scipy.linalg.LinAlgError(
                                    f"Weighted design matrix is rank deficient during "
                                    f"{'lower' if which == -1 else 'upper'} bound CI "
                                    f"for parameter {idx}."
                                )
                            elif status == _STATUS_LINALG_FAIL:
                                raise scipy.linalg.LinAlgError(
                                    f"Weighted design QR factorization failed during "
                                    f"{'lower' if which == -1 else 'upper'} bound CI "
                                    f"for parameter {idx}."
                                )
                            result = ProfileCIBoundResult(
                                bound=bound,
                                converged=(status == _STATUS_CONVERGED),
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
        offset: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """
        Return the linear predictor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to the linear predictor. If not provided, no
            offset is applied.

        Returns
        -------
        ndarray of shape (n_samples,)
            Linear predictor ``X @ coef_ + intercept_ + offset``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)  # for mypy
        offset_arr = self._validate_offset(offset, X.shape[0])
        return X @ self.coef_ + self.intercept_ + offset_arr

    def predict_proba(
        self,
        X: ArrayLike,
        offset: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """
        Return class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to the linear predictor. If not provided, no
            offset is applied.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probability estimates for the two classes.
        """
        scores = self.decision_function(X, offset=offset)
        p1 = expit(scores)
        return np.column_stack([1 - p1, p1])

    def predict(
        self,
        X: ArrayLike,
        offset: ArrayLike | None = None,
    ) -> NDArray[np.int_]:
        """
        Return predicted class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to the linear predictor. If not provided, no
            offset is applied.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X, offset=offset)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_log_proba(
        self,
        X: ArrayLike,
        offset: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """
        Return log class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to the linear predictor. If not provided, no
            offset is applied.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Log probability estimates for the two classes.
        """
        return np.log(self.predict_proba(X, offset=offset))

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        offset: ArrayLike | None = None,
    ) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for ``X``.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        offset : array-like of shape (n_samples,), default=None
            Fixed offset added to the linear predictor. If not provided, no
            offset is applied.

        Returns
        -------
        float
            Mean accuracy of the predictions.
        """
        return float(
            accuracy_score(
                y,
                self.predict(X, offset=offset),
                sample_weight=sample_weight,
            )
        )

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
        if self.max_step < 0:
            raise ValueError(f"max_step must be non-negative, got {self.max_step}")
        if self.gtol < 0 or self.xtol < 0:
            raise ValueError("gtol and xtol must be non-negative.")
        if self.penalty_weight < 0 or not math.isfinite(self.penalty_weight):
            raise ValueError(
                f"penalty_weight must be non-negative and finite, got {self.penalty_weight}"
            )
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

    @staticmethod
    def _validate_offset(
        offset: ArrayLike | None, n_samples: int
    ) -> NDArray[np.float64]:
        """Validate an offset vector, defaulting to zeros."""
        if offset is None:
            return np.zeros(n_samples, dtype=np.float64)

        offset_arr = cast(
            NDArray[np.float64],
            check_array(offset, ensure_2d=False, dtype=np.float64, input_name="offset"),
        )
        if offset_arr.ndim != 1:
            raise ValueError("offset must be a one-dimensional array")
        if offset_arr.shape[0] != n_samples:
            raise ValueError(
                f"Length of offset ({offset_arr.shape[0]}) does not match "
                f"number of samples ({n_samples})"
            )
        return offset_arr


@dataclass
class LogisticQuantities:
    """Quantities needed for one Newton-Raphson iteration"""

    loglik: float
    modified_score: NDArray[
        np.float64
    ]  # (n_features,) U* = X'[weights*(y - p) + 2 * penalty_weight * h * (0.5 - p)]
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
        "temp_k",  # note - this is used as a buffer in numpy, and as modified_score in numba
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
            self.eye_k,
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
    workspace: _Workspace,
    penalty_weight: float = 0.5,
) -> LogisticQuantities:
    """Compute all quantities needed for one Newton-Raphson iteration."""
    n, k = X.shape

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

    if penalty_weight == 0.0:
        # loglik = sample_weight @ (y * eta - np.logaddexp(0, eta))
        np.multiply(y, ws.eta, out=ws.w)  # reuse, w = y * eta
        np.logaddexp(
            0, ws.eta, out=ws.sqrt_w_aug
        )  # reuse, sqrt_w_aug = log(1 + exp(eta))
        np.subtract(ws.w, ws.sqrt_w_aug, out=ws.w)  # reuse, w = y*eta - log(1+exp(eta))
        loglik = float(sample_weight @ ws.w)

        # score = X'[weights*(y-p)]
        np.subtract(y, ws.p, out=ws.residual)
        np.multiply(sample_weight, ws.residual, out=ws.residual)
        modified_score = X.T @ ws.residual

        return LogisticQuantities(
            loglik=loglik,
            modified_score=modified_score,
            fisher_info=ws.fisher_info,
        )

    try:
        L, info = dpotrf(ws.fisher_info, lower=1, overwrite_a=0)
        if info != 0:
            raise scipy.linalg.LinAlgError("dpotrf failed")

        # check pivots, dpotrf can return info=0 on a singular matrix on Accelerate
        L_diag = L.diagonal()
        chol_tol = np.sqrt(max(n, k) * np.finfo(np.float64).eps) * L_diag.max()
        if L_diag.min() <= chol_tol:
            raise scipy.linalg.LinAlgError("Cholesky pivot below tolerance")

        np.log(L_diag, out=ws.temp_k)
        logdet = 2.0 * ws.temp_k.sum()

        inv_fisher_info, info = dpotrs(L, ws.eye_k, lower=1)
        if info != 0:
            raise scipy.linalg.LinAlgError("dpotrs failed")

        np.matmul(inv_fisher_info, ws.XtW, out=ws.solved)

        # h_i = solved[:,i] · XtW[:,i]
        np.einsum("ij,ij->j", ws.solved, ws.XtW, out=ws.h)

    except scipy.linalg.LinAlgError:
        # XtW.T is C-order, explicitly copy to F-order
        XW = np.asfortranarray(ws.XtW.T)

        # pivoting QR
        qr, jpvt, tau, _, info = dgeqp3(XW, overwrite_a=1)
        if info != 0:
            raise scipy.linalg.LinAlgError("dgeqp3 failed")

        R = np.triu(qr[:k, :k])

        # check rank
        tol = max(XW.shape) * np.finfo(np.float64).eps * abs(R[0, 0])
        rank = (np.abs(R.diagonal()) > tol).sum()
        if rank < k:
            raise scipy.linalg.LinAlgError("Weighted design matrix is rank deficient")

        Q, _, info = dorgqr(qr, tau, overwrite_a=1)
        if info != 0:
            raise scipy.linalg.LinAlgError("dorgqr failed")

        np.log(np.abs(R.diagonal()), out=ws.temp_k)
        logdet = 2.0 * ws.temp_k.sum()

        # hat diag: h_i = sum_j Q_ij^2
        np.einsum("ij, ij->i", Q, Q, out=ws.h)

    penalty_scale = 2.0 * penalty_weight

    # augmented fisher information
    # w_aug = (sample_weight + 2*penalty_weight*h) * p * (1 - p)
    np.multiply(ws.h, penalty_scale, out=ws.w_aug)
    np.add(sample_weight, ws.w_aug, out=ws.w_aug)
    np.subtract(1.0, ws.p, out=ws.sqrt_w_aug)  # reuse
    np.multiply(ws.w_aug, ws.p, out=ws.w_aug)
    np.multiply(ws.w_aug, ws.sqrt_w_aug, out=ws.w_aug)

    np.sqrt(ws.w_aug, out=ws.sqrt_w_aug)
    np.multiply(X.T, ws.sqrt_w_aug, out=ws.XtW_aug)
    np.matmul(ws.XtW_aug, ws.XtW_aug.T, out=ws.fisher_info_aug)

    # loglik = sample_weight @ (y * eta - np.logaddexp(0, eta)) + penalty_weight * logdet
    np.multiply(y, ws.eta, out=ws.w)  # reuse, w = y * eta
    np.logaddexp(0, ws.eta, out=ws.sqrt_w_aug)  # reuse, sqrt_w_aug = log(1 + exp(eta))
    np.subtract(ws.w, ws.sqrt_w_aug, out=ws.w)  # reuse, w = y*eta - log(1+exp(eta))
    loglik = sample_weight @ ws.w + penalty_weight * logdet

    # modified score U* = X'[weights*(y-p) + 2*penalty_weight*h*(0.5-p)]
    # residual = sample_weight * (y - p) + 2*penalty_weight*h * (0.5 - p)
    np.subtract(y, ws.p, out=ws.residual)
    np.multiply(sample_weight, ws.residual, out=ws.residual)
    np.subtract(0.5, ws.p, out=ws.w)  # reuse, w = 0.5 - p
    np.multiply(ws.h, ws.w, out=ws.w)
    np.multiply(ws.w, penalty_scale, out=ws.w)
    np.add(ws.residual, ws.w, out=ws.residual)
    modified_score = X.T @ ws.residual

    return LogisticQuantities(
        loglik=loglik,
        modified_score=modified_score,
        fisher_info=ws.fisher_info_aug,
    )
