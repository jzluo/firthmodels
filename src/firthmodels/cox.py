from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal, Self, Sequence, cast

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import dpotrf, dpotrs, dpstrf
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data

from firthmodels import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from firthmodels._numba.cox import (
        _STATUS_CONVERGED,
        _STATUS_LINALG_FAIL,
        _STATUS_MAX_ITER,
        _STATUS_RANK_DEFICIENT,
        _STATUS_STEP_HALVING_FAILED,
        concordance_index,
        constrained_lrt_1df_cox,
        newton_raphson_cox,
        precompute_cox,
        profile_ci_bound_cox,
    )

from firthmodels._lrt import LRTResult, constrained_lrt_1df
from firthmodels._profile_ci import ProfileCIBoundResult, profile_ci_bound
from firthmodels._solvers import newton_raphson
from firthmodels._utils import FirthResult, resolve_feature_indices


class FirthCoxPH(BaseEstimator):
    """
    Cox proportional hazards with Firth's bias reduction.

    Parameters
    ----------
    backend : {'auto', 'numba', 'numpy'}, default='auto'
        Computational backend to use.
        - 'auto': uses the Numba implementation when available, otherwise falls back to
          the NumPy/SciPy path.
        - 'numba': forces the Numba backend (raises ImportError if Numba isn't installed).
        - 'numpy': forces the NumPy/SciPy implementation.
    max_iter : int, default=50
        Maximum number of Newton-Raphson iterations.
    max_step : float, default=5.0
        Maximum step size per coefficient.
    max_halfstep : int, default=5
        Maximum number of step-halvings per iteration.
    gtol : float, default=1e-4
        Gradient convergence criteria. Converged when max|gradient| < gtol.
    xtol : float, default=1e-6
        Parameter convergence criteria. Converged when max|delta| < xtol.
    penalty_weight : float, default=0.5
        Weight of the Firth penalty term. The default 0.5 corresponds to the standard
        Firth bias reduction method (Heinze and Schemper, 2001), equivalent to using
        Jeffreys' invariant prior. Set to 0 for unpenalized Cox partial likelihood
        estimation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted coefficients.
    loglik_ : float
        Fitted penalized log partial likelihood.
    n_iter_ : int
        Number of iterations run.
    converged_ : bool
        Whether the solver converged within `max_iter`.
    bse_ : ndarray of shape (n_features,)
        Wald standard errors.
    pvalues_ : ndarray of shape (n_features,)
        Wald p-values.
    lrt_pvalues_ : ndarray of shape (n_features,)
        Likelihood ratio test p-values. Computed by `lrt()`. Values are
        NaN until computed.
    lrt_bse_ : ndarray of shape (n_features,)
        Back-corrected standard errors from LRT. Computed by `lrt()`.
        Values are NaN until computed.
    unique_times_ : ndarray of shape (n_events,)
        Unique event times in ascending order.
    cum_baseline_hazard_ : ndarray of shape (n_events,)
        Breslow cumulative baseline hazard at each unique event time.
    baseline_survival_ : ndarray of shape (n_events,)
        Baseline survival function at each unique event time.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (if X has feature names).

    References
    ----------
    Heinze, G., Schemper, M. (2001). A Solution to the Problem of Monotone
    Likelihood in Cox Regression. Biometrics 57(1):114-119.
    """

    def __init__(
        self,
        backend: Literal["auto", "numba", "numpy"] = "auto",
        max_iter: int = 50,
        max_step: float = 5.0,
        max_halfstep: int = 5,
        gtol: float = 1e-4,
        xtol: float = 1e-6,
        penalty_weight: float = 0.5,
    ) -> None:
        self.backend = backend
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.gtol = gtol
        self.xtol = xtol
        self.penalty_weight = penalty_weight

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

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the Firth-penalized Cox model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like or tuple
            Survival outcome encoding. Accepts either:
            - a NumPy structured array with one boolean field (event indicator) and
            one float field (time), or
            - a two-tuple `(event, time)` where `event` is a bool array indicating whether
            the event occurred, and `time` is the observed time (event or censoring).

        Returns
        -------
        self : FirthCoxPH
            Fitted estimator.
        """
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
                f"penalty_weight must be non-negative and finite, "
                f"got {self.penalty_weight}"
            )

        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        X = cast(NDArray[np.float64], X)
        event, time = _validate_survival_y(y, n_samples=X.shape[0])

        backend = self._resolve_backend()
        # Precompute sorted data and block structure
        precomputed = _CoxPrecomputed.from_data(X, time, event, backend=backend)
        workspace = _Workspace(precomputed.n_samples, precomputed.n_features)

        if backend == "numba":
            beta, loglik, fisher_info, n_iter, status = newton_raphson_cox(
                X=precomputed.X,
                block_ends=precomputed.block_ends,
                block_d=precomputed.block_d,
                block_s=precomputed.block_s,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
                workspace=workspace.numba_buffers(),
                penalty_weight=self.penalty_weight,
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
                raise scipy.linalg.LinAlgError("Fisher information is rank deficient.")
            elif status == _STATUS_LINALG_FAIL:
                raise scipy.linalg.LinAlgError(
                    "dpstrf failed - Fisher information is not PSD."
                )

            result = FirthResult(
                beta=beta,
                loglik=loglik,
                fisher_info=fisher_info,
                n_iter=n_iter,
                converged=status == _STATUS_CONVERGED,
            )

            self.converged_ = result.converged

        else:  # numpy backend
            # Create closure for newton_raphson
            def quantities(beta: NDArray[np.float64]) -> CoxQuantities:
                return compute_cox_quantities(
                    beta=beta,
                    precomputed=precomputed,
                    workspace=workspace,
                    penalty_weight=self.penalty_weight,
                )

            # Run optimizer
            result = newton_raphson(
                compute_quantities=quantities,
                n_features=precomputed.n_features,
                max_iter=self.max_iter,
                max_step=self.max_step,
                max_halfstep=self.max_halfstep,
                gtol=self.gtol,
                xtol=self.xtol,
            )
            self.converged_ = result.converged

        self.coef_ = result.beta
        self.loglik_ = result.loglik
        self.n_iter_ = result.n_iter

        # Wald
        self._cov = None
        if not np.all(np.isfinite(result.fisher_info)):
            warnings.warn(
                "Fisher information matrix is not finite; "
                "standard errors and p-values cannot be computed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.bse_ = np.full(precomputed.n_features, np.nan)
        else:
            L, info = dpotrf(result.fisher_info, lower=1, overwrite_a=0)
            if info == 0:
                k = result.fisher_info.shape[0]
                eye_k = np.eye(k, dtype=np.float64, order="F")
                inv_fisher_info, info = dpotrs(L, eye_k, lower=1)
                if info == 0:
                    self._cov = inv_fisher_info
                    self.bse_ = np.sqrt(self._cov.diagonal())
            if info != 0:
                warnings.warn(
                    "Fisher information is not positive definite; "
                    "standard errors and p-values cannot be computed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.bse_ = np.full(precomputed.n_features, np.nan)

        z = result.beta / self.bse_
        self.pvalues_ = 2 * scipy.stats.norm.sf(np.abs(z))

        # need these for LRT and profile CI
        self._precomputed = precomputed
        self._workspace = workspace

        self.lrt_pvalues_ = np.full(len(result.beta), np.nan)
        self.lrt_bse_ = np.full(len(result.beta), np.nan)

        self._compute_baseline_hazard(self._precomputed)

        # _profile_ci_cache and _profile_ci_computed are keyed by (alpha, tol, max_iter)
        self._profile_ci_cache: dict[tuple[float, float, int], NDArray[np.float64]] = {}
        # tracks completed bound computations; False means never tried or interrupted
        self._profile_ci_computed: dict[
            tuple[float, float, int], NDArray[np.bool_]
        ] = {}

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
        warm_start : bool, default=True
            If True, warm-start constrained fits using the covariance from the full
            model (when available).

        Returns
        -------
        self : FirthCoxPH
        """
        check_is_fitted(self)
        indices = self._resolve_feature_indices(features)

        # compute LRT
        for idx in indices:
            if np.isnan(self.lrt_pvalues_[idx]):
                self._compute_single_lrt(idx, warm_start=warm_start)
        return self

    def _resolve_feature_indices(
        self,
        features: int | str | Sequence[int | str] | None,
    ) -> list[int]:
        """Convert feature names and/or indices to list of parameter indices."""
        return resolve_feature_indices(
            features,
            n_params=len(self.coef_),
            feature_names_in=getattr(self, "feature_names_in_", None),
        )

    def _compute_single_lrt(self, idx: int, *, warm_start: bool = True) -> None:
        """
        Fit constrained model with `beta[idx]=0` and compute LRT p-value and
        back-corrected standard error.

        Parameters
        ----------
        idx : int
            Index of the coefficient to test.
        warm_start : bool, default=True
            If True, warm-start constrained fits using the covariance from the full
            model (when available).
        """
        beta_hat_full = self.coef_

        k = beta_hat_full.shape[0]
        free_idx = np.array([i for i in range(k) if i != idx], dtype=np.intp)
        beta_free = beta_hat_full[free_idx]
        beta_j = beta_hat_full[idx]
        beta_init_free = None

        # Warm start for constrained LRT (beta_j=0) using cov from the full fit
        # (Schur complement).
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
            constrained_loglik, n_iter, status = constrained_lrt_1df_cox(
                X=self._precomputed.X,
                block_ends=self._precomputed.block_ends,
                block_d=self._precomputed.block_d,
                block_s=self._precomputed.block_s,
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
            elif status == _STATUS_RANK_DEFICIENT:
                raise scipy.linalg.LinAlgError("Fisher information is rank deficient.")
            elif status == _STATUS_LINALG_FAIL:
                raise scipy.linalg.LinAlgError(
                    "dpstrf failed - Fisher information is not PSD."
                )

            chi2 = max(0.0, 2.0 * (self.loglik_ - constrained_loglik))
            pval = scipy.stats.chi2.sf(chi2, df=1)

            # back-corrected SE: |beta|/sqrt(chi2), ensures (beta/SE)^2 = chi2
            bse = abs(self.coef_[idx]) / math.sqrt(chi2) if chi2 > 0 else math.inf
            result = LRTResult(chi2=chi2, pvalue=pval, bse_backcorrected=bse)

        else:

            def compute_quantities_full(beta):
                return compute_cox_quantities(
                    beta=beta,
                    precomputed=self._precomputed,
                    workspace=self._workspace,
                    penalty_weight=self.penalty_weight,
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

        self.lrt_pvalues_[idx] = result.pvalue
        self.lrt_bse_[idx] = result.bse_backcorrected

    def _compute_baseline_hazard(self, precomputed: _CoxPrecomputed) -> None:
        """Compute Breslow baseline cumulative hazard from fitted model."""
        eta = precomputed.X @ self.coef_
        c = np.max(eta)
        risk_scaled = np.exp(eta - c)

        event_times = []
        log_hazard_increments = []

        S0_scaled = 0.0
        start = 0
        for b in range(precomputed.n_blocks):
            end = precomputed.block_ends[b]
            S0_scaled += risk_scaled[start:end].sum()

            d = precomputed.block_d[b]
            if d > 0:
                t = precomputed.time[end - 1]
                # log(d / S0_true) = log(d) - log(S0_scaled) - c
                log_h = np.log(d) - np.log(S0_scaled) - c
                event_times.append(t)
                log_hazard_increments.append(log_h)

            start = end

        # Reverse to ascending time order
        self.unique_times_ = np.array(event_times[::-1])
        hazard_increments = np.exp(log_hazard_increments[::-1])
        self.cum_baseline_hazard_ = np.cumsum(hazard_increments)
        self.baseline_survival_ = np.exp(-self.cum_baseline_hazard_)

    def conf_int(
        self,
        alpha: float = 0.05,
        method: Literal["wald", "pl"] = "wald",
        features: int | str | Sequence[int | str] | None = None,
        max_iter: int = 50,
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
        max_iter : int, default=50
            Maximum number of iterations per bound (only used for `method='pl'`)
        tol : float, default=1e-4
            Convergence tolerance (only used for `method='pl'`)

        Returns
        -------
        ndarray, shape(n_features, 2)
            Column 0: lower bounds, Column 1: upper bounds

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

            def compute_quantities_full(beta: NDArray[np.float64]):
                return compute_cox_quantities(
                    beta=beta,
                    precomputed=self._precomputed,
                    workspace=self._workspace,
                    penalty_weight=self.penalty_weight,
                )

            theta_hat = self.coef_.copy()

            D0 = -compute_quantities_full(theta_hat).fisher_info  # hessian at MLE
            for idx in indices:
                for bound_idx, which in enumerate([-1, 1]):  # lower, upper
                    if not computed[idx, bound_idx]:
                        which = cast(Literal[-1, 1], which)  # mypy -_-
                        if self._resolve_backend() == "numba":
                            bound, converged, iterations = profile_ci_bound_cox(
                                X=self._precomputed.X,
                                block_ends=self._precomputed.block_ends,
                                block_d=self._precomputed.block_d,
                                block_s=self._precomputed.block_s,
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

                            result = ProfileCIBoundResult(
                                bound=bound,
                                converged=converged == _STATUS_CONVERGED,
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

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)
        return X @ self.coef_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the concordance index for the given test data."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)
        event, time = _validate_survival_y(y, n_samples=X.shape[0])
        risk = self.predict(X)
        if self._resolve_backend() == "numba":
            return concordance_index(event, time, risk)
        return _concordance_index(event, time, risk)  # numpy

    def predict_cumulative_hazard_function(
        self, X: ArrayLike, return_array: bool = True
    ) -> NDArray[np.float64]:
        """
        Predict cumulative hazard function for each sample.
        Returns array of shape (n_samples, n_event_times) evaluated at `unique_times_`.
        """
        if not return_array:
            raise NotImplementedError(
                "sksurv StepFunction output not supported; use return_array=True"
            )
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)
        risk = np.exp(X @ self.coef_)
        return self.cum_baseline_hazard_ * risk[:, None]

    def predict_survival_function(
        self, X: ArrayLike, return_array: bool = True
    ) -> NDArray[np.float64]:
        """
        Predict survival function for each sample.
        Returns array of shape (n_samples, n_event_times) evaluated at `unique_times_`.
        """
        if not return_array:
            raise NotImplementedError(
                "sksurv StepFunction output not supported; use return_array=True"
            )
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        X = cast(NDArray[np.float64], X)
        risk = np.exp(X @ self.coef_)
        return self.baseline_survival_ ** risk[:, None]


@dataclass
class CoxQuantities:
    """Quantities needed for one Newton-Raphson iteration."""

    loglik: float
    modified_score: NDArray[np.float64]  # shape (n_features,)
    fisher_info: NDArray[np.float64]  # shape (n_features, n_features)


@dataclass(frozen=True)
class _CoxPrecomputed:
    """
    Precomputed data for Cox likelihood evaluation.

    Samples are sorted by descending time and partitioned into blocks of equal time.
    Risk-set sums can then be computed in a single backward pass.

    Attributes
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix sorted by time descending.
    time : ndarray, shape (n_samples,)
        Observed times, sorted descending.
    event : ndarray, shape (n_samples,)
        Event indicators, sorted descending.
    block_ends : ndarray, shape (n_blocks,)
        End index (exclusive) of each time block in the sorted arrays.
    block_d : ndarray, shape (n_blocks,)
        Number of events in each block.
    block_s : ndarray, shape (n_blocks, n_features)
        Sum of features over events in each block.
    """

    X: NDArray[np.float64]
    time: NDArray[np.float64]
    event: NDArray[np.bool_]
    block_ends: NDArray[np.intp]
    block_d: NDArray[np.intp]
    block_s: NDArray[np.float64]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_blocks(self) -> int:
        return len(self.block_ends)

    @classmethod
    def from_data(
        cls,
        X: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.bool_],
        backend: Literal["numba", "numpy"] = "numpy",
    ) -> Self:
        """
        Sort samples and compute block structure.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.
        time : ndarray, shape (n_samples,)
            Observed times.
        event : ndarray, shape (n_samples,)
            Event indicators.

        Returns
        -------
        _CoxPrecomputed
            Precomputed data for likelihood evaluation.
        """
        if backend == "numba":
            if not NUMBA_AVAILABLE:
                raise ImportError("backend='numba' but numba is not installed.")
            X, time, event, block_ends, block_d, block_s = precompute_cox(
                X, time, event
            )
            return cls(
                X=X,
                time=time,
                event=event,
                block_ends=block_ends,
                block_d=block_d,
                block_s=block_s,
            )

        # d = number of deaths at time t
        # s = vector sum of the covariates of the d individuals
        n, k = X.shape

        # Sort by time descending (stable sort for reproducibility with ties)
        order = np.argsort(-time, kind="stable")
        X = X[order]
        time = time[order]
        event = event[order]

        # Identify block boundaries (blocks are contiguous runs of equal time)
        # Note that np.diff(time) assumes exact equality for ties. If there are floating
        # point differences like 1.0 and 1.0 + 1e-15, they will be treated as different
        # times. So floating point times should be rounded appropriately before fitting.
        time_changes = np.flatnonzero(np.diff(time)) + 1
        block_ends = np.concatenate([time_changes, [n]])

        # Compute per-block event counts and covariate sums
        n_blocks = len(block_ends)
        block_d = np.zeros(n_blocks, dtype=np.int64)
        block_s = np.zeros((n_blocks, k), dtype=np.float64)

        start = 0
        for b in range(n_blocks):
            end = block_ends[b]
            event_mask = event[start:end]
            block_d[b] = event_mask.sum()
            if block_d[b] > 0:
                block_s[b] = X[start:end][event_mask].sum(axis=0)
            start = end

        return cls(
            X=X,
            time=time,
            event=event,
            block_ends=block_ends,
            block_d=block_d,
            block_s=block_s,
        )


def _validate_survival_y(
    y: ArrayLike, *, n_samples: int
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """
    Validate and extract survival outcomes.

    Parameters
    ----------
    y : array-like or tuple
        Survival outcome encoding. Accepts either:
        - a NumPy structured array with one boolean field (event indicator) and
          one float field (time), or
        - a two-tuple `(event, time)` where `event` is a bool array indicating whether
          the event occurred, and `time` is the observed time (event or censoring).
    n_samples : int
        Expected number of samples.

    Returns
    -------
    event : ndarray, shape (n_samples,), dtype=bool
        Boolean array where `True` indicates an event.
    time : ndarray, shape (n_samples,), dtype=float64
        Observed times.
    """
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = y.dtype.names
        if len(names) != 2:
            raise ValueError(f"Structured y must have 2 fields, got {len(names)}")

        # Find which field is bool (event) and which is float (time)
        event_field = time_field = None
        for name in names:
            dtype = y.dtype.fields[name][0]
            if np.issubdtype(dtype, np.bool_):
                event_field = name
            elif np.issubdtype(dtype, np.floating):
                time_field = name

        if event_field is None or time_field is None:
            raise ValueError(
                "Structured y must have one bool field (event) and one float field (time)"
            )
        event = np.asarray(y[event_field])
        time = np.asarray(y[time_field])
    elif isinstance(y, (tuple, list)) and len(y) == 2:
        event, time = np.asarray(y[0]), np.asarray(y[1])
    else:
        raise ValueError(
            "y must be a structured array with fields ('event', 'time') or an "
            "(event, time) tuple."
        )

    if event.ndim != 1 or time.ndim != 1:
        raise ValueError("event and time must be 1-dimensional.")
    if event.shape[0] != time.shape[0]:
        raise ValueError(
            f"event and time must have same length, "
            f"got event: {event.shape[0]} and time: {time.shape[0]}."
        )
    if event.shape[0] != n_samples:
        raise ValueError(f"y has {event.shape[0]} samples, expected {n_samples}.")

    if not np.all((event == 0) | (event == 1)):
        raise ValueError("event must contain only 0/1 or boolean values.")
    event = event.astype(bool)

    time = time.astype(np.float64)
    if not np.all(np.isfinite(time)):
        raise ValueError("time must be finite.")
    if np.any(time < 0):
        raise ValueError("time must be non-negative.")
    if not event.any():
        raise ValueError("At least one event is required to fit a Cox model.")

    return event, time


class _Workspace:
    """Pre-allocated arrays for compute_cox_quantities"""

    __slots__ = (
        "wX",
        "S0_cumsum",
        "S1_cumsum",
        "S2_cumsum",
        "wXh",
        "A_cumsum",
        "B_cumsum",
        "eye_k",
        "eta",
        "risk",
        "XI",
        "h",
        "fisher_info",
    )

    def __init__(self, n_samples: int, n_features: int) -> None:
        n, k = n_samples, n_features
        self.wX = np.empty((n, k), dtype=np.float64)
        self.S0_cumsum = np.empty(n, dtype=np.float64)
        self.S1_cumsum = np.empty((n, k), dtype=np.float64)
        self.S2_cumsum = np.empty((n, k, k), dtype=np.float64)
        self.wXh = np.empty((n, k), dtype=np.float64)
        self.A_cumsum = np.empty((n, k), dtype=np.float64)
        self.B_cumsum = np.empty(n, dtype=np.float64)
        self.eye_k = np.eye(k, dtype=np.float64, order="F")
        self.eta = np.empty(n, dtype=np.float64)
        self.risk = np.empty(n, dtype=np.float64)
        self.XI = np.empty((n, k), dtype=np.float64)
        self.h = np.empty(n, dtype=np.float64)
        self.fisher_info = np.empty((k, k), dtype=np.float64, order="F")

    def numba_buffers(self):  # for numba
        return (
            self.wX,
            self.S0_cumsum,
            self.S1_cumsum,
            self.S2_cumsum,
            self.wXh,
            self.A_cumsum,
            self.B_cumsum,
            self.eye_k,
            self.eta,
            self.risk,
            self.h,
            self.fisher_info,
        )


def compute_cox_quantities(
    beta: NDArray[np.float64],
    precomputed: _CoxPrecomputed,
    workspace: _Workspace,
    penalty_weight: float = 0.5,
) -> CoxQuantities:
    """
    Compute all quantities needed for one Newton-Raphson iteration.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Current coefficient estimates.
    precomputed : _CoxPrecomputed
        Precomputed data from _CoxPrecomputed.from_data().
    workspace : _Workspace
        Pre-allocated arrays to avoid repeated memory allocation.
    penalty_weight : float, default=0.5
        Weight of the Firth penalty term. The default 0.5 corresponds to the standard
        Firth bias reduction method (Heinze and Schemper, 2001), equivalent to using
        Jeffreys' invariant prior. Set to 0 for unpenalized Cox partial likelihood
        estimation.

    Returns
    -------
    CoxQuantities
        Penalized log-likelihood, modified score, and Fisher information.
    """
    X = precomputed.X
    block_ends = precomputed.block_ends
    block_d = precomputed.block_d
    block_s = precomputed.block_s
    k = precomputed.n_features
    ws = workspace

    # Subtract max(eta) to avoid exp overflow. This rescales all exp(eta) by a
    # constant, so ratios like S1/S0 are unchanged; we add c back in loglik.
    np.matmul(X, beta, out=ws.eta)
    c = ws.eta.max()
    np.subtract(ws.eta, c, out=ws.risk)
    np.exp(ws.risk, out=ws.risk)

    # S0, S1, S2 are cumulative sums over the risk set (everyone with time >= t).
    np.multiply(X, ws.risk[:, None], out=ws.wX)
    np.cumsum(ws.risk, out=ws.S0_cumsum)
    np.cumsum(ws.wX, axis=0, out=ws.S1_cumsum)
    np.multiply(ws.wX[:, :, None], X[:, None, :], out=ws.S2_cumsum)
    np.cumsum(ws.S2_cumsum, axis=0, out=ws.S2_cumsum)

    # Index at block boundaries to get risk-set sums at each unique time
    block_end_indices = block_ends - 1
    S0_at_blocks = ws.S0_cumsum[block_end_indices]
    S1_at_blocks = ws.S1_cumsum[block_end_indices]
    S2_at_blocks = ws.S2_cumsum[block_end_indices]

    # Filter to event blocks only
    event_mask = block_d > 0
    d_events = block_d[event_mask]  # (n_event_blocks,)
    s_events = block_s[event_mask]  # (n_event_blocks, k)
    S0_events = S0_at_blocks[event_mask]  # (n_event_blocks,)
    S1_events = S1_at_blocks[event_mask]  # (n_event_blocks, k)
    S2_events = S2_at_blocks[event_mask]  # (n_event_blocks, k, k)

    S0_inv = 1.0 / S0_events  # (n_event_blocks,)
    S0_inv2 = S0_inv * S0_inv
    # Risk-set weighted mean covariate vector.
    x_bar = S1_events * S0_inv[:, None]

    # Add c to undo the scaling exp(eta - c) in the risk-set sum.
    loglik = float((s_events @ beta - d_events * (c + np.log(S0_events))).sum())

    score = (s_events - d_events[:, None] * x_bar).sum(axis=0)

    # Fisher info using the paper's weighted-covariance form.
    # Section 2: each event time contributes d_j times the weighted covariance of X
    # in the risk set, i.e. V = S2/S0 - x_bar x_bar^T.
    V = S2_events * S0_inv[:, None, None] - x_bar[:, :, None] * x_bar[:, None, :]
    np.einsum("b,brt->rt", d_events, V, out=ws.fisher_info)

    if penalty_weight == 0.0:
        return CoxQuantities(
            loglik=loglik,
            modified_score=score,
            fisher_info=ws.fisher_info,
        )

    try:
        L, info = dpotrf(ws.fisher_info, lower=1, overwrite_a=0)
        if info != 0:
            raise scipy.linalg.LinAlgError("dpotrf failed")

        logdet = 2.0 * np.log(L.diagonal()).sum()

        inv_fisher_info, info = dpotrs(L, ws.eye_k, lower=1)
        if info != 0:
            raise scipy.linalg.LinAlgError("dpotrs failed")
    except scipy.linalg.LinAlgError:
        # fall back to pivoted Cholesky
        diag_max = np.abs(ws.fisher_info.diagonal()).max()
        tol = max(1, k) * np.finfo(np.float64).eps * diag_max

        C, piv, rank, info = dpstrf(ws.fisher_info, tol=tol, lower=1)
        if info == 0 and rank == k:
            logdet = 2.0 * np.log(C.diagonal()).sum()

            inv_perm, info = dpotrs(C, ws.eye_k, lower=1)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrs failed on pivoted Cholesky")

            # undo permutation: inv_fisher_info = P @ inv_perm @ P.T
            p = piv - 1  # piv is 1-indexed
            inv_fisher_info = np.empty_like(ws.fisher_info)
            inv_fisher_info[np.ix_(p, p)] = inv_perm
        elif rank < k:
            raise scipy.linalg.LinAlgError("Fisher information is rank deficient")
        else:
            raise scipy.linalg.LinAlgError(
                "dpstrf failed - Fisher information is not PSD"
            )

    # avoid O(n k^3) S3 tensor by swapping the summation order:
    # sum_{r,s} I_inv[r,s] * S3[t,r,s] = sum_i w[i]*X[i,t]*h[i]
    # where h[i] = X[i] @ I_inv @ X[i] is the hat matrix diagonal.
    np.matmul(X, inv_fisher_info, out=ws.XI)
    np.einsum("ij,ij->i", ws.XI, X, out=ws.h)

    # Cumulative sums for the contracted S3 term and trace term
    # A[i,t] = sum_{j<=i} w[j] * X[j,t] * h[j]
    # B[i] = sum_{j<=i} w[j] * h[j] = trace(I_inv @ S2) at sample i
    np.multiply(ws.wX, ws.h[:, None], out=ws.wXh)
    np.cumsum(ws.wXh, axis=0, out=ws.A_cumsum)
    np.cumsum(ws.risk * ws.h, out=ws.B_cumsum)

    # Index at event block boundaries
    event_block_indices = block_end_indices[event_mask]
    A_events = ws.A_cumsum[event_block_indices]  # (n_event_blocks, k)
    B_events = ws.B_cumsum[event_block_indices]  # (n_event_blocks,)

    # term1 contracted with I_inv: A/S0 - B*S1/S0^2
    term1_contrib = (
        A_events * S0_inv[:, None] - B_events[:, None] * S1_events * S0_inv2[:, None]
    )

    # term2 and term3 contracted with I_inv give the same result (symmetry):
    # sum_{r,s} I_inv[r,s] * x_bar[s] * V[r,t] = sum_r V[r,t] * (I_inv @ x_bar)[r]
    Ix = x_bar @ inv_fisher_info  # (n_event_blocks, k)
    term23_contrib = np.einsum("brt,br->bt", V, Ix)

    # Firth correction: penalty_weight * sum over event blocks of d * (term1 - 2*term23)
    firth_per_block = term1_contrib - 2 * term23_contrib
    firth_correction = penalty_weight * np.einsum("b,bt->t", d_events, firth_per_block)
    modified_score = score + firth_correction
    loglik = loglik + penalty_weight * logdet

    return CoxQuantities(
        loglik=loglik,
        modified_score=modified_score,
        fisher_info=ws.fisher_info,
    )


def _concordance_index(
    event: NDArray[np.bool_],
    time: NDArray[np.float64],
    risk: NDArray[np.float64],
) -> float:
    """Compute concordance index (C-index) for survival predictions."""

    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    # TODO: this is awful
    for i in range(n):
        for j in range(i + 1, n):
            # Determine if pair is comparable and who has shorter survival
            if event[i] and event[j]:
                if time[i] == time[j]:
                    continue  # tied times, not comparable
                shorter, longer = (i, j) if time[i] < time[j] else (j, i)
            elif event[i] and not event[j]:
                # i had event, j censored: comparable if i's event <= j's censoring
                # (j was at risk when i's event occurred)
                if time[i] > time[j]:
                    continue
                shorter, longer = i, j
            elif event[j] and not event[i]:
                # j had event, i censored
                if time[j] > time[i]:
                    continue
                shorter, longer = j, i
            else:
                # Both censored: not comparable
                continue

            # Compare risk scores: higher risk should have shorter survival
            if risk[shorter] > risk[longer]:
                concordant += 1
            elif risk[shorter] < risk[longer]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5  # no comparable pairs
    return (concordant + 0.5 * tied_risk) / total
