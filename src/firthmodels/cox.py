from __future__ import annotations

import numpy as np
import scipy

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from typing import Self, cast

from firthmodels._solvers import newton_raphson


class FirthCoxPH(BaseEstimator):
    """
    Cox proportional hazards with Firth's bias reduction.

    Parameters
    ----------
    max_iter : int, default=25
        Maximum number of Newton-Raphson iterations.
    max_step : float, default=5.0
        Maximum step size per coefficient.
    max_halfstep : int, default=25
        Maximum number of step-halvings per iteration.
    gtol : float, default=1e-4
        Gradient convergence criteria. Converged when max|gradient| < gtol.
    xtol : float, default=1e-6
        Parameter convergence criteria. Converged when max|delta| < xtol.

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
        max_iter: int = 25,
        max_step: float = 5.0,
        max_halfstep: int = 25,
        gtol: float = 1e-4,
        xtol: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.gtol = gtol
        self.xtol = xtol

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
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        X = cast(NDArray[np.float64], X)
        event, time = _validate_survival_y(y, n_samples=X.shape[0])

        # Precompute sorted data and block structure
        data = _CoxPrecomputed.from_data(X, time, event)

        # Create closure for newton_raphson
        def quantities(beta: NDArray[np.float64]) -> CoxQuantities:
            return compute_cox_quantities(beta, data)

        # Run optimizer
        result = newton_raphson(
            compute_quantities=quantities,
            n_features=data.n_features,
            max_iter=self.max_iter,
            max_step=self.max_step,
            max_halfstep=self.max_halfstep,
            gtol=self.gtol,
            xtol=self.xtol,
        )

        self.coef_ = result.beta
        self.loglik_ = result.loglik
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged

        # Wald
        try:
            cov = np.linalg.inv(result.fisher_info)
            self.bse_ = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            self.bse_ = np.full(data.n_features, np.nan)

        z = result.beta / self.bse_
        self.pvalues_ = 2 * scipy.stats.norm.sf(np.abs(z))

        self._compute_baseline_hazard(data)

        return self

    def _compute_baseline_hazard(self, data: _CoxPrecomputed) -> None:
        """Compute Breslow baseline cumulative hazard from fitted model."""
        eta = data.X @ self.coef_
        c = np.max(eta)
        risk_scaled = np.exp(eta - c)

        event_times = []
        log_hazard_increments = []

        S0_scaled = 0.0
        start = 0
        for b in range(data.n_blocks):
            end = data.block_ends[b]
            S0_scaled += risk_scaled[start:end].sum()

            d = data.block_d[b]
            if d > 0:
                t = data.time[end - 1]
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
        return _concordance_index(event, time, risk)

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
        block_d = np.zeros(n_blocks, dtype=np.intp)
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


def compute_cox_quantities(
    beta: NDArray[np.float64],
    data: _CoxPrecomputed,
) -> CoxQuantities:
    """
    Compute all quantities needed for one Newton-Raphson iteration.

    Parameters
    ----------
    beta : ndarray, shape (n_features,)
        Current coefficient estimates.
    data : _CoxPrecomputed
        Precomputed data from _CoxPrecomputed.from_data().

    Returns
    -------
    CoxQuantities
        Penalized log-likelihood, modified score, and Fisher information.
    """
    X = data.X
    block_ends = data.block_ends
    block_d = data.block_d
    block_s = data.block_s
    n_blocks = data.n_blocks
    k = data.n_features

    # Subtract max(eta) to avoid exp overflow. This rescales all exp(eta) by a
    # constant, so ratios like S1/S0 are unchanged; we add c back in loglik.
    eta = X @ beta
    c = np.max(eta)
    risk = np.exp(eta - c)

    # Initialize (scaled) risk-set sums computed with risk = exp(eta - c).
    S0 = 0.0
    S1 = np.zeros(k, dtype=np.float64)
    S2 = np.zeros((k, k), dtype=np.float64)
    S3 = np.zeros((k, k, k), dtype=np.float64)

    loglik = 0.0
    score = np.zeros(k, dtype=np.float64)
    fisher_info = np.zeros((k, k), dtype=np.float64)
    dI_dbeta = np.zeros((k, k, k), dtype=np.float64)
    # dI_dbeta[t, r, s] = dI_rs / d beta_t.

    # Walk blocks in descending time order
    start = 0
    for b in range(n_blocks):
        end = block_ends[b]

        block_X = X[start:end]
        block_risk = risk[start:end]

        # update risk-set sums (everyone in the block enters the risk set)
        S0 += block_risk.sum()
        S1 += block_X.T @ block_risk
        S2 += (block_X * block_risk[:, None]).T @ block_X
        S3 += np.einsum(
            "i,ir,is,it->rst",
            block_risk,
            block_X,
            block_X,
            block_X,
            optimize=True,
        )

        d = block_d[b]
        if d == 0:
            start = end
            continue

        s = block_s[b]
        S0_inv = 1.0 / S0
        # Risk-set weighted mean covariate vector.
        x_bar = S1 * S0_inv

        # Add c to undo the scaling exp(eta - c) in the risk-set sum.
        loglik += beta @ s - d * (c + np.log(S0))

        score += s - d * x_bar

        # Fisher info using the paper's weighted-covariance form.
        # Section 2: each event time contributes d_j times the weighted covariance of X
        # in the risk set, i.e. V = S2/S0 - x_bar x_bar^T.
        V = S2 * S0_inv - np.outer(x_bar, x_bar)
        fisher_info += d * V

        # dI/d beta for Firth correction (Section 2 of paper).
        # We implement the per-block contribution using S0/S1/S2/S3 and V, where
        # V_{rt} = S_{rt}/S0 - S_r S_t/S0^2.
        S0_inv2 = S0_inv * S0_inv
        term1 = S3.transpose(2, 0, 1) * S0_inv - (  # S3 is (r,s,t)
            np.einsum("rs,t->trs", S2, S1, optimize=True) * S0_inv2
        )
        term2 = np.einsum("s,rt->trs", S1 * S0_inv, V, optimize=True)
        term3 = np.einsum("r,st->trs", S1 * S0_inv, V, optimize=True)
        dI_dbeta += d * (term1 - term2 - term3)

        start = end

    # log|I| and I^{-1} for Firth correction
    try:
        cho = scipy.linalg.cho_factor(fisher_info, lower=True, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
        inv_fisher_info = scipy.linalg.cho_solve(
            cho, np.eye(k, dtype=np.float64), check_finite=False
        )
    except scipy.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(fisher_info)
        if sign <= 0:
            logdet = -np.inf
        inv_fisher_info = np.linalg.pinv(fisher_info)

    # Firth correction: a_t = 0.5 * trace(I^{-1} * dI/d beta_t).
    firth_correction = 0.5 * np.einsum(
        "rs,trs->t", inv_fisher_info, dI_dbeta, optimize=True
    )
    modified_score = score + firth_correction
    loglik = loglik + 0.5 * logdet

    return CoxQuantities(
        loglik=loglik,
        modified_score=modified_score,
        fisher_info=fisher_info,
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
