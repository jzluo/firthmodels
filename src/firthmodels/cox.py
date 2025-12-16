import numpy as np

from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data
from typing import Self, cast


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
    tol : float, default=1e-4
        Tolerance for stopping criteria

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
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (if X has feature names).

    References
    ----------
    Heinze, G, Schemper, M. (2001). A Solution to the Problem of Monotone
    Likelihood in Cox Regression. Biometrics 57(1):114-119.
    """

    def __init__(
        self,
        max_iter: int = 25,
        max_step: float = 5.0,
        max_halfstep: int = 25,
        tol: float = 1e-4,
    ) -> None:
        self.max_iter = max_iter
        self.max_step = max_step
        self.max_halfstep = max_halfstep
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Fit the Firth-penalized Cox model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like or tuple
            Survival outcome encoding. Accepts either:
            - a NumPy structured array with fields `("event", "time")` (scikit-survival
            convention), or
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

        # TODO: precomputation (sort, identify event blocks)
        # TODO: call newton_raphson with compute_cox_quantities closure
        # TODO: extract results, compute Wald SEs

        raise NotImplementedError("fit() not yet implemented")

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return X @ self.coef_


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
        CoxData
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
        - a NumPy structured array with fields `("event", "time")` (scikit-survival
          convention), or
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
        names = set(y.dtype.names)
        if not {"event", "time"}.issubset(names):
            raise ValueError(
                f"Structured y must have fields {{'event', 'time'}}, "
                f"got {sorted(names)}"
            )
        event = np.asarray(y["event"])
        time = np.asarray(y["time"])
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
