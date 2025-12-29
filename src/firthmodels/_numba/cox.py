import numpy as np
from numba import njit
from numpy.typing import NDArray

from firthmodels._numba.blas_abi import BLAS_INT_DTYPE
from firthmodels._numba.linalg import (
    _alloc_f_order,
    dgemm,
    dgemv,
    dgetrf,
    dgetrs,
    dpotrf,
    dpotri,
    dpotrs,
    dsyrk,
    symmetrize_lower,
)

# Solver exit status codes
_STATUS_CONVERGED = 0
_STATUS_STEP_HALVING_FAILED = 1
_STATUS_MAX_ITER = 2


@njit(cache=True)
def precompute_cox(
    X: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.bool_],
):
    # d = number of deaths at time t
    # s = vector sum of the covariates of the d individuals
    n, k = X.shape

    order = np.argsort(-time, kind="mergesort")  # stable sort
    X_sorted = np.empty((n, k), dtype=np.float64)
    time_sorted = np.empty(n, dtype=np.float64)
    event_sorted = np.empty(n, dtype=np.bool_)

    for i in range(n):
        idx = order[i]
        time_sorted[i] = time[idx]
        event_sorted[i] = event[idx]
        for j in range(k):
            X_sorted[i, j] = X[idx, j]

    # Identify block boundaries (blocks are contiguous runs of equal time)
    # time_changes = np.flatnonzero(np.diff(time)) + 1
    # block_ends = np.concatenate([time_changes, [n]])
    n_blocks = 1
    for i in range(1, n):
        if time_sorted[i] != time_sorted[i - 1]:
            n_blocks += 1

    block_ends = np.empty(n_blocks, dtype=np.intp)
    pos = 0
    for i in range(1, n):
        if time_sorted[i] != time_sorted[i - 1]:
            block_ends[pos] = i
            pos += 1
    block_ends[pos] = n

    # Compute per-block event counts and covariate sums
    block_d = np.zeros(n_blocks, dtype=np.int64)
    block_s = np.zeros((n_blocks, k), dtype=np.float64)

    start = 0
    for b in range(n_blocks):
        end = block_ends[b]
        for i in range(start, end):
            if event_sorted[i]:
                block_d[b] += 1
                for j in range(k):
                    block_s[b, j] += X_sorted[i, j]
        start = end

    return X_sorted, time_sorted, event_sorted, block_ends, block_d, block_s
