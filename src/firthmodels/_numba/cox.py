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


@njit(fastmath=False, cache=True)
def max_abs(vec: NDArray[np.float64]) -> float:
    max_val = 0.0
    for i in range(vec.shape[0]):
        val = vec[i]
        if np.isnan(val):
            return np.inf
        abs_val = abs(val)
        if abs_val > max_val:
            max_val = abs_val
    return max_val


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


@njit(fastmath=True, inline="always", cache=True)
def _compute_risk_sum_sets(
    X: NDArray[np.float64],
    beta: NDArray[np.float64],
    eta: NDArray[np.float64],
    risk: NDArray[np.float64],
    wX: NDArray[np.float64],
    S0_cumsum: NDArray[np.float64],
    S1_cumsum: NDArray[np.float64],
    S2_cumsum: NDArray[np.float64],
):
    """Compute eta, risk, and cumulative risk-set sums"""
    n, k = X.shape

    # eta = X @ beta and stable exp scaling via c
    c = -np.inf
    for i in range(n):
        total = 0.0
        for j in range(k):
            total += X[i, j] * beta[j]
        eta[i] = total
        if total > c:
            c = total

    # risk and cumulative S0/S1/S2 over risk sets
    for i in range(n):
        risk_i = np.exp(eta[i] - c)
        risk[i] = risk_i
        if i == 0:
            S0_cumsum[i] = risk_i
        else:
            S0_cumsum[i] = S0_cumsum[i - 1] + risk_i

        for j in range(k):
            wX_ij = X[i, j] * risk_i
            wX[i, j] = wX_ij
            if i == 0:
                S1_cumsum[i, j] = wX_ij
            else:
                S1_cumsum[i, j] = S1_cumsum[i - 1, j] + wX_ij

        for r in range(k):
            x_r = X[i, r]
            for s in range(k):
                val = risk_i * x_r * X[i, s]
                if i == 0:
                    S2_cumsum[i, r, s] = val
                else:
                    S2_cumsum[i, r, s] = S2_cumsum[i - 1, r, s] + val

    return c


@njit(fastmath=True, inline="always", cache=True)
def _compute_score_fisher_loglik(
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    block_s: NDArray[np.float64],
    beta: NDArray[np.float64],
    S0_cumsum: NDArray[np.float64],
    S1_cumsum: NDArray[np.float64],
    S2_cumsum: NDArray[np.float64],
    x_bar: NDArray[np.float64],
    modified_score: NDArray[np.float64],
    fisher_info: NDArray[np.float64],
    c: float,
):
    """Accumulate score and Fisher info across event blocks."""
    k = beta.shape[0]

    # zero score and Fisher info accumulators
    for i in range(k):
        modified_score[i] = 0.0
        for j in range(k):
            fisher_info[i, j] = 0.0

    # Index at block boundaries to get risk-set sums at each unique time.
    # Filter to event blocks only.
    # loop over time blocks, accumulate score and Fisher info
    loglik = 0.0
    n_blocks = block_ends.shape[0]
    for block in range(n_blocks):
        d = block_d[block]
        if d <= 0:
            continue
        end_idx = block_ends[block] - 1
        S0 = S0_cumsum[end_idx]
        S0_inv = 1.0 / S0

        total = 0.0
        for r in range(k):
            # risk-set weighted mean covariate vector
            x_bar[r] = S1_cumsum[end_idx, r] * S0_inv
            # score = (s_events - d_events[:, None] * x_bar).sum(axis=0)
            modified_score[r] += block_s[block, r] - d * x_bar[r]
            total += block_s[block, r] * beta[r]
        # Add c to undo the scaling exp(eta - c) in the risk-set sum.
        loglik += total - d * (c + np.log(S0))

        # V = S2_events * S0_inv[:, None, None] - x_bar[:, :, None] * x_bar[:, None, :]
        # fisher_info = np.einsum("b,brt->rt", d_events, V)
        for r in range(k):
            x_r = x_bar[r]
            for s in range(k):
                val = S2_cumsum[end_idx, r, s] * S0_inv - x_r * x_bar[s]
                fisher_info[r, s] += d * val

    return loglik


@njit(fastmath=True, inline="always", cache=True)
def _compute_firth_correction(
    X: NDArray[np.float64],
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    S0_cumsum: NDArray[np.float64],
    S1_cumsum: NDArray[np.float64],
    S2_cumsum: NDArray[np.float64],
    risk: NDArray[np.float64],
    wX: NDArray[np.float64],
    fisher_inv: NDArray[np.float64],
    modified_score: NDArray[np.float64],
    XI: NDArray[np.float64],
    h: NDArray[np.float64],
    wXh: NDArray[np.float64],
    A_cumsum: NDArray[np.float64],
    B_cumsum: NDArray[np.float64],
    x_bar: NDArray[np.float64],
    Ix: NDArray[np.float64],
    term1: NDArray[np.float64],
    term23: NDArray[np.float64],
):
    """Add Firth penalty term to score (modifies score in-place)."""
    n, k = X.shape

    # XI = X @ inv_fisher_info
    # TODO: benchmark BLAS-based XI/Ix (with transa using X.T or F-order X) vs loops.
    # since X is C-order, we would need to add a transa option to the dgemm wrapper.
    # just use loops for now
    for i in range(n):
        for j in range(k):
            total = 0.0
            for r in range(k):
                total += X[i, r] * fisher_inv[r, j]
            XI[i, j] = total

    # h = np.einsum("ij,ij->i", XI, X)
    for i in range(n):
        total = 0.0
        for j in range(k):
            total += XI[i, j] * X[i, j]
        h[i] = total

    # np.multiply(ws.wX, h[:, None], out=ws.wXh)
    # np.cumsum(ws.wXh, axis=0, out=ws.A_cumsum)
    # np.cumsum(risk * h, out=ws.B_cumsum)
    for i in range(n):
        risk_h = risk[i] * h[i]
        if i == 0:
            B_cumsum[i] = risk_h
        else:
            B_cumsum[i] = B_cumsum[i - 1] + risk_h
        for j in range(k):
            wXh_ij = wX[i, j] * h[i]
            wXh[i, j] = wXh_ij
            if i == 0:
                A_cumsum[i, j] = wXh_ij
            else:
                A_cumsum[i, j] = A_cumsum[i - 1, j] + wXh_ij

    # Index at event block boundaries
    n_blocks = block_ends.shape[0]
    for block in range(n_blocks):
        d = block_d[block]
        if d <= 0:
            continue
        end_idx = block_ends[block] - 1
        S0 = S0_cumsum[end_idx]
        S0_inv = 1.0 / S0
        S0_inv2 = S0_inv * S0_inv

        for r in range(k):
            x_bar[r] = S1_cumsum[end_idx, r] * S0_inv

        B_events = B_cumsum[end_idx]
        # term1 contracted with I_inv: A/S0 - B*S1/S0^2
        # term1_contrib = (
        #     A_events * S0_inv[:, None] - B_events[:, None] * S1_events * S0_inv2[:, None]
        # )
        for t in range(k):
            A_events = A_cumsum[end_idx, t]
            S1_events = S1_cumsum[end_idx, t]
            term1[t] = A_events * S0_inv - B_events * S1_events * S0_inv2

        # Ix = x_bar @ inv_fisher_info  # (n_event_blocks, k)
        # term23_contrib = np.einsum("brt,br->bt", V, Ix)
        for t in range(k):
            total = 0.0
            for r in range(k):
                total += x_bar[r] * fisher_inv[r, t]
            Ix[t] = total

        for t in range(k):
            total = 0.0
            for r in range(k):
                V_rt = S2_cumsum[end_idx, r, t] * S0_inv - x_bar[r] * x_bar[t]
                total += V_rt * Ix[r]
            term23[t] = total

        # firth_per_block = term1_contrib - 2 * term23_contrib
        # firth_correction = 0.5 * np.einsum("b,bt->t", d_events, firth_per_block)
        # modified_score = score + firth_correction
        for t in range(k):
            modified_score[t] += 0.5 * d * (term1[t] - 2.0 * term23[t])


@njit(cache=True, fastmath=True)
def compute_cox_quantities(
    X: NDArray[np.float64],
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    block_s: NDArray[np.float64],
    beta: NDArray[np.float64],
    eta: NDArray[np.float64],
    risk: NDArray[np.float64],
    wX: NDArray[np.float64],
    S0_cumsum: NDArray[np.float64],
    S1_cumsum: NDArray[np.float64],
    S2_cumsum: NDArray[np.float64],
    fisher_info: NDArray[np.float64],
    fisher_work: NDArray[np.float64],
    XI: NDArray[np.float64],
    h: NDArray[np.float64],
    wXh: NDArray[np.float64],
    A_cumsum: NDArray[np.float64],
    B_cumsum: NDArray[np.float64],
    modified_score: NDArray[np.float64],
    x_bar: NDArray[np.float64],
    Ix: NDArray[np.float64],
    term1: NDArray[np.float64],
    term23: NDArray[np.float64],
):
    n, k = X.shape

    c = _compute_risk_sum_sets(
        X=X,
        beta=beta,
        eta=eta,
        risk=risk,
        wX=wX,
        S0_cumsum=S0_cumsum,
        S1_cumsum=S1_cumsum,
        S2_cumsum=S2_cumsum,
    )

    loglik = _compute_score_fisher_loglik(
        block_ends=block_ends,
        block_d=block_d,
        block_s=block_s,
        beta=beta,
        S0_cumsum=S0_cumsum,
        S1_cumsum=S1_cumsum,
        S2_cumsum=S2_cumsum,
        x_bar=x_bar,
        modified_score=modified_score,
        fisher_info=fisher_info,
        c=c,
    )

    fisher_work[:, :] = fisher_info
    info = dpotrf(fisher_work)  # fisher_work now holds L
    if info == 0:
        logdet = 0.0
        for i in range(k):
            logdet += np.log(fisher_work[i, i])
        logdet *= 2.0
        info = dpotri(fisher_work)  # fisher_work now holds inv(fisher_info)
        if info == 0:
            symmetrize_lower(fisher_work)
    if info != 0:  # dpotrf or dpotri failed
        sign, logdet = np.linalg.slogdet(fisher_info)
        if sign <= 0:
            logdet = -np.inf
        for i in range(k):
            for j in range(k):
                fisher_work[i, j] = 0.0
            fisher_work[i, i] = 1.0
        fisher_work[:, :] = np.linalg.lstsq(fisher_info, fisher_work)[0]
        # fisher_work now holds inv(fisher_info)

    _compute_firth_correction(
        X=X,
        block_ends=block_ends,
        block_d=block_d,
        S0_cumsum=S0_cumsum,
        S1_cumsum=S1_cumsum,
        S2_cumsum=S2_cumsum,
        risk=risk,
        wX=wX,
        fisher_inv=fisher_work,
        modified_score=modified_score,
        XI=XI,
        h=h,
        wXh=wXh,
        A_cumsum=A_cumsum,
        B_cumsum=B_cumsum,
        x_bar=x_bar,
        Ix=Ix,
        term1=term1,
        term23=term23,
    )

    loglik += 0.5 * logdet

    return loglik
