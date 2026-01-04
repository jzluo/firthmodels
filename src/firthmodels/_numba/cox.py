"""
Numba-accelerated Firth Cox proportional hazards.

JIT-compiled versions of the numpy/scipy implementations in src/firthmodels/cox.py,
_solvers.py, _lrt.py, and _profile_ci.py.
"""

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
    dpotrs,
    dpstrf,
    dsyrk,
    set_identity,
    symmetrize_lower,
)

# Solver exit status codes
_STATUS_CONVERGED = 0
_STATUS_STEP_HALVING_FAILED = 1
_STATUS_MAX_ITER = 2
_STATUS_LINALG_FAIL = 3
_STATUS_RANK_DEFICIENT = 4


@njit(cache=True)
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
    fisher_inv: NDArray[np.float64],
    modified_score: NDArray[np.float64],
    x_bar: NDArray[np.float64],
    Ix: NDArray[np.float64],
    term1: NDArray[np.float64],
    term23: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],
):
    """Add Firth penalty term to score (modifies score in-place)."""
    (
        wX,
        S0_cumsum,
        S1_cumsum,
        S2_cumsum,
        wXh,
        A_cumsum,
        B_cumsum,
        eye_k,
        eta,
        risk,
        h,
        fisher_info,
    ) = workspace

    n, k = X.shape
    # using Ix as a scratch buffer
    tmp_k = Ix

    # XI = X @ inv_fisher_info
    # h = np.einsum("ij,ij->i", XI, X)
    # TODO: benchmark BLAS-based XI/Ix (with transa using X.T or F-order X) vs loops.
    # since X is C-order, we would need to add a transa option to the dgemm wrapper.
    # just use loops for now

    # compute h without storing XI
    for i in range(n):
        for j in range(k):
            total = 0.0
            for r in range(k):
                total += X[i, r] * fisher_inv[r, j]
            tmp_k[j] = total

        total = 0.0
        for j in range(k):
            total += tmp_k[j] * X[i, j]
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
    fisher_work: NDArray[np.float64],
    modified_score: NDArray[np.float64],
    x_bar: NDArray[np.float64],
    Ix: NDArray[np.float64],
    term1: NDArray[np.float64],
    term23: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],
) -> tuple[float, int]:
    """Compute loglik, score, and Fisher info for one iteration of Newton-Raphson."""
    n, k = X.shape
    (
        wX,
        S0_cumsum,
        S1_cumsum,
        S2_cumsum,
        wXh,
        A_cumsum,
        B_cumsum,
        eye_k,
        eta,
        risk,
        h,
        fisher_info,
    ) = workspace

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

        set_identity(eye_k)
        info = dpotrs(fisher_work, eye_k)  # eye_k now contains inv(fisher_info)

    if info != 0:  # dpotrf or dpotrs failed
        fisher_work[:, :] = fisher_info

        diag_max = 0.0
        for j in range(k):
            diag_val = abs(fisher_work[j, j])
            if diag_val > diag_max:
                diag_max = diag_val
        tol = max(1, k) * np.finfo(np.float64).eps * diag_max

        piv = np.zeros(k, dtype=BLAS_INT_DTYPE)
        rank, info = dpstrf(fisher_work, piv, tol)
        if info == 0 and rank == k:
            logdet = 0.0
            for j in range(k):
                logdet += np.log(abs(fisher_work[j, j]))
            logdet *= 2.0

            set_identity(eye_k)
            info = dpotrs(fisher_work, eye_k)  # eye_k now contains inv(fisher_info)
            if info == 0:
                # undo the pivoting
                temp = _alloc_f_order(k, k)
                temp[:, :] = eye_k
                for i in range(k):
                    pi = piv[i] - 1  # piv is 1-indexed
                    for j in range(k):
                        pj = piv[j] - 1
                        eye_k[pi, pj] = temp[i, j]
            else:
                return -np.inf, _STATUS_LINALG_FAIL
        elif rank < k:
            return -np.inf, _STATUS_RANK_DEFICIENT
        else:
            return -np.inf, _STATUS_LINALG_FAIL

    _compute_firth_correction(
        X=X,
        block_ends=block_ends,
        block_d=block_d,
        fisher_inv=eye_k,
        modified_score=modified_score,
        x_bar=x_bar,
        Ix=Ix,
        term1=term1,
        term23=term23,
        workspace=workspace,
    )

    loglik += 0.5 * logdet

    return loglik, 0  # status success


@njit(cache=True)
def newton_raphson_cox(
    X: NDArray[np.float64],
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    block_s: NDArray[np.float64],
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
    workspace: tuple[NDArray[np.float64], ...],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], int, int]:
    """
    JIT-compiled Newton-Raphson solver for Firth Cox proportional hazards.

    Returns (beta, loglik, fisher_info, n_iter, status)
    """
    (
        wX,
        S0_cumsum,
        S1_cumsum,
        S2_cumsum,
        wXh,
        A_cumsum,
        B_cumsum,
        eye_k,
        eta,
        risk,
        h,
        fisher_info,
    ) = workspace

    n, k = X.shape
    beta = np.zeros(k, dtype=np.float64)
    fisher_work = _alloc_f_order(k, k)
    modified_score = np.empty(k, dtype=np.float64)
    x_bar = np.empty(k, dtype=np.float64)
    Ix = np.empty(k, dtype=np.float64)
    term1 = np.empty(k, dtype=np.float64)
    term23 = np.empty(k, dtype=np.float64)

    score_col = _alloc_f_order(k, 1)
    delta = np.empty(k, dtype=np.float64)
    beta_new = np.empty(k, dtype=np.float64)

    loglik, status = compute_cox_quantities(
        X=X,
        block_ends=block_ends,
        block_d=block_d,
        block_s=block_s,
        beta=beta,
        fisher_work=fisher_work,
        modified_score=modified_score,
        x_bar=x_bar,
        Ix=Ix,
        term1=term1,
        term23=term23,
        workspace=workspace,
    )
    if status != 0:
        return beta, loglik, fisher_info, 0, status

    for iteration in range(1, max_iter + 1):
        fisher_work[:] = fisher_info

        info = dpotrf(fisher_work)
        if info == 0:
            for i in range(k):
                score_col[i, 0] = modified_score[i]
            info = dpotrs(fisher_work, score_col)
            if info == 0:
                for i in range(k):
                    delta[i] = score_col[i, 0]
        if info != 0:  # dpotrf or dpotrs failed
            delta[:] = np.linalg.lstsq(fisher_info, modified_score)[0]

        max_score = max_abs(modified_score)
        max_delta = max_abs(delta)
        if max_score < gtol and max_delta < xtol:
            return beta, loglik, fisher_info, iteration, _STATUS_CONVERGED

        if max_delta > max_step:
            scale = max_step / max_delta
            for i in range(k):
                delta[i] *= scale

        for i in range(k):
            beta_new[i] = beta[i] + delta[i]

        loglik_new, status = compute_cox_quantities(
            X=X,
            block_ends=block_ends,
            block_d=block_d,
            block_s=block_s,
            beta=beta_new,
            fisher_work=fisher_work,
            modified_score=modified_score,
            x_bar=x_bar,
            Ix=Ix,
            term1=term1,
            term23=term23,
            workspace=workspace,
        )
        if status != 0:
            return beta, loglik, fisher_info, iteration, status

        if loglik_new >= loglik or max_halfstep == 0:
            for i in range(k):
                beta[i] = beta_new[i]
            loglik = loglik_new
        else:
            step_factor = 0.5
            accepted = False
            for _ in range(max_halfstep):
                for i in range(k):
                    beta_new[i] = beta[i] + step_factor * delta[i]
                loglik_new, status = compute_cox_quantities(
                    X=X,
                    block_ends=block_ends,
                    block_d=block_d,
                    block_s=block_s,
                    beta=beta_new,
                    fisher_work=fisher_work,
                    modified_score=modified_score,
                    x_bar=x_bar,
                    Ix=Ix,
                    term1=term1,
                    term23=term23,
                    workspace=workspace,
                )
                if status != 0:
                    return beta, loglik, fisher_info, iteration, status

                if loglik_new >= loglik:
                    for i in range(k):
                        beta[i] = beta_new[i]
                    loglik = loglik_new
                    accepted = True
                    break
                step_factor *= 0.5

            if not accepted:
                # recompute quantities at the last accepted beta
                loglik, status = compute_cox_quantities(
                    X=X,
                    block_ends=block_ends,
                    block_d=block_d,
                    block_s=block_s,
                    beta=beta,
                    fisher_work=fisher_work,
                    modified_score=modified_score,
                    x_bar=x_bar,
                    Ix=Ix,
                    term1=term1,
                    term23=term23,
                    workspace=workspace,
                )
                return beta, loglik, fisher_info, iteration, _STATUS_STEP_HALVING_FAILED

    return beta, loglik, fisher_info, max_iter, _STATUS_MAX_ITER


@njit(fastmath=True, cache=True)
def constrained_lrt_1df_cox(
    X: NDArray[np.float64],
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    block_s: NDArray[np.float64],
    idx: int,
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
    workspace: tuple[NDArray[np.float64], ...],
) -> tuple[float, int, int]:  # loglik, iteration, status
    """
    Fit constrained model with beta[idx]=0 for likelihood ratio test.

    Returns (loglik, n_iter, status)
    """
    (
        wX,
        S0_cumsum,
        S1_cumsum,
        S2_cumsum,
        wXh,
        A_cumsum,
        B_cumsum,
        eye_k,
        eta,
        risk,
        h,
        fisher_info,
    ) = workspace

    n, k = X.shape
    free_k = k - 1

    beta = np.zeros(k, dtype=np.float64)
    fisher_work = _alloc_f_order(k, k)
    modified_score = np.empty(k, dtype=np.float64)
    x_bar = np.empty(k, dtype=np.float64)
    Ix = np.empty(k, dtype=np.float64)
    term1 = np.empty(k, dtype=np.float64)
    term23 = np.empty(k, dtype=np.float64)

    free_idx = np.empty(free_k, dtype=np.intp)
    pos = 0
    for j in range(k):
        if j != idx:
            free_idx[pos] = j
            pos += 1

    fisher_free = _alloc_f_order(free_k, free_k)
    score_free = np.empty(free_k, dtype=np.float64)
    score_col = _alloc_f_order(free_k, 1)
    delta = np.empty(free_k, dtype=np.float64)
    beta_new = np.empty(k, dtype=np.float64)

    loglik, status = compute_cox_quantities(
        X=X,
        block_ends=block_ends,
        block_d=block_d,
        block_s=block_s,
        beta=beta,
        fisher_work=fisher_work,
        modified_score=modified_score,
        x_bar=x_bar,
        Ix=Ix,
        term1=term1,
        term23=term23,
        workspace=workspace,
    )
    if status != 0:
        return loglik, 0, status

    for iteration in range(1, max_iter + 1):
        for i in range(free_k):
            score_free[i] = modified_score[free_idx[i]]

        for i in range(free_k):
            ii = free_idx[i]
            for j in range(free_k):
                jj = free_idx[j]
                fisher_free[i, j] = fisher_info[ii, jj]

        info = dpotrf(fisher_free)
        if info == 0:
            for i in range(free_k):
                score_col[i, 0] = score_free[i]
            info = dpotrs(fisher_free, score_col)
            if info == 0:
                for i in range(free_k):
                    delta[i] = score_col[i, 0]
        if info != 0:  # dpotrf or dpotrs failed
            for i in range(free_k):
                ii = free_idx[i]
                for j in range(free_k):
                    jj = free_idx[j]
                    fisher_free[i, j] = fisher_info[ii, jj]
            delta[:] = np.linalg.lstsq(fisher_free, score_free)[0]

        max_score = max_abs(score_free)
        max_delta = max_abs(delta)
        if max_score < gtol and max_delta < xtol:
            return loglik, iteration, _STATUS_CONVERGED

        if max_delta > max_step:
            scale = max_step / max_delta
            for i in range(free_k):
                delta[i] *= scale

        for i in range(k):
            beta_new[i] = beta[i]
        for i in range(free_k):
            beta_new[free_idx[i]] = beta[free_idx[i]] + delta[i]
        beta_new[idx] = 0.0

        loglik_new, status = compute_cox_quantities(
            X=X,
            block_ends=block_ends,
            block_d=block_d,
            block_s=block_s,
            beta=beta_new,
            fisher_work=fisher_work,
            modified_score=modified_score,
            x_bar=x_bar,
            Ix=Ix,
            term1=term1,
            term23=term23,
            workspace=workspace,
        )
        if status != 0:
            return loglik, iteration, status

        if loglik_new >= loglik or max_halfstep == 0:
            for i in range(k):
                beta[i] = beta_new[i]
            loglik = loglik_new
        else:
            step_factor = 0.5
            accepted = False
            for _ in range(max_halfstep):
                for i in range(free_k):
                    beta_new[free_idx[i]] = beta[free_idx[i]] + step_factor * delta[i]
                beta_new[idx] = 0.0

                loglik_new, status = compute_cox_quantities(
                    X=X,
                    block_ends=block_ends,
                    block_d=block_d,
                    block_s=block_s,
                    beta=beta_new,
                    fisher_work=fisher_work,
                    modified_score=modified_score,
                    x_bar=x_bar,
                    Ix=Ix,
                    term1=term1,
                    term23=term23,
                    workspace=workspace,
                )
                if status != 0:
                    return loglik, iteration, status

                if loglik_new >= loglik:
                    for i in range(k):
                        beta[i] = beta_new[i]
                    loglik = loglik_new
                    accepted = True
                    break
                step_factor *= 0.5

            if not accepted:
                return loglik, iteration, _STATUS_STEP_HALVING_FAILED

    return loglik, max_iter, _STATUS_MAX_ITER


@njit(fastmath=True, cache=True)
def profile_ci_bound_cox(
    X: NDArray[np.float64],
    block_ends: NDArray[np.intp],
    block_d: NDArray[np.int64],
    block_s: NDArray[np.float64],
    idx: int,
    theta_hat: NDArray[np.float64],
    l_star: float,
    which: int,
    chi2_crit: float,
    max_iter: int,
    tol: float,
    D0: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],
) -> tuple[float, int, int]:  # bound, converged, iter
    """Compute one profile likelihood CI bound using Venzon-Moolgavkar algorithm."""
    (
        wX,
        S0_cumsum,
        S1_cumsum,
        S2_cumsum,
        wXh,
        A_cumsum,
        B_cumsum,
        eye_k,
        eta,
        risk,
        h,
        fisher_info,
    ) = workspace

    n, k = X.shape
    theta = theta_hat.copy()

    beta = np.zeros(k, dtype=np.float64)
    fisher_work = _alloc_f_order(k, k)
    modified_score = np.empty(k, dtype=np.float64)
    x_bar = np.empty(k, dtype=np.float64)
    Ix = np.empty(k, dtype=np.float64)
    term1 = np.empty(k, dtype=np.float64)
    term23 = np.empty(k, dtype=np.float64)

    other_idx = np.empty(k - 1, dtype=np.intp)
    pos = 0
    for j in range(k):
        if j != idx:
            other_idx[pos] = j
            pos += 1

    # Appendix step 2: compute dw/dbeta and h
    if k > 1:
        D0_ww = np.empty((k - 1, k - 1), dtype=np.float64)
        D0_bw = np.empty((k - 1), dtype=np.float64)
        for i in range(k - 1):
            ii = other_idx[i]
            D0_bw[i] = D0[idx, ii]
            for j in range(k - 1):
                jj = other_idx[j]
                D0_ww[i, j] = D0[ii, jj]
        sign, logdet = np.linalg.slogdet(D0_ww)
        if sign != 0 and np.isfinite(logdet):
            dw_db = -np.linalg.solve(D0_ww, D0_bw)
        else:
            dw_db = -np.linalg.lstsq(D0_ww, D0_bw)[0]

        d2l_db2 = D0[idx, idx]
        for i in range(k - 1):
            d2l_db2 += D0_bw[i] * dw_db[i]
    else:  # single parameter case
        dw_db = np.empty(0, dtype=np.float64)
        d2l_db2 = D0[idx, idx]

    if d2l_db2 >= 0.0:
        h_step = which * 0.5
    else:
        h_step = which * np.sqrt(chi2_crit / abs(d2l_db2)) / 2.0

    # Appendix step 3: theta(1) = theta_hat + h * [1, dw/dbeta]' (Eq. 4)
    tangent = np.zeros(k, dtype=np.float64)
    tangent[idx] = 1.0
    if k > 1:
        for i in range(k - 1):
            tangent[other_idx[i]] = dw_db[i]

    for i in range(k):
        theta[i] += h_step * tangent[i]

    F = np.empty(k, dtype=np.float64)
    D = np.empty((k, k), dtype=np.float64)
    G = _alloc_f_order(k, k)
    ipiv = np.zeros(k, dtype=BLAS_INT_DTYPE)
    rhs = _alloc_f_order(k, 2)
    v = np.empty(k, dtype=np.float64)
    g_j = np.empty(k, dtype=np.float64)
    Dg = np.empty(k, dtype=np.float64)
    Dv = np.empty(k, dtype=np.float64)
    step1 = np.empty(k, dtype=np.float64)
    step2 = np.empty(k, dtype=np.float64)
    temp = np.empty(k, dtype=np.float64)

    # Appendix steps 4-9: Modified Newton-Raphson
    for iteration in range(1, max_iter + 1):
        # Appendix step 4: compute score and Hessian at theta(i)
        loglik, status = compute_cox_quantities(
            X=X,
            block_ends=block_ends,
            block_d=block_d,
            block_s=block_s,
            beta=theta,
            fisher_work=fisher_work,
            modified_score=modified_score,
            x_bar=x_bar,
            Ix=Ix,
            term1=term1,
            term23=term23,
            workspace=workspace,
        )
        if status != 0:
            return theta[idx], status, iteration

        # Appendix step 5: F = [l - l*, dl/dw]' (eq. 2)
        for i in range(k):
            F[i] = modified_score[i]
        F[idx] = loglik - l_star

        # Appendix step 9: check convergence
        # TODO: the paper checks for relative change in loglik and the coefficients
        # between iterations. We're checking |loglik-l_star| directly, and the
        # |scores| of the nuisance parameters.
        if max_abs(F) <= tol:
            return theta[idx], _STATUS_CONVERGED, iteration

        # D = d2l/dtheta2 at current theta (Appendix step 4)
        for i in range(k):
            for j in range(k):
                val = -fisher_info[i, j]
                D[i, j] = val
                G[i, j] = val

        for j in range(k):
            G[idx, j] = modified_score[j]

        # Appendix step 6: v = G^-1 F (direction to subtract)
        for i in range(k):
            ipiv[i] = 0

        info = dgetrf(G, ipiv)
        if info == 0:
            # Solve G * [v, g_j] = [F, e_idx] without forming G^-1.
            for i in range(k):
                rhs[i, 0] = F[i]
                rhs[i, 1] = 0.0
            rhs[idx, 1] = 1.0
            info = dgetrs(G, ipiv, rhs)
            if info == 0:
                for i in range(k):
                    v[i] = rhs[i, 0]
                    g_j[i] = rhs[i, 1]

        if info != 0:
            # dgetrf or dgetrs failed; reconstruct G from D and use lstsq
            for i in range(k):
                for j in range(k):
                    G[i, j] = D[i, j]
            for j in range(k):
                G[idx, j] = modified_score[j]
            v[:] = np.linalg.lstsq(G, F)[0]
            e_idx_vec = np.zeros(k, dtype=np.float64)
            e_idx_vec[idx] = 1.0
            g_j[:] = np.linalg.lstsq(G, e_idx_vec)[0]

        # Appendix step 7: quadratic correction
        # g'Dg*s^2 + (2v'Dg - 2)*s + v'Dv = 0 (Eq. 8)
        # g_j solves G * g_j = e_idx
        # a = g_j @ D @ g_j
        # b = 2 * v @ D @ g_j - 2
        # c = v @ D @ v

        for i in range(k):
            total = 0.0
            for j in range(k):
                total += D[i, j] * g_j[j]
            Dg[i] = total

        a = 0.0
        for i in range(k):
            a += g_j[i] * Dg[i]

        for i in range(k):
            total = 0.0
            for j in range(k):
                total += D[i, j] * v[j]
            Dv[i] = total

        vDg = 0.0
        for i in range(k):
            vDg += v[i] * Dg[i]
        b = 2.0 * vDg - 2.0

        c = 0.0
        for i in range(k):
            c += v[i] * Dv[i]

        discriminant = b * b - 4.0 * a * c

        if discriminant >= 0.0 and abs(a) > 1e-10:
            sqrt_disc = np.sqrt(discriminant)
            s1 = (-b + sqrt_disc) / (2.0 * a)
            s2 = (-b - sqrt_disc) / (2.0 * a)

            for i in range(k):
                step1[i] = -v[i] - s1 * g_j[i]
                step2[i] = -v[i] - s2 * g_j[i]

            for i in range(k):
                total = 0.0
                for j in range(k):
                    total += -D0[i, j] * step1[j]
                temp[i] = total
            norm1 = 0.0
            for i in range(k):
                norm1 += step1[i] * temp[i]

            for i in range(k):
                total = 0.0
                for j in range(k):
                    total += -D0[i, j] * step2[j]
                temp[i] = total
            norm2 = 0.0
            for i in range(k):
                norm2 += step2[i] * temp[i]

            if norm1 < norm2:
                for i in range(k):
                    theta[i] += step1[i]
            else:
                for i in range(k):
                    theta[i] += step2[i]
        else:
            for i in range(k):
                theta[i] -= 0.1 * v[i]

    return theta[idx], _STATUS_MAX_ITER, max_iter


@njit(cache=True)
def concordance_index(
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
