import numpy as np
from numba import njit
from numpy.typing import NDArray

from firthmodels._numba.linalg import (
    _alloc_f_order,
    dgemm,
    dgetrf,
    dgetri,
    dpotrf,
    dpotri,
    dpotrs,
    dsyrk,
)


@njit(fastmath=True, cache=True)
def expit(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


@njit(fastmath=True, cache=True)
def log1pexp(x: float) -> float:
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(fastmath=True, cache=True)
def max_abs(vec: NDArray[np.float64]) -> float:
    max_val = 0.0
    for i in range(vec.shape[0]):
        abs_val = abs(vec[i])
        if abs_val > max_val:
            max_val = abs_val
    return max_val


@njit(fastmath=True, cache=True)
def compute_logistic_quantities(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    beta: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[float, int]:
    (
        eta,
        p,
        w,
        sqrt_w,
        XtW,
        fisher_info,
        solved,
        h,
        w_aug,
        sqrt_w_aug,
        XtW_aug,
        fisher_info_aug,
        residual,
        modified_score,
    ) = workspace

    n = X.shape[0]
    k = X.shape[1]

    # eta = X @ beta + offset
    # p = expit(eta)
    for i in range(n):
        total = offset[i]
        for j in range(k):
            total += X[i, j] * beta[j]
        eta[i] = total
        p[i] = expit(total)

    for i in range(n):
        w_i = sample_weight[i] * p[i] * (1.0 - p[i])
        w[i] = w_i
        sqrt_w[i] = np.sqrt(w_i)

    # XtW = X.T * sqrt_w
    for i in range(n):
        w_i = sqrt_w[i]
        for j in range(k):
            XtW[j, i] = X[i, j] * w_i

    # fisher_info = XtW @ XtW.T
    # dpotrf L overwrites fisher_info with lower Cholesky factor
    # compute logdet using L diagonal (in fisher_info)
    # dpotri overwrites fisher_info with inv(fisher_info)
    # then reflect upper triangle
    dsyrk(XtW, fisher_info)
    info = dpotrf(fisher_info)
    if info != 0:
        # dpotrf failed and fisher_info was overwritten, so recompute
        dsyrk(XtW, fisher_info)
        sign, logdet = np.linalg.slogdet(fisher_info)
        if sign <= 0:
            return -np.inf, 1
        solved[:, :] = np.linalg.lstsq(fisher_info, XtW)[0]
    else:
        logdet = 0.0
        for i in range(k):
            logdet += np.log(fisher_info[i, i])
        logdet *= 2.0

        info = dpotri(fisher_info)
        if info != 0:
            return -np.inf, info

        for i in range(k):
            for j in range(i + 1, k):
                fisher_info[i, j] = fisher_info[j, i]

        dgemm(fisher_info, XtW, solved)

    # h = np.einsum("ij,ij->j", solved, XtW)
    for i in range(n):
        total = 0.0
        for j in range(k):
            total += solved[j, i] * XtW[j, i]
        h[i] = total

    for i in range(n):
        w_aug_i = (sample_weight[i] + h[i]) * p[i] * (1.0 - p[i])
        w_aug[i] = w_aug_i
        sqrt_w_aug[i] = np.sqrt(w_aug_i)

    for i in range(n):
        w_i = sqrt_w_aug[i]
        for j in range(k):
            XtW_aug[j, i] = X[i, j] * w_i

    dsyrk(XtW_aug, fisher_info_aug)
    for i in range(k):
        for j in range(i + 1, k):
            fisher_info_aug[i, j] = fisher_info_aug[j, i]

    loglik = 0.0
    for i in range(n):
        loglik += sample_weight[i] * (y[i] * eta[i] - log1pexp(eta[i]))
    loglik += 0.5 * logdet

    for i in range(n):
        residual[i] = sample_weight[i] * (y[i] - p[i]) + h[i] * (0.5 - p[i])

    for j in range(k):
        total = 0.0
        for i in range(n):
            total += X[i, j] * residual[i]
        modified_score[j] = total

    return loglik, 0


@njit(fastmath=True, cache=True)
def newton_raphson_logistic(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[
    NDArray[np.float64], float, NDArray[np.float64], int, bool
]:  # beta, loglik, fisher_info_aug, max_iter, converge_yn
    (
        eta,
        p,
        w,
        sqrt_w,
        XtW,
        fisher_info,  # use this as a scratch array
        solved,
        h,
        w_aug,
        sqrt_w_aug,
        XtW_aug,
        fisher_info_aug,
        residual,
        modified_score,
    ) = workspace

    n = X.shape[0]
    k = X.shape[1]
    beta = np.zeros(k, dtype=np.float64)
    beta_new = np.zeros(k, dtype=np.float64)
    score_col = _alloc_f_order(k, 1)
    delta = np.zeros(k, dtype=np.float64)

    loglik, info = compute_logistic_quantities(
        X, y, beta, sample_weight, offset, workspace
    )
    if info != 0:
        return beta, loglik, fisher_info_aug, 0, False

    for iteration in range(1, max_iter + 1):
        fisher_info[:, :] = fisher_info_aug

        info = dpotrf(fisher_info)
        if info != 0:
            delta[:] = np.linalg.lstsq(fisher_info_aug, modified_score)[0]
        else:
            for i in range(k):
                score_col[i, 0] = modified_score[i]

            info = dpotrs(fisher_info, score_col)
            if info != 0:
                return beta, loglik, fisher_info_aug, iteration, False

            for i in range(k):
                delta[i] = score_col[i, 0]

        max_score = max_abs(modified_score)
        max_delta = max_abs(delta)
        if max_score < gtol and max_delta < xtol:
            return beta, loglik, fisher_info_aug, iteration, True

        if max_delta > max_step:
            scale = max_step / max_delta
            for i in range(k):
                delta[i] *= scale

        # try full step first
        for i in range(k):
            beta_new[i] = beta[i] + delta[i]

        loglik_new, info = compute_logistic_quantities(
            X, y, beta_new, sample_weight, offset, workspace
        )
        if info != 0:
            return beta, loglik, fisher_info_aug, iteration, False

        if loglik_new >= loglik or max_halfstep == 0:
            for i in range(k):
                beta[i] = beta_new[i]
            loglik = loglik_new
        else:
            # Step-halving until loglik improves
            step_factor = 0.5
            accepted = False
            for _ in range(max_halfstep):
                for i in range(k):
                    beta_new[i] = beta[i] + step_factor * delta[i]

                loglik_new, info = compute_logistic_quantities(
                    X, y, beta_new, sample_weight, offset, workspace
                )
                if info != 0:
                    return beta, loglik, fisher_info_aug, iteration, False

                if loglik_new >= loglik:
                    for i in range(k):
                        beta[i] = beta_new[i]
                    loglik = loglik_new
                    accepted = True
                    break
                step_factor *= 0.5

            if not accepted:  # step-halving failed, return early
                return beta, loglik, fisher_info_aug, iteration, False

    return beta, loglik, fisher_info_aug, iteration, False


@njit(fastmath=True, cache=True)
def constrained_lrt_1df_logistic(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
    idx: int,
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[float, int, bool]:  # loglik, iteration, converged
    (
        eta,
        p,
        w,
        sqrt_w,
        XtW,
        fisher_info,  # use this as a scratch array
        solved,
        h,
        w_aug,
        sqrt_w_aug,
        XtW_aug,
        fisher_info_aug,
        residual,
        modified_score,
    ) = workspace

    n, k = X.shape
    free_k = k - 1
    beta = np.zeros(k, dtype=np.float64)
    beta_new = np.zeros(k, dtype=np.float64)
    score_free = np.empty(free_k, dtype=np.float64)
    score_col = _alloc_f_order(free_k, 1)
    delta = np.empty(free_k, dtype=np.float64)
    fisher_free = _alloc_f_order(free_k, free_k)

    free_idx = np.empty(free_k, dtype=np.intp)
    pos = 0
    for j in range(k):
        if j != idx:
            free_idx[pos] = j
            pos += 1

    loglik, info = compute_logistic_quantities(
        X, y, beta, sample_weight, offset, workspace
    )
    if info != 0:
        return loglik, 0, False

    for iteration in range(1, max_iter + 1):
        for i in range(free_k):
            score_free[i] = modified_score[free_idx[i]]

        for i in range(free_k):
            ii = free_idx[i]
            for j in range(free_k):
                jj = free_idx[j]
                fisher_free[i, j] = fisher_info_aug[ii, jj]

        info = dpotrf(fisher_free)
        if info != 0:
            # Recompute fisher_free since dpotrf overwrote it
            for i in range(free_k):
                ii = free_idx[i]
                for j in range(free_k):
                    jj = free_idx[j]
                    fisher_free[i, j] = fisher_info_aug[ii, jj]
            delta[:] = np.linalg.lstsq(fisher_free, score_free)[0]
        else:
            for i in range(free_k):
                score_col[i, 0] = score_free[i]

            info = dpotrs(fisher_free, score_col)
            if info != 0:
                return loglik, iteration, False

            for i in range(free_k):
                delta[i] = score_col[i, 0]

        max_score = max_abs(score_free)
        max_delta = max_abs(delta)
        if max_score < gtol and max_delta < xtol:
            return loglik, iteration, True

        if max_delta > max_step:
            scale = max_step / max_delta
            for i in range(free_k):
                delta[i] *= scale

        for i in range(k):
            beta_new[i] = beta[i]
        for i in range(free_k):
            beta_new[free_idx[i]] = beta[free_idx[i]] + delta[i]
        beta_new[idx] = 0.0

        loglik_new, info = compute_logistic_quantities(
            X, y, beta_new, sample_weight, offset, workspace
        )
        if info != 0:
            return loglik, iteration, False

        if loglik_new >= loglik or max_halfstep == 0:
            for i in range(k):
                beta[i] = beta_new[i]
            loglik = loglik_new
        else:
            # Step-halving until loglik improves
            step_factor = 0.5
            accepted = False
            for _ in range(max_halfstep):
                for i in range(free_k):
                    beta_new[free_idx[i]] = beta[free_idx[i]] + step_factor * delta[i]
                beta_new[idx] = 0.0

                loglik_new, info = compute_logistic_quantities(
                    X, y, beta_new, sample_weight, offset, workspace
                )
                if info != 0:
                    return loglik, iteration, False

                if loglik_new >= loglik:
                    for i in range(k):
                        beta[i] = beta_new[i]
                    loglik = loglik_new
                    accepted = True
                    break
                step_factor *= 0.5

            if not accepted:  # step-halving failed, return early
                return loglik, iteration, False

    return loglik, max_iter, False
