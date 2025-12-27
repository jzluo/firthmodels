import numpy as np
from numba import njit
from numpy.typing import NDArray

from firthmodels._blas_abi import BLAS_INT_DTYPE
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


@njit(cache=True, fastmath=False)
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
        symmetrize_lower(fisher_info)
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

        symmetrize_lower(fisher_info)

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
    symmetrize_lower(fisher_info_aug)

    loglik = 0.0
    for i in range(n):
        loglik += sample_weight[i] * (y[i] * eta[i] - log1pexp(eta[i]))
    loglik += 0.5 * logdet

    for i in range(n):
        residual[i] = sample_weight[i] * (y[i] - p[i]) + h[i] * (0.5 - p[i])

    dgemv(X.T, residual, modified_score)

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
        fisher_info_aug[:, :] = np.nan  # poison fisher_info_aug on failure
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
                fisher_info_aug[:, :] = np.nan  # poison fisher_info_aug on failure
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
            fisher_info_aug[:, :] = np.nan  # poison fisher_info_aug on failure
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
                    fisher_info_aug[:, :] = np.nan  # poison fisher_info_aug on failure
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


@njit(fastmath=True, cache=True)
def profile_ci_bound_logistic(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    offset: NDArray[np.float64],
    idx: int,
    theta_hat: NDArray[np.float64],
    l_star: float,
    which: int,
    chi2_crit: float,
    max_iter: int,
    tol: float,
    D0: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[float, bool, int]:  # bound, converged, iter
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
    theta = theta_hat.copy()
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
    else:  # single paramete case
        dw_db = np.empty(0, dtype=np.float64)
        d2l_db2 = D0[idx, idx]

    if d2l_db2 >= 0.0:
        h = which * 0.5
    else:
        h = which * np.sqrt(chi2_crit / abs(d2l_db2)) / 2.0

    # Appendix step 3: theta(1) = theta_hat + h * [1, dw/dbeta]' (Eq. 4)
    tangent = np.zeros(k, dtype=np.float64)
    tangent[idx] = 1.0
    if k > 1:
        for i in range(k - 1):
            tangent[other_idx[i]] = dw_db[i]

    for i in range(k):
        theta[i] += h * tangent[i]

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
        loglik, info = compute_logistic_quantities(
            X, y, theta, sample_weight, offset, workspace
        )
        if info != 0:
            return theta[idx], False, iteration

        # Appendix step 5: F = [l - l*, dl/dw]' (eq. 2)
        for i in range(k):
            F[i] = modified_score[i]
        F[idx] = loglik - l_star

        # Appendix step 9: check convergence
        # TODO: the paper checks for relative change in loglik and the coefficients
        # between iterations. We're checking |loglik-l_star| directly, and the
        # |scores| of the nuisance parameters.
        if max_abs(F) <= tol:
            return theta[idx], True, iteration

        # D = d2l/dtheta2 at current theta (Appendix step 4)
        for i in range(k):
            for j in range(k):
                val = -fisher_info_aug[i, j]
                D[i, j] = val
                G[i, j] = val

        for j in range(k):
            G[idx, j] = modified_score[j]

        # Appendix step 6: v = G^-1 F (direction to subtract)
        for i in range(k):
            ipiv[i] = 0

        info = dgetrf(G, ipiv)
        if info != 0:
            return theta[idx], False, iteration

        # Solve G * [v, g_j] = [F, e_idx] without forming G^-1.
        for i in range(k):
            rhs[i, 0] = F[i]
            rhs[i, 1] = 0.0
        rhs[idx, 1] = 1.0

        info = dgetrs(G, ipiv, rhs)
        if info != 0:
            return theta[idx], False, iteration

        for i in range(k):
            v[i] = rhs[i, 0]
            g_j[i] = rhs[i, 1]

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

    return theta[idx], False, max_iter
