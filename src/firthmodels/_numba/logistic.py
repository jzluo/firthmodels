import numpy as np
from numba import njit
from numpy.typing import NDArray

from firthmodels._numba.linalg import dgemm, dgetrf, dgetri, dpotrf, dpotri, dsyrk


@njit(fastmath=True)
def expit(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


@njit(fastmath=True)
def log1pexp(x: float) -> float:
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(fastmath=True)
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
        return -np.inf, info

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


@njit(fastmath=True)
def newton_raphson_logistic(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    offset: NDArray[np.float64],
    maxiter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[
    NDArray[np.float64], float, NDArray[np.float64], int, bool
]:  # beta, loglik, fisher_info_aug, max_iter, converge_yn
    raise NotImplementedError
