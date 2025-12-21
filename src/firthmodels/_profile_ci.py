from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.lapack import dgetrf, dgetri

from firthmodels._utils import IterationQuantities


@dataclass(frozen=True)
class ProfileCIBoundResult:
    bound: float
    converged: bool
    n_iter: int


def profile_ci_bound(
    *,
    idx: int,
    theta_hat: NDArray[np.float64],
    l_star: float,
    which: Literal[-1, 1],
    max_iter: int,
    tol: float,
    chi2_crit: float,
    compute_quantities_full: Callable[[NDArray[np.float64]], IterationQuantities],
    D0: NDArray[np.float64] | None = None,
) -> ProfileCIBoundResult:
    """
    Compute one profile CI bound using Venzon-Moolgavkar (1988) algorithm.

    Solves F(theta) = [l(theta) - l*, dl/dw]' = 0 where beta = theta[idx]
    is the parameter of interest and w (omega) are nuisance parameters.

    Parameters
    ----------
    idx : int
        Index of the parameter of interest.
    theta_hat : ndarray of shape (k,)
        MLE coefficient vector.
    l_star : float
        Target log-likelihood (l_hat - chi2_crit / 2).
    which : {-1, 1}
        -1 for lower bound, +1 for upper bound.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    chi2_crit : float
        Chi-squared critical value (e.g., chi2.ppf(0.95, 1) for 95% CI).
    compute_quantities_full : callable
        Function that takes beta vector and returns IterationQuantities.
    D0 : ndarray of shape (k, k), optional
        Hessian at MLE (-fisher_info at theta_hat). If None, computed internally.

    Returns
    -------
    ProfileCIBoundResult
        Contains bound value, convergence status, and iteration count.

    References
    ----------
    Venzon, D.J. and Moolgavkar, S.H. (1988). "A Method for Computing
    Profile-Likelihood-Based Confidence Intervals." Applied Statistics,
    37(1), 87-94.
    """
    k = theta_hat.shape[0]

    # Initialize at MLE
    theta = theta_hat.copy()

    # Appendix step 1: compute and store D0 = d2l/dtheta2 at MLE
    if D0 is None:
        D0 = -compute_quantities_full(theta_hat).fisher_info

    # beta = parameter of interest, omega = nuisance parameters
    other_idx = [i for i in range(k) if i != idx]

    # Appendix step 2: compute dw/dbeta and h
    if len(other_idx) > 0:
        D0_ww = D0[np.ix_(other_idx, other_idx)]  # d2l/dw2
        D0_bw = D0[idx, other_idx]  # d2l/dbeta*dw

        # dw/dbeta = -(d2l/dw2)^-1 @ (d2l/dbeta*dw)  (Eq. 5)
        try:
            dw_db = -np.linalg.solve(D0_ww, D0_bw)
        except np.linalg.LinAlgError:
            dw_db = -np.linalg.lstsq(D0_ww, D0_bw, rcond=None)[0]

        # d2l_bar/dbeta2 = D_bb - D_bw @ D_ww^-1 @ D_wb (eq. 6 denominator)
        d2l_db2 = D0[idx, idx] + D0_bw @ dw_db
    else:
        # Single parameter case
        dw_db = np.array([])
        d2l_db2 = D0[idx, idx]

    # step size h (Eq. 6)
    if d2l_db2 >= 0:
        # Hessian should be negative definite, fallback
        h = which * 0.5
    else:
        h = which * np.sqrt(chi2_crit / abs(d2l_db2)) / 2

    # Appendix step 3: theta(1) = theta_hat + h * [1, dw/dbeta]' (Eq. 4)
    tangent = np.zeros(k)
    tangent[idx] = 1.0
    if len(other_idx) > 0:
        tangent[other_idx] = dw_db
    theta = theta + h * tangent

    # Appendix steps 4-9: Modified Newton-Raphson
    for iteration in range(1, max_iter + 1):
        # Appendix step 4: compute score and Hessian at theta(i)
        q = compute_quantities_full(theta)

        # Appendix step 5: F = [l - l*, dl/dw]' (eq. 2)
        F = q.modified_score.copy()
        F[idx] = q.loglik - l_star

        # Appendix step 9: check convergence
        # TODO: the paper checks for relative change in loglik and the coefficients
        # between iterations. We're checking |loglik-l_star| directly, and the
        # |scores| of the nuisance parameters.
        if np.abs(F).max() <= tol:
            return ProfileCIBoundResult(
                bound=theta[idx],
                converged=True,
                n_iter=iteration,
            )

        # D = d2l/dtheta2 at current theta (Appendix step 4)
        D = -q.fisher_info
        G = D.copy(order="F")
        G[idx, :] = q.modified_score  # Jacobian (Eq. 3)

        # Appendix step 6: v = G^-1 F (direction to subtract)
        try:
            # G_inv = np.linalg.inv(G)
            lu, piv, info = dgetrf(G, overwrite_a=0)
            if info != 0:
                raise np.linalg.LinAlgError("dgetrf failed")
            G_inv, info = dgetri(lu, piv, overwrite_lu=1)
            if info != 0:
                raise np.linalg.LinAlgError("dgetri failed")
            v = G_inv @ F
        except np.linalg.LinAlgError:
            v = cast(NDArray[np.float64], np.linalg.lstsq(G, F, rcond=None)[0])
            try:
                G_inv = np.linalg.pinv(G)
            except np.linalg.LinAlgError:
                # pinv failed (SVD did not converge); take damped step (pg 92)
                theta = theta - 0.1 * v
                continue

        # Appendix step 7: quadratic correction
        # g'Dg*s^2 + (2v'Dg - 2)*s + v'Dv = 0 (Eq. 8)
        g_j = G_inv[:, idx]
        a = g_j @ D @ g_j
        b = 2 * v @ D @ g_j - 2
        c = v @ D @ v

        discriminant = b * b - 4 * a * c

        if discriminant >= 0 and abs(a) > 1e-10:
            sqrt_disc = np.sqrt(discriminant)
            s1 = (-b + sqrt_disc) / (2 * a)
            s2 = (-b - sqrt_disc) / (2 * a)

            # Pick root giving smaller step (pg 90)
            step1 = -v - s1 * g_j
            step2 = -v - s2 * g_j
            norm1 = step1 @ (-D0) @ step1
            norm2 = step2 @ (-D0) @ step2

            s = s1 if norm1 < norm2 else s2
            delta = -v - s * g_j  # Eq. 9
        else:
            # No real roots: damped step (pg 92)
            delta = -0.1 * v

        # Appendix step 8: theta(i+1) = theta(i) + delta
        theta = theta + delta

    # failed to converge
    return ProfileCIBoundResult(
        bound=theta[idx],
        converged=False,
        n_iter=max_iter,
    )
