import scipy
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from typing import Callable

from firthmodels._utils import FirthResult, IterationQuantities


def newton_raphson(
    compute_quantities: Callable[[NDArray], IterationQuantities],
    n_features: int,
    max_iter: int = 25,
    max_step: float = 5.0,
    max_halfstep: int = 25,
    tol: float = 1e-4,
) -> FirthResult:
    """
    Newton-Raphson solver

    Parameters
    ----------
    compute_quantities : Callable[[NDArray]]
        Function `callable(beta)` that returns loglik, modified score, and fisher_info
    n_features : int
        Number of features
    max_iter : int, default=25
        Maximum number of iterations
    max_step : float, default=5.0
        Maximum step size per coefficient
    max_halfstep : int, default=25
        Maximum number of step-halvings per iteration
    tol : float, default=1e-4
        Tolerance for stopping criteria

    Returns
    -------
    FirthResult
        Result of Firth-penalized optimization
    """
    beta = np.zeros(n_features, dtype=np.float64)
    q = compute_quantities(beta)

    for iteration in range(1, max_iter + 1):
        # check convergence: max|U*| < tol
        if np.max(np.abs(q.modified_score)) < tol:
            return FirthResult(
                beta=beta,
                loglik=q.loglik,
                fisher_info=q.fisher_info,
                n_iter=iteration,
                converged=True,
            )

        # solve for step: delta = (X'WX)^(-1) @ U*
        try:
            cho = scipy.linalg.cho_factor(q.fisher_info)
            delta = scipy.linalg.cho_solve(cho, q.modified_score)
        except scipy.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(q.fisher_info, q.modified_score, rcond=None)

        # restrict to max_stepsize
        max_delta = np.max(np.abs(delta))
        if max_delta > max_step:
            delta = delta * (max_step / max_delta)

        # Try full step first
        beta_new = beta + delta
        q_new = compute_quantities(beta_new)

        if q_new.loglik >= q.loglik or max_halfstep == 0:
            beta = beta_new
            q = q_new
        else:
            # Step-halving until loglik improves
            step_factor = 0.5
            for _ in range(max_halfstep):
                beta_new = beta + step_factor * delta
                q_new = compute_quantities(beta_new)
                if q_new.loglik >= q.loglik:
                    beta = beta_new
                    q = q_new
                    break
                step_factor *= 0.5
            else:
                warnings.warn(
                    "Step-halving failed to converge.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
                beta = beta_new
                q = q_new
    # max_iter reached without convergence
    warning_msg = "Maximum number of iterations reached without convergence."
    warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)

    return FirthResult(
        beta=beta,
        loglik=q.loglik,
        fisher_info=q.fisher_info,
        n_iter=max_iter,
        converged=False,
    )
