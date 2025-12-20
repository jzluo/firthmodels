import warnings
from typing import Callable

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.linalg.lapack import dpotrf, dpotrs
from sklearn.exceptions import ConvergenceWarning

from firthmodels._utils import FirthResult, IterationQuantities


def newton_raphson(
    compute_quantities: Callable[[NDArray], IterationQuantities],
    n_features: int,
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
) -> FirthResult:
    """
    Newton-Raphson solver

    Parameters
    ----------
    compute_quantities : Callable[[NDArray]]
        Function `callable(beta)` that returns loglik, modified score, and fisher_info
    n_features : int
        Number of features
    max_iter : int
        Maximum number of iterations
    max_step : float
        Maximum step size per coefficient
    max_halfstep : int
        Maximum number of step-halvings per iteration
    gtol : float
        Gradient convergence criteria. Converged when max|gradient| < gtol.
    xtol : float
        Parameter convergence criteria. Converged when max|delta| < xtol.

    Returns
    -------
    FirthResult
        Result of Firth-penalized optimization
    """
    beta = np.zeros(n_features, dtype=np.float64)
    q = compute_quantities(beta)

    for iteration in range(1, max_iter + 1):
        # solve for step: delta = (X'WX)^(-1) @ U*
        try:
            L, info = dpotrf(q.fisher_info, lower=1, overwrite_a=0)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrf failed")
            # dpotrs needs F-contiguous RHS, score is 1D so reshape to column vector
            score_col = q.modified_score.reshape(-1, 1, order="F")
            delta, info = dpotrs(L, score_col, lower=1)
            if info != 0:
                raise scipy.linalg.LinAlgError("dpotrs failed")
            delta = delta.ravel()
        except scipy.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(q.fisher_info, q.modified_score, rcond=None)

        # check convergence:
        max_score = np.max(np.abs(q.modified_score))
        max_delta = np.max(np.abs(delta))
        if max_score < gtol and max_delta < xtol:
            return FirthResult(
                beta=beta,
                loglik=q.loglik,
                fisher_info=q.fisher_info,
                n_iter=iteration,
                converged=True,
            )

        # clip to max_stepsize
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
                return FirthResult(  # step-halving failed, return early
                    beta=beta,
                    loglik=q.loglik,
                    fisher_info=q.fisher_info,
                    n_iter=iteration,
                    converged=False,
                )
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
