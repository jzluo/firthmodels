import numpy as np
from numpy.typing import NDArray
from typing import Callable

from firthmodels._utils import FirthResult


def newton_raphson(
    compute_quantities: Callable[[NDArray]],
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

    for iteration in range(1, max_iter + 1):
        q = compute_quantities(beta)

        # check convergence: max|U*| < tol
        if np.max(np.abs(q.modified_score)) < tol:
            return FirthResult(
                beta=beta,
                loglik=q.loglik,
                fisher_info=q.fisher_info,
                n_iter=iteration,
                converged=True,
            )
