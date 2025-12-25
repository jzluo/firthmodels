import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True)
def _expit(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


@njit(fastmath=True)
def _log1pexp(x: float) -> float:
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(fastmath=True)
def compute_logistic_quantities(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    offset: NDArray[np.float64],
    workspace: tuple[NDArray[np.float64], ...],  # eta, p, w, sqrt_w, etc
) -> tuple[float, int]:
    raise NotImplementedError


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
