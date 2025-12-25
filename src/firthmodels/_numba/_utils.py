import numpy as np
from numba import njit


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
