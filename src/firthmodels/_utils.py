import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Protocol


@dataclass
class FirthResult:
    """Output from Firth-penalized optimization"""

    beta: NDArray[np.float64]  # (n_features,) fitted coefficients
    loglik: float  # fitted log-likelihood
    fisher_info: NDArray[
        np.float64
    ]  # (n_features, n_features) Fisher information matrix
    n_iter: int  # number of iterations
    converged: bool  # whether optimization converged


class IterationQuantities(Protocol):
    """Quantities computed at each iteration of optimization"""

    loglik: float
    modified_score: NDArray[np.float64]
    fisher_info: NDArray[np.float64]
