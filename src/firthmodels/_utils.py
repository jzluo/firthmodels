import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray


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
