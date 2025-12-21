import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy
from numpy.typing import NDArray

from firthmodels._solvers import newton_raphson
from firthmodels._utils import IterationQuantities


@dataclass(frozen=True)
class LRTResult:
    chi2: float
    pvalue: float
    bse_backcorrected: float


@dataclass
class _SlicedQuantities:
    loglik: float
    modified_score: NDArray[np.float64]
    fisher_info: NDArray[np.float64]


def constrained_lrt_1df(
    *,
    idx: int,
    beta_hat_full: NDArray[np.float64],
    loglik_full: float,
    compute_quantities_full: Callable[[NDArray[np.float64]], IterationQuantities],
    max_iter: int,
    max_step: float,
    max_halfstep: int,
    gtol: float,
    xtol: float,
) -> LRTResult:
    """
    Compute penalized likelihood ratio test for a single coefficient.

    Fits a constrained model with `beta[idx]=0`, then computes the LRT statistic
    comparing the full model to the constrained model. Computes LRT p-value and
    back-corrected standard error. Ensures (beta/SE)^2 = chi2, useful for meta-analysis
    where studies are weighted by 1/SE^2.

    Parameters
    ----------
    idx : int
        Index of the coefficient to test.
    beta_hat_full : ndarray of shape (k,)
        Fitted coefficients from the full model.
    loglik_full : float
        Penalized log-likelihood of the full model.
    compute_quantities_full : callable
        Function that takes a beta vector of shape (k,) and returns loglik,
        modified_score, and fisher_info for the full model.
    max_iter : int
        Maximum Newton-Raphson iterations for constrained fit.
    max_step : float
        Maximum step size per coefficient.
    max_halfstep : int
        Maximum step-halvings per iteration.
    gtol : float
        Gradient convergence tolerance.
    xtol : float
        Parameter convergence tolerance.

    Returns
    -------
    LRTResult
        chi2 : float
            Likelihood ratio test statistic, 2 * (loglik_full - loglik_constrained).
        pvalue : float
            p-value from chi-squared distribution with 1 degree of freedom.
        bse_backcorrected : float
            Back-corrected standard error, |beta[idx]| / sqrt(chi2).
    """
    k = beta_hat_full.shape[0]
    free_indices = [i for i in range(k) if i != idx]

    # pre-allocate index arrays
    beta_full = np.zeros(k, dtype=np.float64)
    free_idx_array = np.array(free_indices, dtype=np.intp)
    ix_grid = np.ix_(free_indices, free_indices)

    def constrained_quantities(beta_free: NDArray[np.float64]) -> _SlicedQuantities:
        # update beta_full in-place
        beta_full[:idx] = beta_free[:idx]
        beta_full[idx] = 0.0
        if idx < k - 1:
            beta_full[idx + 1 :] = beta_free[idx:]
        q = compute_quantities_full(beta_full)
        return _SlicedQuantities(
            loglik=q.loglik,
            modified_score=q.modified_score[free_idx_array],
            fisher_info=q.fisher_info[ix_grid],
        )

    reduced_result = newton_raphson(
        compute_quantities=constrained_quantities,
        n_features=k - 1,
        max_iter=max_iter,
        max_step=max_step,
        max_halfstep=max_halfstep,
        gtol=gtol,
        xtol=xtol,
    )

    chi2 = max(0.0, 2.0 * (loglik_full - reduced_result.loglik))
    pval = scipy.stats.chi2.sf(chi2, df=1)

    # back-corrected SE: |beta|/sqrt(chi2), ensures (beta/SE)^2 = chi2
    bse = abs(beta_hat_full[idx]) / math.sqrt(chi2) if chi2 > 0 else math.inf
    return LRTResult(chi2=chi2, pvalue=pval, bse_backcorrected=bse)
