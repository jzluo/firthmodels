import numpy as np
from numba import njit
from numpy.typing import NDArray

from firthmodels._numba.blas_abi import BLAS_INT_DTYPE
from firthmodels._numba.linalg import (
    _alloc_f_order,
    dgemm,
    dgemv,
    dgetrf,
    dgetrs,
    dpotrf,
    dpotri,
    dpotrs,
    dsyrk,
    symmetrize_lower,
)

# Solver exit status codes
_STATUS_CONVERGED = 0
_STATUS_STEP_HALVING_FAILED = 1
_STATUS_MAX_ITER = 2
