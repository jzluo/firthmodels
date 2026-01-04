"""
Linear algebra routines and BLAS/LAPACK wrappers for Numba.

Numba can't call scipy's BLAS/LAPACK wrappers directly in nopython mode, so we:
1. Register function pointers from scipy's Cython modules with llvmlite.
2. Define @intrinsic functions that emit LLVM calls to those symbols with Fortran-style ABI
   (ie all args passed by pointers)
3. Wrap the intrinsics in @njit functions with the expected signatures.

These wrappers can then be cached and inlined into higher-level Numba functions.
"""

import numpy as np
from llvmlite import binding as ll
from llvmlite import ir
from numba import njit
from numba.core import cgutils, types
from numba.extending import get_cython_function_address, intrinsic
from numpy.typing import NDArray

from firthmodels._numba.blas_abi import BLAS_FLAG_DTYPE, BLAS_INT_DTYPE


@njit(inline="always", cache=True)
def _alloc_f_order(rows: int, cols: int) -> NDArray[np.float64]:
    """Allocate F-contiguous array"""
    temp = np.empty((cols, rows), dtype=np.float64)
    return temp.T


@njit(inline="always", cache=True)
def symmetrize_lower(A: NDArray[np.float64]) -> None:
    """Copy lower triangle to upper, making A symmetric in-place."""
    n = A.shape[0]
    for j in range(1, n):
        for i in range(j):
            A[i, j] = A[j, i]


@njit(inline="always", cache=True)
def set_identity(A: NDArray[np.float64]) -> None:
    """Set A to identity matrix in-place."""
    n = A.shape[0]
    A[:, :] = 0.0
    for i in range(n):
        A[i, i] = 1.0


_SYMBOLS = {
    "firth_dsyrk": ("scipy.linalg.cython_blas", "dsyrk"),
    "firth_dgemm": ("scipy.linalg.cython_blas", "dgemm"),
    "firth_dgemv": ("scipy.linalg.cython_blas", "dgemv"),
    "firth_dpotrf": ("scipy.linalg.cython_lapack", "dpotrf"),
    "firth_dpotri": ("scipy.linalg.cython_lapack", "dpotri"),
    "firth_dpotrs": ("scipy.linalg.cython_lapack", "dpotrs"),
    "firth_dgetrf": ("scipy.linalg.cython_lapack", "dgetrf"),
    "firth_dgetri": ("scipy.linalg.cython_lapack", "dgetri"),
    "firth_dgetrs": ("scipy.linalg.cython_lapack", "dgetrs"),
    "firth_dgeqp3": ("scipy.linalg.cython_lapack", "dgeqp3"),
    "firth_dorgqr": ("scipy.linalg.cython_lapack", "dorgqr"),
    "firth_dpstrf": ("scipy.linalg.cython_lapack", "dpstrf"),
}

for symbol, (module, name) in _SYMBOLS.items():
    ll.add_symbol(symbol, get_cython_function_address(module, name))


def _external_call_codegen(symbol_name, context, builder, signature, args):
    """Emit LLVM call to a registered BLAS/LAPACK symbol using array data pointers."""
    ptrs = []
    for arg_type, arg in zip(signature.args, args):
        arr = context.make_array(arg_type)(context, builder, arg)
        ptrs.append(arr.data)
    fnty = ir.FunctionType(ir.VoidType(), [ptr.type for ptr in ptrs])
    fn = cgutils.get_or_insert_function(builder.module, fnty, symbol_name)
    builder.call(fn, ptrs)
    return context.get_dummy_value()


@intrinsic
def _dsyrk_call(typingctx, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    sig = types.void(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dsyrk", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgemm_call(
    typingctx, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
):
    sig = types.void(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgemm", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgemv_call(typingctx, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    sig = types.void(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgemv", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dpotrf_call(typingctx, uplo, n, A, lda, info):
    sig = types.void(uplo, n, A, lda, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dpotrf", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dpotri_call(typingctx, uplo, n, A, lda, info):
    sig = types.void(uplo, n, A, lda, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dpotri", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dpotrs_call(typingctx, uplo, n, nrhs, A, lda, B, ldb, info):
    sig = types.void(uplo, n, nrhs, A, lda, B, ldb, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dpotrs", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgetrf_call(typingctx, m, n, A, lda, ipiv, info):
    sig = types.void(m, n, A, lda, ipiv, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgetrf", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgetri_call(typingctx, n, A, lda, ipiv, work, lwork, info):
    sig = types.void(n, A, lda, ipiv, work, lwork, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgetri", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgetrs_call(typingctx, trans, n, nrhs, A, lda, ipiv, B, ldb, info):
    sig = types.void(trans, n, nrhs, A, lda, ipiv, B, ldb, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgetrs", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dgeqp3_call(typingctx, m, n, A, lda, jpvt, tau, work, lwork, info):
    sig = types.void(m, n, A, lda, jpvt, tau, work, lwork, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dgeqp3", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dorgqr_call(typingctx, m, n, k, A, lda, tau, work, lwork, info):
    sig = types.void(m, n, k, A, lda, tau, work, lwork, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dorgqr", context, builder, signature, args)

    return sig, codegen


@intrinsic
def _dpstrf_call(typingctx, uplo, n, A, lda, piv, rank, tol, work, info):
    sig = types.void(uplo, n, A, lda, piv, rank, tol, work, info)

    def codegen(context, builder, signature, args):
        return _external_call_codegen("firth_dpstrf", context, builder, signature, args)

    return sig, codegen


@njit(cache=True)
def dsyrk(A: np.ndarray, C: np.ndarray) -> None:
    """C = A @ A.T (lower triangle only). Overwrites C."""
    k = A.shape[0]
    n = A.shape[1]

    uplo = np.array([ord("L")], dtype=BLAS_FLAG_DTYPE)
    trans = np.array([ord("N")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([k], dtype=BLAS_INT_DTYPE)
    k_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    alpha = np.array([1.0], dtype=np.float64)
    lda = np.array([k], dtype=BLAS_INT_DTYPE)
    beta = np.array([0.0], dtype=np.float64)
    ldc = np.array([k], dtype=BLAS_INT_DTYPE)

    _dsyrk_call(uplo, trans, n_arr, k_arr, alpha, A, lda, beta, C, ldc)


@njit(cache=True)
def dgemm(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    """C = A @ B"""
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]

    transa = np.array([ord("N")], dtype=BLAS_FLAG_DTYPE)
    transb = np.array([ord("N")], dtype=BLAS_FLAG_DTYPE)
    m_arr = np.array([m], dtype=BLAS_INT_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    k_arr = np.array([k], dtype=BLAS_INT_DTYPE)
    alpha = np.array([1.0], dtype=np.float64)
    lda = np.array([m], dtype=BLAS_INT_DTYPE)
    ldb = np.array([k], dtype=BLAS_INT_DTYPE)
    beta = np.array([0.0], dtype=np.float64)
    ldc = np.array([m], dtype=BLAS_INT_DTYPE)

    _dgemm_call(
        transa, transb, m_arr, n_arr, k_arr, alpha, A, lda, B, ldb, beta, C, ldc
    )


@njit(cache=True)
def dgemv(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """y = A @ x"""
    m = A.shape[0]
    n = A.shape[1]
    trans = np.array([ord("N")], dtype=BLAS_FLAG_DTYPE)
    m_arr = np.array([m], dtype=BLAS_INT_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    alpha = np.array([1.0], dtype=np.float64)
    lda = np.array([m], dtype=BLAS_INT_DTYPE)
    incx = np.array([1], dtype=BLAS_INT_DTYPE)
    beta = np.array([0.0], dtype=np.float64)
    incy = np.array([1], dtype=BLAS_INT_DTYPE)

    _dgemv_call(trans, m_arr, n_arr, alpha, A, lda, x, incx, beta, y, incy)


@njit(cache=True)
def dpotrf(A: np.ndarray) -> int:
    """Cholesky: A = L @ L.T in-place (lower). Returns 0 on success."""
    n = A.shape[0]
    uplo = np.array([ord("L")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dpotrf_call(uplo, n_arr, A, lda, info)
    return int(info[0])


@njit(cache=True)
def dpotri(L: np.ndarray) -> int:
    """Invert from Cholesky factor in-place (lower). Returns 0 on success."""
    n = L.shape[0]
    uplo = np.array([ord("L")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dpotri_call(uplo, n_arr, L, lda, info)
    return int(info[0])


@njit(cache=True)
def dpotrs(L: np.ndarray, B: np.ndarray) -> int:
    """Solve L @ L.T @ X = B using Cholesky factor. B overwritten with X. Returns 0 on success."""
    n = L.shape[0]
    nrhs = B.shape[1]
    uplo = np.array([ord("L")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    nrhs_arr = np.array([nrhs], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    ldb = np.array([n], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dpotrs_call(uplo, n_arr, nrhs_arr, L, lda, B, ldb, info)
    return int(info[0])


@njit(cache=True)
def dgetrf(A: np.ndarray, ipiv: np.ndarray) -> int:
    """LU factorization with pivoting. Overwrites A, pivots in ipiv. Returns 0 on success."""
    m = A.shape[0]
    n = A.shape[1]
    m_arr = np.array([m], dtype=BLAS_INT_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([m], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dgetrf_call(m_arr, n_arr, A, lda, ipiv, info)
    return int(info[0])


@njit(cache=True)
def dgetri(A: np.ndarray, ipiv: np.ndarray, work: np.ndarray) -> int:
    """Invert from LU factors in-place. Returns 0 on success."""
    n = A.shape[0]
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    lwork = np.array([work.shape[0]], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dgetri_call(n_arr, A, lda, ipiv, work, lwork, info)
    return int(info[0])


@njit(cache=True)
def dgetrs(A: np.ndarray, ipiv: np.ndarray, B: np.ndarray) -> int:
    """Solve A @ X = B using LU factors. B overwritten with X. Returns 0 on success."""
    n = A.shape[0]
    nrhs = B.shape[1]
    trans = np.array([ord("N")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    nrhs_arr = np.array([nrhs], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    ldb = np.array([n], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dgetrs_call(trans, n_arr, nrhs_arr, A, lda, ipiv, B, ldb, info)
    return int(info[0])


@njit(cache=True)
def dgeqp3(A: np.ndarray, jpvt: np.ndarray, tau: np.ndarray, work: np.ndarray) -> int:
    """QR with column pivoting: A[:,jpvt] = Q @ R. Overwrites A."""
    m = A.shape[0]
    n = A.shape[1]
    m_arr = np.array([m], dtype=BLAS_INT_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([m], dtype=BLAS_INT_DTYPE)
    lwork = np.array([work.shape[0]], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dgeqp3_call(m_arr, n_arr, A, lda, jpvt, tau, work, lwork, info)
    return int(info[0])


@njit(cache=True)
def dorgqr(A: np.ndarray, tau: np.ndarray, work: np.ndarray, k: int) -> int:
    """Get Q from QR factorization. A overwritten with first k columns of Q."""
    m = A.shape[0]
    n = A.shape[1]
    m_arr = np.array([m], dtype=BLAS_INT_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    k_arr = np.array([k], dtype=BLAS_INT_DTYPE)
    lda = np.array([m], dtype=BLAS_INT_DTYPE)
    lwork = np.array([work.shape[0]], dtype=BLAS_INT_DTYPE)
    info = np.array([0], dtype=BLAS_INT_DTYPE)

    _dorgqr_call(m_arr, n_arr, k_arr, A, lda, tau, work, lwork, info)
    return int(info[0])


@njit(cache=True)
def dpstrf(A: np.ndarray, piv: np.ndarray, tol: float) -> tuple[int, int]:
    """Pivoted Cholesky: P.T @ A @ P = L @ L.T. Overwrites A with L.

    Returns (rank, info)
    """
    n = A.shape[0]
    uplo = np.array([ord("L")], dtype=BLAS_FLAG_DTYPE)
    n_arr = np.array([n], dtype=BLAS_INT_DTYPE)
    lda = np.array([n], dtype=BLAS_INT_DTYPE)
    rank_arr = np.array([0], dtype=BLAS_INT_DTYPE)
    tol_arr = np.array([tol], dtype=np.float64)
    info = np.array([0], dtype=BLAS_INT_DTYPE)
    work = np.empty(2 * n, dtype=np.float64)

    _dpstrf_call(uplo, n_arr, A, lda, piv, rank_arr, tol_arr, work, info)
    return int(rank_arr[0]), int(info[0])
