import ctypes

import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
from numpy.typing import NDArray


@njit(inline="always")
def _alloc_f_order(rows: int, cols: int) -> NDArray[np.float64]:
    temp = np.empty((cols, rows), dtype=np.float64)
    return temp.T


# https://numba.discourse.group/t/extending-lapack-functions-into-numba/1394/4

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_int = ctypes.c_int64

_ptr_dbl = _PTR(_dble)
_ptr_int = _PTR(_int)

# BLAS/LAPACK pointers (scipy provides the cython-level wrappers).
_dsyrk_addr = get_cython_function_address("scipy.linalg.cython_blas", "dsyrk")
_dgemm_addr = get_cython_function_address("scipy.linalg.cython_blas", "dgemm")
_dpotrf_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dpotrf")
_dpotri_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dpotri")
_dpotrs_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dpotrs")
_dgetrf_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dgetrf")
_dgetri_addr = get_cython_function_address("scipy.linalg.cython_lapack", "dgetri")

_dsyrk_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # uplo
    _ptr_int,  # trans
    _ptr_int,  # n
    _ptr_int,  # k
    _ptr_dbl,  # alpha
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_dbl,  # beta
    _ptr_dbl,  # c
    _ptr_int,  # ldc
)
_dsyrk = _dsyrk_functype(_dsyrk_addr)

_dgemm_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # transa
    _ptr_int,  # transb
    _ptr_int,  # m
    _ptr_int,  # n
    _ptr_int,  # k
    _ptr_dbl,  # alpha
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_dbl,  # b
    _ptr_int,  # ldb
    _ptr_dbl,  # beta
    _ptr_dbl,  # c
    _ptr_int,  # ldc
)
_dgemm = _dgemm_functype(_dgemm_addr)

_dpotrf_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # uplo
    _ptr_int,  # n
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_int,  # info
)
_dpotrf = _dpotrf_functype(_dpotrf_addr)

_dpotri_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # uplo
    _ptr_int,  # n
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_int,  # info
)
_dpotri = _dpotri_functype(_dpotri_addr)

_dpotrs_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # uplo
    _ptr_int,  # n
    _ptr_int,  # nrhs
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_dbl,  # b
    _ptr_int,  # ldb
    _ptr_int,  # info
)
_dpotrs = _dpotrs_functype(_dpotrs_addr)

_dgetrf_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # m
    _ptr_int,  # n
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_int,  # ipiv
    _ptr_int,  # info
)
_dgetrf = _dgetrf_functype(_dgetrf_addr)

_dgetri_functype = ctypes.CFUNCTYPE(
    None,
    _ptr_int,  # n
    _ptr_dbl,  # a
    _ptr_int,  # lda
    _ptr_int,  # ipiv
    _ptr_dbl,  # work
    _ptr_int,  # lwork
    _ptr_int,  # info
)
_dgetri = _dgetri_functype(_dgetri_addr)


@njit
def dsyrk(A: np.ndarray, C: np.ndarray) -> None:
    k = A.shape[0]
    n = A.shape[1]

    uplo = np.array([ord("L")], dtype=np.int64)
    trans = np.array([ord("N")], dtype=np.int64)
    n_arr = np.array([k], dtype=np.int64)
    k_arr = np.array([n], dtype=np.int64)
    alpha = np.array([1.0], dtype=np.float64)
    lda = np.array([k], dtype=np.int64)
    beta = np.array([0.0], dtype=np.float64)
    ldc = np.array([k], dtype=np.int64)

    _dsyrk(
        uplo.ctypes,
        trans.ctypes,
        n_arr.ctypes,
        k_arr.ctypes,
        alpha.ctypes,
        A.ctypes,
        lda.ctypes,
        beta.ctypes,
        C.ctypes,
        ldc.ctypes,
    )


@njit
def dgemm(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]

    transa = np.array([ord("N")], dtype=np.int64)
    transb = np.array([ord("N")], dtype=np.int64)
    m_arr = np.array([m], dtype=np.int64)
    n_arr = np.array([n], dtype=np.int64)
    k_arr = np.array([k], dtype=np.int64)
    alpha = np.array([1.0], dtype=np.float64)
    lda = np.array([m], dtype=np.int64)
    ldb = np.array([k], dtype=np.int64)
    beta = np.array([0.0], dtype=np.float64)
    ldc = np.array([m], dtype=np.int64)

    _dgemm(
        transa.ctypes,
        transb.ctypes,
        m_arr.ctypes,
        n_arr.ctypes,
        k_arr.ctypes,
        alpha.ctypes,
        A.ctypes,
        lda.ctypes,
        B.ctypes,
        ldb.ctypes,
        beta.ctypes,
        C.ctypes,
        ldc.ctypes,
    )


@njit
def dpotrf(A: np.ndarray) -> int:
    n = A.shape[0]
    uplo = np.array([ord("L")], dtype=np.int64)
    n_arr = np.array([n], dtype=np.int64)
    lda = np.array([n], dtype=np.int64)
    info = np.array([0], dtype=np.int64)

    _dpotrf(
        uplo.ctypes,
        n_arr.ctypes,
        A.ctypes,
        lda.ctypes,
        info.ctypes,
    )
    return int(info[0])


@njit
def dpotri(L: np.ndarray) -> int:
    n = L.shape[0]
    uplo = np.array([ord("L")], dtype=np.int64)
    n_arr = np.array([n], dtype=np.int64)
    lda = np.array([n], dtype=np.int64)
    info = np.array([0], dtype=np.int64)

    _dpotri(
        uplo.ctypes,
        n_arr.ctypes,
        L.ctypes,
        lda.ctypes,
        info.ctypes,
    )
    return int(info[0])


@njit
def dpotrs(L: np.ndarray, B: np.ndarray) -> int:
    n = L.shape[0]
    nrhs = B.shape[1]
    uplo = np.array([ord("L")], dtype=np.int64)
    n_arr = np.array([n], dtype=np.int64)
    nrhs_arr = np.array([nrhs], dtype=np.int64)
    lda = np.array([n], dtype=np.int64)
    ldb = np.array([n], dtype=np.int64)
    info = np.array([0], dtype=np.int64)

    _dpotrs(
        uplo.ctypes,
        n_arr.ctypes,
        nrhs_arr.ctypes,
        L.ctypes,
        lda.ctypes,
        B.ctypes,
        ldb.ctypes,
        info.ctypes,
    )
    return int(info[0])


@njit
def dgetrf(A: np.ndarray, ipiv: np.ndarray) -> int:
    m = A.shape[0]
    n = A.shape[1]
    m_arr = np.array([m], dtype=np.int64)
    n_arr = np.array([n], dtype=np.int64)
    lda = np.array([m], dtype=np.int64)
    info = np.array([0], dtype=np.int64)

    _dgetrf(
        m_arr.ctypes,
        n_arr.ctypes,
        A.ctypes,
        lda.ctypes,
        ipiv.ctypes,
        info.ctypes,
    )
    return int(info[0])


@njit
def dgetri(A: np.ndarray, ipiv: np.ndarray, work: np.ndarray) -> int:
    n = A.shape[0]
    n_arr = np.array([n], dtype=np.int64)
    lda = np.array([n], dtype=np.int64)
    lwork = np.array([work.shape[0]], dtype=np.int64)
    info = np.array([0], dtype=np.int64)

    _dgetri(
        n_arr.ctypes,
        A.ctypes,
        lda.ctypes,
        ipiv.ctypes,
        work.ctypes,
        lwork.ctypes,
        info.ctypes,
    )
    return int(info[0])
