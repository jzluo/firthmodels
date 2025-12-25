import ctypes

import numpy as np
import pytest

from firthmodels import _blas_abi


def test_extract_int_token_matches_samples():
    sig_int = "void (char *, int *, double *, int *)"
    sig_long = "void (char *, long *, double *, long *)"
    sig_long_long = "void (char *, long long int *, double *, long long *)"

    assert _blas_abi._extract_int_token(sig_int) == "int"
    assert _blas_abi._extract_int_token(sig_long) == "long"
    assert _blas_abi._extract_int_token(sig_long_long) == "long long"


def test_blas_int_dtype_matches_ctype_size():
    dtype_size = np.dtype(_blas_abi.BLAS_INT_DTYPE).itemsize
    ctype_size = ctypes.sizeof(_blas_abi.BLAS_INT_CTYPE)
    assert dtype_size == ctype_size
    assert _blas_abi.BLAS_FLAG_DTYPE == np.uint8


def test_blas_int_token_matches_scipy_capsule():
    pytest.importorskip("scipy.linalg.cython_blas")
    sig = _blas_abi._capsule_signature("scipy.linalg.cython_blas", "dgemm")
    if not sig:
        pytest.skip("SciPy capsule signature unavailable")
    token = _blas_abi._extract_int_token(sig)
    if token is None:
        pytest.skip("Could not parse SciPy capsule signature")
    assert token == _blas_abi.BLAS_INT_TOKEN
