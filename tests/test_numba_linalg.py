import numpy as np
import pytest

from firthmodels import NUMBA_AVAILABLE

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")

if NUMBA_AVAILABLE:
    from firthmodels._numba.linalg import (
        dgemm,
        dgemv,
        dgetrf,
        dgetri,
        dgetrs,
        dpotrf,
        dpotri,
        dpotrs,
        dsyrk,
    )

from firthmodels._blas_abi import BLAS_INT_DTYPE


class TestBLAS:
    def test_dsyrk(self):
        rng = np.random.default_rng(0)
        A = np.asfortranarray(rng.standard_normal((5, 5)))
        C = np.zeros((5, 5), dtype=np.float64, order="F")

        # numpy's A @ A.T does dsyrk and then reflects to make symmetric
        dsyrk(A, C)
        C = np.tril(C) + np.tril(C, -1).T
        np.testing.assert_allclose(C, A @ A.T, rtol=1e-14)

    def test_dgemm(self):
        rng = np.random.default_rng(0)
        A = np.asfortranarray(rng.standard_normal((3, 4)))
        B = np.asfortranarray(rng.standard_normal((4, 5)))
        C = np.zeros((3, 5), dtype=np.float64, order="F")

        dgemm(A, B, C)
        np.testing.assert_allclose(C, A @ B, rtol=1e-14)

    def test_dgemv(self):
        rng = np.random.default_rng(0)
        A = np.asfortranarray(rng.standard_normal((4, 5)))
        x = np.ascontiguousarray(rng.standard_normal(5))
        y = np.zeros(4, dtype=np.float64)

        dgemv(A, x, y)
        np.testing.assert_allclose(y, A @ x, rtol=1e-14)


class TestLAPACK:
    def test_dpotrf_dpotri(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((4, 4))
        M = M @ M.T + np.eye(4)  # positive definite
        A = np.asfortranarray(M.copy())

        # dpotrf: A -> L
        info = dpotrf(A)
        assert info == 0
        L = np.tril(A)
        np.testing.assert_allclose(L @ L.T, M, rtol=1e-14)

        # dpotrs: solve M @ x = b using L
        b = rng.standard_normal((4, 1))
        b = np.asfortranarray(b.copy())
        b_orig = b.copy()
        info = dpotrs(A, b)  # A is still L
        assert info == 0
        np.testing.assert_allclose(M @ b, b_orig, rtol=1e-14)

        # dpotri: L -> inv(M)
        info = dpotri(A)
        assert info == 0
        A = np.tril(A) + np.tril(A, -1).T
        np.testing.assert_allclose(A, np.linalg.inv(M), rtol=1e-14)

    def test_dgetrf_dgetrs(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((4, 4))
        M = M + np.eye(4) * 2  # ensure non-singular
        A = np.asfortranarray(M.copy())
        ipiv = np.zeros(4, dtype=BLAS_INT_DTYPE)

        info = dgetrf(A, ipiv)
        assert info == 0

        b = rng.standard_normal((4, 2))
        b = np.asfortranarray(b.copy())
        b_orig = b.copy()
        info = dgetrs(A, ipiv, b)
        assert info == 0

        np.testing.assert_allclose(M @ b, b_orig, rtol=1e-14)
