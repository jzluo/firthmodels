"""
Infer BLAS/LAPACK integer ABI from scipy Cython capsules.

Parses scipy's capsule signatures for BLAS/LAPACK symbols to infer the
integer type used by the linked libraries. If no signature is available, fall
back to "int".

Exports:
    BLAS_INT_TOKEN: detected token ("int", "long", "long long")
    BLAS_INT_CTYPE: ctypes type for BLAS integers
    BLAS_INT_DTYPE: numpy dtype for BLAS integers
    BLAS_FLAG_CTYPE: ctypes type for character flags (uint8)
    BLAS_FLAG_DTYPE: numpy dtype for character flags (uint8)
"""

from __future__ import annotations

import ctypes
import importlib
import re

import numpy as np

_PYCAPSULE_GETNAME = ctypes.pythonapi.PyCapsule_GetName
_PYCAPSULE_GETNAME.restype = ctypes.c_char_p
_PYCAPSULE_GETNAME.argtypes = [ctypes.py_object]

_CANDIDATES = (
    ("scipy.linalg.cython_blas", "dgemm"),
    ("scipy.linalg.cython_blas", "dsyrk"),
    ("scipy.linalg.cython_lapack", "dpotrf"),
)


def _capsule_signature(modname: str, funcname: str) -> str:
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return ""
    capi = getattr(mod, "__pyx_capi__", None)
    if not capi:
        return ""
    capsule = capi.get(funcname)
    if capsule is None:
        return ""
    sig = _PYCAPSULE_GETNAME(capsule)
    return sig.decode("utf-8") if sig else ""


def _extract_int_token(sig: str) -> str | None:
    if not sig:
        return None
    patterns = (
        ("long long", r"\blong\s+long(?:\s+int)?\s*\*"),
        ("long", r"\blong(?:\s+int)?\s*\*"),
        ("int", r"\bint\s*\*"),
    )
    for token, pattern in patterns:
        if re.search(pattern, sig):
            return token
    return None


def _detect_blas_int_token() -> str:
    tokens = []
    for modname, funcname in _CANDIDATES:
        token = _extract_int_token(_capsule_signature(modname, funcname))
        if token:
            tokens.append(token)
    if not tokens:
        return "int"
    unique = set(tokens)
    if len(unique) > 1:
        raise RuntimeError(f"Inconsistent BLAS/LAPACK int types: {sorted(unique)}")
    return tokens[0]


def _dtype_for_ctype(ctype: type[ctypes._SimpleCData]) -> type[np.integer]:
    size = ctypes.sizeof(ctype)
    if size == 4:
        return np.int32
    if size == 8:
        return np.int64
    raise RuntimeError(f"Unsupported BLAS int size: {size} bytes")


_INT_TOKEN_TO_CTYPE: dict[str, type[ctypes._SimpleCData]] = {
    "int": ctypes.c_int,
    "long": ctypes.c_long,
    "long long": ctypes.c_longlong,
}

BLAS_INT_TOKEN = _detect_blas_int_token()
BLAS_INT_CTYPE = _INT_TOKEN_TO_CTYPE[BLAS_INT_TOKEN]
BLAS_INT_DTYPE = _dtype_for_ctype(BLAS_INT_CTYPE)
BLAS_FLAG_DTYPE = np.uint8
BLAS_FLAG_CTYPE = ctypes.c_uint8
