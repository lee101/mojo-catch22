"""ctypes loading and ndarray conversion for the Mojo shared library."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_CATCH22_LIB") or os.path.join(ROOT, "dist", "libmojo-catch22.so")
_I = ctypes.c_int64
_P = ctypes.c_void_p
_F = ctypes.c_double

_SIGNATURES = {
    "mcc_mean": ([_P, _I], _F),
    "mcc_std": ([_P, _I], _F),
    "mcc_histogram_mode_5": ([_P, _I], _F),
    "mcc_histogram_mode_10": ([_P, _I], _F),
    "mcc_f1ecac": ([_P, _I], _F),
    "mcc_first_min_ac": ([_P, _I], _F),
    "mcc_histogram_ami": ([_P, _I], _F),
    "mcc_trev": ([_P, _I], _F),
    "mcc_pnn40": ([_P, _I], _F),
    "mcc_mean_longstretch1": ([_P, _I], _F),
    "mcc_diff_longstretch0": ([_P, _I], _F),
    "mcc_mean1_tauresrat": ([_P, _I], _F),
    "mcc_mean3_stderr": ([_P, _I], _F),
}


class BuildError(RuntimeError):
    pass


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise BuildError("compiled library is missing; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("catch22 characteristics require a one-dimensional time series")
    if arr.dtype != np.float64:
        raise TypeError("time series must have dtype float64; convert explicitly with numpy.asarray(data, dtype=numpy.float64)")
    if arr.size == 0:
        raise ValueError("time series must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("time series must contain only finite float64 values")
    # Mojo receives a mutable pointer type.  Require a private writable C-order
    # buffer even though current kernels are read-only.
    return np.require(arr, dtype=np.float64, requirements=["C", "W"])


def call(name: str, data) -> float:
    arr = f64(data)
    # arr remains strongly referenced through the native call; the C ABI receives
    # a pointer to an owned, contiguous float64 buffer and its element count.
    return float(getattr(lib(), name)(ctypes.c_void_p(arr.ctypes.data), arr.size))
