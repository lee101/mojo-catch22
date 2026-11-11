"""A/B the native kernels against a saved build, on identical buffers.

Run with `MOJO_CATCH22_BEFORE=/path/to/old/libmojo-catch22.so pixi run python
bench/ab_kernels.py`. Each kernel is called on the same numpy buffer by both
libraries, alternating best-of-N in one process, so the numbers isolate kernel
work instead of Python conversion or machine drift. The output is the
before/after table in the README; lengths include the SIMD tail cases.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojocatch22 import _lib

BEFORE = os.environ.get("MOJO_CATCH22_BEFORE")
if not BEFORE:
    raise SystemExit("set MOJO_CATCH22_BEFORE to the shared library to compare against")


def load(path):
    library = ctypes.CDLL(path)
    for name, (argtypes, restype) in _lib._SIGNATURES.items():
        fn = getattr(library, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return library


def best(fn, pointer, size, args=(), repeat=21):
    def run():
        fn(pointer, size, *args)

    run()
    fastest = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        run()
        fastest = min(fastest, time.perf_counter() - start)
    return fastest


def main():
    before = load(BEFORE)
    after = load(_lib.LIB)
    rng = np.random.default_rng(0)
    large = np.ascontiguousarray(rng.normal(size=2_000_000))
    small = np.ascontiguousarray(rng.normal(size=200_000))
    auto = np.ascontiguousarray(rng.normal(size=4_096))
    zlarge = np.ascontiguousarray((large - large.mean()) / large.std(ddof=1))
    zsmall = np.ascontiguousarray((small - small.mean()) / small.std(ddof=1))
    zauto = np.ascontiguousarray((auto - auto.mean()) / auto.std(ddof=1))
    ami_scale = float(small.std(ddof=1))

    cases = (
        ("DN_Mean", "mcc_mean", large, ()),
        ("DN_Spread_Std", "mcc_std", large, ()),
        ("CO_trev_1_num", "mcc_trev", zlarge, ()),
        ("MD_hrv_classic_pnn40", "mcc_pnn40", zlarge, ()),
        ("DN_HistogramMode_5", "mcc_histogram_mode_5", zsmall, ()),
        ("DN_HistogramMode_10", "mcc_histogram_mode_10", zsmall, ()),
        ("SB_BinaryStats_mean_longstretch1", "mcc_mean_longstretch1", zlarge, ()),
        ("SB_BinaryStats_diff_longstretch0", "mcc_diff_longstretch0", zlarge, ()),
        ("CO_HistogramAMI_even_2_5", "mcc_histogram_ami_raw", small, (ami_scale,)),
        ("CO_f1ecac", "mcc_f1ecac", zauto, ()),
        ("CO_FirstMin_ac", "mcc_first_min_ac", zauto, ()),
        ("FC_LocalSimple_mean1_tauresrat", "mcc_mean1_tauresrat", zauto, ()),
        ("FC_LocalSimple_mean3_stderr", "mcc_mean3_stderr", zsmall, ()),
    )
    print("| kernel | input | before | after | speedup | result |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for label, symbol, buffer, args in cases:
        pointer = ctypes.c_void_p(buffer.ctypes.data)
        old = best(getattr(before, symbol), pointer, buffer.size, args)
        new = best(getattr(after, symbol), pointer, buffer.size, args)
        old_value = getattr(before, symbol)(pointer, buffer.size, *args)
        new_value = getattr(after, symbol)(pointer, buffer.size, *args)
        if old_value == new_value or (old_value != old_value and new_value != new_value):
            agree = "bitwise"
        elif abs(old_value - new_value) <= 1e-9 * max(abs(old_value), 1.0):
            agree = f"within {abs(old_value - new_value):.1e}"
        else:
            agree = f"differs by {abs(old_value - new_value):.3g}"
        scale = 1e6 if old < 1e-3 else 1e3
        unit = "us" if old < 1e-3 else "ms"
        print(
            f"| {label} | {buffer.size:,} | {old * scale:.1f} {unit} | "
            f"{new * scale:.1f} {unit} | {old / new:.2f}x | {agree} |"
        )


if __name__ == "__main__":
    main()
