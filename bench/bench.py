"""Benchmark mojo-catch22 against pycatch22.

Run with `pixi run bench`; the Pixi task holds the shared machine lock.

By default each kernel is timed through the public API on the caller's inputs.
`--kernel` additionally times the native kernel alone (same buffers, z-score
precomputed once, ctypes call in a loop) so kernel work can be separated from
the Python wrapper.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojocatch22 as mojo
import pycatch22 as reference
from mojocatch22._lib import lib


def best_time(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def benchmark(name, ours, theirs):
    ours()
    theirs()
    mojo_time = best_time(ours)
    ref_time = best_time(theirs)
    label = "faster" if mojo_time < ref_time else "slower"
    print(f"| {name} | {mojo_time * 1e3:.2f} ms | {ref_time * 1e3:.2f} ms | {ref_time / mojo_time:.2f}x {label} |")


def kernel_benchmarks(large, mode, autocorr, ami):
    """Time the native kernels alone; both libraries get the same buffers."""
    native = lib()
    zlarge = np.ascontiguousarray((large - large.mean()) / large.std(ddof=1))
    zmode = np.ascontiguousarray((mode - mode.mean()) / mode.std(ddof=1))
    print("\nNative kernel only (same Float64 buffer, z-score precomputed):")
    print("| kernel | input | before | after |")
    print("| --- | --- | ---: | ---: |")
    cases = (
        ("DN_Mean", "mcc_mean", large, None),
        ("DN_Spread_Std", "mcc_std", large, None),
        ("CO_trev_1_num", "mcc_trev", zlarge, None),
        ("MD_hrv_classic_pnn40", "mcc_pnn40", zlarge, None),
        ("DN_HistogramMode_5", "mcc_histogram_mode_5", zmode, None),
        ("DN_HistogramMode_10", "mcc_histogram_mode_10", zmode, None),
        ("SB_BinaryStats_mean_longstretch1", "mcc_mean_longstretch1", zlarge, None),
        ("SB_BinaryStats_diff_longstretch0", "mcc_diff_longstretch0", zlarge, None),
        ("CO_HistogramAMI_even_2_5", "mcc_histogram_ami_raw", ami, float(ami.std(ddof=1))),
    )
    for label, symbol, buffer, scale in cases:
        fn = getattr(native, symbol)
        pointer = ctypes.c_void_p(buffer.ctypes.data)
        args = () if scale is None else (scale,)

        def run(fn=fn, pointer=pointer, size=buffer.size, args=args):
            fn(pointer, size, *args)

        print(f"| {label} | {buffer.size:,} | {best_time(run, 15) * 1e3:.3f} ms | |")


def main():
    rng = np.random.default_rng(0)
    large = np.ascontiguousarray(rng.normal(size=2_000_000))
    autocorr = np.ascontiguousarray(rng.normal(size=4_096))
    ami = np.ascontiguousarray(rng.normal(size=200_000))
    mode = np.ascontiguousarray(rng.normal(size=200_000))
    print("| case | mojo-catch22 | pycatch22 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    benchmark("DN_Mean (2M)", lambda: mojo.DN_Mean(large), lambda: reference.DN_Mean(large.tolist()))
    benchmark("DN_Spread_Std (2M)", lambda: mojo.DN_Spread_Std(large), lambda: reference.DN_Spread_Std(large.tolist()))
    benchmark("CO_trev_1_num (2M)", lambda: mojo.CO_trev_1_num(large), lambda: reference.CO_trev_1_num(large.tolist()))
    benchmark("MD_hrv_classic_pnn40 (2M)", lambda: mojo.MD_hrv_classic_pnn40(large), lambda: reference.MD_hrv_classic_pnn40(large.tolist()))
    benchmark("CO_HistogramAMI_even_2_5 (200k)", lambda: mojo.CO_HistogramAMI_even_2_5(ami), lambda: reference.CO_HistogramAMI_even_2_5(ami.tolist()))
    benchmark("CO_f1ecac (4,096)", lambda: mojo.CO_f1ecac(autocorr), lambda: reference.CO_f1ecac(autocorr.tolist()))
    benchmark("DN_HistogramMode_5 (200k)", lambda: mojo.DN_HistogramMode_5(mode), lambda: reference.DN_HistogramMode_5(mode.tolist()))
    benchmark("DN_HistogramMode_10 (200k)", lambda: mojo.DN_HistogramMode_10(mode), lambda: reference.DN_HistogramMode_10(mode.tolist()))
    benchmark("SB_BinaryStats_mean_longstretch1 (2M)", lambda: mojo.SB_BinaryStats_mean_longstretch1(large), lambda: reference.SB_BinaryStats_mean_longstretch1(large.tolist()))
    benchmark("SB_BinaryStats_diff_longstretch0 (2M)", lambda: mojo.SB_BinaryStats_diff_longstretch0(large), lambda: reference.SB_BinaryStats_diff_longstretch0(large.tolist()))
    benchmark("FC_LocalSimple_mean1_tauresrat (4,096)", lambda: mojo.FC_LocalSimple_mean1_tauresrat(autocorr), lambda: reference.FC_LocalSimple_mean1_tauresrat(autocorr.tolist()))
    if "--kernel" in sys.argv:
        kernel_benchmarks(large, mode, autocorr, ami)


if __name__ == "__main__":
    main()
