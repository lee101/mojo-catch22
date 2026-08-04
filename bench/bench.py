"""Run with `pixi run bench`; the Pixi task holds the shared machine lock."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojocatch22 as mojo
import pycatch22 as reference


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


def main():
    rng = np.random.default_rng(0)
    large = np.ascontiguousarray(rng.normal(size=2_000_000))
    autocorr = np.ascontiguousarray(rng.normal(size=4_096))
    ami = np.ascontiguousarray(rng.normal(size=200_000))
    print("| case | mojo-catch22 | pycatch22 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    benchmark("DN_Mean (2M)", lambda: mojo.DN_Mean(large), lambda: reference.DN_Mean(large.tolist()))
    benchmark("DN_Spread_Std (2M)", lambda: mojo.DN_Spread_Std(large), lambda: reference.DN_Spread_Std(large.tolist()))
    benchmark("CO_trev_1_num (2M)", lambda: mojo.CO_trev_1_num(large), lambda: reference.CO_trev_1_num(large.tolist()))
    benchmark("MD_hrv_classic_pnn40 (2M)", lambda: mojo.MD_hrv_classic_pnn40(large), lambda: reference.MD_hrv_classic_pnn40(large.tolist()))
    benchmark("CO_HistogramAMI_even_2_5 (200k)", lambda: mojo.CO_HistogramAMI_even_2_5(ami), lambda: reference.CO_HistogramAMI_even_2_5(ami.tolist()))
    benchmark("CO_f1ecac (4,096)", lambda: mojo.CO_f1ecac(autocorr), lambda: reference.CO_f1ecac(autocorr.tolist()))


if __name__ == "__main__":
    main()
