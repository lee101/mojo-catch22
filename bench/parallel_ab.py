"""Paired serial vs task-parallel mean: why the parallel path was reverted.

Runs the committed serial kernel (`mcc_mean`) and the measurement-only
task-parallel path (`mcc_mean_taskparallel`, the reverted chunked mean)
alternately on the same buffer, so both see the same host conditions. Worker
counts 1/2/4/8 are all exercised: 1 means one worker for the whole buffer, 2 and
4 are the cap region, 8 is over the cap. Also checks that both paths agree.
Run under the shared machine lock:

    flock /tmp/mojo-bench.lock pixi run python bench/parallel_ab.py
"""

from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojocatch22._lib import lib

ROUNDS = int(os.environ.get("MCC_ROUNDS", "9"))
SIZES = [int(s) for s in os.environ.get("MCC_SIZES", "262144,1000000,2000000").split(",")]
WORKERS = [2, 4, 8]


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e6


def main():
    native = lib()
    print("| n | workers | serial | parallel | serial / parallel best | median | rounds won by parallel | max abs gap |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for n in SIZES:
        x = np.ascontiguousarray(np.random.default_rng(n).normal(size=n))
        pointer = ctypes.c_void_p(x.ctypes.data)
        serial_fn = lambda: native.mcc_mean(pointer, n)
        serial_value = serial_fn()
        serial = [timed(serial_fn) for _ in range(ROUNDS)]
        for workers in [1] + WORKERS:
            par_fn = lambda workers=workers: native.mcc_mean_taskparallel(pointer, n, workers)
            par = [timed(par_fn) for _ in range(ROUNDS)]
            value = par_fn()
            agree = "bitwise" if value == serial_value else f"{abs(value - serial_value):.2g}"
            ratios = [s / p for s, p in zip(serial, par)]
            if workers == 1:
                # One worker for the whole buffer is the serial kernel again;
                # there is no ratio to report, only the control measurement.
                print(
                    f"| {n:,} | 1 (control) | {min(serial) * 1e3:.1f} us | "
                    f"{min(par) * 1e3:.1f} us | - | - | - | {agree} |"
                )
                continue
            print(
                f"| {n:,} | {workers} | {min(serial) * 1e3:.1f} us | {min(par) * 1e3:.1f} us | "
                f"{max(ratios):.2f}x | {statistics.median(ratios):.2f}x | "
                f"{sum(1 for r in ratios if r > 1)}/{ROUNDS} | {agree} |"
            )


if __name__ == "__main__":
    main()
