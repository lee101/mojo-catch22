# mojo-catch22

`mojo-catch22` is a standalone Mojo port of a subset of
[pycatch22](https://pypi.org/project/pycatch22/). It provides canonical
univariate time-series characteristics through a small native library and a
Python API with the upstream function names.

## Coverage

Supported: `DN_HistogramMode_5`, `DN_HistogramMode_10`, `CO_f1ecac`,
`CO_FirstMin_ac`, `CO_HistogramAMI_even_2_5`, `CO_trev_1_num`,
`MD_hrv_classic_pnn40`, `SB_BinaryStats_mean_longstretch1`,
`SB_BinaryStats_diff_longstretch0`, `FC_LocalSimple_mean1_tauresrat`, and
`FC_LocalSimple_mean3_stderr`. `DN_Mean` and `DN_Spread_Std` are additionally
available when `catch24=True`.

`catch22_all` returns only that supported subset, in canonical order. It is not
a full replacement for upstream catch22. It does not include transition-matrix,
periodicity, embedding-distance, Gaussian mutual-information, outlier,
spectral, motif, or fluctuation-analysis characteristics.

Every supported public characteristic is compared with pycatch22 in the test
suite, along with bundle order and boundary validation.

## Install and use

```bash
pixi install
pixi run build
pixi run python - <<'PY'
import numpy as np
import mojocatch22 as catch22

x = np.sin(np.linspace(0, 16, 512, dtype=np.float64))
print(catch22.CO_f1ecac(x))
print(catch22.catch22_all(x, catch24=True)["names"])
PY
```

Inputs must be one-dimensional NumPy `float64` arrays. Non-contiguous arrays
are safely copied for the call. Inputs must be finite; z-scored characteristics
also require a non-constant signal and enough samples for their calculation.

## Benchmark

Measured by `pixi run bench` on this checkout. Timings are the best of three
calls and include public-API conversion for both libraries. The `before` column
runs the same task against the library saved from the build before this pass
(`MOJO_CATCH22_LIB=... pixi run bench`), `after` against the current build. The
machine is shared, so the before and after runs below are one run each, not
medians of a batch: read the multipliers as one-decimal-scale, and the kernel
table underneath as the steadier measure.

| case | mojo-catch22 before | mojo-catch22 after | pycatch22 | after vs before | after vs pycatch22 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DN_Mean (2M) | 1.32 ms | 1.39 ms | 131.12 ms | 0.95x | 94.49x faster |
| DN_Spread_Std (2M) | 2.70 ms | 2.24 ms | 132.05 ms | 1.21x | 58.99x faster |
| CO_trev_1_num (2M) | 19.21 ms | 15.14 ms | 214.77 ms | 1.27x | 14.18x faster |
| MD_hrv_classic_pnn40 (2M) | 16.19 ms | 15.87 ms | 168.94 ms | 1.02x | 10.64x faster |
| CO_HistogramAMI_even_2_5 (200k) | 1.59 ms | 1.45 ms | 17.06 ms | 1.10x | 11.75x faster |
| CO_f1ecac (4,096) | 0.12 ms | 0.10 ms | 3.13 ms | 1.20x | 31.81x faster |
| DN_HistogramMode_5 (200k) | 1.91 ms | 1.87 ms | 10.29 ms | 1.02x | 5.49x faster |
| DN_HistogramMode_10 (200k) | 1.90 ms | 1.90 ms | 10.30 ms | 1.00x | 5.42x faster |
| SB_BinaryStats_mean_longstretch1 (2M) | 19.91 ms | 20.16 ms | 182.24 ms | 0.99x | 9.04x faster |
| SB_BinaryStats_diff_longstretch0 (2M) | 18.45 ms | 18.72 ms | 168.05 ms | 0.99x | 8.98x faster |
| FC_LocalSimple_mean1_tauresrat (4,096) | 0.17 ms | 0.11 ms | 5.75 ms | 1.55x | 51.59x faster |

`DN_Mean`, `CO_trev_1_num`, `MD_hrv_classic_pnn40` and the two longstretch
features are timed through the public API on the *unnormalised* input, so a
large part of each cell is numpy's z-score in `_zscore`, which this pass does not
touch; the kernel-only table below is what reflects the kernels. The longstretch
features in particular show no end-to-end change even though their kernel is
~1.2x faster, because the rest of the call dominates.

Kernel-only, on identical buffers in one process, both libraries alternating
best-of-21 (`MOJO_CATCH22_BEFORE=... pixi run python bench/ab_kernels.py`):

| kernel | input | before | after | speedup | result agreement |
| --- | ---: | ---: | ---: | ---: | --- |
| DN_Mean | 2,000,000 | 489.5 us | 489.9 us | 1.00x | bitwise |
| DN_Spread_Std | 2,000,000 | 997.9 us | 511.4 us | 1.95x | within 2.1e-15 |
| CO_trev_1_num | 2,000,000 | 1.7 ms | 0.7 ms | 2.36x | bitwise |
| MD_hrv_classic_pnn40 | 2,000,000 | 1.8 ms | 0.8 ms | 2.37x | bitwise |
| DN_HistogramMode_5 | 200,000 | 701.4 us | 580.7 us | 1.21x | bitwise |
| DN_HistogramMode_10 | 200,000 | 689.2 us | 582.9 us | 1.18x | bitwise |
| SB_BinaryStats_mean_longstretch1 | 2,000,000 | 4.7 ms | 3.9 ms | 1.21x | bitwise |
| SB_BinaryStats_diff_longstretch0 | 2,000,000 | 4.3 ms | 3.4 ms | 1.27x | bitwise |
| CO_HistogramAMI_even_2_5 | 200,000 | 895.6 us | 708.3 us | 1.26x | bitwise |
| CO_f1ecac | 4,096 | 32.7 us | 6.3 us | 5.23x | bitwise |
| CO_FirstMin_ac | 4,096 | 65.5 us | 12.1 us | 5.39x | bitwise |
| FC_LocalSimple_mean1_tauresrat | 4,096 | 88.7 us | 31.7 us | 2.80x | bitwise |
| FC_LocalSimple_mean3_stderr | 200,000 | 795.7 us | 120.8 us | 6.59x | within 2.2e-15 |

`DN_Spread_Std` and `FC_LocalSimple_mean3_stderr` no longer produce bit-identical
values because their sums are now accumulated raw and centred at the end; the
difference is 2e-15 relative, far inside the 2e-8 test tolerance. Everything else
is bitwise identical to the previous build.

## How it works

The Mojo kernels live in `src/capi.mojo` and are built as one shared library.
Contiguous reductions use the host's native Float64 SIMD width (4 lanes here)
with scalar tails. The two-pass reductions — mean/std, the square-deviation sum
and `FC_LocalSimple_mean3_stderr` — add with four independent partial
accumulators. Where two stages read the same buffer, this pass fused them into
one pass:

- Mean and standard deviation are computed in a single pass. The mean is not
  known until the end, so raw `x` and `x*x` sums are accumulated together and
  the centred sum is recovered as `Σx² - 2mΣx + nm²`. `DN_Spread_Std` re-reads
  the series once instead of twice.
- `FC_LocalSimple_mean3_stderr` fused its mean pass and its residual pass over
  the same sliding window, using the same raw-moment trick.
- The finite-value scan that every kernel runs before starting is itself
  vectorised, and `histogram_mode` computes the range it needs with one combined
  minimum/maximum pass instead of two, as does AMI.
- Autocorrelation no longer recomputes the mean and the zero-lag divisor for
  every lag: `CO_f1ecac`, `CO_FirstMin_ac` and `FC_LocalSimple_mean1_tauresrat`
  compute them once per call and then run one fused numerator pass per lag.

`histogram_mode` counts every bin in a single pass instead of re-scanning the
series once per bin. The Python wrapper validates the array, retains its
contiguous `float64` buffer for the whole `ctypes` call, and passes its address
and element count to the C ABI. The AMI kernel applies z-score scaling to its
bin boundaries instead of allocating a normalized copy, and classifies bins with
SIMD before reducing its 25 counters. Kernels do not allocate or retain the
caller's memory. Autocorrelation is calculated directly over that buffer;
unsupported upstream features are not silently substituted.

The buffers are the ones the FFI is handed: `_lib.f64` runs the array through
`np.require(..., requirements=["C"])`, so every kernel sees a contiguous
`float64` array at least 8-byte aligned, and the SIMD loads are unaligned
(`unsafe_load[width=W]`), which needs no stronger promise. Every loop bound is
written so the vector body stays within the element count and the scalar tail
finishes the remainder.

The multi-accumulator and raw-moment reductions change the order of
floating-point summation. `DN_Spread_Std` and `FC_LocalSimple_mean3_stderr`
therefore differ from the previous build by about 2e-15 relative; every other
kernel is bitwise identical. Test tolerances are unchanged. Every SIMD kernel is
tested at lengths 0, 1, W-1, W, W+1 and 12345 (W = 4 on this machine) against
pycatch22, so both the full-vector body and the scalar tail are exercised.
Those tests also cover edge behaviour the existing suite never reached: every
kernel is now called at lengths 0, 1, W-1, W and W+1, and the wrappers reject
only the lengths they document. Where pycatch22 itself corrupts the heap (it
does on an empty input), this port raises `ValueError` instead.

This pass tried and **reverted** two further vectorisations because the
benchmark did not show a gain at real caller sizes:

- `SB_BinaryStats_mean_longstretch1` and `SB_BinaryStats_diff_longstretch0`:
  classifying lanes with a vector compare and a per-lane flag test compiled to a
  longer dependency chain than the committed scalar loop. A backward
  "largest matching index" vector scan was much faster in isolation but could
  not be made to reproduce the scalar reduction correctly under the pinned
  compiler, so it was dropped; the committed scalar loop is kept. What they do
  gain (about 1.2x) includes the vectorised finite-value scan they share with
  every other kernel, so it is not all attributable to these two loops.
- A branchless lane-select form of the same stretch update does not parse on
  the pinned compiler, so it was not retained.

GPU offload is intentionally omitted. The covered kernels are streaming
reductions, comparisons, and small histograms with arithmetic intensity below
the roughly 2-flop-per-byte point where device transfer and launch overhead can
be justified.

## License

[MIT](LICENSE)
