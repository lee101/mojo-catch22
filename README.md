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
calls and include public-API conversion for both libraries. Both `mojo-catch22`
columns come from that task: `before` runs it against the kernel library saved
before the unroll pass (`MOJO_CATCH22_LIB=... pixi run bench`), `after` against
the current build. Each cell is the median of fifteen runs per function, and the
speedup column is computed from the `before` and `after` cells next to it.

| case | mojo-catch22 before | mojo-catch22 after | pycatch22 | speedup vs pycatch22 |
| --- | ---: | ---: | ---: | ---: |
| DN_Mean (2M) | 1.50 ms | 1.36 ms | 127.97 ms | 85.31-94.10x faster |
| DN_Spread_Std (2M) | 3.32 ms | 2.69 ms | 116.82 ms | 35.19-43.43x faster |
| CO_trev_1_num (2M) | 19.69 ms | 17.44 ms | 213.14 ms | 10.82-12.22x faster |
| MD_hrv_classic_pnn40 (2M) | 19.97 ms | 18.32 ms | 159.98 ms | 8.01-8.73x faster |
| CO_HistogramAMI_even_2_5 (200k) | 1.61 ms | 1.61 ms | 14.57 ms | 9.05x faster |
| CO_f1ecac (4,096) | 0.11 ms | 0.11 ms | 2.94 ms | 26.73x faster |
| DN_HistogramMode_5 (200k) | 3.89 ms | 2.03 ms | 7.83 ms | 2.01-3.86x faster |
| DN_HistogramMode_10 (200k) | 6.00 ms | 2.05 ms | 7.42 ms | 1.24-3.62x faster |
| SB_BinaryStats_mean_longstretch1 (2M) | 24.00 ms | 22.23 ms | 170.33 ms | 7.10-7.66x faster |
| SB_BinaryStats_diff_longstretch0 (2M) | 19.87 ms | 20.25 ms | 158.16 ms | 7.81-7.96x faster |
| FC_LocalSimple_mean1_tauresrat (4,096) | 0.17 ms | 0.13 ms | 5.79 ms | 34.06-44.54x faster |

This machine is shared, so end-to-end numbers drift by tens of percent between
runs. `MD_hrv_classic_pnn40` and `SB_BinaryStats_diff_longstretch0` in
particular vary by more than the change does, which is why their `before` and
`after` cells can invert direction from run to run. A second comparison on
identical buffers in one process (native kernels only, z-score precomputed)
is the steadier measure of kernel work:

| native kernel | input | before | after |
| --- | --- | ---: | ---: |
| DN_Mean | 2M | 0.610 ms | 0.490 ms |
| DN_Spread_Std | 2M | 1.615 ms | 1.004 ms |
| CO_trev_1_num | 2M | 1.926 ms | 1.667 ms |
| MD_hrv_classic_pnn40 | 2M | 1.806 ms | 1.782 ms |
| DN_HistogramMode_5 | 200k | 2.525 ms | 0.711 ms |
| DN_HistogramMode_10 | 200k | 4.704 ms | 0.708 ms |
| CO_HistogramAMI_even_2_5 | 200k | 0.896 ms | 0.898 ms |

All seven features that use the changed kernels, called together the way
`catch22_all` does (numpy z-score included), measure, per size, `after` then
`before`:

| input | after | before |
| --- | ---: | ---: |
| n=4,096 | 0.13 ms | 0.28 ms |
| n=200k | 4.81 ms | 10.73 ms |
| n=2M | 50.4 ms | 111.7 ms |

`pixi run bench` now also covers `DN_HistogramMode_5/10`,
`SB_BinaryStats_mean_longstretch1`, `SB_BinaryStats_diff_longstretch0` and
`FC_LocalSimple_mean1_tauresrat`, and `pixi run bench -- --kernel` prints the
native-kernel table with the current build filled in.

## How it works

The Mojo kernels live in `src/capi.mojo` and are built as one shared library.
Contiguous reductions use the host's native Float64 SIMD width with scalar tails
and four independent partial accumulators, combined once after the loop so the
adds do not serialise on floating-point latency. `histogram_mode` counts every
bin in a single pass instead of re-scanning the series once per bin.
The Python wrapper validates the array, retains its contiguous `float64` buffer
for the whole `ctypes` call, and passes its address and element count to the C
ABI. The AMI kernel applies z-score scaling to its bin boundaries instead of
allocating a normalized copy, and classifies bins with SIMD before reducing its
25 counters. Kernels do not allocate or retain the caller's memory.
Autocorrelation is calculated directly over that buffer; unsupported upstream
features are not silently substituted.

The multi-accumulator reductions change the order of floating-point summation,
so reductions can differ from an earlier build, or from pycatch22, in the last
bits. Test tolerances are unchanged. Two things this pass did *not* do:

- `DN_HistogramMode_5` and `DN_HistogramMode_10` remain identical to an earlier
  build. In theory the single counting pass can disagree with the old per-bin
  re-scan by one bin, but only if a sample sat exactly on a bin boundary where
  the two forms of the index expression round differently. A 4,000-case random
  sweep, and the whole suite, found no such value. Quantised input does not
  reproduce the cases where these two kernels differ from pycatch22 (they
  already differed from it on such input before this pass, and the two builds
  agree with each other there).
- No new vector form was introduced this pass. Every kernel keeps the SIMD shape
  it already had, including `CO_trev_1_num` and the SIMD slice of `pnn40`;
  `pnn40`'s scalar tail still counts its comparison in scalar form. The change
  is in loop structure, accumulator count and pass count. Broader SIMD work is
  the next phase.

GPU offload is intentionally omitted. The covered kernels are streaming
reductions, comparisons, and small histograms with arithmetic intensity below
the roughly 2-flop-per-byte point where device transfer and launch overhead can
be justified.

## License

[MIT](LICENSE)
