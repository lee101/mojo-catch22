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
calls and include public-API conversion for both libraries.

| case | mojo-catch22 | pycatch22 | ratio |
| --- | ---: | ---: | ---: |
| DN_Mean (2M) | 1.50 ms | 153.68 ms | 102.48x faster |
| DN_Spread_Std (2M) | 3.29 ms | 160.08 ms | 48.69x faster |
| CO_trev_1_num (2M) | 22.09 ms | 260.58 ms | 11.80x faster |
| MD_hrv_classic_pnn40 (2M) | 22.71 ms | 188.25 ms | 8.29x faster |
| CO_HistogramAMI_even_2_5 (200k) | 1.60 ms | 18.56 ms | 11.63x faster |
| CO_f1ecac (4,096) | 0.11 ms | 2.94 ms | 25.87x faster |

## How it works

The Mojo kernels live in `src/capi.mojo` and are built as one shared library.
Contiguous reductions use the host's native Float64 SIMD width with scalar tails;
independent accumulators are reduced once after the vector loop.
The Python wrapper validates the array, retains its contiguous `float64` buffer
for the whole `ctypes` call, and passes its address and element count to the C
ABI. The AMI kernel applies z-score scaling to its bin boundaries instead of
allocating a normalized copy, and classifies bins with SIMD before reducing its
25 counters. Kernels do not allocate or retain the caller's memory.
Autocorrelation is calculated directly over that buffer; unsupported upstream
features are not silently substituted.

GPU offload is intentionally omitted. The covered kernels are streaming
reductions, comparisons, and small histograms with arithmetic intensity below
the roughly 2-flop-per-byte point where device transfer and launch overhead can
be justified.

## License

[MIT](LICENSE)
