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

## Parallelism

**The parallel path was built, measured, and reverted.** Nothing in this checkout
calls `parallelize`; the serial kernels are what ship. What follows is why.

The API was `parallelize` from `max.algorithm` (the `std.algorithm` and
`std.threading` forms do not compile on the pinned `1.2.0.dev2026092605`
toolchain), behind a size threshold, capped at 4 workers. `mean` was converted
first as the cheapest kernel: split the buffer into 4 contiguous chunks on a
`4*W` granule boundary, reduce each chunk in a worker, sum the partial means.
When the chunk size divides the length the result differs by only rounding
(about 1e-17 absolute on the mean). When it does not — the last chunk is a short
tail — each partial is taken over a differently sized slice and the gap grows:
up to 2.74e-7 absolute on a signal peaking near 5. That is still tiny, but it is
why the parallel test compares with a bound of `1e-7 * max|x|` instead of the
suite's relative 2e-8, which is useless for a mean that sits near zero.

The blocker is dispatch cost, not the arithmetic. From a ctypes host, **one**
`parallelize` call costs about **4.6 ms** (best of 9, measured 4.58–4.80 ms;
`lib().mcc_parallel_dispatch_us(pointer, n, workers, items)`):

| workers | work items | one parallelize call |
| ---: | ---: | ---: |
| 2 | 2 | 4,800 us |
| 2 | 8 | 4,715 us |
| 2 | 64 | 4,689 us |
| 4 | 2 | 4,581 us |
| 4 | 8 | 4,720 us |
| 4 | 64 | 4,655 us |

Sixty-four work items cost the same as two, and four workers cost the same as
two: the cost is the task runtime waking workers per call, not per item, and no
split of the work avoids it. (A control where one worker reduces the whole
buffer through `parallelize` also cost that much.) A kernel that runs through
`parallelize` once per call therefore has to recover over 4 ms, and it does not:
below roughly a million elements the serial kernel has already finished by then.

`bench/parallel_ab.py` runs the committed serial kernel and a measurement-only
copy of the reverted path (`mcc_mean_taskparallel`) alternately on one buffer,
so both see the same host. The two symbols `mcc_mean_taskparallel` and
`mcc_parallel_dispatch_us` are the only parallel code in the library, both
benchmark-only. One
run, from this checkout (that day the host was not throttled — cgroup
`nr_throttled` stayed at 0 — but was heavily loaded, load1 around 18. The two
runs taken give the same ordering, with the serial columns moving by up to 20%
between them and the parallel columns by much more):

| n | workers | serial | parallel | serial / parallel, best | median | rounds won by parallel | max abs gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | 1 (control) | 64,808 us | 65,171 us | - | - | - | bitwise |
| 262,144 | 2 | 64,808 us | 4,857,984 us | 0.02x | 0.01x | 0/7 | 5e-18 |
| 262,144 | 4 | 64,808 us | 5,178,100 us | 0.01x | 0.01x | 0/7 | 7.6e-18 |
| 262,144 | 8 | 64,808 us | 10,018,403 us | 0.01x | 0.01x | 0/7 | 7.6e-18 |
| 2,000,000 | 1 (control) | 489,398 us | 486,308 us | - | - | - | bitwise |
| 2,000,000 | 2 | 489,398 us | 5,282,871 us | 0.11x | 0.10x | 0/7 | 8.1e-19 |
| 2,000,000 | 4 | 489,398 us | 4,753,845 us | 0.12x | 0.10x | 0/7 | 9.9e-18 |
| 2,000,000 | 8 | 489,398 us | 4,673,987 us | 0.13x | 0.11x | 0/7 | 1.1e-18 |
| 8,000,000 | 1 (control) | 5,925,096 us | 5,366,734 us | - | - | - | bitwise |
| 8,000,000 | 2 | 5,925,096 us | 9,033,382 us | 0.75x | 0.53x | 0/7 | 4.3e-19 |
| 8,000,000 | 4 | 5,925,096 us | 8,704,001 us | 0.73x | 0.64x | 0/7 | 8.7e-19 |
| 8,000,000 | 8 | 5,925,096 us | 7,571,128 us | 0.90x | 0.73x | 0/7 | 4.4e-18 |

At 1, 2, 4 and 8 workers the parallel path loses **every** round at every size.
It does not flatten out and it does not recover with more workers: going 2 → 4 →
8 is 4.86 → 5.18 → 10.02 ms at 256k and 9.03 → 8.70 → 7.57 ms at 8M — jumps
around the dispatch cost, not scaling. The best cell in the table, 8M at 8
workers, is still 1.3x *slower* than serial. Both paths agreed to 1e-17 or
better in every row, so correctness was never the problem.

The full before/after on `mcc_mean` from the frozen pre-change library, same
process, best of 15, one run: 4,096 elements 1.6 → 1.6 us; 200,000 47.6 → 49.5 us
(below any threshold, unchanged); 262,144 **85.5 → 5,156.8 us**; 1,000,000
**254.8 → 4,809.3 us**; 2,000,000 **603.2 → 5,189.0 us**. Shipped, it would have
made 256k-scale callers about 20x *slower* and million-element callers about
10x slower. That run happened to cross a spell of host contention around the 2M
row, so its serial 2M and 8M figures are not reusable — the paired table above
is the one to read — but the direction held in every measurement taken, and the
paired table shows the same conclusion at all three sizes.

So `mcc_mean` is back to the serial kernel, there is no threshold to tune, and
the only parallel code left is `mcc_mean_taskparallel`, compiled in so the table
above can be reproduced from this checkout: `MCC_SIZES=262144,2000000,8000000
flock /tmp/mojo-bench.lock pixi run python bench/parallel_ab.py`. `max` is now a
pinned dependency (`max ==26.7.0.dev2026092605`, alongside the mojo pin) so that
symbol builds and so a future attempt starts from a compiling `max.algorithm`
import rather than rediscovering that it moved. To retry: convert one kernel
behind a threshold, then run `bench/parallel_ab.py` before touching the README —
if it does not win the paired rounds at real caller sizes, revert it again.

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
- Task parallelism through `max.algorithm.parallelize`: built for `DN_Mean`,
  measured through 1/2/4/8 workers at 256k, 2M and 8M elements, and reverted
  because one dispatch costs ~4.6 ms and the parallel path lost every paired
  round to the serial kernel. The table and the reasons are in the
  **Parallelism** section above.

## GPU

**CPU only: memory-bound.** No kernel in this port ships a device path, and that
is the gate's answer, not a gap left open. Nothing in `src/capi.mojo` imports
`max.gpu`; `bench/gpu_probe.mojo` (driven by `bench/gpu_probe.py`) is the
measurement-only evidence for this section and exports nothing.

The gate is arithmetic intensity: a kernel is only a GPU candidate above roughly
2 floating-point operations per byte of input it reads. A `float64` element is 8
bytes, so the bar is about 16 flops per element. Counted from the committed
kernels in `src/capi.mojo`, the heaviest pass is `FC_LocalSimple_mean3_stderr`
at about 5 flops per element (0.63 flops/byte). There is no `bench/analysis.md`
in this checkout and none in its history, so the counts below are taken from the
kernel bodies rather than quoted from a separate analysis; each is per input
element that the kernel reads, with the comparisons and the bin arithmetic
counted as one operation each:

| kernel | dominant work per element | flops/byte | gate |
| --- | --- | ---: | --- |
| `FC_LocalSimple_mean3_stderr` | window sum, residual, square, two accumulations | 0.63 | below |
| `CO_trev_1_num` | difference, two multiplies, add | 0.50 | below |
| `MD_hrv_classic_pnn40` | difference, scale, two compares | 0.50 | below |
| `CO_f1ecac`, `CO_FirstMin_ac`, `FC_LocalSimple_mean1_tauresrat` | two differences, multiply, add, per lag pass | 0.50 | below |
| `CO_HistogramAMI_even_2_5` | eight compares, bin select, counter | 0.25 | below |
| `DN_HistogramMode_5/10` | subtract, divide, clamp, counter | 0.38 | below |
| `DN_Mean`, `DN_Spread_Std` | add (and multiply-add) | 0.25 | below |
| `SB_BinaryStats_*_longstretch*` | difference, compare | 0.13 | below |

The autocorrelation family is the one case worth naming: its *total* work is
O(n²), but that comes from repeating a pass that is itself 0.5 flops/byte, one
pass per lag, re-reading the same buffer each time. Aggregate work grows with
n; arithmetic intensity per byte read does not, which is what the gate asks
about.

### Measured round-trip cost

The gate alone would be an assumption, so the device cost was measured on this
host (NVIDIA GeForce RTX 5090, 32,607 MiB total, 5,500–6,292 MiB free during
these runs — above the 4,000 MiB floor, so the phase was not skipped on memory
grounds). `pixi run python bench/gpu_probe.py`, one run, 101 timed rounds each,
best of the rounds. The kernel is a grid-strided copy-and-scale: as close to no
arithmetic as a kernel can be, so these are the floor a real kernel pays *on top
of* its own compute.

| elements | full round trip, best | round trip, mean | copy in + sync | launch + sync |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 34.3 us | 41.3 us | 13.4 us | 11.0 us |
| 200,000 | 553.9 us | 1,260.0 us | 227.6 us | 9.9 us |
| 2,000,000 | 3,512.6 us | 4,234.8 us | 1,392.2 us | 15.4 us |

The host is shared and the spread is real: repeat runs of the same probe gave
21.3–34.3 us at 4,096, 440.0–553.9 us at 200,000 and 3,350.3–3,512.6 us at
2,000,000 for the best round trip. Read these as one-decimal-scale, the same way
the **Benchmark** section's numbers are meant to be read; the conclusion below
holds across the whole spread, not at one point in it.

Against the committed CPU kernels, same buffers, best of 25 calls through the
native ABI (`mcc_*` symbols directly, z-score precomputed where the kernel needs
it), one run:

| kernel | 4,096 CPU | 200,000 CPU | 2,000,000 CPU | GPU floor at 2M |
| --- | ---: | ---: | ---: | ---: |
| `DN_Mean` | 1.9 us | 58.4 us | 564.4 us | 3,512.6 us |
| `DN_Spread_Std` | 2.2 us | 60.7 us | 584.5 us | 3,512.6 us |
| `CO_trev_1_num` | 2.9 us | 85.0 us | 826.8 us | 3,512.6 us |
| `MD_hrv_classic_pnn40` | 3.1 us | 97.5 us | 947.1 us | 3,512.6 us |
| `DN_HistogramMode_5` | 18.0 us | 724.7 us | 6,462.2 us | 3,512.6 us |
| `CO_HistogramAMI_even_2_5` | 19.6 us | 862.0 us | 7,423.7 us | 3,512.6 us |
| `SB_BinaryStats_mean_longstretch1` | 10.3 us | 466.5 us | 4,120.2 us | 3,512.6 us |
| `FC_LocalSimple_mean3_stderr` | 4.1 us | 146.1 us | 1,239.4 us | 3,512.6 us |

The 4,096 and 200,000 columns are the sizes the benchmark and the autocorrelation
tests actually call, so those are the caller sizes that matter here. At 200,000
— the size `CO_HistogramAMI_even_2_5` and `DN_HistogramMode_*` are benchmarked at
— the cheapest possible round trip is 553.9 us against an 862.0 us CPU kernel:
the transfer alone consumes 64% of the budget, before the kernel does any work,
and on the faster repeat run (440.0 us) it is still 51%. At 4,096 the floor is
34.3 us against a 19.6 us CPU kernel — the round trip alone is 1.8x the entire
CPU call. At 2,000,000 the floor is 3,512.6 us, which is 6.2x the CPU `DN_Mean`
and 52% of the CPU `CO_HistogramAMI_even_2_5`, the slowest kernel in the port.
Even for that kernel there is no headroom: the device-side binning, the 25-bin
reduction and the copy back would all have to fit inside the remaining 48%, and
every other kernel is further away than that. That is the same conclusion the
intensity table reaches from the other side.

So the GPU path was not built. A kernel-and-gear implementation would have been
kept only if it won at the sizes above, and the intensity ceiling says it cannot:
the arithmetic is not there to hide the copy. The CPU path is the only path, and
there is no threshold to tune.

### Tests

No GPU test exists and none was skipped: with no device path shipped there is
nothing device-specific to assert, and `pixi run test` runs the full 223-test
CPU suite on this host, which has the device. `bench/gpu_probe.py` is the only
device-touching code in the tree, it is a benchmark rather than part of
`pixi run test`, and it skips cleanly on a CPU-only machine: when no driver is
present it prints `no CUDA device; probe skipped` and exits 0. That check has to
live in the wrapper rather than in Mojo, because `DeviceContext()` aborts the
process on a host with no driver instead of raising, so a `try:` cannot catch
it.

To re-check this decision after a toolchain or hardware change, run
`pixi run python bench/gpu_probe.py` and compare the 4,096, 200,000 and
2,000,000 rows against the CPU columns above. If the round trip ever falls below
the CPU time of a single kernel at a real caller size, that kernel becomes worth
converting; until then the numbers say CPU only.

## License

## License

[MIT](LICENSE)
