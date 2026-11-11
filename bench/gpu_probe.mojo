"""Measurement-only GPU transport probe: why this port ships CPU only.

This is the evidence behind the "GPU" section of the README. It ships no kernel
and exports nothing; it exists so the GPU decision can be re-checked rather than
taken on trust.

It measures, for the element counts real callers pass (4,096 / 200,000 /
2,000,000 Float64 elements), the *cheapest possible* device round trip: allocate,
copy the input in, launch one trivial kernel, copy the result back, synchronise.
Each component is timed separately so the cost can be attributed:

  - host-to-device copy (enqueue_copy + synchronise)
  - one launch + synchronise, with nothing re-uploaded
  - the full round trip a real kernel would pay per call

The kernel body is a grid-strided copy-and-scale: it does as close to no
arithmetic as a kernel can, so the timing is the host API and the transfer, not
the kernel. A real catch22 kernel would add its own compute on top of these
numbers.

Device memory is 2 x 16 MB at the largest size; the cap is 2 GB.

Run: pixi run python bench/gpu_probe.py

That wrapper checks for a CUDA driver first and prints "no CUDA device; probe
skipped" instead of running this file, because `DeviceContext()` aborts the
process on a host with no driver rather than raising, so the `try:` below cannot
catch a missing device on its own. Run this file directly only where a device is
known to be present.
"""

from std.time import perf_counter_ns

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std._gpu import block_dim, grid_dim

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime THREADS = 128
comptime LOOPS = 101
comptime HUGE = 1_000_000_000_000_000_000


def probe_kernel(src: FPtr, dst: FPtr, n_value: Int64):
    var n = Int(n_value)
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(i, src.unsafe_load(i) * 0.5 + 0.25)
        i += stride


def probe(ctx: DeviceContext, n: Int) raises:
    var host = List[Float64](length=n, fill=1.0)
    var host_out = List[Float64](length=n, fill=0.0)
    var p_in = FPtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var p_out = FPtr(unsafe_from_address=Int(host_out.unsafe_ptr()))
    var d_src = ctx.enqueue_create_buffer[DType.float64](n)
    var d_dst = ctx.enqueue_create_buffer[DType.float64](n)
    var blocks = (n + 2 * THREADS - 1) // (2 * THREADS)

    # Warm: the first launch pays kernel compilation, which no caller repeats.
    ctx.enqueue_copy(d_src, p_in)
    ctx.enqueue_function[probe_kernel](
        d_src, d_dst, Int64(n), grid_dim=blocks, block_dim=THREADS
    )
    ctx.enqueue_copy(p_out, d_dst)
    ctx.synchronize()

    var trip_best = HUGE
    var trip_total = 0
    var copy_best = HUGE
    var launch_best = HUGE
    for _ in range(LOOPS):
        var t0 = perf_counter_ns()
        ctx.enqueue_copy(d_src, p_in)
        ctx.enqueue_function[probe_kernel](
            d_src, d_dst, Int64(n), grid_dim=blocks, block_dim=THREADS
        )
        ctx.enqueue_copy(p_out, d_dst)
        ctx.synchronize()
        var elapsed = perf_counter_ns() - t0
        trip_total += elapsed
        if elapsed < trip_best:
            trip_best = elapsed

        var t1 = perf_counter_ns()
        ctx.enqueue_copy(d_src, p_in)
        ctx.synchronize()
        var copy_elapsed = perf_counter_ns() - t1
        if copy_elapsed < copy_best:
            copy_best = copy_elapsed

        var t2 = perf_counter_ns()
        ctx.enqueue_function[probe_kernel](
            d_src, d_dst, Int64(n), grid_dim=blocks, block_dim=THREADS
        )
        ctx.synchronize()
        var launch_elapsed = perf_counter_ns() - t2
        if launch_elapsed < launch_best:
            launch_best = launch_elapsed

    print(
        "|",
        n,
        "|",
        Float64(trip_best) / 1000.0,
        "|",
        Float64(trip_total // LOOPS) / 1000.0,
        "|",
        Float64(copy_best) / 1000.0,
        "|",
        Float64(launch_best) / 1000.0,
        "|",
    )


def main() raises:
    var available: Bool
    try:
        var ctx = DeviceContext()
        print("device: ", ctx.name())
        available = True
        print(
            "| elements | round trip best us | round trip mean us | "
            "copy in+sync best us | launch+sync best us |"
        )
        print("| ---: | ---: | ---: | ---: | ---: |")
        var sizes = List[Int]()
        sizes.append(4096)
        sizes.append(200_000)
        sizes.append(2_000_000)
        for n in sizes:
            probe(ctx, n)
    except:
        available = False
    if not available:
        print("no CUDA device; probe skipped")
