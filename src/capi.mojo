"""Allocation-free C ABI kernels for the covered catch22 characteristics."""

from std.math import exp, isnan, log, sqrt
from std.memory import stack_allocation
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def invalid(a: Ptr, n: Int) -> Bool:
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        if a.unsafe_load[width=W](i).ne(a.unsafe_load[width=W](i)).reduce_or():
            return True
        i += W
    while i < n:
        if isnan(a[i]):
            return True
        i += 1
    return n < 2


def mean_and_stddev(a: Ptr, n: Int) -> Tuple[Float64, Float64]:
    """Mean and sample standard deviation of `a[:n]` in one pass over `a`.

    The caller does not know the mean until the end, so the second moment is
    accumulated raw (`x`) next to `x*x`, and the centred sum is recovered as
    `sum(x*x) - 2*m*sum(x) + n*m*m`. Both SIMD sums use four independent
    accumulators so the adds do not serialise on floating-point latency.
    """
    comptime W = simd_width_of[DType.float64]()
    var s0 = SIMD[DType.float64, W](0.0)
    var s1 = SIMD[DType.float64, W](0.0)
    var s2 = SIMD[DType.float64, W](0.0)
    var s3 = SIMD[DType.float64, W](0.0)
    var q0 = SIMD[DType.float64, W](0.0)
    var q1 = SIMD[DType.float64, W](0.0)
    var q2 = SIMD[DType.float64, W](0.0)
    var q3 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        comptime for q in range(4):
            var v = a.unsafe_load[width=W](i + q * W)
            var square = v * v
            if q == 0:
                s0 += v
                q0 += square
            elif q == 1:
                s1 += v
                q1 += square
            elif q == 2:
                s2 += v
                q2 += square
            else:
                s3 += v
                q3 += square
        i += 4 * W
    while i + W <= n:
        var v = a.unsafe_load[width=W](i)
        s0 += v
        q0 += v * v
        i += W
    var total = ((s0 + s1) + (s2 + s3)).reduce_add()
    var square_total = ((q0 + q1) + (q2 + q3)).reduce_add()
    while i < n:
        var v = a[i]
        total += v
        square_total += v * v
        i += 1
    var m = total / Float64(n)
    var centred = square_total - 2.0 * m * total + Float64(n) * m * m
    if centred < 0.0:
        centred = 0.0
    return Tuple(m, sqrt(centred / Float64(n - 1)))


def mean(a: Ptr, n: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    # Four independent partial sums: one accumulator would serialise on
    # floating-point add latency (about four cycles per add on this machine).
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        acc0 += a.unsafe_load[width=W](i)
        acc1 += a.unsafe_load[width=W](i + W)
        acc2 += a.unsafe_load[width=W](i + 2 * W)
        acc3 += a.unsafe_load[width=W](i + 3 * W)
        i += 4 * W
    while i + W <= n:
        acc0 += a.unsafe_load[width=W](i)
        i += W
    var total = ((acc0 + acc1) + (acc2 + acc3)).reduce_add()
    while i < n:
        total += a[i]
        i += 1
    return total / Float64(n)


def stddev(a: Ptr, n: Int) -> Float64:
    return mean_and_stddev(a, n)[1]


def min_and_max(a: Ptr, n: Int) -> Tuple[Float64, Float64]:
    """Smallest and largest value of `a[:n]` from one pass over `a`."""
    comptime W = simd_width_of[DType.float64]()
    if n < W:
        var scalar_minimum = a[0]
        var scalar_maximum = a[0]
        for i in range(1, n):
            if a[i] < scalar_minimum:
                scalar_minimum = a[i]
            if a[i] > scalar_maximum:
                scalar_maximum = a[i]
        return Tuple(scalar_minimum, scalar_maximum)
    var i = 0
    var minimum = a.unsafe_load[width=W](i)
    var maximum = minimum
    i += W
    while i <= n - W:
        var values = a.unsafe_load[width=W](i)
        minimum = min(minimum, values)
        maximum = max(maximum, values)
        i += W
    var low = minimum.reduce_min()
    var high = maximum.reduce_max()
    while i < n:
        if a[i] < low:
            low = a[i]
        if a[i] > high:
            high = a[i]
        i += 1
    return Tuple(low, high)


def min_value(a: Ptr, n: Int) -> Float64:
    return min_and_max(a, n)[0]


def max_value(a: Ptr, n: Int) -> Float64:
    return min_and_max(a, n)[1]


def sum_squared_deviation(a: Ptr, n: Int, m: Float64) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var mean_values = SIMD[DType.float64, W](m)
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        comptime for q in range(4):
            var d = a.unsafe_load[width=W](i + q * W) - mean_values
            var square = d * d
            if q == 0:
                acc0 += square
            elif q == 1:
                acc1 += square
            elif q == 2:
                acc2 += square
            else:
                acc3 += square
        i += 4 * W
    while i + W <= n:
        var d = a.unsafe_load[width=W](i) - mean_values
        acc0 += d * d
        i += W
    var total = ((acc0 + acc1) + (acc2 + acc3)).reduce_add()
    while i < n:
        var tail = a[i] - m
        total += tail * tail
        i += 1
    return total


def autocorr_lag(a: Ptr, n: Int, lag: Int, m: Float64, divisor: Float64) -> Float64:
    """Lag-`lag` autocorrelation given the already-computed `m` and `divisor`.

    The mean and the zero-lag divisor are shared by every lag and are computed
    once by the caller. A single accumulator measured at least as fast as four
    partials here, so the multi-chain form only cost registers.
    """
    comptime W = simd_width_of[DType.float64]()
    var mean_values = SIMD[DType.float64, W](m)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n - lag:
        acc += (a.unsafe_load[width=W](i) - mean_values) * (
            a.unsafe_load[width=W](i + lag) - mean_values
        )
        i += W
    var numerator = acc.reduce_add()
    while i < n - lag:
        numerator += (a[i] - m) * (a[i + lag] - m)
        i += 1
    return numerator / divisor


def diff_autocorr_lag(a: Ptr, n: Int, lag: Int, rm: Float64, divisor: Float64) -> Float64:
    """Lag-`lag` autocorrelation of the first difference, given `rm` and `divisor`."""
    var rn = n - 1
    var numerator = 0.0
    for i in range(rn - lag):
        var left = (a[i + 1] - a[i]) - rm
        var right = (a[i + lag + 1] - a[i + lag]) - rm
        numerator += left * right
    return numerator / divisor


def first_zero(a: Ptr, n: Int, differences: Bool) -> Int:
    var ac_n = n - 1 if differences else n
    # The mean (and the zero-lag divisor, which is a total of squares rather than
    # a mean) do not depend on the lag, so they are computed once here instead of
    # once per lag inside the loop below.
    var m: Float64
    var divisor: Float64
    if differences:
        var rn = n - 1
        m = (a[n - 1] - a[0]) / Float64(rn)
        var total = 0.0
        for i in range(rn):
            var d = (a[i + 1] - a[i]) - m
            total += d * d
        divisor = total
    else:
        m = mean(a, n)
        divisor = sum_squared_deviation(a, n, m)
    for lag in range(ac_n):
        var ac: Float64
        if differences:
            ac = diff_autocorr_lag(a, n, lag, m, divisor)
        else:
            ac = autocorr_lag(a, n, lag, m, divisor)
        if ac <= 0.0:
            return lag
    return ac_n


def histogram_mode(a: Ptr, n: Int, bins: Int) -> Float64:
    var extremes = min_and_max(a, n)
    var lo = extremes[0]
    var hi = extremes[1]
    if lo == hi:
        return lo
    var step = (hi - lo) / Float64(bins)
    var counts = stack_allocation[10, Int]()
    for b in range(bins):
        counts[b] = 0
    # One pass over the series, counting every bin at once. The per-bin
    # comparison this replaces re-scanned the whole series once per bin.
    for i in range(n):
        var index = Int((a[i] - lo) / step)
        if index < 0:
            index = 0
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    var best = counts[0]
    var ties = 0
    var centers = 0.0
    for b in range(bins):
        var count = counts[b]
        if count > best:
            best = count
            ties = 1
            centers = lo + (Float64(b) + 0.5) * step
        elif count == best:
            ties += 1
            centers += lo + (Float64(b) + 0.5) * step
    return centers / Float64(ties)


def ami_bin(value: Float64, lo: Float64, step: Float64, padding: Float64) -> Int:
    for edge in range(6):
        if value < lo + step * Float64(edge) - padding:
            return edge
    return 0


def histogram_ami_count_range(
    a: Ptr,
    start: Int,
    end: Int,
    lo: Float64,
    step: Float64,
    padding: Float64,
    counts: Pointer[Int, MutUntrackedOrigin],
):
    comptime W = simd_width_of[DType.float64]()
    for bucket in range(25):
        counts[bucket] = 0
    var k = start
    while k + W <= end:
        var left_values = a.unsafe_load[width=W](k)
        var right_values = a.unsafe_load[width=W](k + 2)
        var left_bins = SIMD[DType.int64, W](0)
        var right_bins = SIMD[DType.int64, W](0)
        comptime for edge in range(4):
            var threshold = SIMD[DType.float64, W](
                lo + Float64(edge + 1) * step - padding
            )
            left_bins += left_values.ge(threshold).cast[DType.int64]()
            right_bins += right_values.ge(threshold).cast[DType.int64]()
        var joint_bins = left_bins * 5 + right_bins
        comptime for lane in range(W):
            counts[Int(joint_bins[lane])] += 1
        k += W
    while k < end:
        var left = ami_bin(a[k], lo, step, padding)
        var right = ami_bin(a[k + 2], lo, step, padding)
        counts[(left - 1) * 5 + right - 1] += 1
        k += 1


def histogram_ami(a: Ptr, n: Int, padding: Float64) -> Float64:
    var extremes = min_and_max(a, n)
    var lo = extremes[0]
    var step = (extremes[1] - lo + 2.0 * padding) / 5.0
    var total = Float64(n - 2)
    var counts = stack_allocation[25, Int]()
    histogram_ami_count_range(a, 0, n - 2, lo, step, padding, counts)
    var rows = stack_allocation[5, Int]()
    var columns = stack_allocation[5, Int]()
    for bin in range(5):
        rows[bin] = 0
        columns[bin] = 0
    for row in range(5):
        for column in range(5):
            rows[row] += counts[row * 5 + column]
            columns[column] += counts[row * 5 + column]
    var result = 0.0
    for row in range(5):
        for column in range(5):
            result += ami_term(
                counts[row * 5 + column], rows[row], columns[column], total
            )
    return result


def ami_term(joint: Int, row: Int, column: Int, total: Float64) -> Float64:
    if joint == 0:
        return 0.0
    var p = Float64(joint) / total
    return p * log(p / ((Float64(row) / total) * (Float64(column) / total)))


def autocorrelation_stats(a: Ptr, n: Int) -> Tuple[Float64, Float64]:
    """Mean and zero-lag sum of squared deviations, shared by every lag."""
    var stats = mean_and_stddev(a, n)
    var m = stats[0]
    return Tuple(m, sum_squared_deviation(a, n, m))


def f1ecac(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0
    var threshold = 1.0 / exp(1.0)
    var stats = autocorrelation_stats(a, n)
    var previous = autocorr_lag(a, n, 0, stats[0], stats[1])
    for lag in range(1, n - 1):
        var current = autocorr_lag(a, n, lag, stats[0], stats[1])
        if current < threshold:
            return Float64(lag - 1) + (threshold - previous) / (current - previous)
        previous = current
    return Float64(n)


def first_min_ac(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0
    var stats = autocorrelation_stats(a, n)
    var previous = autocorr_lag(a, n, 0, stats[0], stats[1])
    var current = autocorr_lag(a, n, 1, stats[0], stats[1])
    for lag in range(1, n - 1):
        var following = autocorr_lag(a, n, lag + 1, stats[0], stats[1])
        if current < previous and current < following:
            return Float64(lag)
        previous = current
        current = following
    return Float64(n)


def trev(a: Ptr, n: Int) -> Float64:
    if n < 2:
        return 0.0 / 0.0
    comptime W = simd_width_of[DType.float64]()
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    # Four independent partials over a fixed four-wide body: one chain would
    # serialise on the multiply, multiply, add latency of the cube.
    while i + 4 * W < n:
        comptime for q in range(4):
            var d = a.unsafe_load[width=W](i + q * W + 1) - a.unsafe_load[width=W](i + q * W)
            var cube = d * d * d
            if q == 0:
                acc0 += cube
            elif q == 1:
                acc1 += cube
            elif q == 2:
                acc2 += cube
            else:
                acc3 += cube
        i += 4 * W
    while i + W < n:
        var d = a.unsafe_load[width=W](i + 1) - a.unsafe_load[width=W](i)
        acc0 += d * d * d
        i += W
    var total = ((acc0 + acc1) + (acc2 + acc3)).reduce_add()
    # Lanes are still loaded a vector at a time and summed in scalar order, so
    # the scalar tail contributes exactly what it did before.
    while i < n - 1:
        comptime for q in range(W):
            if i + q < n - 1:
                var d = a[i + q + 1] - a[i + q]
                total += d * d * d
        i += W
    return total / Float64(n - 1)


def pnn40(a: Ptr, n: Int) -> Float64:
    if n < 2:
        return 0.0 / 0.0
    comptime W = simd_width_of[DType.float64]()
    var limit = SIMD[DType.float64, W](40.0)
    var c0 = SIMD[DType.int64, W](0)
    var c1 = SIMD[DType.int64, W](0)
    var c2 = SIMD[DType.int64, W](0)
    var c3 = SIMD[DType.int64, W](0)
    var i = 0
    # Independent counters keep the mispredicted `> 40` test off a single
    # dependency chain; the lane masks accumulate as integer adds.
    while i + 4 * W < n:
        comptime for q in range(4):
            var d = (a.unsafe_load[width=W](i + q * W + 1) - a.unsafe_load[width=W](i + q * W)) * 1000.0
            var hit = d.gt(limit).cast[DType.int64]() + d.lt(-limit).cast[DType.int64]()
            if q == 0:
                c0 += hit
            elif q == 1:
                c1 += hit
            elif q == 2:
                c2 += hit
            else:
                c3 += hit
        i += 4 * W
    while i + W < n:
        var d = (a.unsafe_load[width=W](i + 1) - a.unsafe_load[width=W](i)) * 1000.0
        c0 += d.gt(limit).cast[DType.int64]() + d.lt(-limit).cast[DType.int64]()
        i += W
    var count = ((c0 + c1) + (c2 + c3)).reduce_add()
    while i < n - 1:
        comptime for q in range(W):
            if i + q < n - 1:
                var d = (a[i + q + 1] - a[i + q]) * 1000.0
                count += Int64(d > 40.0 or d < -40.0)
        i += W
    return Float64(count) / Float64(n - 1)


def mean_longstretch1(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0 / 0.0
    var m = mean(a, n)
    var maximum = 0
    var last = 0
    for i in range(n - 1):
        if a[i] - m <= 0.0 or i == n - 2:
            var stretch = i - last
            if stretch > maximum:
                maximum = stretch
            last = i
    return Float64(maximum)


def diff_longstretch0(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0 / 0.0
    var maximum = 0
    var last = 0
    for i in range(n - 1):
        if a[i + 1] - a[i] >= 0.0 or i == n - 2:
            var stretch = i - last
            if stretch > maximum:
                maximum = stretch
            last = i
    return Float64(maximum)


def mean1_tauresrat(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0 / 0.0
    var residual_zero = first_zero(a, n, True)
    var series_zero = first_zero(a, n, False)
    return Float64(residual_zero) / Float64(series_zero)


def mean3_stderr(a: Ptr, n: Int) -> Float64:
    if invalid(a, n) or n <= 3:
        return 0.0 / 0.0
    comptime W = simd_width_of[DType.float64]()
    var rn = n - 3
    var third = SIMD[DType.float64, W](1.0 / 3.0)
    var s0 = SIMD[DType.float64, W](0.0)
    var s1 = SIMD[DType.float64, W](0.0)
    var s2 = SIMD[DType.float64, W](0.0)
    var s3 = SIMD[DType.float64, W](0.0)
    var q0 = SIMD[DType.float64, W](0.0)
    var q1 = SIMD[DType.float64, W](0.0)
    var q2 = SIMD[DType.float64, W](0.0)
    var q3 = SIMD[DType.float64, W](0.0)
    # The residual window overlaps, so both moments are accumulated as raw sums
    # over the same pass: the centred sum is recovered at the end exactly as in
    # `mean_and_stddev` instead of re-reading the series a second time.
    var i = 0
    while i + 4 * W <= rn:
        comptime for q in range(4):
            var r = a.unsafe_load[width=W](i + q * W + 3) - (
                a.unsafe_load[width=W](i + q * W)
                + a.unsafe_load[width=W](i + q * W + 1)
                + a.unsafe_load[width=W](i + q * W + 2)
            ) * third
            var square = r * r
            if q == 0:
                s0 += r
                q0 += square
            elif q == 1:
                s1 += r
                q1 += square
            elif q == 2:
                s2 += r
                q2 += square
            else:
                s3 += r
                q3 += square
        i += 4 * W
    while i + W <= rn:
        var r = a.unsafe_load[width=W](i + 3) - (
            a.unsafe_load[width=W](i)
            + a.unsafe_load[width=W](i + 1)
            + a.unsafe_load[width=W](i + 2)
        ) * third
        s0 += r
        q0 += r * r
        i += W
    var total = ((s0 + s1) + (s2 + s3)).reduce_add()
    var square_total = ((q0 + q1) + (q2 + q3)).reduce_add()
    while i < rn:
        var r = a[i + 3] - (a[i] + a[i + 1] + a[i + 2]) / 3.0
        total += r
        square_total += r * r
        i += 1
    var m = total / Float64(rn)
    var centred = square_total - 2.0 * m * total + Float64(rn) * m * m
    if centred < 0.0:
        centred = 0.0
    return sqrt(centred / Float64(rn - 1))


def p(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@export("mcc_mean")
def mcc_mean(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 1:
        return 0.0 / 0.0
    return mean(p(a), n)


@export("mcc_std")
def mcc_std(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return stddev(p(a), n)


@export("mcc_histogram_mode_5")
def mcc_histogram_mode_5(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 1:
        return 0.0 / 0.0
    return histogram_mode(p(a), n, 5)


@export("mcc_histogram_mode_10")
def mcc_histogram_mode_10(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 1:
        return 0.0 / 0.0
    return histogram_mode(p(a), n, 10)


@export("mcc_f1ecac")
def mcc_f1ecac(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return f1ecac(p(a), n)


@export("mcc_first_min_ac")
def mcc_first_min_ac(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return first_min_ac(p(a), n)


@export("mcc_histogram_ami")
def mcc_histogram_ami(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 3:
        return 0.0 / 0.0
    if invalid(p(a), n):
        return 0.0 / 0.0
    return histogram_ami(p(a), n, 0.1)


@export("mcc_histogram_ami_raw")
def mcc_histogram_ami_raw(a: Int, n: Int, scale: Float64) abi("C") -> Float64:
    if a == 0 or n < 3:
        return 0.0 / 0.0
    if invalid(p(a), n):
        return 0.0 / 0.0
    return histogram_ami(p(a), n, 0.1 * scale)


@export("mcc_trev")
def mcc_trev(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return trev(p(a), n)


@export("mcc_pnn40")
def mcc_pnn40(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return pnn40(p(a), n)


@export("mcc_mean_longstretch1")
def mcc_mean_longstretch1(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return mean_longstretch1(p(a), n)


@export("mcc_diff_longstretch0")
def mcc_diff_longstretch0(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return diff_longstretch0(p(a), n)


@export("mcc_mean1_tauresrat")
def mcc_mean1_tauresrat(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 2:
        return 0.0 / 0.0
    return mean1_tauresrat(p(a), n)


@export("mcc_mean3_stderr")
def mcc_mean3_stderr(a: Int, n: Int) abi("C") -> Float64:
    if a == 0 or n < 5:
        return 0.0 / 0.0
    return mean3_stderr(p(a), n)
