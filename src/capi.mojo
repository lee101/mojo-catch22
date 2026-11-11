"""Allocation-free C ABI kernels for the covered catch22 characteristics."""

from std.math import exp, isnan, log, sqrt
from std.memory import stack_allocation
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def invalid(a: Ptr, n: Int) -> Bool:
    for i in range(n):
        if isnan(a[i]):
            return True
    return n < 2


def mean(a: Ptr, n: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i]
        i += 1
    return total / Float64(n)


def stddev(a: Ptr, n: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var m = mean(a, n)
    var means = SIMD[DType.float64, W](m)
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var d = a.unsafe_load[width=W](i) - means
        acc += d * d
        i += W
    var total = acc.reduce_add()
    while i < n:
        var tail = a[i] - m
        total += tail * tail
        i += 1
    return sqrt(total / Float64(n - 1))


def min_value(a: Ptr, n: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    if n < W:
        var scalar_minimum = a[0]
        for i in range(1, n):
            if a[i] < scalar_minimum:
                scalar_minimum = a[i]
        return scalar_minimum
    var i = 0
    var values = a.unsafe_load[width=W](i)
    var minimum = values
    i += W
    while i <= n - W:
        values = a.unsafe_load[width=W](i)
        minimum = min(minimum, values)
        i += W
    var value = minimum.reduce_min()
    while i < n:
        if a[i] < value:
            value = a[i]
        i += 1
    return value


def max_value(a: Ptr, n: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    if n < W:
        var scalar_maximum = a[0]
        for i in range(1, n):
            if a[i] > scalar_maximum:
                scalar_maximum = a[i]
        return scalar_maximum
    var i = 0
    var values = a.unsafe_load[width=W](i)
    var maximum = values
    i += W
    while i <= n - W:
        values = a.unsafe_load[width=W](i)
        maximum = max(maximum, values)
        i += W
    var value = maximum.reduce_max()
    while i < n:
        if a[i] > value:
            value = a[i]
        i += 1
    return value


def autocorr_lag(a: Ptr, n: Int, lag: Int) -> Float64:
    var m = mean(a, n)
    var divisor = 0.0
    var numerator = 0.0
    for i in range(n):
        var d = a[i] - m
        divisor += d * d
    for i in range(n - lag):
        numerator += (a[i] - m) * (a[i + lag] - m)
    return numerator / divisor


def diff_autocorr_lag(a: Ptr, n: Int, lag: Int) -> Float64:
    var rn = n - 1
    var rm = (a[n - 1] - a[0]) / Float64(rn)
    var divisor = 0.0
    var numerator = 0.0
    for i in range(rn):
        var d = (a[i + 1] - a[i]) - rm
        divisor += d * d
    for i in range(rn - lag):
        var left = (a[i + 1] - a[i]) - rm
        var right = (a[i + lag + 1] - a[i + lag]) - rm
        numerator += left * right
    return numerator / divisor


def first_zero(a: Ptr, n: Int, differences: Bool) -> Int:
    var ac_n = n - 1 if differences else n
    for lag in range(ac_n):
        var ac = diff_autocorr_lag(a, n, lag) if differences else autocorr_lag(a, n, lag)
        if ac <= 0.0:
            return lag
    return ac_n


def histogram_mode(a: Ptr, n: Int, bins: Int) -> Float64:
    var lo = min_value(a, n)
    var hi = max_value(a, n)
    if lo == hi:
        return lo
    var step = (hi - lo) / Float64(bins)
    var best = 0
    var ties = 0
    var centers = 0.0
    for b in range(bins):
        var count = 0
        for i in range(n):
            var index = Int((a[i] - lo) / step)
            if index < 0:
                index = 0
            if index >= bins:
                index = bins - 1
            if index == b:
                count += 1
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
    var lo = min_value(a, n)
    var step = (max_value(a, n) - lo + 2.0 * padding) / 5.0
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


def f1ecac(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0
    var threshold = 1.0 / exp(1.0)
    var previous = autocorr_lag(a, n, 0)
    for lag in range(1, n - 1):
        var current = autocorr_lag(a, n, lag)
        if current < threshold:
            return Float64(lag - 1) + (threshold - previous) / (current - previous)
        previous = current
    return Float64(n)


def first_min_ac(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0
    var previous = autocorr_lag(a, n, 0)
    var current = autocorr_lag(a, n, 1)
    for lag in range(1, n - 1):
        var following = autocorr_lag(a, n, lag + 1)
        if current < previous and current < following:
            return Float64(lag)
        previous = current
        current = following
    return Float64(n)


def trev(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0 / 0.0
    comptime W = simd_width_of[DType.float64]()
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W < n:
        var d = a.unsafe_load[width=W](i + 1) - a.unsafe_load[width=W](i)
        acc += d * d * d
        i += W
    var total = acc.reduce_add()
    while i < n - 1:
        var tail = a[i + 1] - a[i]
        total += tail * tail * tail
        i += 1
    return total / Float64(n - 1)


def pnn40(a: Ptr, n: Int) -> Float64:
    if invalid(a, n):
        return 0.0 / 0.0
    var count = 0
    for i in range(n - 1):
        var d = (a[i + 1] - a[i]) * 1000.0
        if d < 0.0:
            d = -d
        if d > 40.0:
            count += 1
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
    var rn = n - 3
    var m = 0.0
    for i in range(rn):
        m += a[i + 3] - (a[i] + a[i + 1] + a[i + 2]) / 3.0
    m /= Float64(rn)
    var total = 0.0
    for i in range(rn):
        var r = a[i + 3] - (a[i] + a[i + 1] + a[i + 2]) / 3.0 - m
        total += r * r
    return sqrt(total / Float64(rn - 1))


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
