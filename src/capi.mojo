"""Allocation-free C ABI kernels for the covered catch22 characteristics."""

from std.math import exp, isnan, log, sqrt
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
        acc += a.load[width=W](i)
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
        var d = a.load[width=W](i) - means
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
    var values = a.load[width=W](i)
    var minimum = values
    i += W
    while i <= n - W:
        values = a.load[width=W](i)
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
    var values = a.load[width=W](i)
    var maximum = values
    i += W
    while i <= n - W:
        values = a.load[width=W](i)
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


def ami_bin(value: Float64, lo: Float64, step: Float64) -> Int:
    for edge in range(6):
        if value < lo + step * Float64(edge) - 0.1:
            return edge
    return 0


def histogram_ami(a: Ptr, n: Int) -> Float64:
    var lo = min_value(a, n)
    var step = (max_value(a, n) - lo + 0.2) / 5.0
    var total = Float64(n - 2)
    var row1 = 0
    var row2 = 0
    var row3 = 0
    var row4 = 0
    var row5 = 0
    var column1 = 0
    var column2 = 0
    var column3 = 0
    var column4 = 0
    var column5 = 0
    var joint11 = 0
    var joint12 = 0
    var joint13 = 0
    var joint14 = 0
    var joint15 = 0
    var joint21 = 0
    var joint22 = 0
    var joint23 = 0
    var joint24 = 0
    var joint25 = 0
    var joint31 = 0
    var joint32 = 0
    var joint33 = 0
    var joint34 = 0
    var joint35 = 0
    var joint41 = 0
    var joint42 = 0
    var joint43 = 0
    var joint44 = 0
    var joint45 = 0
    var joint51 = 0
    var joint52 = 0
    var joint53 = 0
    var joint54 = 0
    var joint55 = 0
    for k in range(n - 2):
        var left = ami_bin(a[k], lo, step)
        var right = ami_bin(a[k + 2], lo, step)
        if left == 1:
            row1 += 1
            if right == 1:
                joint11 += 1
            elif right == 2:
                joint12 += 1
            elif right == 3:
                joint13 += 1
            elif right == 4:
                joint14 += 1
            else:
                joint15 += 1
        elif left == 2:
            row2 += 1
            if right == 1:
                joint21 += 1
            elif right == 2:
                joint22 += 1
            elif right == 3:
                joint23 += 1
            elif right == 4:
                joint24 += 1
            else:
                joint25 += 1
        elif left == 3:
            row3 += 1
            if right == 1:
                joint31 += 1
            elif right == 2:
                joint32 += 1
            elif right == 3:
                joint33 += 1
            elif right == 4:
                joint34 += 1
            else:
                joint35 += 1
        elif left == 4:
            row4 += 1
            if right == 1:
                joint41 += 1
            elif right == 2:
                joint42 += 1
            elif right == 3:
                joint43 += 1
            elif right == 4:
                joint44 += 1
            else:
                joint45 += 1
        else:
            row5 += 1
            if right == 1:
                joint51 += 1
            elif right == 2:
                joint52 += 1
            elif right == 3:
                joint53 += 1
            elif right == 4:
                joint54 += 1
            else:
                joint55 += 1
        if right == 1:
            column1 += 1
        elif right == 2:
            column2 += 1
        elif right == 3:
            column3 += 1
        elif right == 4:
            column4 += 1
        else:
            column5 += 1
    var result = 0.0
    result += ami_term(joint11, row1, column1, total)
    result += ami_term(joint12, row1, column2, total)
    result += ami_term(joint13, row1, column3, total)
    result += ami_term(joint14, row1, column4, total)
    result += ami_term(joint15, row1, column5, total)
    result += ami_term(joint21, row2, column1, total)
    result += ami_term(joint22, row2, column2, total)
    result += ami_term(joint23, row2, column3, total)
    result += ami_term(joint24, row2, column4, total)
    result += ami_term(joint25, row2, column5, total)
    result += ami_term(joint31, row3, column1, total)
    result += ami_term(joint32, row3, column2, total)
    result += ami_term(joint33, row3, column3, total)
    result += ami_term(joint34, row3, column4, total)
    result += ami_term(joint35, row3, column5, total)
    result += ami_term(joint41, row4, column1, total)
    result += ami_term(joint42, row4, column2, total)
    result += ami_term(joint43, row4, column3, total)
    result += ami_term(joint44, row4, column4, total)
    result += ami_term(joint45, row4, column5, total)
    result += ami_term(joint51, row5, column1, total)
    result += ami_term(joint52, row5, column2, total)
    result += ami_term(joint53, row5, column3, total)
    result += ami_term(joint54, row5, column4, total)
    result += ami_term(joint55, row5, column5, total)
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
        var d = a.load[width=W](i + 1) - a.load[width=W](i)
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
    return histogram_ami(p(a), n)


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
