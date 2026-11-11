import ctypes

import numpy as np
import pytest

import mojocatch22 as mojo
import pycatch22 as reference
from mojocatch22._lib import f64, lib


FEATURES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_BinaryStats_diff_longstretch0",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "DN_Mean",
    "DN_Spread_Std",
]


@pytest.fixture(params=[17, 64, 257], ids=lambda n: f"n={n}")
def series(request):
    rng = np.random.default_rng(request.param)
    t = np.arange(request.param, dtype=np.float64)
    return np.ascontiguousarray(0.5 * np.sin(t / 5.0) + rng.normal(size=request.param))


@pytest.mark.parametrize("name", FEATURES)
def test_individual_features_match_pycatch22(name, series):
    actual = getattr(mojo, name)(series)
    expected = getattr(reference, name)(series.tolist())
    if isinstance(expected, int):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=2e-8, abs=2e-10)


def test_affine_invariance_matches_reference():
    x = np.array([-2.0, 0.5, 4.0, 1.0, 3.0, -1.0, 2.5, 5.0, 0.0, 1.5])
    for name in FEATURES[:-2]:
        assert getattr(mojo, name)(x) == pytest.approx(getattr(mojo, name)(7.0 * x + 3.0))


def _tail_series(n: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + n)
    t = np.arange(n, dtype=np.float64)
    return np.ascontiguousarray(0.5 * np.sin(t / 5.0) + rng.normal(size=n))


# The SIMD kernels run a vector body of W Float64 lanes plus a scalar tail, so a
# length of 0, 1, W-1, W or W+1 must agree with pycatch22, as must a large odd
# length that leaves a partial tail after many full vectors.
SIMD_WIDTH = 4
TAIL_LENGTHS = [0, 1, 3, 4, 5, 12345]
TAIL_IDS = ["n=0", "n=1", "n=W-1", "n=W", "n=W+1", "n=12345"]

# Smallest length the public wrapper accepts, and the smallest length pycatch22
# itself survives (it corrupts the heap at n=0, so parity starts at 1).
MIN_SIZE = {
    "DN_Mean": 1,
    "DN_Spread_Std": 2,
    "FC_LocalSimple_mean3_stderr": 5,
}
REFERENCE_MIN_SIZE = 2


@pytest.mark.parametrize("n", TAIL_LENGTHS, ids=TAIL_IDS)
@pytest.mark.parametrize("name", FEATURES)
def test_simd_tail_lengths_match_pycatch22(name, n):
    x = _tail_series(n)
    if n < MIN_SIZE.get(name, 3):
        with pytest.raises(ValueError):
            getattr(mojo, name)(x)
        return
    if n < REFERENCE_MIN_SIZE:
        # pycatch22 corrupts its heap here; this port defines the answer itself.
        assert getattr(mojo, name)(x) == pytest.approx(x[0])
        return
    actual = getattr(mojo, name)(x)
    expected = getattr(reference, name)(x.tolist())
    if isinstance(expected, int):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, rel=2e-8, abs=2e-10)


@pytest.mark.parametrize("n", TAIL_LENGTHS, ids=TAIL_IDS)
@pytest.mark.parametrize("name", FEATURES)
def test_simd_tail_lengths_are_repeatable(name, n):
    """Every tail length must run the vector body and the scalar tail without
    reading past the buffer, and repeat calls must agree bit for bit."""
    x = _tail_series(n)
    if n < MIN_SIZE.get(name, 3):
        return
    assert getattr(mojo, name)(x) == getattr(mojo, name)(x)


@pytest.mark.parametrize("n", [3, 5, 6, 7, 9, 10, 11])
def test_histogram_ami_simd_tail_matches_pycatch22(n):
    rng = np.random.default_rng(n)
    x = np.ascontiguousarray(rng.normal(size=n))
    assert mojo.CO_HistogramAMI_even_2_5(x) == pytest.approx(
        reference.CO_HistogramAMI_even_2_5(x.tolist()), rel=2e-8, abs=2e-10
    )


@pytest.mark.parametrize("n", [200_000])
def test_mean3_stderr_matches_pycatch22_at_benchmark_size(n):
    rng = np.random.default_rng(n)
    x = np.ascontiguousarray(0.5 * np.sin(np.arange(n) / 50.0) + rng.normal(size=n))
    assert mojo.FC_LocalSimple_mean3_stderr(x) == pytest.approx(
        reference.FC_LocalSimple_mean3_stderr(x.tolist()), rel=2e-8, abs=2e-10
    )


def test_catch22_all_subset_order_and_catch24():
    x = np.linspace(-1.0, 1.0, 31) + np.sin(np.arange(31))
    result = mojo.catch22_all(x, catch24=True, short_names=True)
    assert result["names"] == [
        "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac",
        "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
        "SB_BinaryStats_mean_longstretch1", "FC_LocalSimple_mean1_tauresrat",
        "SB_BinaryStats_diff_longstretch0", "FC_LocalSimple_mean3_stderr",
        "DN_Mean", "DN_Spread_Std",
    ]
    assert len(result["values"]) == len(result["names"]) == len(result["short_names"])
    assert result["values"][-2:] == pytest.approx([reference.DN_Mean(x.tolist()), reference.DN_Spread_Std(x.tolist())])


def test_input_validation():
    with pytest.raises(ValueError, match="one-dimensional"):
        mojo.DN_Mean(np.ones((2, 2)))
    with pytest.raises(TypeError, match="dtype float64"):
        mojo.DN_Mean(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        mojo.DN_Mean(np.array([0.0, np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="non-constant"):
        mojo.CO_trev_1_num(np.ones(4, dtype=np.float64))
    with pytest.raises(ValueError, match="at least 5"):
        mojo.FC_LocalSimple_mean3_stderr(np.arange(4, dtype=np.float64))


def test_noncontiguous_float64_input_is_copied_safely():
    x = np.arange(40, dtype=np.float64)[::2]
    assert mojo.CO_trev_1_num(x) == pytest.approx(reference.CO_trev_1_num(x.tolist()))


def test_readonly_contiguous_input_stays_zero_copy():
    x = np.arange(20, dtype=np.float64)
    x.flags.writeable = False
    assert f64(x) is x
    assert mojo.CO_trev_1_num(x) == pytest.approx(reference.CO_trev_1_num(x.tolist()))


def test_native_abi_rejects_null_and_short_buffers():
    native = lib()
    assert np.isnan(native.mcc_mean(ctypes.c_void_p(), 0))
    assert np.isnan(native.mcc_mean3_stderr(ctypes.c_void_p(), 4))
