"""A Mojo implementation of a performance-oriented subset of pycatch22."""

from __future__ import annotations

import numpy as np

from ._lib import call, f64

__version__ = "0.1.0"


def _zscore(data, *, min_size: int = 2) -> np.ndarray:
    arr = f64(data)
    if arr.size < min_size:
        raise ValueError(f"time series must contain at least {min_size} values")
    scale = arr.std(ddof=1)
    if scale == 0.0:
        raise ValueError("z-scored characteristics require a non-constant time series")
    return np.ascontiguousarray((arr - arr.mean()) / scale)


def DN_Mean(data):
    return call("mcc_mean", data)


def DN_Spread_Std(data):
    arr = f64(data)
    if arr.size < 2:
        raise ValueError("time series must contain at least 2 values")
    return call("mcc_std", arr)


def DN_HistogramMode_5(data):
    return call("mcc_histogram_mode_5", _zscore(data))


def DN_HistogramMode_10(data):
    return call("mcc_histogram_mode_10", _zscore(data))


def CO_f1ecac(data):
    return call("mcc_f1ecac", _zscore(data))


def CO_FirstMin_ac(data):
    return int(call("mcc_first_min_ac", _zscore(data)))


def CO_HistogramAMI_even_2_5(data):
    return call("mcc_histogram_ami", _zscore(data, min_size=3))


def CO_trev_1_num(data):
    return call("mcc_trev", _zscore(data))


def MD_hrv_classic_pnn40(data):
    return call("mcc_pnn40", _zscore(data))


def SB_BinaryStats_mean_longstretch1(data):
    return call("mcc_mean_longstretch1", _zscore(data))


def SB_BinaryStats_diff_longstretch0(data):
    return call("mcc_diff_longstretch0", _zscore(data))


def FC_LocalSimple_mean1_tauresrat(data):
    return call("mcc_mean1_tauresrat", _zscore(data))


def FC_LocalSimple_mean3_stderr(data):
    return call("mcc_mean3_stderr", _zscore(data, min_size=5))


_FEATURES = (
    ("DN_HistogramMode_5", DN_HistogramMode_5, "mode_5"),
    ("DN_HistogramMode_10", DN_HistogramMode_10, "mode_10"),
    ("CO_f1ecac", CO_f1ecac, "acf_timescale"),
    ("CO_FirstMin_ac", CO_FirstMin_ac, "acf_first_min"),
    ("CO_HistogramAMI_even_2_5", CO_HistogramAMI_even_2_5, "ami2"),
    ("CO_trev_1_num", CO_trev_1_num, "trev"),
    ("MD_hrv_classic_pnn40", MD_hrv_classic_pnn40, "high_fluctuation"),
    ("SB_BinaryStats_mean_longstretch1", SB_BinaryStats_mean_longstretch1, "stretch_high"),
    ("FC_LocalSimple_mean1_tauresrat", FC_LocalSimple_mean1_tauresrat, "whiten_timescale"),
    ("SB_BinaryStats_diff_longstretch0", SB_BinaryStats_diff_longstretch0, "stretch_decreasing"),
    ("FC_LocalSimple_mean3_stderr", FC_LocalSimple_mean3_stderr, "forecast_error"),
)


def catch22_all(data, catch24=False, short_names=False):
    """Return the covered characteristics in pycatch22's dictionary format.

    This intentionally returns only the subset implemented by this project.
    """
    entries = list(_FEATURES)
    if catch24:
        entries.extend((("DN_Mean", DN_Mean, "mean"), ("DN_Spread_Std", DN_Spread_Std, "std")))
    result = {"names": [name for name, _, _ in entries], "values": [fn(data) for _, fn, _ in entries]}
    if short_names:
        result["short_names"] = [short for _, _, short in entries]
    return result


catch22 = catch22_all

__all__ = [name for name, _, _ in _FEATURES] + ["DN_Mean", "DN_Spread_Std", "catch22", "catch22_all"]
