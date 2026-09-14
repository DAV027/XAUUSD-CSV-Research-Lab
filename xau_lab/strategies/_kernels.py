from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _window_all_finite(values: np.ndarray, start: int, end: int) -> bool:
    if start < 0 or end > len(values) or start >= end:
        return False
    for j in range(start, end):
        if not np.isfinite(values[j]):
            return False
    return True


@njit(cache=True)
def _window_mean_std(values: np.ndarray, start: int, end: int):
    if not _window_all_finite(values, start, end):
        return False, np.nan, np.nan
    n = end - start
    total = 0.0
    for j in range(start, end):
        total += values[j]
    mean = total / n
    sq = 0.0
    for j in range(start, end):
        d = values[j] - mean
        sq += d * d
    return True, mean, math.sqrt(sq / n)


@njit(cache=True)
def _window_min_max(values: np.ndarray, start: int, end: int):
    if not _window_all_finite(values, start, end):
        return False, np.nan, np.nan
    low = values[start]
    high = values[start]
    for j in range(start + 1, end):
        value = values[j]
        if value < low:
            low = value
        if value > high:
            high = value
    return True, low, high


@njit(cache=True)
def _linear_quantile_sorted_window(
    values: np.ndarray,
    start: int,
    end: int,
    q: float,
    scratch: np.ndarray,
) -> float:
    n = end - start
    if start < 0 or end > len(values) or n <= 0 or n > len(scratch) or q < 0.0 or q > 1.0:
        return np.nan
    for j in range(n):
        value = values[start + j]
        if not np.isfinite(value):
            return np.nan
        scratch[j] = value
    ordered = np.sort(scratch[:n])
    h = (n - 1) * q
    lo = int(math.floor(h))
    fraction = h - lo
    if fraction == 0.0:
        return ordered[lo]
    hi = lo + 1
    difference = ordered[hi] - ordered[lo]
    if fraction >= 0.5:
        return ordered[hi] - difference * (1.0 - fraction)
    return ordered[lo] + difference * fraction


@njit(cache=True)
def _body_direction(open_: float, close: float) -> np.int8:
    if not np.isfinite(open_) or not np.isfinite(close):
        return np.int8(0)
    if close > open_:
        return np.int8(1)
    if close < open_:
        return np.int8(-1)
    return np.int8(0)
