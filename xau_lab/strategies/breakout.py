from __future__ import annotations

import math

import numpy as np
from numba import njit

from xau_lab.strategies._kernels import (
    _linear_quantile_sorted_window,
    _window_mean_std,
    _window_min_max,
)
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _finite(values: np.ndarray) -> bool:
    return len(values) > 0 and np.isfinite(values).all()


@njit(cache=True)
def _max_heap_push(heap: np.ndarray, size: int, value: float) -> int:
    index = size
    heap[index] = value
    while index > 0:
        parent = (index - 1) // 2
        if heap[parent] >= heap[index]:
            break
        tmp = heap[parent]
        heap[parent] = heap[index]
        heap[index] = tmp
        index = parent
    return size + 1


@njit(cache=True)
def _min_heap_push(heap: np.ndarray, size: int, value: float) -> int:
    index = size
    heap[index] = value
    while index > 0:
        parent = (index - 1) // 2
        if heap[parent] <= heap[index]:
            break
        tmp = heap[parent]
        heap[parent] = heap[index]
        heap[index] = tmp
        index = parent
    return size + 1


@njit(cache=True)
def _max_heap_pop(heap: np.ndarray, size: int) -> tuple[float, int]:
    root = heap[0]
    new_size = size - 1
    if new_size == 0:
        return root, 0

    heap[0] = heap[new_size]
    index = 0
    while True:
        left = 2 * index + 1
        if left >= new_size:
            break
        right = left + 1
        child = left
        if right < new_size and heap[right] > heap[left]:
            child = right
        if heap[index] >= heap[child]:
            break
        tmp = heap[index]
        heap[index] = heap[child]
        heap[child] = tmp
        index = child
    return root, new_size


@njit(cache=True)
def _min_heap_pop(heap: np.ndarray, size: int) -> tuple[float, int]:
    root = heap[0]
    new_size = size - 1
    if new_size == 0:
        return root, 0

    heap[0] = heap[new_size]
    index = 0
    while True:
        left = 2 * index + 1
        if left >= new_size:
            break
        right = left + 1
        child = left
        if right < new_size and heap[right] < heap[left]:
            child = right
        if heap[index] <= heap[child]:
            break
        tmp = heap[index]
        heap[index] = heap[child]
        heap[child] = tmp
        index = child
    return root, new_size


@njit(cache=True)
def _expanding_linear_quantiles_kernel(values: np.ndarray, q: float) -> np.ndarray:
    """Exact NumPy-linear quantile of every strict prefix values[:i]."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= 1:
        return out

    lower = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    lower_size = 0
    upper_size = 0
    history_valid = True

    for i in range(1, n):
        value = values[i - 1]
        if not np.isfinite(value):
            history_valid = False
        if not history_valid:
            continue

        if lower_size == 0 or value <= lower[0]:
            lower_size = _max_heap_push(lower, lower_size, value)
        else:
            upper_size = _min_heap_push(upper, upper_size, value)

        history_size = i
        h = (history_size - 1) * q
        floor_rank = int(math.floor(h))
        desired_lower_size = floor_rank + 1

        while lower_size > desired_lower_size:
            moved, lower_size = _max_heap_pop(lower, lower_size)
            upper_size = _min_heap_push(upper, upper_size, moved)
        while lower_size < desired_lower_size:
            moved, upper_size = _min_heap_pop(upper, upper_size)
            lower_size = _max_heap_push(lower, lower_size, moved)

        lower_value = lower[0]
        fraction = h - floor_rank
        if fraction == 0.0:
            out[i] = lower_value
        else:
            upper_value = upper[0]
            difference = upper_value - lower_value
            if fraction >= 0.5:
                out[i] = upper_value - difference * (1.0 - fraction)
            else:
                out[i] = lower_value + difference * fraction

    return out


def _expanding_linear_quantiles(values: np.ndarray, q: float) -> np.ndarray:
    """Return exact default-linear quantiles for strict expanding prefixes."""
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    array = np.ascontiguousarray(array)
    return _expanding_linear_quantiles_kernel(array, q)


@njit(cache=True)
def _nbar_breakout_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        start = i - lookback
        valid_high, _, prior_high = _window_min_max(high, start, i)
        valid_low, prior_low, _ = _window_min_max(low, start, i)
        current = close[i]
        if not valid_high or not valid_low or not np.isfinite(current):
            continue
        if current > prior_high:
            out[i] = 1
        elif current < prior_low:
            out[i] = -1
    return out


@njit(cache=True)
def _nbar_failed_breakout_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        start = i - lookback
        valid_high, _, prior_high = _window_min_max(high, start, i)
        valid_low, prior_low, _ = _window_min_max(low, start, i)
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        if (
            not valid_high
            or not valid_low
            or not np.isfinite(current_high)
            or not np.isfinite(current_low)
            or not np.isfinite(current_close)
        ):
            continue
        if current_high > prior_high and current_close < prior_high:
            out[i] = -1
        elif current_low < prior_low and current_close > prior_low:
            out[i] = 1
    return out


@njit(cache=True)
def _true_ranges_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    if np.isfinite(high[0]) and np.isfinite(low[0]):
        tr[0] = high[0] - low[0]
    for i in range(1, n):
        h = high[i]
        l = low[i]
        previous_close = close[i - 1]
        if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(previous_close):
            continue
        a = h - l
        b = abs(h - previous_close)
        c = abs(l - previous_close)
        maximum = a
        if b > maximum:
            maximum = b
        if c > maximum:
            maximum = c
        tr[i] = maximum
    return tr


@njit(cache=True)
def _range_expansion_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    multiple: float,
) -> np.ndarray:
    tr = _true_ranges_kernel(high, low, close)
    out = np.zeros(len(close), dtype=np.int8)
    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, len(close)):
        current = tr[i]
        if not np.isfinite(current):
            continue
        median = _linear_quantile_sorted_window(tr, i - lookback, i, 0.5, scratch)
        if not np.isfinite(median) or median <= 0.0 or current / median < multiple:
            continue
        body = close[i] - open_[i]
        if np.isfinite(body) and body != 0.0:
            out[i] = np.int8(1 if body > 0.0 else -1)
    return out


@njit(cache=True)
def _bollinger_expansion_kernel(
    close: np.ndarray,
    lookback: int,
    multiplier: float,
    percentile: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    bandwidths = np.full(n, np.nan, dtype=np.float64)
    lowers = np.full(n, np.nan, dtype=np.float64)
    uppers = np.full(n, np.nan, dtype=np.float64)

    for i in range(lookback - 1, n):
        valid, mean, std = _window_mean_std(close, i - lookback + 1, i + 1)
        if not valid or mean == 0.0:
            continue
        upper = mean + multiplier * std
        lower = mean - multiplier * std
        lowers[i] = lower
        uppers[i] = upper
        bandwidths[i] = (upper - lower) / abs(mean)

    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(2 * lookback - 1, n):
        current_width = bandwidths[i]
        if not np.isfinite(current_width):
            continue
        threshold = _linear_quantile_sorted_window(
            bandwidths,
            i - lookback,
            i,
            percentile,
            scratch,
        )
        if not np.isfinite(threshold) or current_width < threshold:
            continue
        current_close = close[i]
        if current_close > uppers[i]:
            out[i] = 1
        elif current_close < lowers[i]:
            out[i] = -1
    return out


@njit(cache=True)
def _compression_breakout_signal_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ranges: np.ndarray,
    thresholds: np.ndarray,
    compression_lookback: int,
    breakout_lookback: int,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    warmup = max(compression_lookback, breakout_lookback)

    for i in range(warmup, n):
        threshold = thresholds[i]
        if not np.isfinite(threshold):
            continue

        compressed_sum = 0.0
        compressed_valid = True
        for j in range(i - compression_lookback, i):
            value = ranges[j]
            if not np.isfinite(value):
                compressed_valid = False
                break
            compressed_sum += value
        if not compressed_valid:
            continue
        if compressed_sum / compression_lookback > threshold:
            continue

        prior_high = high[i - breakout_lookback]
        prior_low = low[i - breakout_lookback]
        if not np.isfinite(prior_high) or not np.isfinite(prior_low):
            continue
        levels_valid = True
        for j in range(i - breakout_lookback + 1, i):
            h = high[j]
            l = low[j]
            if not np.isfinite(h) or not np.isfinite(l):
                levels_valid = False
                break
            if h > prior_high:
                prior_high = h
            if l < prior_low:
                prior_low = l
        if not levels_valid:
            continue

        current_close = close[i]
        if not np.isfinite(current_close):
            continue
        if current_close > prior_high:
            out[i] = 1
        elif current_close < prior_low:
            out[i] = -1

    return out


def nbar_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _nbar_breakout_kernel(ctx.high, ctx.low, ctx.close, int(params["lookback"]))


def nbar_failed_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _nbar_failed_breakout_kernel(ctx.high, ctx.low, ctx.close, int(params["lookback"]))


def range_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _range_expansion_kernel(
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["median_lookback"]),
        float(params["range_multiple"]),
    )


def bollinger_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _bollinger_expansion_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["std_multiplier"]),
        float(params["bandwidth_percentile"]),
    )


def compression_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    compression_lookback = int(params["compression_lookback"])
    compression_percentile = float(params["compression_percentile"])
    breakout_lookback = int(params["breakout_lookback"])
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    thresholds = _expanding_linear_quantiles(ranges, compression_percentile)
    return _compression_breakout_signal_kernel(
        ctx.high,
        ctx.low,
        ctx.close,
        ranges,
        thresholds,
        compression_lookback,
        breakout_lookback,
    )


register_strategy(StrategyDefinition("breakout", "nbar_breakout", nbar_breakout, {"lookback": (2, 100)}))
register_strategy(
    StrategyDefinition("breakout", "nbar_failed_breakout", nbar_failed_breakout, {"lookback": (2, 100)})
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "range_expansion",
        range_expansion,
        {"median_lookback": (5, 100), "range_multiple": (1.1, 3.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "bollinger_expansion",
        bollinger_expansion,
        {"lookback": (10, 80), "std_multiplier": (1.0, 3.0), "bandwidth_percentile": (0.5, 0.95)},
    )
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "compression_breakout",
        compression_breakout,
        {
            "compression_lookback": (10, 100),
            "compression_percentile": (0.05, 0.40),
            "breakout_lookback": (2, 50),
        },
    )
)
