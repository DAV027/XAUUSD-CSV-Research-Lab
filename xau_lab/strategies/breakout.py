from __future__ import annotations

import math

import numpy as np
from numba import njit

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

    # The two heaps partition the finite history around the requested floor rank.
    # Only insertions are required because the quantile is expanding, not rolling.
    lower = np.empty(n, dtype=np.float64)  # max heap, ranks <= floor(h)
    upper = np.empty(n, dtype=np.float64)  # min heap, ranks > floor(h)
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
            # Match NumPy's stable linear interpolation branch in _lerp.
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


def nbar_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior_high = ctx.high[i - lookback : i]
        prior_low = ctx.low[i - lookback : i]
        close = float(ctx.close[i])
        if not _finite(prior_high) or not _finite(prior_low) or not np.isfinite(close):
            continue
        if close > float(np.max(prior_high)):
            out[i] = 1
        elif close < float(np.min(prior_low)):
            out[i] = -1
    return out


def nbar_failed_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior_highs = ctx.high[i - lookback : i]
        prior_lows = ctx.low[i - lookback : i]
        if not _finite(prior_highs) or not _finite(prior_lows):
            continue
        high = float(ctx.high[i])
        low = float(ctx.low[i])
        close = float(ctx.close[i])
        if not np.isfinite([high, low, close]).all():
            continue
        prior_high = float(np.max(prior_highs))
        prior_low = float(np.min(prior_lows))
        if high > prior_high and close < prior_high:
            out[i] = -1
        elif low < prior_low and close > prior_low:
            out[i] = 1
    return out


def _true_ranges(ctx: StrategyContext) -> np.ndarray:
    n = len(ctx)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    if np.isfinite(ctx.high[0]) and np.isfinite(ctx.low[0]):
        tr[0] = float(ctx.high[0] - ctx.low[0])
    for i in range(1, n):
        values = np.array(
            [
                ctx.high[i] - ctx.low[i],
                abs(ctx.high[i] - ctx.close[i - 1]),
                abs(ctx.low[i] - ctx.close[i - 1]),
            ],
            dtype=np.float64,
        )
        if np.isfinite(values).all():
            tr[i] = float(np.max(values))
    return tr


def range_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["median_lookback"])
    multiple = float(params["range_multiple"])
    tr = _true_ranges(ctx)
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior = tr[i - lookback : i]
        if not _finite(prior) or not np.isfinite(tr[i]):
            continue
        median = float(np.median(prior))
        if median <= 0.0 or tr[i] / median < multiple:
            continue
        body = float(ctx.close[i] - ctx.open[i])
        if np.isfinite(body) and body != 0.0:
            out[i] = 1 if body > 0.0 else -1
    return out


def _band_stats(values: np.ndarray, multiplier: float) -> tuple[float, float, float] | None:
    if not _finite(values):
        return None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if mean == 0.0:
        return None
    upper = mean + multiplier * std
    lower = mean - multiplier * std
    bandwidth = (upper - lower) / abs(mean)
    return lower, upper, bandwidth


def bollinger_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    multiplier = float(params["std_multiplier"])
    percentile = float(params["bandwidth_percentile"])
    out = np.zeros(len(ctx), dtype=np.int8)
    bandwidths = np.full(len(ctx), np.nan, dtype=np.float64)
    lowers = np.full(len(ctx), np.nan, dtype=np.float64)
    uppers = np.full(len(ctx), np.nan, dtype=np.float64)
    for i in range(lookback - 1, len(ctx)):
        stats = _band_stats(ctx.close[i - lookback + 1 : i + 1], multiplier)
        if stats is not None:
            lowers[i], uppers[i], bandwidths[i] = stats
    for i in range(2 * lookback - 1, len(ctx)):
        prior_widths = bandwidths[i - lookback : i]
        if not _finite(prior_widths) or not np.isfinite(bandwidths[i]):
            continue
        threshold = float(np.quantile(prior_widths, percentile))
        if bandwidths[i] < threshold:
            continue
        close = float(ctx.close[i])
        if close > uppers[i]:
            out[i] = 1
        elif close < lowers[i]:
            out[i] = -1
    return out


def compression_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    compression_lookback = int(params["compression_lookback"])
    compression_percentile = float(params["compression_percentile"])
    breakout_lookback = int(params["breakout_lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    thresholds = _expanding_linear_quantiles(ranges, compression_percentile)
    warmup = max(compression_lookback, breakout_lookback)
    for i in range(warmup, len(ctx)):
        compressed = ranges[i - compression_lookback : i]
        breakout_highs = ctx.high[i - breakout_lookback : i]
        breakout_lows = ctx.low[i - breakout_lookback : i]
        threshold = thresholds[i]
        if not (
            _finite(compressed)
            and np.isfinite(threshold)
            and _finite(breakout_highs)
            and _finite(breakout_lows)
        ):
            continue
        if float(np.mean(compressed)) > threshold:
            continue
        close = float(ctx.close[i])
        if not np.isfinite(close):
            continue
        if close > float(np.max(breakout_highs)):
            out[i] = 1
        elif close < float(np.min(breakout_lows)):
            out[i] = -1
    return out


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
