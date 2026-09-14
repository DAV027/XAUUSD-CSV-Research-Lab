from __future__ import annotations

import math

import numpy as np
from numba import njit

from xau_lab.strategies._kernels import _window_mean_std, _window_min_max
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _return_continuation_kernel(close: np.ndarray, lookback: int, threshold: float) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        current = close[i]
        prior = close[i - lookback]
        if not np.isfinite(current) or not np.isfinite(prior) or prior == 0.0:
            continue
        value = (current / prior - 1.0) * 100.0
        if value > threshold:
            out[i] = 1
        elif value < -threshold:
            out[i] = -1
    return out


@njit(cache=True)
def _rolling_autocorr_direction_kernel(
    close: np.ndarray,
    lookback: int,
    min_abs: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    returns = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, len(close)):
        start = i - lookback
        valid = True
        for j in range(lookback):
            prior = close[start + j]
            current = close[start + j + 1]
            if not np.isfinite(prior) or not np.isfinite(current) or prior == 0.0:
                valid = False
                break
            returns[j] = current / prior - 1.0
        if not valid or lookback < 3:
            continue

        pair_count = lookback - 1
        left_sum = 0.0
        right_sum = 0.0
        for j in range(pair_count):
            left_sum += returns[j]
            right_sum += returns[j + 1]
        left_mean = left_sum / pair_count
        right_mean = right_sum / pair_count

        left_ss = 0.0
        right_ss = 0.0
        cross = 0.0
        for j in range(pair_count):
            left_delta = returns[j] - left_mean
            right_delta = returns[j + 1] - right_mean
            left_ss += left_delta * left_delta
            right_ss += right_delta * right_delta
            cross += left_delta * right_delta
        if left_ss == 0.0 or right_ss == 0.0:
            continue
        corr = cross / math.sqrt(left_ss * right_ss)
        if not np.isfinite(corr) or abs(corr) < min_abs:
            continue
        latest = returns[lookback - 1]
        if latest == 0.0:
            continue
        direction = 1 if latest > 0.0 else -1
        if corr < 0.0:
            direction = -direction
        out[i] = np.int8(direction)
    return out


@njit(cache=True)
def _range_position_reversal_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    edge_fraction: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback - 1, len(close)):
        start = i - lookback + 1
        valid_high, _, highest = _window_min_max(high, start, i + 1)
        valid_low, lowest, _ = _window_min_max(low, start, i + 1)
        current = close[i]
        if not valid_high or not valid_low or not np.isfinite(current):
            continue
        span = highest - lowest
        if span <= 0.0:
            continue
        position = (current - lowest) / span
        if position >= 1.0 - edge_fraction:
            out[i] = -1
        elif position <= edge_fraction:
            out[i] = 1
    return out


@njit(cache=True)
def _standardized_return_signal_kernel(
    close: np.ndarray,
    lookback: int,
    threshold: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    if n < 2:
        return out
    returns = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        current = close[i]
        prior = close[i - 1]
        if np.isfinite(current) and np.isfinite(prior) and prior != 0.0:
            returns[i] = current / prior - 1.0
    for i in range(lookback + 1, n):
        current = returns[i]
        if not np.isfinite(current):
            continue
        valid, mean, std = _window_mean_std(returns, i - lookback, i)
        if not valid or std <= 0.0:
            continue
        z = (current - mean) / std
        if z >= threshold:
            out[i] = 1
        elif z <= -threshold:
            out[i] = -1
    return out


def return_continuation(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _return_continuation_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params.get("threshold_pct", 0.0)),
    )


def return_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return np.ascontiguousarray(-return_continuation(ctx, params), dtype=np.int8)


def rolling_autocorr_direction(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _rolling_autocorr_direction_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params.get("min_abs_autocorr", 0.0)),
    )


def range_position_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _range_position_reversal_kernel(
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["edge_fraction"]),
    )


def standardized_return_signal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _standardized_return_signal_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["z_threshold"]),
    )


register_strategy(
    StrategyDefinition(
        "statistical",
        "return_continuation",
        return_continuation,
        {"lookback": (1, 60), "threshold_pct": (0.0, 1.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "statistical",
        "return_reversal",
        return_reversal,
        {"lookback": (1, 60), "threshold_pct": (0.0, 1.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "statistical",
        "rolling_autocorr_direction",
        rolling_autocorr_direction,
        {"lookback": (5, 100), "min_abs_autocorr": (0.0, 0.8)},
    )
)
register_strategy(
    StrategyDefinition(
        "statistical",
        "range_position_reversal",
        range_position_reversal,
        {"lookback": (5, 100), "edge_fraction": (0.05, 0.40)},
    )
)
register_strategy(
    StrategyDefinition(
        "statistical",
        "standardized_return_signal",
        standardized_return_signal,
        {"lookback": (5, 100), "z_threshold": (0.5, 3.5)},
    )
)
