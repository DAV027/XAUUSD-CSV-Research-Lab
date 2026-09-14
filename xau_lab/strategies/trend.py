from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _direction_kernel(value: float, threshold: float) -> np.int8:
    if not np.isfinite(value):
        return np.int8(0)
    if value > threshold:
        return np.int8(1)
    if value < -threshold:
        return np.int8(-1)
    return np.int8(0)


@njit(cache=True)
def _finite_mean_kernel(values: np.ndarray, start: int, end: int) -> tuple[bool, float]:
    if start < 0 or end > len(values) or start >= end:
        return False, np.nan
    total = 0.0
    for j in range(start, end):
        value = values[j]
        if not np.isfinite(value):
            return False, np.nan
        total += value
    return True, total / (end - start)


@njit(cache=True)
def _ema_window_kernel(values: np.ndarray, start: int, end: int) -> tuple[bool, float]:
    if start < 0 or end > len(values) or start >= end:
        return False, np.nan
    first = values[start]
    if not np.isfinite(first):
        return False, np.nan
    length = end - start
    alpha = 2.0 / (length + 1.0)
    ema = first
    for j in range(start + 1, end):
        value = values[j]
        if not np.isfinite(value):
            return False, np.nan
        ema = alpha * value + (1.0 - alpha) * ema
    return True, ema


@njit(cache=True)
def _sma_slope_kernel(
    close: np.ndarray,
    atr14: np.ndarray,
    lookback: int,
    horizon: int,
    threshold_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close)):
        if not np.isfinite(atr14[i]) or atr14[i] <= 0.0:
            continue
        current_start = i - lookback + 1
        previous_end = i - horizon + 1
        previous_start = previous_end - lookback
        valid_current, current_mean = _finite_mean_kernel(close, current_start, i + 1)
        valid_previous, previous_mean = _finite_mean_kernel(close, previous_start, previous_end)
        if not valid_current or not valid_previous:
            continue
        slope = current_mean - previous_mean
        out[i] = _direction_kernel(slope, threshold_atr * atr14[i])
    return out


@njit(cache=True)
def _ema_slope_kernel(
    close: np.ndarray,
    atr14: np.ndarray,
    lookback: int,
    horizon: int,
    threshold_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close)):
        if not np.isfinite(atr14[i]) or atr14[i] <= 0.0:
            continue
        current_start = i - lookback + 1
        previous_end = i - horizon + 1
        previous_start = previous_end - lookback
        valid_current, current_ema = _ema_window_kernel(close, current_start, i + 1)
        valid_previous, previous_ema = _ema_window_kernel(close, previous_start, previous_end)
        if not valid_current or not valid_previous:
            continue
        out[i] = _direction_kernel(current_ema - previous_ema, threshold_atr * atr14[i])
    return out


@njit(cache=True)
def _regression_slope_kernel(
    close: np.ndarray,
    atr14: np.ndarray,
    lookback: int,
    threshold_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    x_mean = (lookback - 1) / 2.0
    denominator = 0.0
    for j in range(lookback):
        centered = j - x_mean
        denominator += centered * centered

    for i in range(len(close)):
        if not np.isfinite(atr14[i]) or atr14[i] <= 0.0:
            continue
        start = i - lookback + 1
        valid, mean = _finite_mean_kernel(close, start, i + 1)
        if not valid:
            continue
        numerator = 0.0
        for j in range(lookback):
            numerator += (j - x_mean) * (close[start + j] - mean)
        slope = numerator / denominator
        out[i] = _direction_kernel(slope, threshold_atr * atr14[i])
    return out


@njit(cache=True)
def _roc_momentum_kernel(close: np.ndarray, lookback: int, threshold_pct: float) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        current = close[i]
        prior = close[i - lookback]
        if not np.isfinite(current) or not np.isfinite(prior) or prior == 0.0:
            continue
        roc_pct = (current / prior - 1.0) * 100.0
        out[i] = _direction_kernel(roc_pct, threshold_pct)
    return out


@njit(cache=True)
def _efficiency_trend_kernel(close: np.ndarray, lookback: int, er_threshold: float) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        start = i - lookback
        first = close[start]
        if not np.isfinite(first):
            continue
        previous = first
        path = 0.0
        valid = True
        for j in range(start + 1, i + 1):
            current = close[j]
            if not np.isfinite(current):
                valid = False
                break
            path += abs(current - previous)
            previous = current
        if not valid or path <= 0.0:
            continue
        net = close[i] - first
        er = abs(net) / path
        if er >= er_threshold and net != 0.0:
            out[i] = np.int8(1 if net > 0.0 else -1)
    return out


def sma_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _sma_slope_kernel(
        ctx.close,
        ctx.atr14,
        int(params["lookback"]),
        int(params["slope_horizon"]),
        float(params.get("threshold_atr", 0.0)),
    )


def ema_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _ema_slope_kernel(
        ctx.close,
        ctx.atr14,
        int(params["lookback"]),
        int(params["slope_horizon"]),
        float(params.get("threshold_atr", 0.0)),
    )


def regression_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _regression_slope_kernel(
        ctx.close,
        ctx.atr14,
        int(params["lookback"]),
        float(params.get("threshold_atr", 0.0)),
    )


def roc_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _roc_momentum_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params.get("threshold_pct", 0.0)),
    )


def efficiency_trend(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _efficiency_trend_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["er_threshold"]),
    )


register_strategy(
    StrategyDefinition(
        "trend_momentum",
        "sma_slope",
        sma_slope,
        {"lookback": (5, 100), "slope_horizon": (1, 10), "threshold_atr": (0.0, 0.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "trend_momentum",
        "ema_slope",
        ema_slope,
        {"lookback": (5, 100), "slope_horizon": (1, 10), "threshold_atr": (0.0, 0.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "trend_momentum",
        "regression_slope",
        regression_slope,
        {"lookback": (5, 120), "threshold_atr": (0.0, 0.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "trend_momentum",
        "roc_momentum",
        roc_momentum,
        {"lookback": (2, 60), "threshold_pct": (0.0, 1.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "trend_momentum",
        "efficiency_trend",
        efficiency_trend,
        {"lookback": (5, 80), "er_threshold": (0.1, 0.8)},
    )
)
