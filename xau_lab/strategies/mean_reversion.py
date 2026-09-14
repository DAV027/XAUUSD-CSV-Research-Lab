from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies._kernels import _window_mean_std, _window_min_max
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _zscore_reversion_kernel(close: np.ndarray, lookback: int, threshold: float) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback - 1, len(close)):
        valid, mean, std = _window_mean_std(close, i - lookback + 1, i + 1)
        if not valid or std <= 0.0:
            continue
        z = (close[i] - mean) / std
        if z >= threshold:
            out[i] = -1
        elif z <= -threshold:
            out[i] = 1
    return out


@njit(cache=True)
def _bollinger_reversion_kernel(close: np.ndarray, lookback: int, multiplier: float) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback - 1, len(close)):
        valid, mean, std = _window_mean_std(close, i - lookback + 1, i + 1)
        if not valid or std <= 0.0:
            continue
        current = close[i]
        upper = mean + multiplier * std
        lower = mean - multiplier * std
        if current >= upper:
            out[i] = -1
        elif current <= lower:
            out[i] = 1
    return out


@njit(cache=True)
def _rsi_extreme_kernel(close: np.ndarray, lookback: int, lower: float) -> np.ndarray:
    upper = 100.0 - lower
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        start = i - lookback
        previous = close[start]
        if not np.isfinite(previous):
            continue
        gain_sum = 0.0
        loss_sum = 0.0
        valid = True
        for j in range(start + 1, i + 1):
            current = close[j]
            if not np.isfinite(current):
                valid = False
                break
            change = current - previous
            if change > 0.0:
                gain_sum += change
            elif change < 0.0:
                loss_sum -= change
            previous = current
        if not valid:
            continue
        avg_gain = gain_sum / lookback
        avg_loss = loss_sum / lookback
        if avg_gain == 0.0 and avg_loss == 0.0:
            continue
        if avg_loss == 0.0:
            rsi = 100.0
        elif avg_gain == 0.0:
            rsi = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        if rsi <= lower:
            out[i] = 1
        elif rsi >= upper:
            out[i] = -1
    return out


@njit(cache=True)
def _stochastic_extreme_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    lower: float,
) -> np.ndarray:
    upper = 100.0 - lower
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
        k = 100.0 * (current - lowest) / span
        if k <= lower:
            out[i] = 1
        elif k >= upper:
            out[i] = -1
    return out


@njit(cache=True)
def _cci_extreme_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    threshold: float,
) -> np.ndarray:
    typical = (high + low + close) / 3.0
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback - 1, len(close)):
        start = i - lookback + 1
        total = 0.0
        valid = True
        for j in range(start, i + 1):
            value = typical[j]
            if not np.isfinite(value):
                valid = False
                break
            total += value
        if not valid:
            continue
        mean = total / lookback
        deviation_sum = 0.0
        for j in range(start, i + 1):
            deviation_sum += abs(typical[j] - mean)
        mean_deviation = deviation_sum / lookback
        if mean_deviation <= 0.0:
            continue
        cci = (typical[i] - mean) / (0.015 * mean_deviation)
        if cci >= threshold:
            out[i] = -1
        elif cci <= -threshold:
            out[i] = 1
    return out


@njit(cache=True)
def _williams_r_extreme_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    edge_band: float,
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
        percent_r = -100.0 * (highest - current) / span
        if percent_r >= -edge_band:
            out[i] = -1
        elif percent_r <= -(100.0 - edge_band):
            out[i] = 1
    return out


def zscore_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _zscore_reversion_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["z_threshold"]),
    )


def bollinger_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _bollinger_reversion_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["std_multiplier"]),
    )


def rsi_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _rsi_extreme_kernel(
        ctx.close,
        int(params["lookback"]),
        float(params["lower"]),
    )


def stochastic_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _stochastic_extreme_kernel(
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["lower"]),
    )


def cci_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _cci_extreme_kernel(
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["threshold"]),
    )


def williams_r_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _williams_r_extreme_kernel(
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["edge_band"]),
    )


register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "zscore_reversion",
        zscore_reversion,
        {"lookback": (10, 200), "z_threshold": (1.0, 3.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "bollinger_reversion",
        bollinger_reversion,
        {"lookback": (10, 100), "std_multiplier": (1.0, 3.5)},
    )
)
register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "rsi_extreme",
        rsi_extreme,
        {"lookback": (5, 40), "lower": (10.0, 40.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "stochastic_extreme",
        stochastic_extreme,
        {"lookback": (5, 40), "lower": (5.0, 35.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "cci_extreme",
        cci_extreme,
        {"lookback": (5, 60), "threshold": (50.0, 250.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "mean_reversion",
        "williams_r_extreme",
        williams_r_extreme,
        {"lookback": (5, 40), "edge_band": (5.0, 35.0)},
    )
)
