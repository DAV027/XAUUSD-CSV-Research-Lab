from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _atr_ok(ctx: StrategyContext, i: int) -> bool:
    return np.isfinite(ctx.atr14[i]) and ctx.atr14[i] > 0.0


def _direction(value: float, threshold: float) -> np.int8:
    if not np.isfinite(value):
        return np.int8(0)
    if value > threshold:
        return np.int8(1)
    if value < -threshold:
        return np.int8(-1)
    return np.int8(0)


def _window(values: np.ndarray, end: int, length: int) -> np.ndarray | None:
    start = end - length + 1
    if start < 0:
        return None
    out = values[start : end + 1]
    if len(out) != length or not np.isfinite(out).all():
        return None
    return out


def _ema_of_window(values: np.ndarray) -> float:
    alpha = 2.0 / (len(values) + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return ema


def sma_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    horizon = int(params["slope_horizon"])
    threshold_atr = float(params.get("threshold_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        current = _window(ctx.close, i, lookback)
        previous = _window(ctx.close, i - horizon, lookback)
        if current is None or previous is None or not _atr_ok(ctx, i):
            continue
        slope = float(np.mean(current) - np.mean(previous))
        out[i] = _direction(slope, threshold_atr * float(ctx.atr14[i]))
    return out


def ema_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    horizon = int(params["slope_horizon"])
    threshold_atr = float(params.get("threshold_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        current = _window(ctx.close, i, lookback)
        previous = _window(ctx.close, i - horizon, lookback)
        if current is None or previous is None or not _atr_ok(ctx, i):
            continue
        slope = _ema_of_window(current) - _ema_of_window(previous)
        out[i] = _direction(slope, threshold_atr * float(ctx.atr14[i]))
    return out


def regression_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold_atr = float(params.get("threshold_atr", 0.0))
    x = np.arange(lookback, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        values = _window(ctx.close, i, lookback)
        if values is None or not _atr_ok(ctx, i):
            continue
        slope = float(np.dot(x_centered, values - values.mean()) / denominator)
        out[i] = _direction(slope, threshold_atr * float(ctx.atr14[i]))
    return out


def roc_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold_pct = float(params.get("threshold_pct", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        current = float(ctx.close[i])
        prior = float(ctx.close[i - lookback])
        if not np.isfinite(current) or not np.isfinite(prior) or prior == 0.0:
            continue
        roc_pct = (current / prior - 1.0) * 100.0
        out[i] = _direction(roc_pct, threshold_pct)
    return out


def efficiency_trend(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    er_threshold = float(params["er_threshold"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        values = ctx.close[i - lookback : i + 1]
        if not np.isfinite(values).all():
            continue
        net = float(values[-1] - values[0])
        path = float(np.abs(np.diff(values)).sum())
        if path <= 0.0:
            continue
        er = abs(net) / path
        if er >= er_threshold and net != 0.0:
            out[i] = np.int8(1 if net > 0.0 else -1)
    return out


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
