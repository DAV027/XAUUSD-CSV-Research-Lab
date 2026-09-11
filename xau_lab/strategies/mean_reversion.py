from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _finite_window(values: np.ndarray, end: int, length: int) -> np.ndarray | None:
    start = end - length + 1
    if start < 0:
        return None
    window = values[start : end + 1]
    if len(window) != length or not np.isfinite(window).all():
        return None
    return window


def zscore_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold = float(params["z_threshold"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        window = _finite_window(ctx.close, i, lookback)
        if window is None:
            continue
        std = float(np.std(window, ddof=0))
        if std <= 0.0:
            continue
        z = (float(window[-1]) - float(np.mean(window))) / std
        if z >= threshold:
            out[i] = -1
        elif z <= -threshold:
            out[i] = 1
    return out


def bollinger_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    multiplier = float(params["std_multiplier"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        window = _finite_window(ctx.close, i, lookback)
        if window is None:
            continue
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        if std <= 0.0:
            continue
        close = float(window[-1])
        upper = mean + multiplier * std
        lower = mean - multiplier * std
        if close >= upper:
            out[i] = -1
        elif close <= lower:
            out[i] = 1
    return out


def rsi_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    lower = float(params["lower"])
    upper = 100.0 - lower
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        window = _finite_window(ctx.close, i, lookback + 1)
        if window is None:
            continue
        changes = np.diff(window)
        gains = np.maximum(changes, 0.0)
        losses = np.maximum(-changes, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
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


def stochastic_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    lower = float(params["lower"])
    upper = 100.0 - lower
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        highs = _finite_window(ctx.high, i, lookback)
        lows = _finite_window(ctx.low, i, lookback)
        close = float(ctx.close[i])
        if highs is None or lows is None or not np.isfinite(close):
            continue
        highest = float(np.max(highs))
        lowest = float(np.min(lows))
        span = highest - lowest
        if span <= 0.0:
            continue
        k = 100.0 * (close - lowest) / span
        if k <= lower:
            out[i] = 1
        elif k >= upper:
            out[i] = -1
    return out


def cci_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold = float(params["threshold"])
    typical = (ctx.high + ctx.low + ctx.close) / 3.0
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        window = _finite_window(typical, i, lookback)
        if window is None:
            continue
        mean = float(np.mean(window))
        mean_deviation = float(np.mean(np.abs(window - mean)))
        if mean_deviation <= 0.0:
            continue
        cci = (float(window[-1]) - mean) / (0.015 * mean_deviation)
        if cci >= threshold:
            out[i] = -1
        elif cci <= -threshold:
            out[i] = 1
    return out


def williams_r_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    edge_band = float(params["edge_band"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        highs = _finite_window(ctx.high, i, lookback)
        lows = _finite_window(ctx.low, i, lookback)
        close = float(ctx.close[i])
        if highs is None or lows is None or not np.isfinite(close):
            continue
        highest = float(np.max(highs))
        lowest = float(np.min(lows))
        span = highest - lowest
        if span <= 0.0:
            continue
        percent_r = -100.0 * (highest - close) / span
        if percent_r >= -edge_band:
            out[i] = -1
        elif percent_r <= -(100.0 - edge_band):
            out[i] = 1
    return out


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
