from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _aggregate_return_pct(close: np.ndarray, i: int, lookback: int) -> float | None:
    if i < lookback:
        return None
    current = float(close[i])
    prior = float(close[i - lookback])
    if not np.isfinite(current) or not np.isfinite(prior) or prior == 0.0:
        return None
    return (current / prior - 1.0) * 100.0


def return_continuation(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold = float(params.get("threshold_pct", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        value = _aggregate_return_pct(ctx.close, i, lookback)
        if value is None:
            continue
        if value > threshold:
            out[i] = 1
        elif value < -threshold:
            out[i] = -1
    return out


def return_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return np.ascontiguousarray(-return_continuation(ctx, params), dtype=np.int8)


def rolling_autocorr_direction(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    min_abs = float(params.get("min_abs_autocorr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prices = ctx.close[i - lookback : i + 1]
        if not np.isfinite(prices).all() or np.any(prices[:-1] == 0.0):
            continue
        returns = prices[1:] / prices[:-1] - 1.0
        if len(returns) < 3:
            continue
        left = returns[:-1]
        right = returns[1:]
        if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if not np.isfinite(corr) or abs(corr) < min_abs:
            continue
        latest = float(returns[-1])
        if latest == 0.0:
            continue
        direction = 1 if latest > 0.0 else -1
        if corr < 0.0:
            direction = -direction
        out[i] = np.int8(direction)
    return out


def range_position_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    edge_fraction = float(params["edge_fraction"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback - 1, len(ctx)):
        highs = ctx.high[i - lookback + 1 : i + 1]
        lows = ctx.low[i - lookback + 1 : i + 1]
        close = float(ctx.close[i])
        if not np.isfinite(highs).all() or not np.isfinite(lows).all() or not np.isfinite(close):
            continue
        high = float(np.max(highs))
        low = float(np.min(lows))
        span = high - low
        if span <= 0.0:
            continue
        position = (close - low) / span
        if position >= 1.0 - edge_fraction:
            out[i] = -1
        elif position <= edge_fraction:
            out[i] = 1
    return out


def standardized_return_signal(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold = float(params["z_threshold"])
    out = np.zeros(len(ctx), dtype=np.int8)
    if len(ctx) < 2:
        return out
    returns = np.full(len(ctx), np.nan, dtype=np.float64)
    valid = np.isfinite(ctx.close[1:]) & np.isfinite(ctx.close[:-1]) & (ctx.close[:-1] != 0.0)
    returns[1:][valid] = ctx.close[1:][valid] / ctx.close[:-1][valid] - 1.0
    for i in range(lookback + 1, len(ctx)):
        prior = returns[i - lookback : i]
        current = float(returns[i])
        if not np.isfinite(prior).all() or not np.isfinite(current):
            continue
        mean = float(np.mean(prior))
        std = float(np.std(prior, ddof=0))
        if std <= 0.0:
            continue
        z = (current - mean) / std
        if z >= threshold:
            out[i] = 1
        elif z <= -threshold:
            out[i] = -1
    return out


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
