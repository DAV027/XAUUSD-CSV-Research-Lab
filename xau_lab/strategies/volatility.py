from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _direction_from_body(ctx: StrategyContext, i: int) -> np.int8:
    open_ = float(ctx.open[i])
    close = float(ctx.close[i])
    if not np.isfinite(open_) or not np.isfinite(close):
        return np.int8(0)
    if close > open_:
        return np.int8(1)
    if close < open_:
        return np.int8(-1)
    return np.int8(0)


def atr_percentile_regime(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    percentile = float(params["percentile"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior = ctx.atr14[i - lookback : i]
        current = float(ctx.atr14[i])
        if not np.isfinite(prior).all() or not np.isfinite(current):
            continue
        threshold = float(np.quantile(prior, percentile))
        if current >= threshold:
            out[i] = _direction_from_body(ctx, i)
    return out


def volatility_expansion_direction(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    multiple = float(params["range_multiple"])
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior = ranges[i - lookback : i]
        current = float(ranges[i])
        if not np.isfinite(prior).all() or not np.isfinite(current):
            continue
        baseline = float(np.median(prior))
        if baseline > 0.0 and current >= multiple * baseline:
            out[i] = _direction_from_body(ctx, i)
    return out


def volatility_contraction_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    percentile = float(params["percentile"])
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior = ranges[i - lookback : i]
        current = float(ranges[i])
        if not np.isfinite(prior).all() or not np.isfinite(current):
            continue
        threshold = float(np.quantile(prior, percentile))
        if current <= threshold:
            direction = _direction_from_body(ctx, i)
            out[i] = np.int8(-direction)
    return out


register_strategy(
    StrategyDefinition(
        "volatility",
        "atr_percentile_regime",
        atr_percentile_regime,
        {"lookback": (10, 200), "percentile": (0.5, 0.95)},
    )
)
register_strategy(
    StrategyDefinition(
        "volatility",
        "volatility_expansion_direction",
        volatility_expansion_direction,
        {"lookback": (5, 100), "range_multiple": (1.1, 3.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "volatility",
        "volatility_contraction_reversion",
        volatility_contraction_reversion,
        {"lookback": (5, 100), "percentile": (0.05, 0.40)},
    )
)
