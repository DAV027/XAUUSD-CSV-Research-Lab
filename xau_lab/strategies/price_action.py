from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def engulfing(ctx: StrategyContext, params: dict) -> np.ndarray:
    min_body_atr = float(params.get("min_body_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(1, len(ctx)):
        values = np.array(
            [ctx.open[i - 1], ctx.close[i - 1], ctx.open[i], ctx.close[i], ctx.atr14[i]],
            dtype=float,
        )
        if not np.isfinite(values).all() or ctx.atr14[i] <= 0.0:
            continue
        prev_open, prev_close, cur_open, cur_close, atr = map(float, values)
        body = abs(cur_close - cur_open)
        if body < min_body_atr * atr:
            continue
        bullish = (
            prev_close < prev_open
            and cur_close > cur_open
            and cur_open <= prev_close
            and cur_close >= prev_open
        )
        bearish = (
            prev_close > prev_open
            and cur_close < cur_open
            and cur_open >= prev_close
            and cur_close <= prev_open
        )
        if bullish:
            out[i] = 1
        elif bearish:
            out[i] = -1
    return out


def rejection_candle(ctx: StrategyContext, params: dict) -> np.ndarray:
    ratio = float(params.get("wick_body_ratio", 2.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        values = np.array([ctx.open[i], ctx.high[i], ctx.low[i], ctx.close[i]], dtype=float)
        if not np.isfinite(values).all():
            continue
        open_, high, low, close = map(float, values)
        body = abs(close - open_)
        if body <= 0.0:
            continue
        upper = high - max(open_, close)
        lower = min(open_, close) - low
        if lower >= ratio * body and lower > upper and close > open_:
            out[i] = 1
        elif upper >= ratio * body and upper > lower and close < open_:
            out[i] = -1
    return out


def inside_bar_break(ctx: StrategyContext, params: dict) -> np.ndarray:
    buffer_atr = float(params.get("break_buffer_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(2, len(ctx)):
        values = np.array(
            [
                ctx.high[i - 2], ctx.low[i - 2], ctx.high[i - 1], ctx.low[i - 1],
                ctx.close[i], ctx.atr14[i],
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or ctx.atr14[i] <= 0.0:
            continue
        mother_high, mother_low, inside_high, inside_low, close, atr = map(float, values)
        if inside_high > mother_high or inside_low < mother_low:
            continue
        buffer = buffer_atr * atr
        if close > mother_high + buffer:
            out[i] = 1
        elif close < mother_low - buffer:
            out[i] = -1
    return out


def outside_bar(ctx: StrategyContext, params: dict) -> np.ndarray:
    min_range_atr = float(params.get("min_range_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(1, len(ctx)):
        values = np.array(
            [ctx.open[i], ctx.high[i], ctx.low[i], ctx.close[i], ctx.high[i - 1], ctx.low[i - 1], ctx.atr14[i]],
            dtype=float,
        )
        if not np.isfinite(values).all() or ctx.atr14[i] <= 0.0:
            continue
        open_, high, low, close, prev_high, prev_low, atr = map(float, values)
        if high <= prev_high or low >= prev_low or (high - low) < min_range_atr * atr:
            continue
        if close > open_:
            out[i] = 1
        elif close < open_:
            out[i] = -1
    return out


def level_sweep_reclaim(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior_highs = ctx.high[i - lookback : i]
        prior_lows = ctx.low[i - lookback : i]
        values = np.array([ctx.high[i], ctx.low[i], ctx.close[i]], dtype=float)
        if not np.isfinite(prior_highs).all() or not np.isfinite(prior_lows).all() or not np.isfinite(values).all():
            continue
        prior_high = float(np.max(prior_highs))
        prior_low = float(np.min(prior_lows))
        high, low, close = map(float, values)
        if low < prior_low and close > prior_low:
            out[i] = 1
        elif high > prior_high and close < prior_high:
            out[i] = -1
    return out


register_strategy(StrategyDefinition("price_action", "engulfing", engulfing, {"min_body_atr": (0.0, 1.0)}))
register_strategy(StrategyDefinition("price_action", "rejection_candle", rejection_candle, {"wick_body_ratio": (1.5, 5.0)}))
register_strategy(StrategyDefinition("price_action", "inside_bar_break", inside_bar_break, {"break_buffer_atr": (0.0, 0.5)}))
register_strategy(StrategyDefinition("price_action", "outside_bar", outside_bar, {"min_range_atr": (0.0, 2.0)}))
register_strategy(StrategyDefinition("price_action", "level_sweep_reclaim", level_sweep_reclaim, {"lookback": (2, 100)}))
