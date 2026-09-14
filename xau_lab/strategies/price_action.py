from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies._kernels import _window_min_max
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _engulfing_kernel(
    open_: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    min_body_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(1, len(close)):
        prev_open = open_[i - 1]
        prev_close = close[i - 1]
        cur_open = open_[i]
        cur_close = close[i]
        atr = atr14[i]
        if (
            not np.isfinite(prev_open)
            or not np.isfinite(prev_close)
            or not np.isfinite(cur_open)
            or not np.isfinite(cur_close)
            or not np.isfinite(atr)
            or atr <= 0.0
        ):
            continue
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


@njit(cache=True)
def _rejection_candle_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ratio: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close)):
        current_open = open_[i]
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        if (
            not np.isfinite(current_open)
            or not np.isfinite(current_high)
            or not np.isfinite(current_low)
            or not np.isfinite(current_close)
        ):
            continue
        body = abs(current_close - current_open)
        if body <= 0.0:
            continue
        upper = current_high - max(current_open, current_close)
        lower = min(current_open, current_close) - current_low
        if lower >= ratio * body and lower > upper and current_close > current_open:
            out[i] = 1
        elif upper >= ratio * body and upper > lower and current_close < current_open:
            out[i] = -1
    return out


@njit(cache=True)
def _inside_bar_break_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    buffer_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(2, len(close)):
        mother_high = high[i - 2]
        mother_low = low[i - 2]
        inside_high = high[i - 1]
        inside_low = low[i - 1]
        current_close = close[i]
        atr = atr14[i]
        if (
            not np.isfinite(mother_high)
            or not np.isfinite(mother_low)
            or not np.isfinite(inside_high)
            or not np.isfinite(inside_low)
            or not np.isfinite(current_close)
            or not np.isfinite(atr)
            or atr <= 0.0
        ):
            continue
        if inside_high > mother_high or inside_low < mother_low:
            continue
        buffer = buffer_atr * atr
        if current_close > mother_high + buffer:
            out[i] = 1
        elif current_close < mother_low - buffer:
            out[i] = -1
    return out


@njit(cache=True)
def _outside_bar_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    min_range_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(1, len(close)):
        current_open = open_[i]
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        atr = atr14[i]
        if (
            not np.isfinite(current_open)
            or not np.isfinite(current_high)
            or not np.isfinite(current_low)
            or not np.isfinite(current_close)
            or not np.isfinite(prev_high)
            or not np.isfinite(prev_low)
            or not np.isfinite(atr)
            or atr <= 0.0
        ):
            continue
        if current_high <= prev_high or current_low >= prev_low or (current_high - current_low) < min_range_atr * atr:
            continue
        if current_close > current_open:
            out[i] = 1
        elif current_close < current_open:
            out[i] = -1
    return out


@njit(cache=True)
def _level_sweep_reclaim_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(lookback, len(close)):
        highs_valid, _, prior_high = _window_min_max(high, i - lookback, i)
        lows_valid, prior_low, _ = _window_min_max(low, i - lookback, i)
        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        if (
            not highs_valid
            or not lows_valid
            or not np.isfinite(current_high)
            or not np.isfinite(current_low)
            or not np.isfinite(current_close)
        ):
            continue
        if current_low < prior_low and current_close > prior_low:
            out[i] = 1
        elif current_high > prior_high and current_close < prior_high:
            out[i] = -1
    return out


def engulfing(ctx: StrategyContext, params: dict) -> np.ndarray:
    min_body_atr = float(params.get("min_body_atr", 0.0))
    return _engulfing_kernel(ctx.open, ctx.close, ctx.atr14, min_body_atr)


def rejection_candle(ctx: StrategyContext, params: dict) -> np.ndarray:
    ratio = float(params.get("wick_body_ratio", 2.0))
    return _rejection_candle_kernel(ctx.open, ctx.high, ctx.low, ctx.close, ratio)


def inside_bar_break(ctx: StrategyContext, params: dict) -> np.ndarray:
    buffer_atr = float(params.get("break_buffer_atr", 0.0))
    return _inside_bar_break_kernel(ctx.high, ctx.low, ctx.close, ctx.atr14, buffer_atr)


def outside_bar(ctx: StrategyContext, params: dict) -> np.ndarray:
    min_range_atr = float(params.get("min_range_atr", 0.0))
    return _outside_bar_kernel(ctx.open, ctx.high, ctx.low, ctx.close, ctx.atr14, min_range_atr)


def level_sweep_reclaim(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    return _level_sweep_reclaim_kernel(ctx.high, ctx.low, ctx.close, lookback)


register_strategy(StrategyDefinition("price_action", "engulfing", engulfing, {"min_body_atr": (0.0, 1.0)}))
register_strategy(StrategyDefinition("price_action", "rejection_candle", rejection_candle, {"wick_body_ratio": (1.5, 5.0)}))
register_strategy(StrategyDefinition("price_action", "inside_bar_break", inside_bar_break, {"break_buffer_atr": (0.0, 0.5)}))
register_strategy(StrategyDefinition("price_action", "outside_bar", outside_bar, {"min_range_atr": (0.0, 2.0)}))
register_strategy(StrategyDefinition("price_action", "level_sweep_reclaim", level_sweep_reclaim, {"lookback": (2, 100)}))
