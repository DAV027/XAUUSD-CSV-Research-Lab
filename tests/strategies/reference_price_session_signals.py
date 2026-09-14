from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext


def reference_engulfing(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_rejection_candle(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_inside_bar_break(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_outside_bar(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_level_sweep_reclaim(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def _session_flag(ctx: StrategyContext, session: str) -> np.ndarray:
    return np.asarray(ctx.feature(f"session_{session}"), dtype=bool)


def reference_session_open_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    threshold_atr = float(params.get("threshold_atr", 0.0))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        if not flags[i] or (i > 0 and flags[i - 1]):
            continue
        values = np.array([ctx.open[i], ctx.close[i], ctx.atr14[i]], dtype=float)
        if not np.isfinite(values).all() or ctx.atr14[i] <= 0.0:
            continue
        move = float(ctx.close[i] - ctx.open[i])
        threshold = threshold_atr * float(ctx.atr14[i])
        if move > threshold:
            out[i] = 1
        elif move < -threshold:
            out[i] = -1
    return out


def reference_prior_session_high_low_break(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    out = np.zeros(len(ctx), dtype=np.int8)
    previous_high: float | None = None
    previous_low: float | None = None
    active_high: float | None = None
    active_low: float | None = None
    in_session = False

    for i in range(len(ctx)):
        if flags[i]:
            if not in_session:
                active_high = None
                active_low = None
                in_session = True
            close = float(ctx.close[i])
            if previous_high is not None and np.isfinite(close):
                if close > previous_high:
                    out[i] = 1
                elif close < previous_low:
                    out[i] = -1
            high = float(ctx.high[i])
            low = float(ctx.low[i])
            if np.isfinite(high) and np.isfinite(low):
                active_high = high if active_high is None else max(active_high, high)
                active_low = low if active_low is None else min(active_low, low)
        elif in_session:
            if active_high is not None and active_low is not None:
                previous_high = active_high
                previous_low = active_low
            active_high = None
            active_low = None
            in_session = False
    return out


def reference_opening_range_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    opening_bars = int(params["opening_range_bars"])
    if opening_bars < 1:
        raise ValueError("opening_range_bars must be >= 1")
    out = np.zeros(len(ctx), dtype=np.int8)
    range_high: float | None = None
    range_low: float | None = None
    session_bar = 0

    for i in range(len(ctx)):
        if not flags[i]:
            range_high = None
            range_low = None
            session_bar = 0
            continue
        high = float(ctx.high[i])
        low = float(ctx.low[i])
        close = float(ctx.close[i])
        if not np.isfinite([high, low, close]).all():
            session_bar += 1
            continue
        if session_bar < opening_bars:
            range_high = high if range_high is None else max(range_high, high)
            range_low = low if range_low is None else min(range_low, low)
        elif range_high is not None and range_low is not None:
            if close > range_high:
                out[i] = 1
            elif close < range_low:
                out[i] = -1
        session_bar += 1
    return out
