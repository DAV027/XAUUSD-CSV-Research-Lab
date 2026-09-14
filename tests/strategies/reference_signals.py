from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.breakout import _expanding_linear_quantiles


def _finite(values: np.ndarray) -> bool:
    return len(values) > 0 and np.isfinite(values).all()


def _finite_window(values: np.ndarray, end: int, length: int) -> np.ndarray | None:
    start = end - length + 1
    if start < 0:
        return None
    window = values[start : end + 1]
    if len(window) != length or not np.isfinite(window).all():
        return None
    return window


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


def _trend_window(values: np.ndarray, end: int, length: int) -> np.ndarray | None:
    start = end - length + 1
    if start < 0:
        return None
    out = values[start : end + 1]
    if len(out) != length or not np.isfinite(out).all():
        return None
    return out


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


def reference_regression_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    threshold_atr = float(params.get("threshold_atr", 0.0))
    x = np.arange(lookback, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(len(ctx)):
        values = _trend_window(ctx.close, i, lookback)
        if values is None or not _atr_ok(ctx, i):
            continue
        slope = float(np.dot(x_centered, values - values.mean()) / denominator)
        out[i] = _direction(slope, threshold_atr * float(ctx.atr14[i]))
    return out


def reference_compression_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    compression_lookback = int(params["compression_lookback"])
    compression_percentile = float(params["compression_percentile"])
    breakout_lookback = int(params["breakout_lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    thresholds = _expanding_linear_quantiles(ranges, compression_percentile)
    warmup = max(compression_lookback, breakout_lookback)
    for i in range(warmup, len(ctx)):
        compressed = ranges[i - compression_lookback : i]
        breakout_highs = ctx.high[i - breakout_lookback : i]
        breakout_lows = ctx.low[i - breakout_lookback : i]
        threshold = thresholds[i]
        if not (
            _finite(compressed)
            and np.isfinite(threshold)
            and _finite(breakout_highs)
            and _finite(breakout_lows)
        ):
            continue
        if float(np.mean(compressed)) > threshold:
            continue
        close = float(ctx.close[i])
        if not np.isfinite(close):
            continue
        if close > float(np.max(breakout_highs)):
            out[i] = 1
        elif close < float(np.min(breakout_lows)):
            out[i] = -1
    return out


def reference_bollinger_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_zscore_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_opening_range_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = np.asarray(ctx.feature(f"session_{str(params['session'])}"), dtype=bool)
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


def reference_volatility_contraction_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_standardized_return_signal(ctx: StrategyContext, params: dict) -> np.ndarray:
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
