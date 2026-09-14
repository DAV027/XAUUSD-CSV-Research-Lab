from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext


def _finite_window(values: np.ndarray, end: int, length: int) -> np.ndarray | None:
    start = end - length + 1
    if start < 0:
        return None
    window = values[start : end + 1]
    if len(window) != length or not np.isfinite(window).all():
        return None
    return window


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


def reference_rsi_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_stochastic_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_cci_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_williams_r_extreme(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def _aggregate_return_pct(close: np.ndarray, i: int, lookback: int) -> float | None:
    if i < lookback:
        return None
    current = float(close[i])
    prior = float(close[i - lookback])
    if not np.isfinite(current) or not np.isfinite(prior) or prior == 0.0:
        return None
    return (current / prior - 1.0) * 100.0


def reference_return_continuation(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_return_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return np.ascontiguousarray(-reference_return_continuation(ctx, params), dtype=np.int8)


def reference_rolling_autocorr_direction(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_range_position_reversal(ctx: StrategyContext, params: dict) -> np.ndarray:
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
