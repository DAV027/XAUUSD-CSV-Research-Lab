from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.breakout import _expanding_linear_quantiles


def _finite(values: np.ndarray) -> bool:
    return len(values) > 0 and np.isfinite(values).all()


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


def reference_sma_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_ema_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_regression_slope(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_roc_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_efficiency_trend(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def reference_nbar_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior_high = ctx.high[i - lookback : i]
        prior_low = ctx.low[i - lookback : i]
        close = float(ctx.close[i])
        if not _finite(prior_high) or not _finite(prior_low) or not np.isfinite(close):
            continue
        if close > float(np.max(prior_high)):
            out[i] = 1
        elif close < float(np.min(prior_low)):
            out[i] = -1
    return out


def reference_nbar_failed_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior_highs = ctx.high[i - lookback : i]
        prior_lows = ctx.low[i - lookback : i]
        if not _finite(prior_highs) or not _finite(prior_lows):
            continue
        high = float(ctx.high[i])
        low = float(ctx.low[i])
        close = float(ctx.close[i])
        if not np.isfinite([high, low, close]).all():
            continue
        prior_high = float(np.max(prior_highs))
        prior_low = float(np.min(prior_lows))
        if high > prior_high and close < prior_high:
            out[i] = -1
        elif low < prior_low and close > prior_low:
            out[i] = 1
    return out


def _true_ranges(ctx: StrategyContext) -> np.ndarray:
    n = len(ctx)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    if np.isfinite(ctx.high[0]) and np.isfinite(ctx.low[0]):
        tr[0] = float(ctx.high[0] - ctx.low[0])
    for i in range(1, n):
        values = np.array(
            [
                ctx.high[i] - ctx.low[i],
                abs(ctx.high[i] - ctx.close[i - 1]),
                abs(ctx.low[i] - ctx.close[i - 1]),
            ],
            dtype=np.float64,
        )
        if np.isfinite(values).all():
            tr[i] = float(np.max(values))
    return tr


def reference_range_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["median_lookback"])
    multiple = float(params["range_multiple"])
    tr = _true_ranges(ctx)
    out = np.zeros(len(ctx), dtype=np.int8)
    for i in range(lookback, len(ctx)):
        prior = tr[i - lookback : i]
        if not _finite(prior) or not np.isfinite(tr[i]):
            continue
        median = float(np.median(prior))
        if median <= 0.0 or tr[i] / median < multiple:
            continue
        body = float(ctx.close[i] - ctx.open[i])
        if np.isfinite(body) and body != 0.0:
            out[i] = 1 if body > 0.0 else -1
    return out


def _band_stats(values: np.ndarray, multiplier: float) -> tuple[float, float, float] | None:
    if not _finite(values):
        return None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if mean == 0.0:
        return None
    upper = mean + multiplier * std
    lower = mean - multiplier * std
    bandwidth = (upper - lower) / abs(mean)
    return lower, upper, bandwidth


def reference_bollinger_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
    lookback = int(params["lookback"])
    multiplier = float(params["std_multiplier"])
    percentile = float(params["bandwidth_percentile"])
    out = np.zeros(len(ctx), dtype=np.int8)
    bandwidths = np.full(len(ctx), np.nan, dtype=np.float64)
    lowers = np.full(len(ctx), np.nan, dtype=np.float64)
    uppers = np.full(len(ctx), np.nan, dtype=np.float64)
    for i in range(lookback - 1, len(ctx)):
        stats = _band_stats(ctx.close[i - lookback + 1 : i + 1], multiplier)
        if stats is not None:
            lowers[i], uppers[i], bandwidths[i] = stats
    for i in range(2 * lookback - 1, len(ctx)):
        prior_widths = bandwidths[i - lookback : i]
        if not _finite(prior_widths) or not np.isfinite(bandwidths[i]):
            continue
        threshold = float(np.quantile(prior_widths, percentile))
        if bandwidths[i] < threshold:
            continue
        close = float(ctx.close[i])
        if close > uppers[i]:
            out[i] = 1
        elif close < lowers[i]:
            out[i] = -1
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
