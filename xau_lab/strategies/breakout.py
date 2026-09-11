from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _finite(values: np.ndarray) -> bool:
    return len(values) > 0 and np.isfinite(values).all()


def nbar_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def nbar_failed_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def range_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def bollinger_expansion(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def compression_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    compression_lookback = int(params["compression_lookback"])
    compression_percentile = float(params["compression_percentile"])
    breakout_lookback = int(params["breakout_lookback"])
    out = np.zeros(len(ctx), dtype=np.int8)
    ranges = np.asarray(ctx.high - ctx.low, dtype=np.float64)
    warmup = max(compression_lookback, breakout_lookback)
    for i in range(warmup, len(ctx)):
        compressed = ranges[i - compression_lookback : i]
        history = ranges[:i]
        breakout_highs = ctx.high[i - breakout_lookback : i]
        breakout_lows = ctx.low[i - breakout_lookback : i]
        if not (_finite(compressed) and _finite(history) and _finite(breakout_highs) and _finite(breakout_lows)):
            continue
        threshold = float(np.quantile(history, compression_percentile))
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


register_strategy(StrategyDefinition("breakout", "nbar_breakout", nbar_breakout, {"lookback": (2, 100)}))
register_strategy(
    StrategyDefinition("breakout", "nbar_failed_breakout", nbar_failed_breakout, {"lookback": (2, 100)})
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "range_expansion",
        range_expansion,
        {"median_lookback": (5, 100), "range_multiple": (1.1, 3.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "bollinger_expansion",
        bollinger_expansion,
        {"lookback": (10, 80), "std_multiplier": (1.0, 3.0), "bandwidth_percentile": (0.5, 0.95)},
    )
)
register_strategy(
    StrategyDefinition(
        "breakout",
        "compression_breakout",
        compression_breakout,
        {
            "compression_lookback": (10, 100),
            "compression_percentile": (0.05, 0.40),
            "breakout_lookback": (2, 50),
        },
    )
)
