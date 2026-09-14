from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies._kernels import _body_direction, _linear_quantile_sorted_window
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _atr_percentile_regime_kernel(
    open_: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    lookback: int,
    percentile: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, n):
        current = atr14[i]
        if not np.isfinite(current):
            continue
        threshold = _linear_quantile_sorted_window(
            atr14,
            i - lookback,
            i,
            percentile,
            scratch,
        )
        if not np.isfinite(threshold):
            continue
        if current >= threshold:
            out[i] = _body_direction(open_[i], close[i])
    return out


@njit(cache=True)
def _volatility_expansion_direction_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    multiple: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    ranges = high - low
    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, n):
        current = ranges[i]
        if not np.isfinite(current):
            continue
        baseline = _linear_quantile_sorted_window(
            ranges,
            i - lookback,
            i,
            0.5,
            scratch,
        )
        if not np.isfinite(baseline):
            continue
        if baseline > 0.0 and current >= multiple * baseline:
            out[i] = _body_direction(open_[i], close[i])
    return out


@njit(cache=True)
def _volatility_contraction_reversion_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int,
    percentile: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    ranges = high - low
    scratch = np.empty(lookback, dtype=np.float64)
    for i in range(lookback, n):
        current = ranges[i]
        if not np.isfinite(current):
            continue
        threshold = _linear_quantile_sorted_window(
            ranges,
            i - lookback,
            i,
            percentile,
            scratch,
        )
        if not np.isfinite(threshold) or current > threshold:
            continue
        out[i] = np.int8(-_body_direction(open_[i], close[i]))
    return out


def atr_percentile_regime(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _atr_percentile_regime_kernel(
        ctx.open,
        ctx.close,
        ctx.atr14,
        int(params["lookback"]),
        float(params["percentile"]),
    )


def volatility_expansion_direction(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _volatility_expansion_direction_kernel(
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["range_multiple"]),
    )


def volatility_contraction_reversion(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _volatility_contraction_reversion_kernel(
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        int(params["lookback"]),
        float(params["percentile"]),
    )


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
