from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _trend_pullback_recovery_kernel(
    close: np.ndarray,
    atr14: np.ndarray,
    trend_lookback: int,
    trend_threshold_atr: float,
    pullback_lookback: int,
    pullback_threshold_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    warmup = max(trend_lookback, pullback_lookback) + 1

    for i in range(warmup, len(close)):
        atr = atr14[i - 1]
        current = close[i]
        prior = close[i - 1]
        prior2 = close[i - 2]
        trend_anchor = close[i - 1 - trend_lookback]
        pullback_anchor = close[i - 1 - pullback_lookback]

        if (
            not np.isfinite(atr)
            or atr <= 0.0
            or not np.isfinite(current)
            or not np.isfinite(prior)
            or not np.isfinite(prior2)
            or not np.isfinite(trend_anchor)
            or not np.isfinite(pullback_anchor)
        ):
            continue

        trend_move = prior - trend_anchor
        pullback_move = prior - pullback_anchor
        trend_threshold = trend_threshold_atr * atr
        pullback_threshold = pullback_threshold_atr * atr

        if trend_move >= trend_threshold:
            if (
                pullback_move <= -pullback_threshold
                and prior < prior2
                and current > prior
            ):
                out[i] = np.int8(1)
        elif trend_move <= -trend_threshold:
            if (
                pullback_move >= pullback_threshold
                and prior > prior2
                and current < prior
            ):
                out[i] = np.int8(-1)

    return out


def trend_pullback_recovery(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _trend_pullback_recovery_kernel(
        ctx.close,
        ctx.atr14,
        int(params["trend_lookback"]),
        float(params["trend_threshold_atr"]),
        int(params["pullback_lookback"]),
        float(params["pullback_threshold_atr"]),
    )


register_strategy(
    StrategyDefinition(
        "edge_b_trend_pullback",
        "trend_pullback_recovery",
        trend_pullback_recovery,
        {
            "trend_lookback": (60, 240),
            "trend_threshold_atr": (1.0, 4.0),
            "pullback_lookback": (3, 20),
            "pullback_threshold_atr": (0.25, 1.5),
        },
    )
)


__all__ = ["trend_pullback_recovery"]
