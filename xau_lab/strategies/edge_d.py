from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


EDGE_D_FAMILY = "edge_d_session_sweep_reclaim"
EDGE_D_STRATEGY = "prior_session_sweep_reclaim"

_TRANSITION_ASIA_TO_LONDON = 0
_TRANSITION_LONDON_PRE_NY_TO_NEW_YORK = 1


@njit(cache=True)
def _latest_positive_atr_before(atr14: np.ndarray, end_exclusive: int) -> float:
    for j in range(end_exclusive - 1, -1, -1):
        value = atr14[j]
        if np.isfinite(value) and value > 0.0:
            return value
    return np.nan


@njit(cache=True)
def _prior_session_sweep_reclaim_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    asia: np.ndarray,
    london: np.ndarray,
    new_york: np.ndarray,
    transition_code: int,
    entry_window_bars: int,
    sweep_atr: float,
    max_reclaim_bars: int,
    reclaim_depth_atr: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)

    asia_active_high = 0.0
    asia_active_low = 0.0
    asia_has_active = False
    asia_completed_high = 0.0
    asia_completed_low = 0.0
    asia_has_completed = False

    london_active_high = 0.0
    london_active_low = 0.0
    london_has_active = False

    armed = False
    locked = False
    invalidated = False
    reference_high = 0.0
    reference_low = 0.0
    frozen_atr = 0.0
    target_bar_index = 0
    sweep_direction = 0  # +1 upper sweep -> short; -1 lower sweep -> long
    sweep_bar_index = -1

    for i in range(n):
        # Finalize a completed Asia range before a same-index London start is armed.
        if i > 0 and asia[i - 1] and not asia[i] and asia_has_active:
            asia_completed_high = asia_active_high
            asia_completed_low = asia_active_low
            asia_has_completed = True
            asia_has_active = False

        london_start = london[i] and (i == 0 or not london[i - 1])
        ny_start = new_york[i] and (i == 0 or not new_york[i - 1])
        target_start = False
        target_flag = False
        can_arm = False
        candidate_high = 0.0
        candidate_low = 0.0

        if transition_code == _TRANSITION_ASIA_TO_LONDON:
            target_start = london_start
            target_flag = london[i]
            if target_start and asia_has_completed:
                candidate_high = asia_completed_high
                candidate_low = asia_completed_low
                can_arm = True
                # Consume this completed Asia range so it cannot arm a second London start.
                asia_has_completed = False
        else:
            target_start = ny_start
            target_flag = new_york[i]
            if target_start and london_has_active:
                candidate_high = london_active_high
                candidate_low = london_active_low
                can_arm = True

        if target_start:
            armed = False
            locked = False
            invalidated = False
            target_bar_index = 0
            sweep_direction = 0
            sweep_bar_index = -1
            if (
                can_arm
                and np.isfinite(candidate_high)
                and np.isfinite(candidate_low)
                and candidate_high > candidate_low
            ):
                lagged_atr = _latest_positive_atr_before(atr14, i)
                if np.isfinite(lagged_atr) and lagged_atr > 0.0:
                    reference_high = candidate_high
                    reference_low = candidate_low
                    frozen_atr = lagged_atr
                    armed = True

        if armed:
            if not target_flag:
                armed = False
            elif target_bar_index >= entry_window_bars:
                armed = False
            elif not locked and not invalidated:
                current_open = open_[i]
                current_high = high[i]
                current_low = low[i]
                current_close = close[i]
                if (
                    np.isfinite(current_open)
                    and np.isfinite(current_high)
                    and np.isfinite(current_low)
                    and np.isfinite(current_close)
                ):
                    upper_threshold = reference_high + sweep_atr * frozen_atr
                    lower_threshold = reference_low - sweep_atr * frozen_atr
                    upper_sweep = current_high > upper_threshold
                    lower_sweep = current_low < lower_threshold

                    if sweep_direction == 0:
                        if upper_sweep and lower_sweep:
                            invalidated = True
                        elif upper_sweep:
                            sweep_direction = 1
                            sweep_bar_index = target_bar_index
                        elif lower_sweep:
                            sweep_direction = -1
                            sweep_bar_index = target_bar_index
                    else:
                        opposite_sweep = (
                            sweep_direction == 1 and lower_sweep
                        ) or (
                            sweep_direction == -1 and upper_sweep
                        )
                        if opposite_sweep:
                            invalidated = True

                    if not invalidated and sweep_direction != 0:
                        age = target_bar_index - sweep_bar_index
                        if age > max_reclaim_bars:
                            invalidated = True
                        elif sweep_direction == 1:
                            reclaim_level = reference_high - reclaim_depth_atr * frozen_atr
                            if current_close < reclaim_level and current_close < current_open:
                                out[i] = np.int8(-1)
                                locked = True
                        else:
                            reclaim_level = reference_low + reclaim_depth_atr * frozen_atr
                            if current_close > reclaim_level and current_close > current_open:
                                out[i] = np.int8(1)
                                locked = True

            target_bar_index += 1

        # Update reference-session ranges after target-start freezing. This ordering is
        # essential for the London-pre-NY transition: the first NY bar is excluded.
        if asia[i]:
            current_high = high[i]
            current_low = low[i]
            if np.isfinite(current_high) and np.isfinite(current_low):
                if asia_has_active:
                    if current_high > asia_active_high:
                        asia_active_high = current_high
                    if current_low < asia_active_low:
                        asia_active_low = current_low
                else:
                    asia_active_high = current_high
                    asia_active_low = current_low
                    asia_has_active = True

        if london[i]:
            current_high = high[i]
            current_low = low[i]
            if np.isfinite(current_high) and np.isfinite(current_low):
                if london_has_active:
                    if current_high > london_active_high:
                        london_active_high = current_high
                    if current_low < london_active_low:
                        london_active_low = current_low
                else:
                    london_active_high = current_high
                    london_active_low = current_low
                    london_has_active = True
        elif i > 0 and london[i - 1]:
            london_has_active = False

    return out


def prior_session_sweep_reclaim(ctx: StrategyContext, params: dict) -> np.ndarray:
    transition = str(params["transition"])
    if transition == "asia_to_london":
        transition_code = _TRANSITION_ASIA_TO_LONDON
    elif transition == "london_pre_ny_to_new_york":
        transition_code = _TRANSITION_LONDON_PRE_NY_TO_NEW_YORK
    else:
        raise ValueError(f"unsupported Edge D transition: {transition!r}")

    entry_window_bars = int(params["entry_window_bars"])
    max_reclaim_bars = int(params["max_reclaim_bars"])
    sweep_atr = float(params["sweep_atr"])
    reclaim_depth_atr = float(params["reclaim_depth_atr"])
    if entry_window_bars < 1:
        raise ValueError("entry_window_bars must be positive")
    if max_reclaim_bars < 0:
        raise ValueError("max_reclaim_bars must be nonnegative")
    if sweep_atr < 0.0 or reclaim_depth_atr < 0.0:
        raise ValueError("ATR thresholds must be nonnegative")

    asia = np.asarray(ctx.feature("session_asia"), dtype=np.bool_)
    london = np.asarray(ctx.feature("session_london"), dtype=np.bool_)
    new_york = np.asarray(ctx.feature("session_new_york"), dtype=np.bool_)

    return _prior_session_sweep_reclaim_kernel(
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        ctx.atr14,
        asia,
        london,
        new_york,
        transition_code,
        entry_window_bars,
        sweep_atr,
        max_reclaim_bars,
        reclaim_depth_atr,
    )


register_strategy(
    StrategyDefinition(
        EDGE_D_FAMILY,
        EDGE_D_STRATEGY,
        prior_session_sweep_reclaim,
        {
            "transition": ("asia_to_london", "london_pre_ny_to_new_york"),
            "entry_window_bars": (30, 180),
            "sweep_atr": (0.10, 1.00),
            "max_reclaim_bars": (0, 10),
            "reclaim_depth_atr": (0.00, 0.50),
        },
    )
)


__all__ = ["EDGE_D_FAMILY", "EDGE_D_STRATEGY", "prior_session_sweep_reclaim"]
