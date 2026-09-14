from __future__ import annotations

import numpy as np
from numba import njit

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _session_flag(ctx: StrategyContext, session: str) -> np.ndarray:
    return np.asarray(ctx.feature(f"session_{session}"), dtype=bool)


@njit(cache=True)
def _session_open_momentum_kernel(
    flags: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    threshold_atr: float,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    for i in range(len(close)):
        if not flags[i] or (i > 0 and flags[i - 1]):
            continue
        current_open = open_[i]
        current_close = close[i]
        atr = atr14[i]
        if (
            not np.isfinite(current_open)
            or not np.isfinite(current_close)
            or not np.isfinite(atr)
            or atr <= 0.0
        ):
            continue
        move = current_close - current_open
        threshold = threshold_atr * atr
        if move > threshold:
            out[i] = 1
        elif move < -threshold:
            out[i] = -1
    return out


@njit(cache=True)
def _prior_session_high_low_break_kernel(
    flags: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    previous_high = 0.0
    previous_low = 0.0
    active_high = 0.0
    active_low = 0.0
    has_previous = False
    has_active = False
    in_session = False

    for i in range(len(close)):
        if flags[i]:
            if not in_session:
                has_active = False
                in_session = True

            current_close = close[i]
            if has_previous and np.isfinite(current_close):
                if current_close > previous_high:
                    out[i] = 1
                elif current_close < previous_low:
                    out[i] = -1

            current_high = high[i]
            current_low = low[i]
            if np.isfinite(current_high) and np.isfinite(current_low):
                if has_active:
                    if current_high > active_high:
                        active_high = current_high
                    if current_low < active_low:
                        active_low = current_low
                else:
                    active_high = current_high
                    active_low = current_low
                    has_active = True
        elif in_session:
            if has_active:
                previous_high = active_high
                previous_low = active_low
                has_previous = True
            has_active = False
            in_session = False

    return out


@njit(cache=True)
def _opening_range_breakout_kernel(
    flags: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    opening_bars: int,
) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.int8)
    range_high = 0.0
    range_low = 0.0
    has_range = False
    session_bar = 0

    for i in range(len(close)):
        if not flags[i]:
            has_range = False
            session_bar = 0
            continue

        current_high = high[i]
        current_low = low[i]
        current_close = close[i]
        if (
            not np.isfinite(current_high)
            or not np.isfinite(current_low)
            or not np.isfinite(current_close)
        ):
            session_bar += 1
            continue

        if session_bar < opening_bars:
            if has_range:
                if current_high > range_high:
                    range_high = current_high
                if current_low < range_low:
                    range_low = current_low
            else:
                range_high = current_high
                range_low = current_low
                has_range = True
        elif has_range:
            if current_close > range_high:
                out[i] = 1
            elif current_close < range_low:
                out[i] = -1

        session_bar += 1

    return out


def session_open_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    threshold_atr = float(params.get("threshold_atr", 0.0))
    return _session_open_momentum_kernel(flags, ctx.open, ctx.close, ctx.atr14, threshold_atr)


def prior_session_high_low_break(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    return _prior_session_high_low_break_kernel(flags, ctx.high, ctx.low, ctx.close)


def opening_range_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
    flags = _session_flag(ctx, str(params["session"]))
    opening_bars = int(params["opening_range_bars"])
    if opening_bars < 1:
        raise ValueError("opening_range_bars must be >= 1")
    return _opening_range_breakout_kernel(flags, ctx.high, ctx.low, ctx.close, opening_bars)


SESSION_DOMAIN = {"session": ("asia", "london", "new_york")}
register_strategy(
    StrategyDefinition(
        "session",
        "session_open_momentum",
        session_open_momentum,
        {**SESSION_DOMAIN, "threshold_atr": (0.0, 1.0)},
    )
)
register_strategy(
    StrategyDefinition(
        "session",
        "prior_session_high_low_break",
        prior_session_high_low_break,
        SESSION_DOMAIN,
    )
)
register_strategy(
    StrategyDefinition(
        "session",
        "opening_range_breakout",
        opening_range_breakout,
        {**SESSION_DOMAIN, "opening_range_bars": (2, 30)},
    )
)
