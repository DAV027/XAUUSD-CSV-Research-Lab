from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


def _session_flag(ctx: StrategyContext, session: str) -> np.ndarray:
    return np.asarray(ctx.feature(f"session_{session}"), dtype=bool)


def session_open_momentum(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def prior_session_high_low_break(ctx: StrategyContext, params: dict) -> np.ndarray:
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


def opening_range_breakout(ctx: StrategyContext, params: dict) -> np.ndarray:
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
