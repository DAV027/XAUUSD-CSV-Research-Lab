from __future__ import annotations

import math

import numpy as np
from numba import njit

from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


@njit(cache=True)
def _compression_breakout_retest_v2_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr14: np.ndarray,
    compression_lookback: int,
    compression_atr_ratio: float,
    breakout_buffer_atr: float,
    breakout_body_atr: float,
    retest_window: int,
    retest_tolerance_atr: float,
    confirmation_atr: float,
) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    if compression_lookback <= 0 or n == 0:
        return out

    capacity = compression_lookback + 1
    high_q = np.empty(capacity, dtype=np.int64)
    low_q = np.empty(capacity, dtype=np.int64)
    high_head = 0
    high_tail = 0
    high_size = 0
    low_head = 0
    low_tail = 0
    low_size = 0
    invalid_count = 0

    SEEK = 0
    ARMED = 1
    RETEST = 2
    state = SEEK
    rearm_ready = True

    range_high = 0.0
    range_low = 0.0
    episode_atr = 0.0
    broken_level = 0.0
    breakout_direction = 0
    breakout_index = -1
    retest_seen = False

    for i in range(n):
        if i > 0:
            add_idx = i - 1
            add_high = high[add_idx]
            add_low = low[add_idx]
            add_valid = np.isfinite(add_high) and np.isfinite(add_low)
            if not add_valid:
                invalid_count += 1
            else:
                while high_size > 0:
                    back = (high_tail - 1 + capacity) % capacity
                    if high[high_q[back]] > add_high:
                        break
                    high_tail = back
                    high_size -= 1
                high_q[high_tail] = add_idx
                high_tail = (high_tail + 1) % capacity
                high_size += 1

                while low_size > 0:
                    back = (low_tail - 1 + capacity) % capacity
                    if low[low_q[back]] < add_low:
                        break
                    low_tail = back
                    low_size -= 1
                low_q[low_tail] = add_idx
                low_tail = (low_tail + 1) % capacity
                low_size += 1

            leaving = i - compression_lookback - 1
            if leaving >= 0:
                if not (np.isfinite(high[leaving]) and np.isfinite(low[leaving])):
                    invalid_count -= 1

            minimum_index = i - compression_lookback
            while high_size > 0 and high_q[high_head] < minimum_index:
                high_head = (high_head + 1) % capacity
                high_size -= 1
            while low_size > 0 and low_q[low_head] < minimum_index:
                low_head = (low_head + 1) % capacity
                low_size -= 1

        compression = False
        candidate_high = 0.0
        candidate_low = 0.0
        candidate_atr = np.nan
        if (
            i >= compression_lookback
            and invalid_count == 0
            and high_size > 0
            and low_size > 0
            and np.isfinite(atr14[i - 1])
            and atr14[i - 1] > 0.0
        ):
            candidate_high = high[high_q[high_head]]
            candidate_low = low[low_q[low_head]]
            candidate_atr = atr14[i - 1]
            width = candidate_high - candidate_low
            volatility_scale = candidate_atr * math.sqrt(float(compression_lookback))
            if np.isfinite(width) and width >= 0.0 and volatility_scale > 0.0:
                compression = width / volatility_scale <= compression_atr_ratio

        if state == RETEST:
            if i <= breakout_index:
                continue
            if i - breakout_index > retest_window:
                state = SEEK
                rearm_ready = False
                continue

            o = open_[i]
            h = high[i]
            l = low[i]
            c = close[i]
            if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
                continue

            tolerance = retest_tolerance_atr * episode_atr
            confirmation = confirmation_atr * episode_atr
            if breakout_direction > 0:
                if c < broken_level - tolerance:
                    state = SEEK
                    rearm_ready = False
                    continue
                if l <= broken_level + tolerance:
                    retest_seen = True
                if retest_seen and c >= broken_level + confirmation and c > o:
                    out[i] = np.int8(1)
                    state = SEEK
                    rearm_ready = False
                    continue
            else:
                if c > broken_level + tolerance:
                    state = SEEK
                    rearm_ready = False
                    continue
                if h >= broken_level - tolerance:
                    retest_seen = True
                if retest_seen and c <= broken_level - confirmation and c < o:
                    out[i] = np.int8(-1)
                    state = SEEK
                    rearm_ready = False
                    continue
            continue

        if state == SEEK:
            if not compression:
                if not rearm_ready:
                    rearm_ready = True
                continue
            if not rearm_ready:
                continue
            range_high = candidate_high
            range_low = candidate_low
            episode_atr = candidate_atr
            state = ARMED

        if state == ARMED:
            o = open_[i]
            c = close[i]
            valid_bar = np.isfinite(o) and np.isfinite(c)
            if valid_bar:
                body = abs(c - o)
                minimum_body = breakout_body_atr * episode_atr
                long_break = (
                    c > range_high + breakout_buffer_atr * episode_atr
                    and c > o
                    and body >= minimum_body
                )
                short_break = (
                    c < range_low - breakout_buffer_atr * episode_atr
                    and c < o
                    and body >= minimum_body
                )
                if long_break or short_break:
                    breakout_direction = 1 if long_break else -1
                    broken_level = range_high if long_break else range_low
                    breakout_index = i
                    retest_seen = False
                    state = RETEST
                    continue

            if not compression:
                state = SEEK
                rearm_ready = True

    return out


def compression_breakout_retest_v2(ctx: StrategyContext, params: dict) -> np.ndarray:
    return _compression_breakout_retest_v2_kernel(
        ctx.open,
        ctx.high,
        ctx.low,
        ctx.close,
        ctx.atr14,
        int(params["compression_lookback"]),
        float(params["compression_atr_ratio"]),
        float(params["breakout_buffer_atr"]),
        float(params["breakout_body_atr"]),
        int(params["retest_window"]),
        float(params["retest_tolerance_atr"]),
        float(params["confirmation_atr"]),
    )


register_strategy(
    StrategyDefinition(
        "edge_c_breakout_retest_v2",
        "compression_breakout_retest_v2",
        compression_breakout_retest_v2,
        {
            "compression_lookback": (20, 120),
            "compression_atr_ratio": (0.50, 1.10),
            "breakout_buffer_atr": (0.25, 1.50),
            "breakout_body_atr": (0.25, 1.50),
            "retest_window": (2, 20),
            "retest_tolerance_atr": (0.10, 0.75),
            "confirmation_atr": (0.10, 1.00),
        },
    )
)


__all__ = ["compression_breakout_retest_v2"]
