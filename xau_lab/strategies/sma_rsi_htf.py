from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xau_lab.data.features import _rolling_mean, _rsi, _wilder
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy


FAST_SMA = 9
SLOW_SMA = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
ATR_PERIOD = 14
HTF_SMA = 200


@dataclass(frozen=True)
class SmaRsiHtfState:
    signal: np.ndarray
    execution_atr: np.ndarray
    m5_close_time: np.ndarray
    m5_source_last_index: np.ndarray
    m5_close: np.ndarray
    m5_sma_fast: np.ndarray
    m5_sma_slow: np.ndarray
    m5_rsi: np.ndarray
    m5_atr: np.ndarray
    m15_close_time: np.ndarray
    m15_close: np.ndarray
    m15_sma200: np.ndarray


@dataclass(frozen=True)
class _Aggregate:
    close_time: np.ndarray
    source_last_index: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def _empty_aggregate() -> _Aggregate:
    return _Aggregate(
        close_time=np.empty(0, dtype=np.int64),
        source_last_index=np.empty(0, dtype=np.int64),
        open=np.empty(0, dtype=np.float64),
        high=np.empty(0, dtype=np.float64),
        low=np.empty(0, dtype=np.float64),
        close=np.empty(0, dtype=np.float64),
    )


def _aggregate_complete_minutes(ctx: StrategyContext, minutes: int) -> _Aggregate:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if len(ctx) == 0:
        return _empty_aggregate()

    seconds = minutes * 60
    time = np.asarray(ctx.time_epoch, dtype=np.int64)

    bucket = time // seconds
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    stops = np.r_[starts[1:], len(time)]

    close_time: list[int] = []
    source_last_index: list[int] = []
    open_: list[float] = []
    high: list[float] = []
    low: list[float] = []
    close: list[float] = []

    for start, stop in zip(starts, stops):
        start = int(start)
        stop = int(stop)
        bucket_start = int(bucket[start]) * seconds
        times = time[start:stop]

        # MT5 still forms an M5/M15 bar when individual M1 minutes inside the
        # interval have no ticks. Aggregate every non-empty time bucket instead
        # of requiring exactly 5/15 source rows.
        if len(times) == 0:
            continue
        if np.any(times < bucket_start) or np.any(times >= bucket_start + seconds):
            continue
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            continue

        o = float(ctx.open[start])
        h = float(np.max(ctx.high[start:stop]))
        l = float(np.min(ctx.low[start:stop]))
        c = float(ctx.close[stop - 1])
        if not all(np.isfinite(value) for value in (o, h, l, c)):
            continue

        close_time.append(bucket_start + seconds)
        source_last_index.append(stop - 1)
        open_.append(o)
        high.append(h)
        low.append(l)
        close.append(c)

    return _Aggregate(
        close_time=np.asarray(close_time, dtype=np.int64),
        source_last_index=np.asarray(source_last_index, dtype=np.int64),
        open=np.asarray(open_, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=np.asarray(close, dtype=np.float64),
    )


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    if len(close) == 0:
        return np.empty(0, dtype=np.float64)
    previous_close = np.roll(close, 1)
    previous_close[0] = close[0]
    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ]
    )
    return _wilder(true_range, window)


def build_sma_rsi_htf_state(ctx: StrategyContext) -> SmaRsiHtfState:
    """Build the frozen V2.1 M5/M15 signal state on an M1 context.

    Signal placement is deliberate: a crossover confirmed by the just-completed
    M5 candle is written on that candle's final M1 row. The lab backtester enters
    one row later, at the next M1 open, which is the new M5 open when the source
    data are continuous. This mirrors the MT5 EA evaluating closed bars on the
    first tick of a new M5 candle.

    A signal may enter on a later minute inside the immediately following M5
    bucket when the first minute(s) contain no ticks. It is never carried across
    an entirely empty M5 bucket or a market-closure gap.
    """

    m5 = _aggregate_complete_minutes(ctx, 5)
    m15 = _aggregate_complete_minutes(ctx, 15)

    n = len(ctx)
    signal = np.zeros(n, dtype=np.int8)
    execution_atr = np.full(n, np.nan, dtype=np.float64)

    m5_fast = _rolling_mean(m5.close, FAST_SMA)
    m5_slow = _rolling_mean(m5.close, SLOW_SMA)
    m5_rsi = _rsi(m5.close, RSI_PERIOD)
    m5_atr = _atr(m5.high, m5.low, m5.close, ATR_PERIOD)
    m15_sma200 = _rolling_mean(m15.close, HTF_SMA)

    for i in range(len(m5.close)):
        source_index = int(m5.source_last_index[i])
        if source_index < 0 or source_index >= n:
            continue

        if np.isfinite(m5_atr[i]):
            execution_atr[source_index] = m5_atr[i]

        if i == 0 or source_index + 1 >= n:
            continue

        # MT5 evaluates on the first tick of the next M5 bar. The first source
        # M1 row may be one or more minutes after the nominal M5 open when those
        # early minutes had no ticks, but it must still belong to the immediately
        # following M5 bucket. Never carry a signal across an entirely empty M5 bar.
        next_time = int(ctx.time_epoch[source_index + 1])
        if next_time // 300 != int(m5.close_time[i]) // 300:
            continue

        fast_now = m5_fast[i]
        slow_now = m5_slow[i]
        fast_prev = m5_fast[i - 1]
        slow_prev = m5_slow[i - 1]
        rsi_now = m5_rsi[i]
        atr_now = m5_atr[i]

        if not all(
            np.isfinite(value)
            for value in (fast_now, slow_now, fast_prev, slow_prev, rsi_now, atr_now)
        ):
            continue

        htf_index = int(np.searchsorted(m15.close_time, m5.close_time[i], side="right") - 1)
        if htf_index < 0:
            continue

        htf_close = m15.close[htf_index]
        htf_sma = m15_sma200[htf_index]
        if not np.isfinite(htf_close) or not np.isfinite(htf_sma):
            continue

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        if cross_up and rsi_now < RSI_OVERBOUGHT and htf_close > htf_sma:
            signal[source_index] = np.int8(1)
        elif cross_down and rsi_now > RSI_OVERSOLD and htf_close < htf_sma:
            signal[source_index] = np.int8(-1)

    signal.setflags(write=False)
    execution_atr.setflags(write=False)

    return SmaRsiHtfState(
        signal=signal,
        execution_atr=execution_atr,
        m5_close_time=m5.close_time,
        m5_source_last_index=m5.source_last_index,
        m5_close=m5.close,
        m5_sma_fast=m5_fast,
        m5_sma_slow=m5_slow,
        m5_rsi=m5_rsi,
        m5_atr=m5_atr,
        m15_close_time=m15.close_time,
        m15_close=m15.close,
        m15_sma200=m15_sma200,
    )


def sma_rsi_htf_v21(ctx: StrategyContext, params: dict) -> np.ndarray:
    if params:
        raise ValueError("sma_rsi_htf_v21 is frozen and does not accept tunable parameters")
    return build_sma_rsi_htf_state(ctx).signal


register_strategy(
    StrategyDefinition(
        family="sma_rsi_htf_reproduction",
        name="sma_rsi_htf_v21",
        signal=sma_rsi_htf_v21,
        parameter_domain={},
    )
)


__all__ = [
    "ATR_PERIOD",
    "FAST_SMA",
    "HTF_SMA",
    "RSI_OVERBOUGHT",
    "RSI_OVERSOLD",
    "RSI_PERIOD",
    "SLOW_SMA",
    "SmaRsiHtfState",
    "build_sma_rsi_htf_state",
    "sma_rsi_htf_v21",
]
