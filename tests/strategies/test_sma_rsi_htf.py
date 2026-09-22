import numpy as np
import pytest

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy
from xau_lab.strategies.sma_rsi_htf import (
    build_sma_rsi_htf_state,
    sma_rsi_htf_v21,
)


def _context_from_m5_closes(m5_close: np.ndarray) -> StrategyContext:
    m5_close = np.asarray(m5_close, dtype=np.float64)
    close = np.repeat(m5_close, 5)
    n = len(close)
    return StrategyContext(
        open=close,
        high=close + 0.05,
        low=close - 0.05,
        close=close,
        spread=np.zeros(n, dtype=np.float64),
        atr14=np.ones(n, dtype=np.float64),
        time_epoch=np.arange(n, dtype=np.int64) * 60,
        features={},
    )


def _bullish_fixture() -> np.ndarray:
    close = 100.0 + np.arange(660, dtype=np.float64) * 0.01
    close[620:641] -= np.linspace(0.0, 1.0, 21)
    close[641:] -= np.linspace(1.0, 0.0, 19)
    return close


def _bearish_fixture() -> np.ndarray:
    close = 120.0 - np.arange(660, dtype=np.float64) * 0.01
    close[620:641] += np.linspace(0.0, 1.0, 21)
    close[641:] += np.linspace(1.0, 0.0, 19)
    return close


def test_bullish_v21_signal_is_written_on_last_m1_before_new_m5_open():
    ctx = _context_from_m5_closes(_bullish_fixture())
    state = build_sma_rsi_htf_state(ctx)

    signal_indexes = np.flatnonzero(state.signal)
    assert signal_indexes.tolist() == [649 * 5 + 4]

    signal_index = int(signal_indexes[0])
    entry_index = signal_index + 1

    assert state.signal[signal_index] == 1
    assert ctx.time_epoch[entry_index] == state.m5_close_time[649]
    assert np.isfinite(state.execution_atr[signal_index])
    assert state.m5_rsi[649] < 70.0

    htf_index = int(
        np.searchsorted(state.m15_close_time, state.m5_close_time[649], side="right") - 1
    )
    assert state.m15_close_time[htf_index] <= state.m5_close_time[649]
    assert state.m15_close[htf_index] > state.m15_sma200[htf_index]


def test_bearish_v21_signal_is_written_on_last_m1_before_new_m5_open():
    ctx = _context_from_m5_closes(_bearish_fixture())
    state = build_sma_rsi_htf_state(ctx)

    signal_indexes = np.flatnonzero(state.signal)
    assert signal_indexes.tolist() == [649 * 5 + 4]

    signal_index = int(signal_indexes[0])
    assert state.signal[signal_index] == -1
    assert np.isfinite(state.execution_atr[signal_index])
    assert state.m5_rsi[649] > 30.0

    htf_index = int(
        np.searchsorted(state.m15_close_time, state.m5_close_time[649], side="right") - 1
    )
    assert state.m15_close_time[htf_index] <= state.m5_close_time[649]
    assert state.m15_close[htf_index] < state.m15_sma200[htf_index]


def test_htf_filter_never_uses_a_m15_bar_that_closes_after_the_m5_signal():
    ctx = _context_from_m5_closes(_bullish_fixture())
    state = build_sma_rsi_htf_state(ctx)
    m5_index = 649

    # This fixture's signal M5 closes 10 minutes into its current M15 block.
    # Therefore the newest usable M15 close must be strictly earlier.
    htf_index = int(
        np.searchsorted(state.m15_close_time, state.m5_close_time[m5_index], side="right") - 1
    )
    assert state.m15_close_time[htf_index] < state.m5_close_time[m5_index]
    if htf_index + 1 < len(state.m15_close_time):
        assert state.m15_close_time[htf_index + 1] > state.m5_close_time[m5_index]


def test_no_signal_is_carried_across_a_gap_after_completed_m5_bar():
    ctx = _context_from_m5_closes(_bullish_fixture())
    signal_index = 649 * 5 + 4

    keep = np.ones(len(ctx), dtype=bool)
    keep[signal_index + 1] = False
    gapped = StrategyContext(
        open=ctx.open[keep],
        high=ctx.high[keep],
        low=ctx.low[keep],
        close=ctx.close[keep],
        spread=ctx.spread[keep],
        atr14=ctx.atr14[keep],
        time_epoch=ctx.time_epoch[keep],
        features={},
    )

    state = build_sma_rsi_htf_state(gapped)
    assert np.count_nonzero(state.signal) == 0


def test_frozen_strategy_rejects_parameter_tuning_and_is_registered():
    ctx = _context_from_m5_closes(_bullish_fixture())

    direct = sma_rsi_htf_v21(ctx, {})
    registered = get_strategy("sma_rsi_htf_v21")

    assert registered.family == "sma_rsi_htf_reproduction"
    assert registered.parameter_domain == {}
    assert np.array_equal(registered.generate(ctx, {}), direct)

    with pytest.raises(ValueError, match="frozen"):
        sma_rsi_htf_v21(ctx, {"fast_sma": 10})
