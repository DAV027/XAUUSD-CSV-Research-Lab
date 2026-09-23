from pathlib import Path

import numpy as np
import pytest

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy
from xau_lab.strategies.sma_rsi_htf import (
    _mt5_iatr,
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


def test_signal_can_enter_on_later_minute_inside_immediately_following_m5_bar():
    ctx = _context_from_m5_closes(_bullish_fixture())
    signal_index = 649 * 5 + 4

    keep = np.ones(len(ctx), dtype=bool)
    keep[signal_index + 1] = False
    sparse = StrategyContext(
        open=ctx.open[keep],
        high=ctx.high[keep],
        low=ctx.low[keep],
        close=ctx.close[keep],
        spread=ctx.spread[keep],
        atr14=ctx.atr14[keep],
        time_epoch=ctx.time_epoch[keep],
        features={},
    )

    state = build_sma_rsi_htf_state(sparse)
    signal_indexes = np.flatnonzero(state.signal)
    assert len(signal_indexes) == 1

    sparse_signal_index = int(signal_indexes[0])
    next_time = int(sparse.time_epoch[sparse_signal_index + 1])
    assert next_time // 300 == int(state.m5_close_time[649]) // 300
    assert next_time > int(state.m5_close_time[649])


def test_no_signal_is_carried_across_an_entirely_empty_next_m5_bar():
    ctx = _context_from_m5_closes(_bullish_fixture())
    signal_index = 649 * 5 + 4

    keep = np.ones(len(ctx), dtype=bool)
    keep[signal_index + 1 : signal_index + 6] = False
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


def test_sparse_minutes_still_form_m5_and_m15_bars_like_mt5():
    ctx = _context_from_m5_closes(_bullish_fixture())

    # Remove one interior M1 minute from a historical M15 block, without
    # removing the whole M5 or M15 bucket.
    remove_index = 500 * 5 + 2
    keep = np.ones(len(ctx), dtype=bool)
    keep[remove_index] = False
    sparse = StrategyContext(
        open=ctx.open[keep],
        high=ctx.high[keep],
        low=ctx.low[keep],
        close=ctx.close[keep],
        spread=ctx.spread[keep],
        atr14=ctx.atr14[keep],
        time_epoch=ctx.time_epoch[keep],
        features={},
    )

    state = build_sma_rsi_htf_state(sparse)

    # The sparse source minute must not delete its enclosing M5/M15 market bar.
    target_epoch = int(ctx.time_epoch[500 * 5]) // 300 * 300 + 300
    assert target_epoch in set(state.m5_close_time.tolist())

    target_m15_epoch = int(ctx.time_epoch[500 * 5]) // 900 * 900 + 900
    assert target_m15_epoch in set(state.m15_close_time.tolist())


def test_frozen_strategy_rejects_parameter_tuning_and_is_registered():
    ctx = _context_from_m5_closes(_bullish_fixture())

    direct = sma_rsi_htf_v21(ctx, {})
    registered = get_strategy("sma_rsi_htf_v21")

    assert registered.family == "sma_rsi_htf_reproduction"
    assert registered.parameter_domain == {}
    assert np.array_equal(registered.generate(ctx, {}), direct)

    with pytest.raises(ValueError, match="frozen"):
        sma_rsi_htf_v21(ctx, {"fast_sma": 10})


def test_parity_export_script_is_syntax_valid():
    path = Path("scripts/export_sma_rsi_htf_parity.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_mt5_parity_comparator_is_syntax_valid():
    path = Path("scripts/compare_sma_rsi_parity_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_mt5_guard_timestamp_regex_matches_tester_lines():
    import runpy

    module = runpy.run_path("scripts/compare_sma_rsi_parity_mt5.py")
    match = module["_SIM_TIME_RE"].search(
        "Core 01 2026.06.01 01:05:00   No trade: spread=181.0 points."
    )
    assert match is not None
    assert match.group(1) == "2026.06.01 01:05:00"


def test_trade_construction_parity_script_is_syntax_valid():
    path = Path("scripts/compare_sma_rsi_trade_construction_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_mt5_iatr_matches_rolling_true_range_average():
    high = np.array([10, 12, 13, 15, 14, 16], dtype=float)
    low = np.array([9, 10, 11, 12, 12, 13], dtype=float)
    close = np.array([9.5, 11, 12, 14, 13, 15], dtype=float)

    atr = _mt5_iatr(high, low, close, 3)

    tr = np.array([
        0.0,
        max(12.0, 9.5) - min(10.0, 9.5),
        max(13.0, 11.0) - min(11.0, 11.0),
        max(15.0, 12.0) - min(12.0, 12.0),
        max(14.0, 14.0) - min(12.0, 14.0),
        max(16.0, 13.0) - min(13.0, 13.0),
    ])

    assert np.isnan(atr[:3]).all()
    assert atr[3] == pytest.approx(np.mean(tr[1:4]))
    assert atr[4] == pytest.approx(np.mean(tr[2:5]))
    assert atr[5] == pytest.approx(np.mean(tr[3:6]))


def test_volume_parity_script_is_syntax_valid():
    path = Path("scripts/compare_sma_rsi_volume_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_lifecycle_parity_script_is_syntax_valid():
    path = Path("scripts/compare_sma_rsi_lifecycle_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_tick_export_script_is_syntax_valid():
    path = Path("scripts/export_mt5_ticks.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_tick_exit_parity_script_is_syntax_valid():
    path = Path("scripts/compare_sma_rsi_tick_exits_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_tick_exit_fallback_logic_is_present():
    text = Path("scripts/compare_sma_rsi_tick_exits_mt5.py").read_text(encoding="utf-8")
    assert "entry_second_level_touch_ambiguities" in text
    assert "entry_second_fallbacks" in text


def test_entry_delay_diagnostic_script_is_syntax_valid():
    path = Path("scripts/diagnose_sma_rsi_entry_delay_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_cost_model_diagnostic_script_is_syntax_valid():
    path = Path("scripts/diagnose_sma_rsi_cost_model_mt5.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_fxify_symbol_cost_probe_is_syntax_valid():
    path = Path("scripts/inspect_fxify_symbol_cost_spec.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_standalone_v21_replay_script_is_syntax_valid():
    path = Path("scripts/replay_sma_rsi_v21_standalone.py")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
