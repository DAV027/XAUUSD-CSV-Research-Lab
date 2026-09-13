import numpy as np
import pytest

import xau_lab.backtest.fast as fast_module
from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.reference import run_reference_backtest


SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
COST = CostModel()
RISK = RiskModel()


class _NoIterSignals(np.ndarray):
    def __iter__(self):
        raise AssertionError("run_fast_backtest must not materialize NumPy signals as a Python list")


def _random_market(n=2000):
    rng = np.random.default_rng(9_212_000)
    opens = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    opens[0] = 2200.0
    closes[0] = opens[0] + rng.normal(0.0, 0.25)
    for i in range(1, n):
        opens[i] = closes[i - 1]
        closes[i] = opens[i] + rng.normal(0.0, 0.25)
    high_pad = rng.uniform(0.05, 0.65, size=n)
    low_pad = rng.uniform(0.05, 0.65, size=n)
    highs = np.maximum(opens, closes) + high_pad
    lows = np.minimum(opens, closes) - low_pad
    spread = rng.integers(10, 41, size=n, dtype=np.int64)
    atr = rng.uniform(0.45, 1.55, size=n)
    signals = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n, p=[0.08, 0.84, 0.08])
    bars = MarketBars(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + 1_700_000_000,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        spread=spread,
        atr=atr,
    )
    return bars, signals


def _assert_results_equal(expected, actual):
    assert actual.risk_skip_count == expected.risk_skip_count
    assert len(actual.trades) == len(expected.trades)
    for expected_trade, actual_trade in zip(expected.trades, actual.trades, strict=True):
        assert actual_trade.entry_index == expected_trade.entry_index
        assert actual_trade.exit_index == expected_trade.exit_index
        assert actual_trade.direction == expected_trade.direction
        assert actual_trade.exit_reason == expected_trade.exit_reason
        assert abs(actual_trade.net_pnl - expected_trade.net_pnl) < 1e-9
        assert abs(actual_trade.entry_price - expected_trade.entry_price) < 1e-9
        assert abs(actual_trade.exit_price - expected_trade.exit_price) < 1e-9
        assert abs(actual_trade.stop_price - expected_trade.stop_price) < 1e-9
        if expected_trade.target_price is None:
            assert actual_trade.target_price is None
        else:
            assert abs(actual_trade.target_price - expected_trade.target_price) < 1e-9


def test_fast_kernel_matches_reference_trade_for_trade_across_exit_modes():
    bars, signals = _random_market()
    exit_specs = [
        ExitSpec(stop_atr=1.0, target_r=1.5),
        ExitSpec(stop_atr=1.0, target_r=None, time_exit_minutes=8),
        ExitSpec(stop_atr=1.5, target_r=2.0, atr_trail=0.8),
        ExitSpec(stop_atr=1.25, target_r=None, time_exit_minutes=12, atr_trail=1.0),
    ]

    for exit_spec in exit_specs:
        reference = run_reference_backtest(bars, signals, SPEC, COST, RISK, exit_spec)
        fast = run_fast_backtest(bars, signals, SPEC, COST, RISK, exit_spec)
        _assert_results_equal(reference, fast)


def test_fast_backtest_accepts_numpy_signals_without_python_iteration():
    bars, signals = _random_market(n=128)
    guarded = signals.view(_NoIterSignals)

    result = run_fast_backtest(
        bars,
        guarded,
        SPEC,
        COST,
        RISK,
        ExitSpec(stop_atr=1.0, target_r=1.5),
    )

    assert result.risk_skip_count >= 0


@pytest.mark.parametrize("invalid", [0.5, 1.5, np.nan, np.inf, -np.inf])
def test_fast_backtest_rejects_non_domain_values_before_int8_coercion(invalid):
    bars, signals = _random_market(n=32)
    float_signals = signals.astype(np.float64)
    float_signals[7] = invalid

    with pytest.raises(ValueError, match=r"signals must contain only -1, 0, \+1"):
        run_fast_backtest(
            bars,
            float_signals,
            SPEC,
            COST,
            RISK,
            ExitSpec(stop_atr=1.0, target_r=1.5),
        )


def test_fast_backtest_sizes_kernel_output_capacity_to_nonzero_signals(
    monkeypatch: pytest.MonkeyPatch,
):
    bars, _ = _random_market(n=4096)
    signals = np.zeros(4096, dtype=np.int8)
    signals[[10, 1000, 3000]] = [1, -1, 1]
    captured = {}

    def fake_kernel(*args):
        captured["capacity"] = args[-1]
        capacity = max(0, int(args[-1]))
        return (
            0,
            0,
            np.zeros(capacity, dtype=np.int8),
            np.full(capacity, -1, dtype=np.int64),
            np.full(capacity, -1, dtype=np.int64),
            np.full(capacity, -1, dtype=np.int64),
            np.zeros(capacity, dtype=np.float64),
            np.zeros(capacity, dtype=np.float64),
            np.zeros(capacity, dtype=np.float64),
            np.full(capacity, np.nan, dtype=np.float64),
            np.zeros(capacity, dtype=np.float64),
            np.zeros(capacity, dtype=np.float64),
            np.zeros(capacity, dtype=np.float64),
            np.zeros(capacity, dtype=np.int8),
        )

    monkeypatch.setattr(fast_module, "_kernel", fake_kernel)

    result = run_fast_backtest(
        bars,
        signals,
        SPEC,
        COST,
        RISK,
        ExitSpec(stop_atr=1.0, target_r=1.5),
    )

    assert result.trades == ()
    assert captured["capacity"] == 3
