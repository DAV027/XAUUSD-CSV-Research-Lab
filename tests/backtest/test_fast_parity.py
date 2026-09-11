import numpy as np

from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.reference import run_reference_backtest


SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
COST = CostModel()
RISK = RiskModel()


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
        assert fast.risk_skip_count == reference.risk_skip_count
        assert len(fast.trades) == len(reference.trades)
        for expected, actual in zip(reference.trades, fast.trades, strict=True):
            assert actual.entry_index == expected.entry_index
            assert actual.exit_index == expected.exit_index
            assert actual.direction == expected.direction
            assert actual.exit_reason == expected.exit_reason
            assert abs(actual.net_pnl - expected.net_pnl) < 1e-9
            assert abs(actual.entry_price - expected.entry_price) < 1e-9
            assert abs(actual.exit_price - expected.exit_price) < 1e-9
            assert abs(actual.stop_price - expected.stop_price) < 1e-9
            if expected.target_price is None:
                assert actual.target_price is None
            else:
                assert abs(actual.target_price - expected.target_price) < 1e-9
