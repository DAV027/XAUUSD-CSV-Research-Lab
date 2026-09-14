import numpy as np
import pytest

from xau_lab.backtest.fast import run_fast_backtest, run_packed_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult


SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
COST = CostModel()
RISK = RiskModel()


def _market(n: int = 256):
    bars = MarketBars(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + 1_700_000_000,
        open=np.full(n, 2000.0),
        high=np.full(n, 2000.7),
        low=np.full(n, 1999.3),
        close=np.full(n, 2000.0),
        spread=np.full(n, 20, dtype=np.int64),
        atr=np.full(n, 1.0),
    )
    signals = np.zeros(n, dtype=np.int8)
    signals[::5] = 1
    return bars, signals


def test_packed_backtest_exposes_kernel_rows_without_trade_objects():
    bars, signals = _market()
    packed = run_packed_backtest(
        bars, signals, SPEC, COST, RISK, ExitSpec(stop_atr=1.0, target_r=1.0)
    )
    assert isinstance(packed, PackedBacktestResult)
    assert 0 <= packed.trade_count <= np.count_nonzero(signals)
    assert packed.risk_skip_count >= 0
    for values in (
        packed.direction,
        packed.signal_index,
        packed.entry_index,
        packed.exit_index,
        packed.raw_entry,
        packed.entry_price,
        packed.initial_stop,
        packed.target,
        packed.lot,
        packed.planned_risk,
        packed.raw_exit,
        packed.reason,
    ):
        assert values.ndim == 1
        assert len(values) >= packed.trade_count


def test_packed_result_rejects_negative_counts():
    arrays = [np.zeros(1, dtype=np.float64) for _ in range(12)]
    with pytest.raises(ValueError, match="packed counts must be nonnegative"):
        PackedBacktestResult(-1, 0, *arrays)


def test_packed_result_rejects_non_vector_trade_arrays():
    arrays = [np.zeros(1, dtype=np.float64) for _ in range(12)]
    arrays[0] = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        PackedBacktestResult(0, 0, *arrays)


def test_packed_result_rejects_short_trade_arrays():
    arrays = [np.zeros(1, dtype=np.float64) for _ in range(12)]
    with pytest.raises(ValueError, match="shorter than trade_count"):
        PackedBacktestResult(2, 0, *arrays)


def test_packed_rows_match_detailed_trade_materialization_exactly():
    bars, signals = _market()
    exit_spec = ExitSpec(stop_atr=1.0, target_r=1.0)
    packed = run_packed_backtest(bars, signals, SPEC, COST, RISK, exit_spec)
    detailed = run_fast_backtest(bars, signals, SPEC, COST, RISK, exit_spec)

    assert packed.trade_count == len(detailed.trades)
    assert packed.risk_skip_count == detailed.risk_skip_count
    reason_by_code = {
        1: "stop",
        2: "stop_same_bar_ambiguous",
        3: "target",
        4: "time",
        5: "end_of_data",
        6: "trail",
    }
    for index, trade in enumerate(detailed.trades):
        assert packed.direction[index] == trade.direction
        assert packed.signal_index[index] == trade.entry_index - 1
        assert packed.entry_index[index] == trade.entry_index
        assert packed.exit_index[index] == trade.exit_index
        assert packed.raw_entry[index] == bars.open[trade.entry_index]
        assert packed.entry_price[index] == trade.entry_price
        assert packed.initial_stop[index] == trade.stop_price
        assert packed.target[index] == trade.target_price
        assert packed.lot[index] == trade.lot_size
        assert packed.planned_risk[index] == trade.planned_risk_usd
        assert reason_by_code[int(packed.reason[index])] == trade.exit_reason
