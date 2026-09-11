import numpy as np
import pytest

from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.reference import run_reference_backtest


SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
RISK = RiskModel(account_equity=5000.0, preferred_risk_usd=2.0, hard_risk_usd=5.0)


def synthetic_bars(opens, highs, lows, closes, spreads, atr, start=1_700_000_000):
    n = len(opens)
    return MarketBars(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + start,
        open=np.asarray(opens, dtype=np.float64),
        high=np.asarray(highs, dtype=np.float64),
        low=np.asarray(lows, dtype=np.float64),
        close=np.asarray(closes, dtype=np.float64),
        spread=np.asarray(spreads, dtype=np.int64),
        atr=np.asarray(atr, dtype=np.float64),
    )


def test_long_signal_enters_next_bar_and_hits_target():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[101, 100.5, 103], lows=[99, 99.5, 99.5], closes=[100, 100, 102],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0))
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.direction == 1
    assert trade.exit_reason == "target"
    assert trade.net_pnl > 0


def test_long_loser_hits_stop():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[101, 100.5, 100.5], lows=[99, 99.5, 98.5], closes=[100, 100, 99],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0))
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.direction == 1
    assert trade.exit_reason == "stop"
    assert trade.net_pnl < 0


def test_short_signal_enters_next_bar_and_hits_target():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[101, 100.5, 100.5], lows=[99, 99.5, 97], closes=[100, 100, 98],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(bars, [-1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0))
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.direction == -1
    assert trade.exit_reason == "target"
    assert trade.net_pnl > 0


def test_short_loser_hits_stop():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[101, 100.5, 101.5], lows=[99, 99.5, 99.5], closes=[100, 100, 101],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(bars, [-1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0))
    trade = result.trades[0]
    assert trade.direction == -1
    assert trade.exit_reason == "stop"
    assert trade.net_pnl < 0


def test_same_bar_stop_and_target_uses_stop_first():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[101, 100.2, 103], lows=[99, 99.8, 97], closes=[100, 100, 100],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0)
    )
    assert result.trades[0].exit_reason == "stop_same_bar_ambiguous"
    assert result.trades[0].net_pnl < 0


def test_signals_while_position_open_are_ignored():
    bars = synthetic_bars(
        opens=[100, 100, 100, 100], highs=[101, 100.2, 100.2, 100.2],
        lows=[99, 99.8, 99.8, 99.8], closes=[100, 100, 100, 100],
        spreads=[0, 0, 0, 0], atr=[10, 10, 10, 10],
    )
    result = run_reference_backtest(
        bars, [1, -1, 1, 0], SPEC, CostModel(0, 0), RISK,
        ExitSpec(stop_atr=1.0, target_r=None, time_exit_minutes=3),
    )
    assert len(result.trades) == 1
    assert result.trades[0].direction == 1


def test_round_trip_commission_is_charged_once():
    bars = synthetic_bars(
        opens=[100, 100, 100], highs=[100.2, 100.2, 100.2], lows=[99.8, 99.8, 99.8], closes=[100, 100, 100],
        spreads=[0, 0, 0], atr=[1, 1, 1],
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(commission_round_trip_per_lot=6.0, slippage_points_per_fill=0),
        RISK, ExitSpec(stop_atr=1.0, target_r=None, time_exit_minutes=1),
    )
    trade = result.trades[0]
    assert trade.lot_size == pytest.approx(0.02)
    assert trade.gross_pnl == pytest.approx(0.0)
    assert trade.commission == pytest.approx(0.12)
    assert trade.net_pnl == pytest.approx(-0.12)


def test_spread_and_slippage_are_attributed_once_for_long_trade():
    bars = synthetic_bars(
        opens=[2000, 2000, 2000], highs=[2000.2, 2000.2, 2000.2], lows=[1999.8, 1999.8, 1999.8], closes=[2000, 2000, 2000],
        spreads=[35, 35, 35], atr=[1, 1, 1],
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(commission_round_trip_per_lot=0, slippage_points_per_fill=5),
        RISK, ExitSpec(stop_atr=1.0, target_r=None, time_exit_minutes=1),
    )
    trade = result.trades[0]
    assert trade.lot_size == pytest.approx(0.02)
    assert trade.entry_price == pytest.approx(2000.40)
    assert trade.exit_price == pytest.approx(1999.95)
    assert trade.spread_cost == pytest.approx(0.70)
    assert trade.slippage_cost == pytest.approx(0.20)
    assert trade.gross_pnl == pytest.approx(0.0)
    assert trade.net_pnl == pytest.approx(-0.90)


def test_infeasible_minimum_lot_is_counted_and_not_traded():
    bars = synthetic_bars(
        opens=[2000, 2000, 2000], highs=[2001, 2001, 2001], lows=[1999, 1999, 1999], closes=[2000, 2000, 2000],
        spreads=[0, 0, 0], atr=[6, 6, 6],
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0)
    )
    assert result.trades == ()
    assert result.risk_skip_count == 1


def test_signal_length_must_match_market_bars():
    bars = synthetic_bars([100, 100], [101, 101], [99, 99], [100, 100], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        run_reference_backtest(bars, [1], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0))
