import numpy as np
import pytest

from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.reference import run_reference_backtest


SPEC = SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)
RISK = RiskModel(account_equity=5000.0, preferred_risk_usd=2.0, hard_risk_usd=5.0)


def synthetic_bars(opens, highs, lows, closes, spreads, atr, start=1_700_000_000):
    n = len(opens)
    return MarketBars(
        np.arange(n, dtype=np.int64) * 60 + start,
        np.asarray(opens, float),
        np.asarray(highs, float),
        np.asarray(lows, float),
        np.asarray(closes, float),
        np.asarray(spreads, np.int64),
        np.asarray(atr, float),
    )


def test_long_signal_enters_next_bar_and_hits_target():
    bars = synthetic_bars(
        [100, 100, 100], [101, 100.5, 103], [99, 99.5, 99.5], [100, 100, 102], [0, 0, 0], [1, 1, 1]
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0, target_r=2.0)
    )
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.direction == 1
    assert trade.exit_reason == "target"
    assert trade.net_pnl > 0


def test_long_loser_hits_stop():
    bars = synthetic_bars(
        [100, 100, 100], [101, 100.5, 100.5], [99, 99.5, 98.5], [100, 100, 99], [0, 0, 0], [1, 1, 1]
    )
    trade = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, 2.0)
    ).trades[0]
    assert trade.entry_index == 1
    assert trade.direction == 1
    assert trade.exit_reason == "stop"
    assert trade.net_pnl < 0


def test_short_signal_enters_next_bar_and_hits_target():
    bars = synthetic_bars(
        [100, 100, 100], [101, 100.5, 100.5], [99, 99.5, 97], [100, 100, 98], [0, 0, 0], [1, 1, 1]
    )
    trade = run_reference_backtest(
        bars, [-1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, 2.0)
    ).trades[0]
    assert trade.entry_index == 1
    assert trade.direction == -1
    assert trade.exit_reason == "target"
    assert trade.net_pnl > 0


def test_short_loser_hits_stop():
    bars = synthetic_bars(
        [100, 100, 100], [101, 100.5, 101.5], [99, 99.5, 99.5], [100, 100, 101], [0, 0, 0], [1, 1, 1]
    )
    trade = run_reference_backtest(
        bars, [-1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, 2.0)
    ).trades[0]
    assert trade.direction == -1
    assert trade.exit_reason == "stop"
    assert trade.net_pnl < 0


def test_same_bar_stop_and_target_uses_stop_first():
    bars = synthetic_bars(
        [100, 100, 100], [101, 100.2, 103], [99, 99.8, 97], [100, 100, 100], [0, 0, 0], [1, 1, 1]
    )
    trade = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, 2.0)
    ).trades[0]
    assert trade.exit_reason == "stop_same_bar_ambiguous"
    assert trade.net_pnl < 0


def test_signals_while_position_open_are_ignored():
    bars = synthetic_bars(
        [100] * 4, [101, 100.2, 100.2, 100.2], [99, 99.8, 99.8, 99.8], [100] * 4, [0] * 4, [4] * 4
    )
    result = run_reference_backtest(
        bars, [1, -1, 1, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, None, 3)
    )
    assert len(result.trades) == 1
    assert result.trades[0].direction == 1


def test_round_trip_commission_is_charged_once():
    bars = synthetic_bars([100] * 3, [100.2] * 3, [99.8] * 3, [100] * 3, [0] * 3, [1] * 3)
    trade = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(6, 0), RISK, ExitSpec(1.0, None, 1)
    ).trades[0]
    assert trade.lot_size == pytest.approx(0.02)
    assert trade.gross_pnl == pytest.approx(0)
    assert trade.commission == pytest.approx(0.12)
    assert trade.net_pnl == pytest.approx(-0.12)


def test_spread_and_slippage_are_attributed_once_for_long_trade():
    bars = synthetic_bars([2000] * 3, [2000.2] * 3, [1999.8] * 3, [2000] * 3, [35] * 3, [1] * 3)
    trade = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 5), RISK, ExitSpec(1.0, None, 1)
    ).trades[0]
    assert trade.lot_size == pytest.approx(0.02)
    assert trade.entry_price == pytest.approx(2000.40)
    assert trade.exit_price == pytest.approx(1999.95)
    assert trade.spread_cost == pytest.approx(0.70)
    assert trade.slippage_cost == pytest.approx(0.20)
    assert trade.gross_pnl == pytest.approx(0)
    assert trade.net_pnl == pytest.approx(-0.90)


def test_infeasible_minimum_lot_is_counted_and_not_traded():
    bars = synthetic_bars([2000] * 3, [2001] * 3, [1999] * 3, [2000] * 3, [0] * 3, [6] * 3)
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0, 2.0)
    )
    assert result.trades == ()
    assert result.risk_skip_count == 1


def test_signal_length_must_match_market_bars():
    bars = synthetic_bars([100] * 2, [101] * 2, [99] * 2, [100] * 2, [0] * 2, [1] * 2)
    with pytest.raises(ValueError):
        run_reference_backtest(bars, [1], SPEC, CostModel(0, 0), RISK, ExitSpec(1.0))


def test_time_exit_occurs_on_fifth_elapsed_bar_after_entry():
    bars = synthetic_bars(
        opens=[100] * 7,
        highs=[100.2] * 7,
        lows=[99.8] * 7,
        closes=[100] * 7,
        spreads=[0] * 7,
        atr=[1] * 7,
    )
    result = run_reference_backtest(
        bars,
        [1, 0, 0, 0, 0, 0, 0],
        SPEC,
        CostModel(0, 0),
        RISK,
        ExitSpec(stop_atr=4.0, target_r=None, time_exit_minutes=5),
    )
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.exit_index == 6
    assert trade.exit_reason == "time"
    assert trade.hold_minutes == pytest.approx(5.0)


def test_atr_trail_uses_completed_bar_only_and_exits_later_bar():
    bars = synthetic_bars(
        opens=[100, 100, 100, 103],
        highs=[100.5, 100.5, 103.5, 103.2],
        lows=[99.5, 99.5, 99.5, 101.5],
        closes=[100, 100, 103, 102],
        spreads=[0, 0, 0, 0],
        atr=[1, 1, 1, 1],
    )
    result = run_reference_backtest(
        bars,
        [1, 0, 0, 0],
        SPEC,
        CostModel(0, 0),
        RISK,
        ExitSpec(stop_atr=4.0, target_r=None, atr_trail=1.0),
    )
    trade = result.trades[0]
    assert trade.entry_index == 1
    assert trade.exit_index == 3
    assert trade.exit_reason == "trail"
    assert trade.exit_price == pytest.approx(102.0)


def test_quantized_zero_stop_distance_is_skipped_instead_of_crashing():
    bars = synthetic_bars(
        opens=[2000, 2000, 2000],
        highs=[2000.01, 2000.01, 2000.01],
        lows=[1999.99, 1999.99, 1999.99],
        closes=[2000, 2000, 2000],
        spreads=[0, 0, 0],
        atr=[0.0001, 0.0001, 0.0001],
    )
    result = run_reference_backtest(
        bars, [1, 0, 0], SPEC, CostModel(0, 0), RISK, ExitSpec(stop_atr=1.0)
    )
    assert result.trades == ()
    assert result.risk_skip_count == 1


def test_trade_dates_and_timezone_follow_explicit_broker_timezone():
    bars = MarketBars(
        time_epoch=np.array([84540, 84600, 84660], dtype=np.int64),
        open=np.array([100.0, 100.0, 100.0]),
        high=np.array([100.2, 100.2, 100.2]),
        low=np.array([99.8, 99.8, 99.8]),
        close=np.array([100.0, 100.0, 100.0]),
        spread=np.array([0, 0, 0], dtype=np.int64),
        atr=np.array([1.0, 1.0, 1.0]),
        broker_timezone="Etc/GMT-2",
    )
    trade = run_reference_backtest(
        bars,
        [1, 0, 0],
        SPEC,
        CostModel(0, 0),
        RISK,
        ExitSpec(stop_atr=4.0, time_exit_minutes=1),
    ).trades[0]
    assert trade.broker_timezone == "Etc/GMT-2"
    assert trade.broker_date == "1970-01-02"
