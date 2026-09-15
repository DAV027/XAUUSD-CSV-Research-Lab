from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from xau_lab.backtest.fast import (
    _materialize_packed_trades, run_fast_backtest, run_packed_backtest,
)
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec
from xau_lab.backtest.packed import PackedBacktestResult
from xau_lab.metrics.packed import summarize_packed_backtest
from xau_lab.metrics.performance import summarize_trades
from xau_lab.runner.single import MarketBundle


SPEC = SymbolSpec(0.01, 2, 100.0, 0.01, 0.01)
COST = CostModel(6.137, 3.17)
RISK = RiskModel()


def _bars(n, **overrides):
    values = dict(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + 1_700_000_000,
        open=np.full(n, 2000.0), high=np.full(n, 2000.7),
        low=np.full(n, 1999.3), close=np.full(n, 2000.0),
        spread=np.arange(n, dtype=np.int64) % 30, atr=np.ones(n),
    )
    values.update(overrides)
    return MarketBars(**values)


def _bundle(bars, dates=None, symbol=SPEC):
    if dates is None:
        dates = [str(date(2023, 12, 30) + timedelta(days=i // 3)) for i in range(len(bars))]
    return MarketBundle(bars, symbol, np.asarray(dates, dtype=object), {})


def _summarize(packed, bundle, cost=COST, risk=RISK):
    return summarize_packed_backtest(
        packed, bundle.bars, bundle.symbol, cost, risk,
        bundle.broker_day_id, bundle.broker_month_id, bundle.broker_year_id,
    )


def _assert_summary(bars, signals, exit_spec, dates=None, cost=COST, risk=RISK):
    bundle = _bundle(bars, dates)
    detailed = run_fast_backtest(bars, signals, SPEC, cost, risk, exit_spec)
    packed = run_packed_backtest(bars, signals, SPEC, cost, risk, exit_spec)
    expected = summarize_trades(detailed.trades, risk.account_equity, broker_dates=bundle.broker_date)
    assert _summarize(packed, bundle, cost, risk) == expected
    assert packed.risk_skip_count == detailed.risk_skip_count
    return detailed, expected


@pytest.mark.parametrize("n", [0, 1, 30])
def test_empty_summary_preserves_zero_and_none_fields(n):
    _, summary = _assert_summary(_bars(n), np.zeros(n, dtype=np.int8), ExitSpec(1.0))
    assert summary["completed_trades"] == 0
    assert summary["net_profit"] == 0.0
    assert summary["median_hold_minutes"] is None


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("reason,exit_spec,high,low,close", [
    ("stop", ExitSpec(0.3), 2000.7, 1999.3, 2000.0),
    ("stop_same_bar_ambiguous", ExitSpec(0.3, target_r=1.0), 2000.7, 1999.3, 2000.0),
    ("target", ExitSpec(2.0, target_r=0.1), 2000.7, 1999.3, 2000.0),
    ("time", ExitSpec(2.0, time_exit_minutes=2), 2000.7, 1999.3, 2000.0),
    ("trail", ExitSpec(2.0, atr_trail=0.1), 2000.7, 1999.3, 2000.0),
    ("end_of_data", ExitSpec(2.0), 2000.7, 1999.3, 2000.0),
])
def test_each_exit_reason_exactly(direction, reason, exit_spec, high, low, close):
    n = 31
    signals = np.zeros(n, dtype=np.int8)
    signals[0] = direction
    detailed, _ = _assert_summary(
        _bars(n, high=np.full(n, high), low=np.full(n, low), close=np.full(n, close)),
        signals, exit_spec,
    )
    assert len(detailed.trades) == 1
    assert detailed.trades[0].exit_reason == reason


@pytest.mark.parametrize("n", [20, 21, 20_000])
def test_dense_mixed_trade_summary_and_calendar_boundaries(n):
    rng = np.random.default_rng(9215000)
    opens = 2000.0 + np.cumsum(rng.normal(0, 0.3, n))
    closes = opens + rng.normal(0, 0.2, n)
    bars = _bars(n, open=opens, close=closes,
                 high=np.maximum(opens, closes) + rng.uniform(0.2, 1.2, n),
                 low=np.minimum(opens, closes) - rng.uniform(0.2, 1.2, n),
                 atr=rng.uniform(0.4, 1.6, n))
    signals = rng.choice(np.array([-1, 1], dtype=np.int8), n)
    detailed, summary = _assert_summary(bars, signals, ExitSpec(0.5, target_r=0.8))
    assert summary["long_trades"] > 0 and summary["short_trades"] > 0
    if n == 20_000:
        assert len(detailed.trades) > 15_000
        assert summary["wins"] > 0 and summary["losses"] > 0


@pytest.mark.parametrize("atr", [0.0, -1.0, np.nan, np.inf, -np.inf, 100.0])
def test_invalid_atr_and_risk_skips(atr):
    detailed, summary = _assert_summary(
        _bars(12, atr=np.full(12, atr)), np.ones(12, dtype=np.int8), ExitSpec(1.0),
    )
    assert summary["completed_trades"] == 0
    assert detailed.risk_skip_count == (11 if atr == 100.0 else 0)


def _synthetic(values, *, lots=None, risks=None, dates=None, cost=CostModel(0.0, 0.0)):
    """Controlled packed rows, finalized by the unchanged detailed oracle."""
    count = len(values)
    n = 2 * count + 2
    bars = _bars(n, spread=np.arange(n, dtype=np.int64) % 30 if cost != CostModel(0, 0) else np.zeros(n, dtype=np.int64))
    direction = np.where(np.arange(count) % 2, -1, 1).astype(np.int8)
    entry = np.arange(count, dtype=np.int64)
    size = np.ones(count) if lots is None else np.asarray(lots, dtype=np.float64)
    raw_exit = np.asarray(values, dtype=np.float64) * direction
    symbol = replace(SPEC, contract_size=1.0)
    packed = PackedBacktestResult(
        count, 0, direction, entry.copy(), entry, entry + np.arange(count),
        np.zeros(count), np.zeros(count), np.zeros(count), np.full(count, np.nan),
        size, np.ones(count) if risks is None else np.asarray(risks, dtype=np.float64),
        raw_exit, np.full(count, 5, dtype=np.int8),
    )
    bundle = _bundle(bars, dates, symbol)
    return packed, bundle, cost


@pytest.mark.parametrize("values", [
    [1.0], [-1.0], [0.0], [1, 2, 3], [-1, -2, -3, -4],
    [7, 7, 7, 7, 7, 7, 7, -2], [5, -2, 0, -3, -4, 8],
    [1e16, 1.0, -1e16], [-1e16, -1, 1e16],
    [1e16, 1, 1, 1, -1e16, -1, -1, -1],
    [1e16, 1, 1, 1, 1, 1, -1e16],
])
def test_adversarial_totals_streaks_top_five_and_odd_even_medians(values):
    packed, bundle, cost = _synthetic(values)
    trades = _materialize_packed_trades(packed, bundle.bars, bundle.symbol, cost)
    expected = summarize_trades(trades, RISK.account_equity, broker_dates=bundle.broker_date)
    actual = _summarize(packed, bundle, cost)
    assert actual == expected
    assert actual["net_profit"] == sum(float(value) for value in values)


def test_recurrent_calendar_keys_and_plain_calendar_accumulation():
    values = [1e16, 1.0, -1e16, 9.0, -3.0, 2.0, 0.0]
    dates = ["2024-02-01", "2024-02-01", "2024-02-01", "2023-12-31", "2024-03-01", "2023-12-31", "2024-02-02"]
    dates += ["2025-01-01"] * (2 * len(values) + 2 - len(dates))
    packed, bundle, cost = _synthetic(values, dates=dates)
    trades = _materialize_packed_trades(packed, bundle.bars, bundle.symbol, cost)
    actual = _summarize(packed, bundle, cost)
    assert actual == summarize_trades(trades, RISK.account_equity, broker_dates=dates)
    assert actual["net_profit"] == 9.0
    assert actual["median_profit_per_active_day"] == 0.0
    assert actual["active_months"] == 3


def test_cost_and_pnl_r_compensated_sums_with_missing_risk():
    packed, bundle, cost = _synthetic(
        [0.0] * 10, lots=[1e16, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        risks=[1, 0, -1, 3, 7, 11, 13, 17, 19, 23], cost=CostModel(1.137, 3.17),
    )
    trades = _materialize_packed_trades(packed, bundle.bars, bundle.symbol, cost)
    assert _summarize(packed, bundle, cost) == summarize_trades(
        trades, RISK.account_equity, broker_dates=bundle.broker_date,
    )


@pytest.mark.parametrize("field", ["broker_day_id", "broker_month_id", "broker_year_id"])
@pytest.mark.parametrize("failure", ["shape", "length", "negative", "large", "float"])
def test_rejects_invalid_calendar_arrays_before_kernel(field, failure):
    packed, bundle, cost = _synthetic([1.0])
    ids = np.array(getattr(bundle, field))
    if failure == "shape":
        ids = ids.reshape(1, -1)
    elif failure == "length":
        ids = ids[:-1]
    elif failure == "negative":
        ids[0] = -1
    elif failure == "large":
        ids[0] = len(bundle.bars)
    else:
        ids = ids.astype(float)
    object.__setattr__(bundle, field, ids)
    with pytest.raises(ValueError):
        _summarize(packed, bundle, cost)


@pytest.mark.parametrize("field", ["entry_index", "exit_index"])
@pytest.mark.parametrize("value", [-1, 4, 2**62])
def test_rejects_out_of_bounds_trade_indices(field, value):
    packed, bundle, cost = _synthetic([1.0])
    getattr(packed, field)[0] = value
    with pytest.raises(ValueError, match="index"):
        _summarize(packed, bundle, cost)


@pytest.mark.parametrize("field,value", [
    ("trade_count", -1), ("trade_count", 1.5), ("trade_count", 99),
    ("risk_skip_count", -1), ("raw_exit", np.zeros((1, 1))),
    ("lot", np.empty(0)), ("entry_index", np.array([0.5])),
])
def test_revalidates_mutated_packed_shapes_counts_and_index_types(field, value):
    packed, bundle, cost = _synthetic([1.0])
    object.__setattr__(packed, field, value)
    with pytest.raises(ValueError):
        _summarize(packed, bundle, cost)


def test_ignores_unused_capacity_rows():
    packed, bundle, cost = _synthetic([3.0, 1e16, -1e16])
    object.__setattr__(packed, "trade_count", 1)
    packed.entry_index[1:] = -999
    trades = _materialize_packed_trades(packed, bundle.bars, bundle.symbol, cost)
    assert _summarize(packed, bundle, cost) == summarize_trades(
        trades, RISK.account_equity, broker_dates=bundle.broker_date,
    )
