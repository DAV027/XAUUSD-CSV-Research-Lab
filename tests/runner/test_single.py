from __future__ import annotations

import numpy as np
import pytest

import xau_lab.runner.single as single_module
from xau_lab.backtest.models import MarketBars, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.single import MarketBundle, run_experiment


def _experiment() -> CompleteExperiment:
    params = {"lookback": 2}
    return CompleteExperiment(
        experiment_id="EXP_DISCOVERY_PATH",
        fingerprint="f" * 64,
        allocation_bucket="breakout",
        family="breakout",
        strategy_name="nbar_breakout",
        parameters=params,
        canonical_parameters_json=canonical_json(params),
        direction_mode="combined",
        stop_atr=1.0,
        exit_type="target_r",
        target_r=1.0,
        time_exit_minutes=None,
        atr_trail=None,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        strategy_seed=1,
        family_seed=2,
    )


def _market() -> MarketBundle:
    n = 16
    close = 2000.0 + np.arange(n, dtype=np.float64)
    open_ = close - 0.2
    high = close + 0.4
    low = close - 0.4
    bars = MarketBars(
        time_epoch=np.arange(n, dtype=np.int64) * 60 + 1_767_312_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        spread=np.full(n, 10, dtype=np.int64),
        atr=np.ones(n, dtype=np.float64),
        broker_timezone="UTC",
    )
    london = np.zeros(n, dtype=bool)
    london[4:10] = True
    return MarketBundle(
        bars=bars,
        symbol=SymbolSpec(
            point=0.01,
            digits=2,
            contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
        ),
        broker_date=np.array(["2026-01-02"] * n, dtype=object),
        features={
            "session_asia": ~london,
            "session_london": london,
            "session_new_york": np.zeros(n, dtype=bool),
            "session_overlap": np.zeros(n, dtype=bool),
        },
    )


def test_market_bundle_derives_contiguous_broker_calendar_group_ids():
    base = _market()
    dates = np.array(
        [
            "2025-12-31",
            "2025-12-31",
            "2026-01-01",
            "2026-01-01",
            "2026-02-01",
            "2026-02-01",
        ],
        dtype=object,
    )
    bars = MarketBars(
        time_epoch=np.arange(6, dtype=np.int64) * 60 + 1_767_312_000,
        open=np.ones(6),
        high=np.ones(6) + 1,
        low=np.ones(6) - 1,
        close=np.ones(6),
        spread=np.zeros(6, dtype=np.int64),
        atr=np.ones(6),
    )

    bundle = MarketBundle(bars=bars, symbol=base.symbol, broker_date=dates, features={})

    assert bundle.broker_day_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert bundle.broker_month_id.tolist() == [0, 0, 1, 1, 2, 2]
    assert bundle.broker_year_id.tolist() == [0, 0, 1, 1, 1, 1]
    assert bundle.broker_day_id.dtype == np.int32
    assert bundle.broker_day_id.flags.c_contiguous
    assert not bundle.broker_day_id.flags.writeable


def test_market_bundle_reuses_ids_for_recurring_calendar_keys():
    base = _market()
    dates = np.array(
        ["2025-12-31", "2026-01-01", "2025-12-31", "2026-02-01"],
        dtype=object,
    )
    bars = MarketBars(
        time_epoch=np.arange(4, dtype=np.int64) * 60 + 1_767_312_000,
        open=np.ones(4),
        high=np.ones(4) + 1,
        low=np.ones(4) - 1,
        close=np.ones(4),
        spread=np.zeros(4, dtype=np.int64),
        atr=np.ones(4),
    )

    bundle = MarketBundle(bars=bars, symbol=base.symbol, broker_date=dates, features={})

    assert bundle.broker_day_id.tolist() == [0, 1, 0, 2]
    assert bundle.broker_month_id.tolist() == [0, 1, 0, 2]
    assert bundle.broker_year_id.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize(
    "invalid_date", ["2026-02-29", "2026-13-01", "2026-01-32", "2026-1x-02"]
)
def test_market_bundle_rejects_invalid_broker_dates(invalid_date: str):
    base = _market()
    bars = MarketBars(
        time_epoch=np.array([1_767_312_000], dtype=np.int64),
        open=np.ones(1),
        high=np.ones(1) + 1,
        low=np.ones(1) - 1,
        close=np.ones(1),
        spread=np.zeros(1, dtype=np.int64),
        atr=np.ones(1),
    )

    with pytest.raises(ValueError, match="broker_date must use valid YYYY-MM-DD dates"):
        MarketBundle(
            bars=bars,
            symbol=base.symbol,
            broker_date=np.array([invalid_date], dtype=object),
            features={},
        )


def test_market_bundle_rejects_non_vector_broker_dates():
    base = _market()
    with pytest.raises(ValueError, match="broker_date must be one-dimensional"):
        MarketBundle(
            bars=base.bars,
            symbol=base.symbol,
            broker_date=np.array([["2026-01-02"] * len(base.bars)], dtype=object),
            features=base.features,
        )


def test_market_bundle_normalizes_supplied_calendar_group_ids():
    base = _market()
    supplied = np.zeros(len(base.bars), dtype=np.int64)

    bundle = MarketBundle(
        bars=base.bars,
        symbol=base.symbol,
        broker_date=base.broker_date,
        features=base.features,
        broker_day_id=supplied,
        broker_month_id=supplied,
        broker_year_id=supplied,
    )

    for values in (bundle.broker_day_id, bundle.broker_month_id, bundle.broker_year_id):
        assert values.dtype == np.int32
        assert values.flags.c_contiguous
        assert not values.flags.writeable


def test_market_bundle_rejects_partial_calendar_group_ids():
    base = _market()
    with pytest.raises(ValueError, match="calendar group IDs must be provided together"):
        MarketBundle(
            bars=base.bars,
            symbol=base.symbol,
            broker_date=base.broker_date,
            features=base.features,
            broker_day_id=np.zeros(len(base.bars), dtype=np.int32),
        )


@pytest.mark.parametrize(
    "invalid_ids",
    [
        np.zeros((2, 8), dtype=np.int32),
        np.zeros(15, dtype=np.int32),
        np.array([-1] + [0] * 15, dtype=np.int32),
        np.array([0] * 8 + [2] * 8, dtype=np.int32),
        np.zeros(16, dtype=np.float64),
        np.array([0.5] + [0.0] * 15, dtype=np.float64),
        np.array([0] * 15 + [np.iinfo(np.int32).max + 1], dtype=np.int64),
    ],
)
def test_market_bundle_rejects_invalid_supplied_calendar_group_ids(
    invalid_ids: np.ndarray,
):
    base = _market()
    with pytest.raises(ValueError, match="broker_day_id"):
        MarketBundle(
            bars=base.bars,
            symbol=base.symbol,
            broker_date=base.broker_date,
            features=base.features,
            broker_day_id=invalid_ids,
            broker_month_id=np.zeros(len(base.bars), dtype=np.int32),
            broker_year_id=np.zeros(len(base.bars), dtype=np.int32),
        )


def test_discovery_path_does_not_annotate_trades(monkeypatch: pytest.MonkeyPatch):
    def forbidden_annotation(*args, **kwargs):
        raise AssertionError("discovery-only runs must not duplicate Trade objects")

    monkeypatch.setattr(single_module, "_annotate_trade", forbidden_annotation)

    outcome = run_experiment(_experiment(), _market(), include_trades=False)

    assert outcome.ok
    assert outcome.trades == ()
    assert outcome.master_result is not None
    assert outcome.master_result["completed_trades"] > 0


def test_discovery_master_metrics_match_full_annotation_path():
    discovery = run_experiment(_experiment(), _market(), include_trades=False)
    full = run_experiment(_experiment(), _market(), include_trades=True)

    assert discovery.master_result == full.master_result
    assert discovery.trades == ()
    assert len(full.trades) > 0


def test_full_trade_path_retains_experiment_and_session_annotations():
    outcome = run_experiment(_experiment(), _market(), include_trades=True)

    assert outcome.trades
    trade = outcome.trades[0]
    assert trade.experiment_id == "EXP_DISCOVERY_PATH"
    assert trade.parameter_set_id == "f" * 64
    assert trade.fingerprint == "f" * 64
    assert trade.broker_date == "2026-01-02"
    assert trade.session_asia is not None
    assert trade.session_london is not None
    assert trade.session_new_york is False
    assert trade.session_overlap is False
    assert trade.broker_timezone == "UTC"
