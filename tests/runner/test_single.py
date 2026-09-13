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
