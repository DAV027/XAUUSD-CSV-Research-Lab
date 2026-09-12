from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from xau_lab.backtest.models import MarketBars, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.single import MarketBundle, run_experiment
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy
from xau_lab.validation.walkforward import summarize_fold_results, validate_candidate


def _regime_signal(ctx: StrategyContext, params: dict) -> np.ndarray:
    out = np.zeros(len(ctx), dtype=np.int8)
    for i, epoch in enumerate(ctx.time_epoch):
        year = datetime.fromtimestamp(int(epoch), tz=timezone.utc).year
        if year == 2023:
            out[i] = -1 if i % 5 == 0 else 1
        else:
            out[i] = -1
    return out


try:
    register_strategy(StrategyDefinition("test", "walkforward_regime_test", _regime_signal, {}))
except ValueError:
    pass


def _market() -> MarketBundle:
    epochs: list[int] = []
    broker_dates: list[str] = []
    prices: list[float] = []
    value = 2000.0
    year, month = 2023, 1
    for _ in range(15):
        for minute in range(10):
            dt = datetime(year, month, 15, 10, minute, tzinfo=timezone.utc)
            epochs.append(int(dt.timestamp()))
            broker_dates.append(dt.date().isoformat())
            prices.append(value)
            value += 0.20
        month += 1
        if month == 13:
            month = 1
            year += 1

    open_ = np.asarray(prices, dtype=float)
    close = open_ + 0.10
    bars = MarketBars(
        time_epoch=np.asarray(epochs, dtype=np.int64),
        open=open_,
        high=np.maximum(open_, close) + 0.05,
        low=np.minimum(open_, close) - 0.05,
        close=close,
        spread=np.zeros(len(open_), dtype=np.int64),
        atr=np.full(len(open_), 4.0),
        broker_timezone="UTC",
    )
    return MarketBundle(
        bars=bars,
        symbol=SymbolSpec(
            point=0.01,
            digits=2,
            contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
        ),
        broker_date=np.asarray(broker_dates, dtype=object),
        features={},
    )


def _experiment() -> CompleteExperiment:
    params = {}
    return CompleteExperiment(
        experiment_id="EXP_WF_REGIME",
        fingerprint="wf-regime-fingerprint",
        allocation_bucket="trend",
        family="test",
        strategy_name="walkforward_regime_test",
        parameters=params,
        canonical_parameters_json=canonical_json(params),
        direction_mode="combined",
        stop_atr=1.0,
        exit_type="time",
        target_r=None,
        time_exit_minutes=1,
        atr_trail=None,
        commission_round_trip_per_lot=0.0,
        slippage_points_per_fill=0.0,
        strategy_seed=1,
        family_seed=1,
    )


def test_validation_reports_regime_degradation_instead_of_hiding_it_in_full_history():
    market = _market()
    experiment = _experiment()
    full = run_experiment(experiment, market)
    assert full.master_result is not None
    assert full.master_result["net_profit"] > 0.0

    results = validate_candidate(experiment, market, "expanding")
    assert len(results) == 2
    research, validation = results
    assert research.segment == "research"
    assert validation.segment == "validation"
    assert research.start == "2023-01"
    assert research.end == "2023-12"
    assert validation.start == "2024-01"
    assert validation.end == "2024-03"
    assert research.net_profit > 0.0
    assert research.pf is not None
    assert validation.net_profit < 0.0
    assert validation.expectancy_usd is not None and validation.expectancy_usd < 0.0


def test_walkforward_summary_exposes_validation_stability_and_degradation_ratio():
    results = validate_candidate(_experiment(), _market(), "rolling")
    summary = summarize_fold_results(results)
    assert summary["validation_fold_count"] == 1
    assert summary["validation_pf_above_1_fraction"] == 0.0
    assert summary["validation_positive_expectancy_fraction"] == 0.0
    assert summary["validation_joint_stability_fraction"] == 0.0
    assert summary["worst_validation_net_profit"] < 0.0
    assert summary["research_to_validation_pf_degradation_ratio"] is not None


def test_scheme_must_be_expanding_or_rolling():
    try:
        validate_candidate(_experiment(), _market(), "random")
    except ValueError as exc:
        assert "scheme" in str(exc)
    else:
        raise AssertionError("invalid walk-forward scheme should fail")
