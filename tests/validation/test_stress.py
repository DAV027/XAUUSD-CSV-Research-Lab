from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from xau_lab.backtest.models import MarketBars, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.single import MarketBundle
from xau_lab.strategies.base import StrategyContext, StrategyDefinition
from xau_lab.strategies.registry import register_strategy
from xau_lab.validation.stress import (
    STRESS_RESULT_FIELDS,
    StressConfig,
    parameter_neighbors,
    run_stress_suite,
    stress_candidate,
    write_stress_results,
)


def test_integer_numeric_neighbors_are_plus_minus_10_and_20_percent_with_domain_clipping():
    rows = parameter_neighbors({"period": 20}, {"period": (10, 23)})
    assert [row["period"] for row in rows] == [16, 18, 22, 23]


def test_float_numeric_neighbors_are_plus_minus_10_and_20_percent():
    rows = parameter_neighbors({"threshold": 1.0}, {"threshold": (0.5, 2.0)})
    assert [row["threshold"] for row in rows] == [0.8, 0.9, 1.1, 1.2]


def test_categorical_parameters_are_not_mutated_and_neighbors_change_one_parameter_at_a_time():
    params = {"period": 20, "threshold": 1.0, "mode": "close"}
    domains = {"period": (10, 30), "threshold": (0.5, 2.0), "mode": ("close", "high")}
    rows = parameter_neighbors(params, domains)
    assert len(rows) == 8
    for row in rows:
        changed = [key for key in params if row[key] != params[key]]
        assert len(changed) == 1
        assert changed[0] != "mode"
        assert row["mode"] == "close"


def _stress_signal(ctx: StrategyContext, params: dict) -> np.ndarray:
    return np.ones(len(ctx), dtype=np.int8)


try:
    register_strategy(
        StrategyDefinition(
            "test",
            "stress_long_test",
            _stress_signal,
            {"period": (10, 30), "mode": ("close", "high")},
        )
    )
except ValueError:
    pass


def _market() -> MarketBundle:
    start = datetime(2026, 1, 2, 10, 0, tzinfo=timezone.utc)
    epochs = np.asarray([int(start.timestamp()) + 60 * i for i in range(80)], dtype=np.int64)
    open_ = 2000.0 + 0.25 * np.arange(len(epochs), dtype=float)
    close = open_ + 0.10
    bars = MarketBars(
        time_epoch=epochs,
        open=open_,
        high=close + 0.05,
        low=open_ - 0.05,
        close=close,
        spread=np.full(len(epochs), 10, dtype=np.int64),
        atr=np.full(len(epochs), 4.0),
        broker_timezone="UTC",
    )
    return MarketBundle(
        bars=bars,
        symbol=SymbolSpec(0.01, 2, 100.0, 0.01, 0.01),
        broker_date=np.asarray(["2026-01-02"] * len(epochs), dtype=object),
        features={},
    )


def _candidate() -> CompleteExperiment:
    params = {"period": 20, "mode": "close"}
    return CompleteExperiment(
        experiment_id="EXP_STRESS",
        fingerprint="stress-fingerprint",
        allocation_bucket="trend",
        family="test",
        strategy_name="stress_long_test",
        parameters=params,
        canonical_parameters_json=canonical_json(params),
        direction_mode="long",
        stop_atr=1.0,
        exit_type="time",
        target_r=None,
        time_exit_minutes=1,
        atr_trail=None,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        strategy_seed=1,
        family_seed=1,
    )


def test_stress_candidate_runs_required_cost_scenarios_and_reports_stability():
    report = stress_candidate(_candidate(), _market(), StressConfig())
    labels = {row.label for row in report.results}
    assert {"slippage:0", "slippage:5", "slippage:10", "slippage:20"} <= labels
    assert {"commission:1", "commission:1.25", "commission:1.5"} <= labels
    assert {"spread:1", "spread:1.25", "spread:1.5"} <= labels
    assert report.parameter_run_count == 4
    assert report.cost_run_count == 10
    assert 0.0 <= report.parameter_stability_pct <= 100.0
    assert 0.0 <= report.cost_stability_pct <= 100.0
    assert report.max_drawdown_pct >= 0.0


def test_approved_run_stress_suite_interface_uses_default_matrix():
    report = run_stress_suite(_candidate(), _market())
    assert report.parameter_run_count == 4
    assert report.cost_run_count == 10


def test_commission_stress_is_pinned_to_approved_six_dollar_baseline():
    standard = run_stress_suite(_candidate(), _market())
    altered_source = run_stress_suite(
        replace(_candidate(), commission_round_trip_per_lot=12.0),
        _market(),
    )
    standard_rows = {
        row.label: (row.pf, row.expectancy_usd, row.net_profit, row.max_drawdown_pct)
        for row in standard.results
        if row.label.startswith("commission:")
    }
    altered_rows = {
        row.label: (row.pf, row.expectancy_usd, row.net_profit, row.max_drawdown_pct)
        for row in altered_source.results
        if row.label.startswith("commission:")
    }
    assert standard_rows == altered_rows


def test_stress_writer_records_all_runs_with_canonical_fields(tmp_path):
    report = run_stress_suite(_candidate(), _market())
    output = tmp_path / "STRESS_RESULTS.csv"
    write_stress_results(output, report.results)
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == STRESS_RESULT_FIELDS
        rows = list(reader)
    assert len(rows) == report.parameter_run_count + report.cost_run_count
    assert {row["stress_type"] for row in rows} == {"parameter", "cost"}


def test_stress_is_deterministic_for_same_candidate_and_market():
    first = stress_candidate(_candidate(), _market(), StressConfig())
    second = stress_candidate(_candidate(), _market(), StressConfig())
    assert first == second
