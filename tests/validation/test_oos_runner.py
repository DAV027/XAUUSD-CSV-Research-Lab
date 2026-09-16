from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from xau_lab.backtest.models import MarketBars, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.single import MarketBundle
from xau_lab.validation.oos_runner import (
    append_oos_rows,
    build_holdout_signals,
    load_holdout_plan,
    run_oos_experiment,
)

START_EPOCH = 1_789_603_200  # 2026-09-17T00:00:00Z
END_EPOCH = 1_797_465_600  # 2026-12-17T00:00:00Z


def _experiment() -> CompleteExperiment:
    params = {"lookback": 2, "threshold_pct": 0.02}
    return CompleteExperiment(
        experiment_id="EXP_OOS_TEST",
        fingerprint="f" * 64,
        allocation_bucket="statistical",
        family="statistical",
        strategy_name="return_reversal",
        parameters=params,
        canonical_parameters_json=canonical_json(params),
        direction_mode="combined",
        stop_atr=1.0,
        exit_type="target_r",
        target_r=1.0,
        time_exit_minutes=None,
        atr_trail=None,
        commission_round_trip_per_lot=0.0,
        slippage_points_per_fill=0.0,
        strategy_seed=1,
        family_seed=2,
    )


def _market() -> MarketBundle:
    n = 18
    start_index = 6
    first_epoch = START_EPOCH - start_index * 60
    close = 2000.0 + np.arange(n, dtype=np.float64)
    bars = MarketBars(
        time_epoch=first_epoch + np.arange(n, dtype=np.int64) * 60,
        open=close.copy(),
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        spread=np.zeros(n, dtype=np.int64),
        atr=np.ones(n, dtype=np.float64),
        broker_timezone="UTC",
    )
    dates = np.array(
        ["2026-09-16"] * start_index + ["2026-09-17"] * (n - start_index),
        dtype=object,
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
        broker_date=dates,
        features={},
    )


def test_holdout_signals_keep_full_history_lookback_but_zero_pre_start():
    market = _market()

    signals = build_holdout_signals(
        _experiment(),
        market,
        start_epoch=START_EPOCH,
        end_exclusive_epoch=END_EPOCH,
    )

    first_oos_index = int(np.searchsorted(market.bars.time_epoch, START_EPOCH))
    assert np.all(signals[:first_oos_index] == 0)
    assert signals[first_oos_index] == -1


def test_run_oos_experiment_never_enters_before_prospective_start():
    outcome = run_oos_experiment(
        _experiment(),
        _market(),
        start_epoch=START_EPOCH,
        end_exclusive_epoch=END_EPOCH,
    )

    assert outcome.ok
    assert outcome.trades
    assert all(trade.entry_time >= START_EPOCH for trade in outcome.trades)
    assert outcome.master_result is not None
    assert outcome.master_result["oos_start_epoch"] == START_EPOCH
    assert outcome.master_result["oos_end_exclusive_epoch"] == END_EPOCH


def test_run_oos_experiment_requires_actual_prospective_rows():
    market = _market()
    after_snapshot = int(market.bars.time_epoch[-1]) + 60

    with pytest.raises(ValueError, match="no prospective OOS rows"):
        run_oos_experiment(
            _experiment(),
            market,
            start_epoch=after_snapshot,
            end_exclusive_epoch=END_EPOCH,
        )


def test_append_oos_rows_is_idempotent_and_rejects_snapshot_drift(tmp_path: Path):
    path = tmp_path / "OOS_RESULTS.csv"
    row = {
        "experiment_id": "EXP1",
        "observed_through_epoch": "1789606800",
        "net_profit": "1.25",
    }

    assert append_oos_rows(path, [row]) == 1
    assert append_oos_rows(path, [row]) == 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        stored = list(csv.DictReader(handle))
    assert stored == [row]

    changed = dict(row, net_profit="2.50")
    with pytest.raises(ValueError, match="immutable OOS snapshot drift"):
        append_oos_rows(path, [changed])


def test_repository_holdout_plan_starts_after_freeze_and_selection_window():
    plan = load_holdout_plan(Path("config/oos_holdout_v1.json"))

    assert plan["freeze_timestamp"] == "2026-09-16T06:46:32Z"
    assert plan["prospective_start_utc"] == "2026-09-17T00:00:00Z"
    assert plan["prospective_start_epoch"] == START_EPOCH
    assert plan["planned_months"] == 3
    assert plan["planned_end_exclusive_utc"] == "2026-12-17T00:00:00Z"
    assert plan["planned_end_exclusive_epoch"] == END_EPOCH
    assert plan["selection_data_end_epoch"] == 1_789_178_340


def test_holdout_plan_rejects_start_not_after_freeze(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test",
                "freeze_timestamp": "2026-09-17T00:00:00Z",
                "prospective_start_utc": "2026-09-17T00:00:00Z",
                "prospective_start_epoch": START_EPOCH,
                "planned_months": 3,
                "planned_end_exclusive_utc": "2026-12-17T00:00:00Z",
                "planned_end_exclusive_epoch": END_EPOCH,
                "selection_data_end_epoch": 1_789_178_340,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prospective start must be after freeze"):
        load_holdout_plan(path)
