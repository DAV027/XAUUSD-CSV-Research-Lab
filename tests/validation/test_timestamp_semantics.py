from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import polars as pl

from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.campaign import load_market_bundle
from xau_lab.runner.single import run_experiment
from xau_lab.strategies.registry import get_strategy


def _write_broker_local_market(tmp_path: Path, rows: int = 180) -> Path:
    start = datetime(2026, 7, 1, 3, 0, 0)
    times = [(start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(rows)]
    x = np.arange(rows, dtype=np.float64)
    close = 2000.0 + np.sin(x / 3.0) * 4.0
    frame = pl.DataFrame(
        {
            "time": times,
            "open": close - 0.05,
            "high": close + 0.35,
            "low": close - 0.35,
            "close": close,
            "spread": np.zeros(rows, dtype=np.int64),
            "atr_14_lag1": np.ones(rows, dtype=np.float64),
            "broker_date": ["2026-07-01"] * rows,
            "entry_allowed": np.ones(rows, dtype=bool),
        }
    )
    path = tmp_path / "broker_local.parquet"
    frame.write_parquet(path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "digits": 2,
                "point": 0.01,
                "trade_contract_size": 100.0,
                "volume_min": 0.01,
                "volume_step": 0.01,
                "volume_max": 100.0,
                "server_timezone": "EET",
            }
        ),
        encoding="utf-8",
    )
    return path


def _experiment() -> CompleteExperiment:
    params = {"lookback": 3, "threshold_pct": 0.05}
    return CompleteExperiment(
        experiment_id="EXP_TIMESTAMP_PARITY",
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


def test_broker_local_to_utc_mode_corrects_epoch_without_changing_signals_or_economics(tmp_path: Path):
    path = _write_broker_local_market(tmp_path)

    legacy = load_market_bundle(path)
    corrected = load_market_bundle(path, timestamp_semantics="broker_local_to_utc")

    legacy_first = int(datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc).timestamp())
    corrected_first = int(datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc).timestamp())
    assert int(legacy.bars.time_epoch[0]) == legacy_first
    assert int(corrected.bars.time_epoch[0]) == corrected_first
    assert int(legacy.bars.time_epoch[0] - corrected.bars.time_epoch[0]) == 3 * 60 * 60

    experiment = _experiment()
    definition = get_strategy(experiment.strategy_name)
    legacy_signals = definition.generate(legacy.strategy_context(), experiment.parameters)
    corrected_signals = definition.generate(corrected.strategy_context(), experiment.parameters)
    np.testing.assert_array_equal(legacy_signals, corrected_signals)

    legacy_result = run_experiment(experiment, legacy, include_trades=True)
    corrected_result = run_experiment(experiment, corrected, include_trades=True)
    assert legacy_result.ok and corrected_result.ok
    assert legacy_result.master_result is not None
    assert corrected_result.master_result is not None
    assert legacy_result.master_result["completed_trades"] > 0

    legacy_metrics = dict(legacy_result.master_result)
    corrected_metrics = dict(corrected_result.master_result)
    legacy_metrics.pop("data_start")
    legacy_metrics.pop("data_end")
    corrected_metrics.pop("data_start")
    corrected_metrics.pop("data_end")
    assert corrected_metrics == legacy_metrics
