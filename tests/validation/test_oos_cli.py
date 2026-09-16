from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.run_oos_holdout import run_oos_snapshot
from xau_lab.experiments.spec import canonical_json

START_EPOCH = 1_789_603_200
END_EPOCH = 1_797_465_600
SELECTION_END = 1_789_178_340


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    params = {"lookback": 2, "threshold_pct": 0.02}
    row = {
        "experiment_id": "EXP_SYNTH",
        "fingerprint": "f" * 64,
        "allocation_bucket": "statistical",
        "family": "statistical",
        "strategy_name": "return_reversal",
        "canonical_parameters_json": canonical_json(params),
        "direction_mode": "combined",
        "stop_atr": "1.0",
        "exit_type": "target_r",
        "target_r": "1.0",
        "time_exit_minutes": "",
        "atr_trail": "",
        "commission_round_trip_per_lot": "0.0",
        "slippage_points_per_fill": "0.0",
        "strategy_seed": "1",
        "family_seed": "2",
        "sampler_version": "v1",
    }
    catalog = tmp_path / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    canonical_feature = tmp_path / "canonical.parquet"
    canonical_feature.write_bytes(b"frozen-selection-feature")

    shortlist = tmp_path / "shortlist.json"
    shortlist.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test_shortlist",
                "expected_candidate_count": 1,
                "selection_source": {
                    "campaign_commit": "a" * 40,
                    "feature_sha256": _sha256(canonical_feature),
                    "catalog_sha256": _sha256(catalog),
                    "campaign_data_end_epoch": SELECTION_END,
                    "campaign_data_end_utc": "2026-09-12T01:59:00Z",
                    "oos_start_exclusive_epoch": SELECTION_END,
                    "oos_rule": "time_epoch > oos_start_exclusive_epoch",
                },
                "candidates": [{"tier": "primary", "experiment": row}],
            }
        ),
        encoding="utf-8",
    )

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test_holdout",
                "status": "frozen",
                "shortlist_config": str(shortlist),
                "freeze_timestamp": "2026-09-16T06:46:32Z",
                "selection_data_end_epoch": SELECTION_END,
                "prospective_start_utc": "2026-09-17T00:00:00Z",
                "prospective_start_epoch": START_EPOCH,
                "planned_months": 3,
                "planned_end_exclusive_utc": "2026-12-17T00:00:00Z",
                "planned_end_exclusive_epoch": END_EPOCH,
            }
        ),
        encoding="utf-8",
    )

    n = 18
    first_epoch = START_EPOCH - 6 * 60
    close = 2000.0 + np.arange(n, dtype=np.float64)
    oos_feature = tmp_path / "fresh_snapshot.parquet"
    pl.DataFrame(
        {
            "time_epoch": first_epoch + np.arange(n, dtype=np.int64) * 60,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "spread": np.zeros(n, dtype=np.int64),
            "atr_14_lag1": np.ones(n, dtype=np.float64),
            "broker_date": ["2026-09-16"] * 6 + ["2026-09-17"] * 12,
        }
    ).write_parquet(oos_feature)
    oos_feature.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "digits": 2,
                "point": 0.01,
                "trade_contract_size": 100.0,
                "volume_min": 0.01,
                "volume_step": 0.01,
                "volume_max": 0.10,
                "server_timezone": "UTC",
            }
        ),
        encoding="utf-8",
    )
    return {
        "catalog": catalog,
        "canonical_feature": canonical_feature,
        "shortlist": shortlist,
        "plan": plan,
        "oos_feature": oos_feature,
    }


def test_run_oos_snapshot_writes_only_frozen_candidate_and_is_idempotent(tmp_path: Path):
    paths = _write_fixture(tmp_path)
    output_root = tmp_path / "oos_results"

    first = run_oos_snapshot(
        shortlist_config=paths["shortlist"],
        holdout_plan=paths["plan"],
        catalog_path=paths["catalog"],
        canonical_feature_path=paths["canonical_feature"],
        oos_feature_path=paths["oos_feature"],
        output_root=output_root,
    )
    second = run_oos_snapshot(
        shortlist_config=paths["shortlist"],
        holdout_plan=paths["plan"],
        catalog_path=paths["catalog"],
        canonical_feature_path=paths["canonical_feature"],
        oos_feature_path=paths["oos_feature"],
        output_root=output_root,
    )

    assert first["candidate_count"] == 1
    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    with (output_root / "OOS_RESULTS.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP_SYNTH"
    assert rows[0]["tier"] == "primary"
    assert rows[0]["holdout_name"] == "test_holdout"
    assert rows[0]["oos_start_epoch"] == str(START_EPOCH)
    assert rows[0]["timestamp_semantics"] == "broker_local_to_utc"
    assert rows[0]["snapshot_feature_sha256"] == _sha256(paths["oos_feature"])


def test_run_oos_snapshot_refuses_to_reuse_canonical_feature_artifact(tmp_path: Path):
    shared = tmp_path / "same.parquet"
    shared.write_bytes(b"same")

    with pytest.raises(ValueError, match="separate snapshot feature artifact"):
        run_oos_snapshot(
            shortlist_config=tmp_path / "shortlist.json",
            holdout_plan=tmp_path / "plan.json",
            catalog_path=tmp_path / "catalog.csv",
            canonical_feature_path=shared,
            oos_feature_path=shared,
            output_root=tmp_path / "out",
        )
