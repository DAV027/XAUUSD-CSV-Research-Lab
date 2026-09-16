from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.run_post_selection_diagnostic import run_diagnostic_snapshot
from xau_lab.experiments.spec import canonical_json

DIAG_START = 1_789_257_600  # 2026-09-13T00:00:00Z
DIAG_END = 1_789_603_200    # 2026-09-17T00:00:00Z
SELECTION_END = 1_789_178_340


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    params = {"lookback": 2, "threshold_pct": 0.02}
    row = {
        "experiment_id": "EXP_DIAG",
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

    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"canonical-selection")
    shortlist = tmp_path / "shortlist.json"
    shortlist.write_text(json.dumps({
        "schema_version": 1,
        "expected_candidate_count": 1,
        "selection_source": {
            "campaign_commit": "a" * 40,
            "feature_sha256": _sha256(canonical),
            "catalog_sha256": _sha256(catalog),
            "campaign_data_end_epoch": SELECTION_END,
            "campaign_data_end_utc": "2026-09-12T01:59:00Z",
            "oos_start_exclusive_epoch": SELECTION_END,
            "oos_rule": "time_epoch > oos_start_exclusive_epoch",
        },
        "candidates": [{"tier": "primary", "experiment": row}],
    }), encoding="utf-8")

    oos_plan = tmp_path / "oos_plan.json"
    oos_plan.write_text(json.dumps({
        "schema_version": 1,
        "name": "prospective",
        "status": "frozen",
        "shortlist_config": str(shortlist),
        "freeze_timestamp": "2026-09-16T06:46:32Z",
        "selection_data_end_epoch": SELECTION_END,
        "prospective_start_utc": "2026-09-17T00:00:00Z",
        "prospective_start_epoch": DIAG_END,
        "planned_months": 3,
        "planned_end_exclusive_utc": "2026-12-17T00:00:00Z",
        "planned_end_exclusive_epoch": 1_797_465_600,
    }), encoding="utf-8")

    diagnostic_plan = tmp_path / "diagnostic.json"
    diagnostic_plan.write_text(json.dumps({
        "schema_version": 1,
        "name": "post_selection_diagnostic_2026_09_13_16",
        "classification": "post_selection_pre_freeze_diagnostic",
        "shortlist_config": str(shortlist),
        "selection_data_end_epoch": SELECTION_END,
        "diagnostic_start_utc": "2026-09-13T00:00:00Z",
        "diagnostic_start_epoch": DIAG_START,
        "diagnostic_end_exclusive_utc": "2026-09-17T00:00:00Z",
        "diagnostic_end_exclusive_epoch": DIAG_END,
        "is_pristine_prospective_oos": False,
    }), encoding="utf-8")

    n = 20
    first = DIAG_START - 5 * 60
    close = 2000.0 + np.arange(n, dtype=np.float64)
    feature = tmp_path / "diagnostic_snapshot.parquet"
    pl.DataFrame({
        "time_epoch": first + np.arange(n, dtype=np.int64) * 60,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "spread": np.zeros(n, dtype=np.int64),
        "atr_14_lag1": np.ones(n, dtype=np.float64),
        "broker_date": ["2026-09-12"] * 5 + ["2026-09-13"] * 15,
    }).write_parquet(feature)
    feature.with_suffix(".metadata.json").write_text(json.dumps({
        "digits": 2,
        "point": 0.01,
        "trade_contract_size": 100.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 0.10,
        "server_timezone": "UTC",
    }), encoding="utf-8")

    return {
        "catalog": catalog,
        "canonical": canonical,
        "shortlist": shortlist,
        "oos_plan": oos_plan,
        "diagnostic_plan": diagnostic_plan,
        "feature": feature,
    }


def test_diagnostic_snapshot_is_separate_labeled_and_idempotent(tmp_path: Path):
    p = _fixture(tmp_path)
    out = tmp_path / "results"
    first = run_diagnostic_snapshot(
        shortlist_config=p["shortlist"],
        oos_holdout_plan=p["oos_plan"],
        diagnostic_plan=p["diagnostic_plan"],
        catalog_path=p["catalog"],
        canonical_feature_path=p["canonical"],
        diagnostic_feature_path=p["feature"],
        output_root=out,
    )
    second = run_diagnostic_snapshot(
        shortlist_config=p["shortlist"],
        oos_holdout_plan=p["oos_plan"],
        diagnostic_plan=p["diagnostic_plan"],
        catalog_path=p["catalog"],
        canonical_feature_path=p["canonical"],
        diagnostic_feature_path=p["feature"],
        output_root=out,
    )

    assert first["candidate_count"] == 1
    assert first["appended_rows"] == 1
    assert second["appended_rows"] == 0
    assert not (out / "OOS_RESULTS.csv").exists()
    with (out / "DIAGNOSTIC_RESULTS.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["evaluation_kind"] == "post_selection_pre_freeze_diagnostic"
    assert row["is_pristine_prospective_oos"] == "False"
    assert row["diagnostic_start_epoch"] == str(DIAG_START)
    assert row["diagnostic_end_exclusive_epoch"] == str(DIAG_END)
    assert "oos_start_epoch" not in row
    assert row["snapshot_feature_sha256"] == _sha256(p["feature"])


def test_diagnostic_plan_must_end_at_or_before_true_oos_start(tmp_path: Path):
    p = _fixture(tmp_path)
    payload = json.loads(p["diagnostic_plan"].read_text(encoding="utf-8"))
    payload["diagnostic_end_exclusive_epoch"] = DIAG_END + 60
    payload["diagnostic_end_exclusive_utc"] = "2026-09-17T00:01:00Z"
    p["diagnostic_plan"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap the prospective OOS window"):
        run_diagnostic_snapshot(
            shortlist_config=p["shortlist"],
            oos_holdout_plan=p["oos_plan"],
            diagnostic_plan=p["diagnostic_plan"],
            catalog_path=p["catalog"],
            canonical_feature_path=p["canonical"],
            diagnostic_feature_path=p["feature"],
            output_root=tmp_path / "out",
        )
