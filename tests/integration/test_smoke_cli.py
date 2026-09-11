from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_market(tmp_path: Path, rows: int = 240) -> Path:
    epoch = np.arange(1_700_000_000, 1_700_000_000 + rows * 60, 60, dtype=np.int64)
    close = 2000.0 + np.sin(np.arange(rows) / 7.0) * 3.0 + np.arange(rows) * 0.01
    open_ = close - 0.05
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    london = np.zeros(rows, dtype=bool)
    london[30:90] = True
    new_york = np.zeros(rows, dtype=bool)
    new_york[80:150] = True
    asia = ~(london | new_york)
    frame = pl.DataFrame(
        {
            "time_epoch": epoch,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "spread": np.full(rows, 35, dtype=np.int64),
            "atr_14_lag1": np.full(rows, 1.25),
            "broker_date": ["2026-01-02"] * rows,
            "session_asia": asia,
            "session_london": london,
            "session_new_york": new_york,
            "session_overlap": london & new_york,
        }
    )
    feature_path = tmp_path / "XAUUSD_M1_FEATURES.parquet"
    frame.write_parquet(feature_path)
    feature_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "digits": 2,
                "point": 0.01,
                "trade_contract_size": 100.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "server_timezone": "UTC",
            }
        ),
        encoding="utf-8",
    )
    return feature_path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_create_catalog_and_run_smoke_produce_exact_unique_count_and_manifest(tmp_path: Path):
    feature_path = _write_market(tmp_path)
    result_root = tmp_path / "results"
    catalog_path = result_root / "EXPERIMENT_CATALOG.csv"

    _run(
        "scripts/create_catalog.py",
        "--budget",
        "20",
        "--seed",
        "9215000",
        "--output",
        str(catalog_path),
    )
    smoke = _run(
        "scripts/run_smoke.py",
        "--catalog",
        str(catalog_path),
        "--features",
        str(feature_path),
        "--result-root",
        str(result_root),
        "--count",
        "10",
        "--workers",
        "1",
        "--seed",
        "9215000",
    )
    assert "10" in smoke.stdout

    with (result_root / "MASTER_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["experiment_id"] for row in rows]
    assert len(ids) == 10
    assert len(set(ids)) == 10

    manifest = json.loads((result_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source_data_path"] == str(feature_path.resolve())
    assert len(manifest["source_data_sha256"]) == 64
    assert len(manifest["catalog_sha256"]) == 64
    assert manifest["workers"] == 1
    assert manifest["campaign_seed"] == 9215000
    assert manifest["cost_model"]["commission_round_trip_per_lot"] == 6.0
    assert manifest["risk_model"]["account_equity"] == 5000.0
    assert manifest["start_timestamp"]
