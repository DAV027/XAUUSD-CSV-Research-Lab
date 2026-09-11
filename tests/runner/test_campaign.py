from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

from xau_lab.experiments.sampler import generate_catalog
from xau_lab.io.checkpoint import CheckpointStore
from xau_lab.io.results import ResultStore
from xau_lab.runner.campaign import run_campaign


def _write_market(tmp_path: Path, rows: int = 240) -> Path:
    epoch = np.arange(1_700_000_000, 1_700_000_000 + rows * 60, 60, dtype=np.int64)
    close = 2000.0 + np.sin(np.arange(rows) / 7.0) * 3.0 + np.arange(rows) * 0.01
    open_ = close - 0.05
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    session_london = np.zeros(rows, dtype=bool)
    session_london[30:90] = True
    session_new_york = np.zeros(rows, dtype=bool)
    session_new_york[80:150] = True
    session_asia = ~(session_london | session_new_york)

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
            "session_asia": session_asia,
            "session_london": session_london,
            "session_new_york": session_new_york,
            "session_overlap": session_london & session_new_york,
        }
    )
    feature_path = tmp_path / "market.parquet"
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
        )
    )
    return feature_path


def _write_catalog(path: Path, count: int = 20) -> list[str]:
    catalog = generate_catalog(total_budget=count, seed=9_215_000)
    rows = [experiment.to_dict() for experiment in catalog]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [experiment.experiment_id for experiment in catalog]


def _normalized_master(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        row.pop("runtime_seconds", None)
        normalized.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return sorted(normalized)


def test_campaign_is_deterministic_across_worker_counts_and_resume(tmp_path: Path):
    feature_path = _write_market(tmp_path)
    catalog_path = tmp_path / "EXPERIMENT_CATALOG.csv"
    catalog_ids = _write_catalog(catalog_path, count=20)

    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    run_campaign(catalog_path, feature_path, root_one, workers=1)
    run_campaign(catalog_path, feature_path, root_two, workers=2)

    assert _normalized_master(root_one / "MASTER_RESULTS.csv") == _normalized_master(
        root_two / "MASTER_RESULTS.csv"
    )

    with (root_one / "MASTER_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        complete_rows = list(csv.DictReader(handle))

    resume_root = tmp_path / "resume"
    checkpoint = CheckpointStore(tmp_path / "resume-state" / "CHECKPOINT.json")
    seed_store = ResultStore(resume_root, catalog_ids=catalog_ids, checkpoint=checkpoint)
    for row in complete_rows[:10]:
        seed_store.append_result(row)
    checkpoint.write(seed_store.completed_ids)

    run_campaign(catalog_path, feature_path, resume_root, workers=2)
    with (resume_root / "MASTER_RESULTS.csv").open(newline="", encoding="utf-8") as handle:
        resumed_rows = list(csv.DictReader(handle))
    ids = [row["experiment_id"] for row in resumed_rows]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert set(ids) == set(catalog_ids)
