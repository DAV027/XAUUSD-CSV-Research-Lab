from __future__ import annotations

import json
from pathlib import Path

import pytest

from xau_lab.io.checkpoint import CheckpointStore
from xau_lab.io.results import ResultStore


def test_resume_reconstructs_completed_ids_from_master_results_and_rejects_duplicates(tmp_path: Path):
    catalog_ids = [f"EXP{i}" for i in range(1, 11)]
    checkpoint = CheckpointStore(tmp_path / "state" / "CHECKPOINT.json")
    store = ResultStore(tmp_path / "results", catalog_ids=catalog_ids, checkpoint=checkpoint)

    for index, experiment_id in enumerate(catalog_ids[:4], start=1):
        store.append_result({"experiment_id": experiment_id, "after_cost_profit": float(index)})
    checkpoint.write(store.completed_ids)

    assert json.loads(checkpoint.path.read_text())["completed_ids"] == catalog_ids[:4]

    reconstructed_checkpoint = CheckpointStore(tmp_path / "state" / "CHECKPOINT.json")
    reconstructed = ResultStore(
        tmp_path / "results",
        catalog_ids=catalog_ids,
        checkpoint=reconstructed_checkpoint,
    )
    assert reconstructed.pending_ids() == catalog_ids[4:]

    with pytest.raises(ValueError, match="duplicate experiment_id"):
        reconstructed.append_result({"experiment_id": "EXP4", "after_cost_profit": 99.0})


def test_checkpoint_loss_does_not_reenable_completed_experiments(tmp_path: Path):
    catalog_ids = ["EXP1", "EXP2", "EXP3"]
    checkpoint_path = tmp_path / "state" / "CHECKPOINT.json"
    checkpoint = CheckpointStore(checkpoint_path)
    store = ResultStore(tmp_path / "results", catalog_ids=catalog_ids, checkpoint=checkpoint)
    store.append_result({"experiment_id": "EXP1", "after_cost_profit": 1.0})
    checkpoint.write(store.completed_ids)
    checkpoint_path.unlink()

    recovered = ResultStore(
        tmp_path / "results",
        catalog_ids=catalog_ids,
        checkpoint=CheckpointStore(checkpoint_path),
    )
    assert recovered.pending_ids() == ["EXP2", "EXP3"]


def test_error_row_writes_metadata_and_full_traceback_file(tmp_path: Path):
    store = ResultStore(tmp_path / "results", catalog_ids=["EXP1"])
    store.append_error(
        experiment_id="EXP1",
        exception_type="RuntimeError",
        message="boom",
        traceback_text="Traceback full text",
    )

    error_csv = (tmp_path / "results" / "ERRORS.csv").read_text()
    assert "EXP1" in error_csv
    assert "RuntimeError" in error_csv
    trace_path = tmp_path / "results" / "errors" / "EXP1.txt"
    assert trace_path.read_text() == "Traceback full text"
