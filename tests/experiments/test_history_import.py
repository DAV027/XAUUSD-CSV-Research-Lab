from __future__ import annotations

import json
import zipfile
from pathlib import Path

import polars as pl

from xau_lab.experiments.history import (
    HISTORICAL_COLUMNS,
    historical_matches,
    import_historical_checkpoint,
)


def _write_member(zf: zipfile.ZipFile, name: str, payload: object) -> None:
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, sort_keys=True)
    zf.writestr(name, text)


def test_imports_matching_historical_evidence_without_inventing_missing_fields(tmp_path: Path):
    archive = tmp_path / "outputs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        _write_member(
            zf,
            "outputs/experiment_ledger.jsonl",
            json.dumps(
                {
                    "historical_id": "LAB001R",
                    "strategy_family": "trend",
                    "strategy_name": "SMA Slope",
                    "status": "rejected",
                }
            )
            + "\n",
        )
        _write_member(
            zf,
            "outputs/LAB001R_results.json",
            {"net_profit": -12.5, "profit_factor": 0.82, "trades": 17},
        )
        _write_member(
            zf,
            "outputs/LAB001R_manifest.json",
            {"source_file": "ReportTester.xlsx", "build": 6140},
        )

    result_root = tmp_path / "results"
    frame, manifest = import_historical_checkpoint(archive, result_root)

    assert frame.columns == HISTORICAL_COLUMNS
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["origin"] == "historical_mt5"
    assert row["historical_id"] == "LAB001R"
    assert row["net_profit"] == -12.5
    assert row["profit_factor"] == 0.82
    assert row["after_cost_profit"] is None
    assert row["after_cost_pf"] is None
    assert row["trades"] == 17
    assert row["manifest_file"] == "outputs/LAB001R_manifest.json"
    assert row["results_file"] == "outputs/LAB001R_results.json"

    assert (result_root / "HISTORICAL_EVIDENCE.csv").exists()
    saved_manifest = json.loads((result_root / "HISTORICAL_IMPORT_MANIFEST.json").read_text())
    assert saved_manifest == manifest
    assert saved_manifest["source_sha256"]
    assert saved_manifest["members"] == sorted(saved_manifest["members"])


def test_historical_matches_normalizes_labels_and_compares_shared_parameters_only():
    evidence = pl.DataFrame(
        {
            "historical_id": ["LAB001R", "LAB002R", "LAB003R"],
            "strategy_name": ["SMA Slope", "sma-slope", "EMA slope"],
            "lookback": [20, 50, 20],
        }
    )

    matches = historical_matches("sma_slope", {"lookback": 20, "threshold": 0.1}, evidence)
    assert matches == ["LAB001R"]
