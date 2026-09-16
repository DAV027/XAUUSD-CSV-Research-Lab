import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_oos_freeze import validate_oos_freeze


def _write_catalog(path: Path, row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    row = {
        "experiment_id": "EXPTEST00000001",
        "fingerprint": "a" * 64,
        "allocation_bucket": "statistical",
        "family": "statistical",
        "strategy_name": "return_reversal",
        "canonical_parameters_json": '{"lookback":37,"threshold_pct":1.25}',
        "direction_mode": "combined",
        "stop_atr": "0.5",
        "exit_type": "atr_trail",
        "target_r": "",
        "time_exit_minutes": "",
        "atr_trail": "0.5",
        "commission_round_trip_per_lot": "6.0",
        "slippage_points_per_fill": "5.0",
        "strategy_seed": "123",
        "family_seed": "456",
        "sampler_version": "v1",
    }
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, row)
    features = tmp_path / "features.parquet"
    features.write_bytes(b"feature-fixture")
    config = {
        "schema_version": 1,
        "name": "test_freeze",
        "expected_candidate_count": 1,
        "selection_source": {
            "campaign_commit": "b" * 40,
            "feature_sha256": _sha256(features),
            "catalog_sha256": _sha256(catalog),
            "campaign_data_end_epoch": 1789178340,
            "campaign_data_end_utc": "2026-09-12T01:59:00Z",
            "oos_start_exclusive_epoch": 1789178340,
            "oos_rule": "time_epoch > oos_start_exclusive_epoch",
        },
        "candidates": [{"tier": "primary", "experiment": row.copy()}],
    }
    config_path = tmp_path / "freeze.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, catalog, features, config


def test_valid_freeze_matches_catalog_and_hashes(tmp_path):
    config, catalog, features, _ = _fixture(tmp_path)
    result = validate_oos_freeze(config, catalog, features)
    assert result == {"candidate_count": 1, "primary_count": 1, "diagnostic_count": 0}


def test_candidate_parameter_drift_is_rejected(tmp_path):
    config_path, catalog, features, config = _fixture(tmp_path)
    config["candidates"][0]["experiment"]["stop_atr"] = "0.75"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate drift"):
        validate_oos_freeze(config_path, catalog, features)


def test_catalog_hash_drift_is_rejected(tmp_path):
    config_path, catalog, features, _ = _fixture(tmp_path)
    with catalog.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="catalog SHA256 mismatch"):
        validate_oos_freeze(config_path, catalog, features)


def test_feature_hash_drift_is_rejected(tmp_path):
    config_path, catalog, features, _ = _fixture(tmp_path)
    features.write_bytes(b"changed")
    with pytest.raises(ValueError, match="feature SHA256 mismatch"):
        validate_oos_freeze(config_path, catalog, features)


def test_duplicate_candidate_ids_are_rejected(tmp_path):
    config_path, catalog, features, config = _fixture(tmp_path)
    config["expected_candidate_count"] = 2
    config["candidates"].append(config["candidates"][0].copy())
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frozen experiment_id"):
        validate_oos_freeze(config_path, catalog, features)


def test_repository_freeze_contains_exact_selected_ids():
    path = Path(__file__).resolve().parents[2] / "config" / "oos_shortlist_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "EXPFEA5F02ABA9D",
        "EXP5DCA824C5E9B",
        "EXP9BC15A17FD42",
        "EXP26056C1AB093",
        "EXP11A4F3EC0EBF",
        "EXP976C07ABB608",
        "EXP8AA07E518720",
        "EXPBD27970D311B",
        "EXP54AD9954C65C",
        "EXP40D1331D4A0C",
    }
    actual = {item["experiment"]["experiment_id"] for item in config["candidates"]}
    assert actual == expected
    assert config["expected_candidate_count"] == 10
    assert sum(item["tier"] == "primary" for item in config["candidates"]) == 6
    assert sum(item["tier"] == "diagnostic" for item in config["candidates"]) == 4
    assert config["selection_source"]["oos_start_exclusive_epoch"] == 1789178340
