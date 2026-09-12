from __future__ import annotations

import json

import pytest

from xau_lab.runner.manifest import (
    sha256_file,
    verify_manifest_artifacts,
    write_or_validate_run_manifest,
)


def _manifest(feature_path, catalog_path, **overrides):
    base = {
        "source_data_path": str(feature_path),
        "source_data_sha256": sha256_file(feature_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "software_git_commit": "abc123",
        "workers": 2,
        "cost_model": {
            "commission_round_trip_per_lot": 6.0,
            "slippage_points_per_fill": 5.0,
        },
        "risk_model": {
            "account_equity": 5000.0,
            "preferred_risk_usd": 10.0,
            "hard_risk_usd": 10.0,
            "max_lot": 0.1,
        },
        "campaign_seed": 9_215_000,
        "start_timestamp": "2026-09-12T06:00:00+00:00",
    }
    return base | overrides


def test_resume_rejects_changed_source_catalog_code_or_model_but_allows_worker_change(tmp_path):
    feature_path = tmp_path / "features.parquet"
    catalog_path = tmp_path / "catalog.csv"
    feature_path.write_bytes(b"feature-v1")
    catalog_path.write_text("catalog-v1\n", encoding="utf-8")
    root = tmp_path / "results"

    original = _manifest(feature_path, catalog_path)
    write_or_validate_run_manifest(root, original)

    compatible = original | {
        "workers": 8,
        "start_timestamp": "2026-09-12T07:00:00+00:00",
    }
    path = write_or_validate_run_manifest(root, compatible)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["workers"] == 2
    assert persisted["start_timestamp"] == "2026-09-12T06:00:00+00:00"

    cases = [
        {"source_data_sha256": "1" * 64},
        {"catalog_sha256": "2" * 64},
        {"software_git_commit": "different"},
        {"campaign_seed": 123},
        {"cost_model": original["cost_model"] | {"slippage_points_per_fill": 10.0}},
        {"risk_model": original["risk_model"] | {"hard_risk_usd": 20.0}},
    ]
    for override in cases:
        with pytest.raises(ValueError, match="resume provenance mismatch"):
            write_or_validate_run_manifest(root, original | override)


def test_validation_rejects_feature_or_catalog_bytes_that_do_not_match_run_manifest(tmp_path):
    feature_path = tmp_path / "features.parquet"
    catalog_path = tmp_path / "catalog.csv"
    feature_path.write_bytes(b"feature-v1")
    catalog_path.write_text("catalog-v1\n", encoding="utf-8")
    root = tmp_path / "results"
    manifest = _manifest(feature_path, catalog_path)
    manifest_path = write_or_validate_run_manifest(root, manifest)

    verified = verify_manifest_artifacts(manifest_path, feature_path, catalog_path)
    assert verified["source_data_sha256"] == sha256_file(feature_path)
    assert verified["catalog_sha256"] == sha256_file(catalog_path)

    feature_path.write_bytes(b"feature-v2")
    with pytest.raises(ValueError, match="source_data_sha256"):
        verify_manifest_artifacts(manifest_path, feature_path, catalog_path)

    feature_path.write_bytes(b"feature-v1")
    catalog_path.write_text("catalog-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="catalog_sha256"):
        verify_manifest_artifacts(manifest_path, feature_path, catalog_path)
