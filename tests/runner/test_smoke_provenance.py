from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from xau_lab.runner.manifest import build_run_manifest, write_run_manifest


def _load_run_smoke_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_resume_rejects_manifest_provenance_change_before_campaign_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_run_smoke_module()
    feature_path = tmp_path / "features.parquet"
    catalog_path = tmp_path / "catalog.csv"
    result_root = tmp_path / "results"
    feature_path.write_bytes(b"feature-v1")
    catalog_path.write_text("catalog-v1\n", encoding="utf-8")

    original = build_run_manifest(
        feature_path=feature_path,
        catalog_path=catalog_path,
        workers=1,
        campaign_seed=9_215_000,
        repo_root=Path(__file__).resolve().parents[2],
    )
    write_run_manifest(result_root, original)

    def campaign_must_not_run(*args, **kwargs):
        raise AssertionError("campaign ran despite a smoke-resume provenance mismatch")

    monkeypatch.setattr(module, "run_campaign", campaign_must_not_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_smoke.py",
            "--features",
            str(feature_path),
            "--catalog",
            str(catalog_path),
            "--result-root",
            str(result_root),
            "--count",
            "1",
            "--workers",
            "1",
            "--seed",
            "123",
        ],
    )

    with pytest.raises(ValueError, match="resume provenance mismatch"):
        module.main()
