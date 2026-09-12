from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from xau_lab.backtest.models import CostModel, RiskModel

RESUME_IMMUTABLE_FIELDS = (
    "source_data_sha256",
    "catalog_sha256",
    "software_git_commit",
    "cost_model",
    "risk_model",
    "campaign_seed",
)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_workers(workers: int | None) -> int:
    value = max(1, (os.cpu_count() or 1) - 1) if workers is None else int(workers)
    if value < 1:
        raise ValueError("workers must be >= 1")
    return value


def git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=None if repo_root is None else Path(repo_root),
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_run_manifest(
    *,
    feature_path: str | Path,
    catalog_path: str | Path,
    workers: int,
    campaign_seed: int,
    repo_root: str | Path | None = None,
) -> dict:
    feature_path = Path(feature_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    cost = CostModel()
    risk = RiskModel()
    return {
        "source_data_path": str(feature_path),
        "source_data_sha256": sha256_file(feature_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "software_git_commit": git_commit(repo_root),
        "workers": int(workers),
        "cost_model": {
            "commission_round_trip_per_lot": cost.commission_round_trip_per_lot,
            "slippage_points_per_fill": cost.slippage_points_per_fill,
        },
        "risk_model": {
            "account_equity": risk.account_equity,
            "preferred_risk_usd": risk.preferred_risk_usd,
            "hard_risk_usd": risk.hard_risk_usd,
            "max_lot": risk.max_lot,
        },
        "campaign_seed": int(campaign_seed),
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"run manifest is required: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run manifest must be a JSON object: {manifest_path}")
    return payload


def assert_resume_compatible(
    existing: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    mismatches = [
        field
        for field in RESUME_IMMUTABLE_FIELDS
        if existing.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(f"resume provenance mismatch: {', '.join(mismatches)}")


def write_run_manifest(result_root: str | Path, manifest: dict) -> Path:
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "RUN_MANIFEST.json"
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def write_or_validate_run_manifest(result_root: str | Path, manifest: dict) -> Path:
    result_root = Path(result_root)
    path = result_root / "RUN_MANIFEST.json"
    if path.exists():
        existing = _load_manifest(path)
        assert_resume_compatible(existing, manifest)
        return path
    return write_run_manifest(result_root, manifest)


def verify_manifest_artifacts(
    manifest_path: str | Path,
    feature_path: str | Path,
    catalog_path: str | Path,
) -> dict:
    manifest = _load_manifest(manifest_path)
    actual_source = sha256_file(feature_path)
    actual_catalog = sha256_file(catalog_path)
    expected_source = manifest.get("source_data_sha256")
    expected_catalog = manifest.get("catalog_sha256")
    if expected_source != actual_source:
        raise ValueError(
            "source_data_sha256 mismatch between run manifest and current feature file"
        )
    if expected_catalog != actual_catalog:
        raise ValueError(
            "catalog_sha256 mismatch between run manifest and current experiment catalog"
        )
    return manifest


__all__ = [
    "RESUME_IMMUTABLE_FIELDS",
    "assert_resume_compatible",
    "build_run_manifest",
    "git_commit",
    "resolved_workers",
    "sha256_file",
    "verify_manifest_artifacts",
    "write_or_validate_run_manifest",
    "write_run_manifest",
]
