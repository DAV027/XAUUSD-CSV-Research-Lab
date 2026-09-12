from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from xau_lab.backtest.models import CostModel, RiskModel


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


def write_run_manifest(result_root: str | Path, manifest: dict) -> Path:
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "RUN_MANIFEST.json"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path


__all__ = [
    "build_run_manifest",
    "git_commit",
    "resolved_workers",
    "sha256_file",
    "write_run_manifest",
]
