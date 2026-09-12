from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path


BUCKET_SIZES = {
    "trend_momentum": 160,
    "breakout": 160,
    "mean_reversion": 160,
    "price_action": 140,
    "session": 100,
    "volatility": 100,
    "statistical": 100,
    "exit_execution": 80,
}
EXPECTED_100 = {
    "trend_momentum": 16,
    "breakout": 16,
    "mean_reversion": 16,
    "price_action": 14,
    "session": 10,
    "volatility": 10,
    "statistical": 10,
    "exit_execution": 8,
}


def _load_run_smoke_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_smoke.py"
    spec = importlib.util.spec_from_file_location("run_smoke_selection_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_grouped_catalog(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for bucket, size in BUCKET_SIZES.items():
        for index in range(size):
            experiment_id = f"{bucket}-{index:04d}"
            rows.append({"experiment_id": experiment_id, "allocation_bucket": bucket})
            lookup[experiment_id] = bucket
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "allocation_bucket"])
        writer.writeheader()
        writer.writerows(rows)
    return lookup


def test_stratified_smoke_selection_matches_full_catalog_allocation(tmp_path: Path):
    module = _load_run_smoke_module()
    catalog_path = tmp_path / "EXPERIMENT_CATALOG.csv"
    lookup = _write_grouped_catalog(catalog_path)

    selected = module.select_stratified_smoke_ids(catalog_path, 100)

    assert len(selected) == 100
    assert len(set(selected)) == 100
    assert set(selected).issubset(lookup)
    assert Counter(lookup[experiment_id] for experiment_id in selected) == Counter(EXPECTED_100)
    assert selected == module.select_stratified_smoke_ids(catalog_path, 100)


def test_stratified_smoke_selection_covers_every_bucket_when_count_allows(tmp_path: Path):
    module = _load_run_smoke_module()
    catalog_path = tmp_path / "EXPERIMENT_CATALOG.csv"
    lookup = _write_grouped_catalog(catalog_path)

    selected = module.select_stratified_smoke_ids(catalog_path, len(BUCKET_SIZES))

    assert len(selected) == len(BUCKET_SIZES)
    assert set(lookup[experiment_id] for experiment_id in selected) == set(BUCKET_SIZES)
