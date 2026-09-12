from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from xau_lab.runner.campaign import run_campaign
from xau_lab.runner.manifest import (
    build_run_manifest,
    resolved_workers,
    write_or_validate_run_manifest,
)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "experiment_id" not in reader.fieldnames:
            raise ValueError("MASTER_RESULTS.csv is missing experiment_id")
        return {
            experiment_id
            for row in reader
            if (experiment_id := (row.get("experiment_id") or "").strip())
        }


def _bucket_quotas(bucket_sizes: dict[str, int], count: int) -> dict[str, int]:
    total = sum(bucket_sizes.values())
    if count <= 0:
        raise ValueError("smoke count must be positive")
    if count > total:
        raise ValueError(f"smoke count {count} exceeds catalog size {total}")

    bucket_order = list(bucket_sizes)
    raw = {bucket: count * size / total for bucket, size in bucket_sizes.items()}
    quotas = {bucket: int(raw[bucket]) for bucket in bucket_order}
    remaining = count - sum(quotas.values())
    ranked = sorted(
        bucket_order,
        key=lambda bucket: (-(raw[bucket] - quotas[bucket]), bucket_order.index(bucket)),
    )
    for bucket in ranked:
        if remaining == 0:
            break
        if quotas[bucket] < bucket_sizes[bucket]:
            quotas[bucket] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("could not allocate exact smoke quotas")

    # When the smoke count is at least the number of non-empty buckets, make the
    # data-compatibility gate exercise every bucket while staying deterministic.
    if count >= len(bucket_order):
        for bucket in bucket_order:
            if quotas[bucket] != 0:
                continue
            donors = [candidate for candidate in bucket_order if quotas[candidate] > 1]
            if not donors:
                raise RuntimeError("could not preserve smoke bucket coverage")
            donor = max(
                donors,
                key=lambda candidate: (
                    quotas[candidate] - raw[candidate],
                    quotas[candidate],
                    -bucket_order.index(candidate),
                ),
            )
            quotas[donor] -= 1
            quotas[bucket] += 1

    return quotas


def select_stratified_smoke_ids(catalog_path: Path, count: int) -> list[str]:
    with Path(catalog_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("experiment catalog is empty")

    required = {"experiment_id", "allocation_bucket"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"experiment catalog missing required columns: {sorted(missing)}")

    bucket_sizes: dict[str, int] = {}
    seen_ids: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for row in rows:
        experiment_id = (row.get("experiment_id") or "").strip()
        bucket = (row.get("allocation_bucket") or "").strip()
        if not experiment_id or not bucket:
            raise ValueError("experiment catalog requires non-empty experiment_id and allocation_bucket")
        if experiment_id in seen_ids:
            raise ValueError(f"experiment catalog contains duplicate experiment_id: {experiment_id}")
        seen_ids.add(experiment_id)
        normalized.append((experiment_id, bucket))
        bucket_sizes[bucket] = bucket_sizes.get(bucket, 0) + 1

    quotas = _bucket_quotas(bucket_sizes, count)
    selected: list[str] = []
    selected_by_bucket: Counter[str] = Counter()
    for experiment_id, bucket in normalized:
        if selected_by_bucket[bucket] >= quotas[bucket]:
            continue
        selected.append(experiment_id)
        selected_by_bucket[bucket] += 1

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} smoke experiments; expected {count}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic bounded XAUUSD smoke campaign")
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=9_215_000)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    workers = resolved_workers(args.workers)
    manifest = build_run_manifest(
        feature_path=args.features,
        catalog_path=args.catalog,
        workers=workers,
        campaign_seed=args.seed,
        repo_root=Path(__file__).resolve().parents[1],
    )
    write_or_validate_run_manifest(args.result_root, manifest)

    selected_ids = select_stratified_smoke_ids(args.catalog, args.count)
    selected_set = set(selected_ids)
    master_path = args.result_root / "MASTER_RESULTS.csv"
    before_ids = _completed_ids(master_path)
    before_selected = len(before_ids.intersection(selected_set))

    started = time.perf_counter()
    run_campaign(
        args.catalog,
        args.features,
        args.result_root,
        workers=workers,
        experiment_ids=selected_ids,
    )
    elapsed = time.perf_counter() - started

    after_ids = _completed_ids(master_path)
    after_selected = len(after_ids.intersection(selected_set))
    completed_now = after_selected - before_selected

    if after_selected != args.count:
        raise RuntimeError(
            f"smoke target has {after_selected} completed selected experiments; expected {args.count}. "
            "Inspect ERRORS.csv before continuing."
        )

    print(
        f"Smoke target complete: {after_selected} selected experiments; "
        f"executed this invocation: {completed_now}; total stored: {len(after_ids)}"
    )

    if args.count == 100 and completed_now > 0:
        per_experiment = elapsed / completed_now
        projected = per_experiment * 50_000
        benchmark = {
            "completed_experiments": completed_now,
            "smoke_target_experiments": args.count,
            "preexisting_selected_experiments": before_selected,
            "elapsed_seconds": elapsed,
            "seconds_per_experiment": per_experiment,
            "projected_50000_seconds": projected,
            "projected_50000_hours": projected / 3600.0,
            "requires_profiling_before_full_run": projected > 24 * 3600,
        }
        (args.result_root / "THROUGHPUT_BENCHMARK.json").write_text(
            json.dumps(benchmark, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if benchmark["requires_profiling_before_full_run"]:
            print("Projected 50k runtime exceeds 24 hours; profile before launching the full campaign.")


if __name__ == "__main__":
    main()
