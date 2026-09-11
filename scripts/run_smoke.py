from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from xau_lab.runner.campaign import run_campaign
from xau_lab.runner.manifest import build_run_manifest, resolved_workers, write_run_manifest


def _completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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
    write_run_manifest(args.result_root, manifest)

    before = _completed_count(args.result_root / "MASTER_RESULTS.csv")
    started = time.perf_counter()
    run_campaign(args.catalog, args.features, args.result_root, workers=workers, limit=args.count)
    elapsed = time.perf_counter() - started
    after = _completed_count(args.result_root / "MASTER_RESULTS.csv")
    completed_now = after - before

    if completed_now != args.count:
        raise RuntimeError(
            f"smoke campaign completed {completed_now} experiments; expected exactly {args.count}. "
            "Inspect ERRORS.csv before continuing."
        )

    print(f"Completed {completed_now} smoke experiments; total stored: {after}")

    if args.count == 100:
        per_experiment = elapsed / completed_now if completed_now else None
        projected = per_experiment * 50_000 if per_experiment is not None else None
        benchmark = {
            "completed_experiments": completed_now,
            "elapsed_seconds": elapsed,
            "seconds_per_experiment": per_experiment,
            "projected_50000_seconds": projected,
            "projected_50000_hours": (projected / 3600.0) if projected is not None else None,
            "requires_profiling_before_full_run": bool(projected is not None and projected > 24 * 3600),
        }
        (args.result_root / "THROUGHPUT_BENCHMARK.json").write_text(
            json.dumps(benchmark, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if benchmark["requires_profiling_before_full_run"]:
            print("Projected 50k runtime exceeds 24 hours; profile before launching the full campaign.")


if __name__ == "__main__":
    main()
