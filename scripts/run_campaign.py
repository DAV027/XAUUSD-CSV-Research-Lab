from __future__ import annotations

import argparse
import json
from pathlib import Path

from xau_lab.runner.campaign import run_campaign
from xau_lab.runner.manifest import (
    build_run_manifest,
    resolved_workers,
    write_or_validate_run_manifest,
)


def _guard_slow_campaign(result_root: Path, allow_slow: bool) -> None:
    benchmark_path = result_root / "THROUGHPUT_BENCHMARK.json"
    if allow_slow or not benchmark_path.exists():
        return
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark.get("requires_profiling_before_full_run"):
        raise RuntimeError(
            "the 100-experiment benchmark projects more than 24 hours for 50k experiments; "
            "profile the runner first or pass --allow-slow after explicitly accepting that runtime"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the complete XAUUSD research campaign")
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet"))
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=9_215_000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-slow", action="store_true")
    args = parser.parse_args()

    workers = resolved_workers(args.workers)
    _guard_slow_campaign(args.result_root, args.allow_slow)
    manifest = build_run_manifest(
        feature_path=args.features,
        catalog_path=args.catalog,
        workers=workers,
        campaign_seed=args.seed,
        repo_root=Path(__file__).resolve().parents[1],
    )
    write_or_validate_run_manifest(args.result_root, manifest)
    run_campaign(
        args.catalog,
        args.features,
        args.result_root,
        workers=workers,
        limit=args.limit,
    )
    print("Campaign run completed or resumed successfully")


if __name__ == "__main__":
    main()
