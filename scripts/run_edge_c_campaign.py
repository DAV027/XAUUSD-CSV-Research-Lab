from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_campaign import _guard_slow_campaign
from xau_lab.experiments.edge_c_sampler import EDGE_C_DEFAULT_SEED
from xau_lab.runner.campaign import run_campaign
from xau_lab.runner.manifest import (
    build_run_manifest,
    resolved_workers,
    write_or_validate_run_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume isolated Edge C compression-breakout-retest research"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("edge_c/results/EXPERIMENT_CATALOG.csv"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument("--result-root", type=Path, default=Path("edge_c/results"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=EDGE_C_DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-slow", action="store_true")
    return parser


def run_edge_c_campaign(
    *,
    catalog_path: Path,
    feature_path: Path,
    result_root: Path,
    workers: int,
    seed: int,
    limit: int | None,
    allow_slow: bool,
) -> None:
    workers = resolved_workers(workers)
    _guard_slow_campaign(result_root, allow_slow)
    manifest = build_run_manifest(
        feature_path=feature_path,
        catalog_path=catalog_path,
        workers=workers,
        campaign_seed=seed,
        repo_root=Path(__file__).resolve().parents[1],
    )
    write_or_validate_run_manifest(result_root, manifest)
    run_campaign(
        catalog_path,
        feature_path,
        result_root,
        workers=workers,
        limit=limit,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_edge_c_campaign(
        catalog_path=args.catalog,
        feature_path=args.features,
        result_root=args.result_root,
        workers=args.workers,
        seed=args.seed,
        limit=args.limit,
        allow_slow=args.allow_slow,
    )
    print("Edge C campaign run completed or resumed successfully")


if __name__ == "__main__":
    main()
