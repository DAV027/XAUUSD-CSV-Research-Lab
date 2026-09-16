from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_campaign import _guard_slow_campaign
from xau_lab.experiments.edge_c_v2_sampler import (
    EDGE_C_V2_DEFAULT_BUDGET,
    EDGE_C_V2_DEFAULT_SEED,
    EDGE_C_V2_FAMILY,
    EDGE_C_V2_SAMPLER_VERSION,
    EDGE_C_V2_STRATEGY,
)
from xau_lab.runner.campaign import run_campaign
from xau_lab.runner.manifest import (
    build_run_manifest,
    resolved_workers,
    sha256_file,
    write_or_validate_run_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume Edge C v2 only after the frozen activation gate passes"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("edge_c_v2/results/EXPERIMENT_CATALOG.csv"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument("--result-root", type=Path, default=Path("edge_c_v2/results"))
    parser.add_argument(
        "--activation-report",
        type=Path,
        default=Path("edge_c_v2/results/ACTIVATION_REPORT.json"),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=EDGE_C_V2_DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-slow", action="store_true")
    return parser


def validate_activation_report(
    report_path: Path,
    catalog_path: Path,
    feature_path: Path,
) -> dict[str, object]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Edge C v2 activation report is required before campaign execution: {report_path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Edge C v2 activation report: {report_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Edge C v2 activation report must be a JSON object")

    expected_identity = {
        "schema_version": 1,
        "name": "edge_c_v2_activation",
        "family": EDGE_C_V2_FAMILY,
        "strategy": EDGE_C_V2_STRATEGY,
        "sampler_version": EDGE_C_V2_SAMPLER_VERSION,
        "campaign_seed": EDGE_C_V2_DEFAULT_SEED,
        "catalog_budget": EDGE_C_V2_DEFAULT_BUDGET,
    }
    mismatches = [
        key for key, expected in expected_identity.items() if payload.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Edge C v2 activation report identity mismatch: " + ", ".join(mismatches)
        )
    if payload.get("passed") is not True:
        raise RuntimeError("Edge C v2 activation did not pass; full campaign is blocked")

    actual_catalog_hash = sha256_file(catalog_path)
    if payload.get("catalog_sha256") != actual_catalog_hash:
        raise ValueError("Edge C v2 activation report catalog hash does not match current catalog")
    actual_source_hash = sha256_file(feature_path)
    if payload.get("source_data_sha256") != actual_source_hash:
        raise ValueError("Edge C v2 activation report feature hash does not match current source data")
    return payload


def run_edge_c_v2_campaign(
    *,
    catalog_path: Path,
    feature_path: Path,
    result_root: Path,
    activation_report: Path,
    workers: int,
    seed: int,
    limit: int | None,
    allow_slow: bool,
) -> None:
    if int(seed) != EDGE_C_V2_DEFAULT_SEED:
        raise ValueError(f"Edge C v2 campaign seed is frozen at {EDGE_C_V2_DEFAULT_SEED}")
    validate_activation_report(activation_report, catalog_path, feature_path)
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
    run_edge_c_v2_campaign(
        catalog_path=args.catalog,
        feature_path=args.features,
        result_root=args.result_root,
        activation_report=args.activation_report,
        workers=args.workers,
        seed=args.seed,
        limit=args.limit,
        allow_slow=args.allow_slow,
    )
    print("Edge C v2 campaign run completed or resumed successfully")


if __name__ == "__main__":
    main()
