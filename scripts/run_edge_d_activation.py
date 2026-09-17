from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xau_lab.experiments.edge_d_sampler import (
    EDGE_D_DEFAULT_BUDGET,
    EDGE_D_DEFAULT_SEED,
    EDGE_D_FAMILY,
    EDGE_D_SAMPLER_VERSION,
    EDGE_D_STRATEGY,
)
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.manifest import sha256_file
from xau_lab.runner.single import run_experiment
from xau_lab.validation.edge_d_activation import (
    ACTIVATION_SAMPLE_SIZE,
    evaluate_activation,
    spread_sample_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic structural activation gate for Edge D v1"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("edge_d/results/EXPERIMENT_CATALOG.csv"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge_d/results/ACTIVATION_REPORT.json"),
    )
    return parser


def _validate_catalog_identity(catalog) -> None:
    if len(catalog) != EDGE_D_DEFAULT_BUDGET:
        raise ValueError(
            f"Edge D activation requires exactly {EDGE_D_DEFAULT_BUDGET} catalog rows"
        )
    for experiment in catalog:
        if experiment.family != EDGE_D_FAMILY:
            raise ValueError("Edge D catalog contains an unexpected family")
        if experiment.allocation_bucket != EDGE_D_FAMILY:
            raise ValueError("Edge D catalog contains an unexpected allocation bucket")
        if experiment.strategy_name != EDGE_D_STRATEGY:
            raise ValueError("Edge D catalog contains an unexpected strategy")
        if experiment.sampler_version != EDGE_D_SAMPLER_VERSION:
            raise ValueError("Edge D catalog contains an unexpected sampler version")
        if experiment.family_seed != EDGE_D_DEFAULT_SEED:
            raise ValueError("Edge D catalog contains an unexpected family seed")
        if experiment.direction_mode != "combined":
            raise ValueError("Edge D activation requires combined direction mode")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run_edge_d_activation(
    *,
    catalog_path: Path,
    feature_path: Path,
    output_path: Path,
) -> dict[str, object]:
    catalog = _read_catalog(catalog_path)
    _validate_catalog_identity(catalog)
    sample_indices = spread_sample_indices(len(catalog), ACTIVATION_SAMPLE_SIZE)
    market = load_market_bundle(feature_path)

    diagnostic_rows: list[dict[str, object]] = []
    for index in sample_indices:
        experiment = catalog[index]
        outcome = run_experiment(experiment, market, include_trades=True)
        if not outcome.ok or outcome.master_result is None:
            raise RuntimeError(
                outcome.error_message
                or f"activation run failed for {experiment.experiment_id}"
            )
        completed_trades = len(outcome.trades)
        end_of_data_exits = sum(
            1 for trade in outcome.trades if trade.exit_reason == "end_of_data"
        )
        diagnostic_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "completed_trades": int(completed_trades),
                "risk_skip_count": int(outcome.master_result["risk_skip_count"]),
                "end_of_data_exits": int(end_of_data_exits),
            }
        )

    decision = evaluate_activation(diagnostic_rows)
    report: dict[str, object] = {
        "schema_version": 1,
        "name": "edge_d_activation",
        "family": EDGE_D_FAMILY,
        "strategy": EDGE_D_STRATEGY,
        "sampler_version": EDGE_D_SAMPLER_VERSION,
        "campaign_seed": EDGE_D_DEFAULT_SEED,
        "catalog_budget": len(catalog),
        "catalog_sha256": sha256_file(catalog_path),
        "source_data_sha256": sha256_file(feature_path),
        "sample_indices": list(sample_indices),
        "diagnostic_rows": diagnostic_rows,
        **decision.to_dict(),
    }
    _write_json_atomic(output_path, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_edge_d_activation(
        catalog_path=args.catalog,
        feature_path=args.features,
        output_path=args.output,
    )
    status = "PASS" if report["passed"] else "FAIL"
    print(f"Edge D activation {status}")
    print(
        "Activation counts: "
        f">=50={report['at_least_50_count']}/{report['sample_size']}, "
        f">=300={report['at_least_300_count']}/{report['sample_size']}, "
        f"completed={report['total_completed_trades']}, "
        f"end_of_data_fraction={report['end_of_data_fraction']}"
    )
    if not report["passed"]:
        print("Full Edge D campaign remains blocked; preserve this activation report.")


if __name__ == "__main__":
    main()
