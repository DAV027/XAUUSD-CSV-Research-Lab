from __future__ import annotations

import argparse
import csv
from pathlib import Path

from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.manifest import verify_manifest_artifacts
from xau_lab.runner.single import run_experiment
from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.reporting import (
    build_robustness_evidence,
    classify_candidates,
    write_candidate_tables,
    write_promoted_trade_logs,
)
from xau_lab.validation.resampling import DEFAULT_RESAMPLES, RESAMPLING_SEED
from xau_lab.validation.stress import run_stress_suite, write_stress_results
from xau_lab.validation.walkforward import validate_candidate, write_fold_results


def _read_master(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _integrity_ok(row: dict[str, str]) -> bool:
    value = row.get("integrity_ok", "true").strip().lower()
    return value not in {"0", "false", "no", "fail", "failed"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Stage-1 promotion, run frozen robustness validation, and write "
            "canonical XAUUSD candidate outputs"
        )
    )
    parser.add_argument("--master", type=Path, default=Path("results/MASTER_RESULTS.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet")
    )
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Discovery RUN_MANIFEST.json; defaults to <result-root>/RUN_MANIFEST.json",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=RESAMPLING_SEED)
    args = parser.parse_args()

    if args.resamples <= 0:
        raise ValueError("--resamples must be positive")

    manifest_path = args.run_manifest or (args.result_root / "RUN_MANIFEST.json")
    verify_manifest_artifacts(manifest_path, args.features, args.catalog)

    master_rows = _read_master(args.master)
    if not master_rows:
        raise RuntimeError("MASTER_RESULTS.csv contains no experiment rows")

    stage1_ids = {
        row["experiment_id"]
        for row in master_rows
        if row.get("experiment_id")
        and stage1_decision(row, integrity_ok=_integrity_ok(row)).passed
    }
    catalog = _read_catalog(args.catalog)
    by_id = {experiment.experiment_id: experiment for experiment in catalog}
    missing = sorted(stage1_ids - set(by_id))
    if missing:
        raise RuntimeError(f"Stage-1 survivors are missing from the experiment catalog: {missing}")

    market = load_market_bundle(args.features)
    robustness_evidence: dict[str, dict[str, object]] = {}
    trade_logs = {}
    fold_rows = []
    stress_rows = []

    # Catalog order is retained deliberately so regenerated evidence files are deterministic.
    for experiment in catalog:
        if experiment.experiment_id not in stage1_ids:
            continue

        expanding = validate_candidate(experiment, market, "expanding")
        rolling = validate_candidate(experiment, market, "rolling")
        fold_rows.extend(expanding)
        fold_rows.extend(rolling)

        stress = run_stress_suite(experiment, market)
        stress_rows.extend(stress.results)

        baseline = run_experiment(experiment, market, include_trades=True)
        if not baseline.ok or baseline.master_result is None:
            raise RuntimeError(
                baseline.error_message
                or f"baseline ledger rerun failed for {experiment.experiment_id}"
            )
        if not baseline.trades:
            raise RuntimeError(
                f"Stage-1 survivor {experiment.experiment_id} produced no baseline trades on rerun"
            )

        robustness_evidence[experiment.experiment_id] = build_robustness_evidence(
            expanding,
            rolling,
            stress,
            baseline.trades,
            resample_n=args.resamples,
            seed=args.seed,
        )
        trade_logs[experiment.experiment_id] = baseline.trades

    args.result_root.mkdir(parents=True, exist_ok=True)
    write_fold_results(args.result_root / "FOLD_RESULTS.csv", fold_rows)
    write_stress_results(args.result_root / "STRESS_RESULTS.csv", stress_rows)

    tables = classify_candidates(master_rows, robustness_evidence)
    write_candidate_tables(args.result_root, tables)
    write_promoted_trade_logs(args.result_root, tables, trade_logs)

    print(
        "Promotion complete: "
        f"rejected={len(tables.rejected)}, "
        f"stage1_survivors={len(tables.survivors)}, "
        f"robust_candidates={len(tables.top_candidates)}"
    )
    print(
        "ROBUST_CANDIDATE is a research-promotion label only; "
        "it is not live-trading approval."
    )


if __name__ == "__main__":
    main()
