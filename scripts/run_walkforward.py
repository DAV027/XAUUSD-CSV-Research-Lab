from __future__ import annotations

import argparse
import csv
from pathlib import Path

from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.walkforward import validate_candidate, write_fold_results


def _read_master(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _integrity_ok(row: dict[str, str]) -> bool:
    value = row.get("integrity_ok", "true").strip().lower()
    return value not in {"0", "false", "no", "fail", "failed"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen-parameter expanding and rolling walk-forward validation for Stage-1 survivors"
    )
    parser.add_argument("--master", type=Path, default=Path("results/MASTER_RESULTS.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("results/FOLD_RESULTS.csv"))
    args = parser.parse_args()

    master_rows = _read_master(args.master)
    promoted_ids = {
        row["experiment_id"]
        for row in master_rows
        if row.get("experiment_id")
        and stage1_decision(row, integrity_ok=_integrity_ok(row)).passed
    }
    catalog = _read_catalog(args.catalog)
    catalog_ids = {experiment.experiment_id for experiment in catalog}
    missing = sorted(promoted_ids - catalog_ids)
    if missing:
        raise RuntimeError(f"Stage-1 survivors are missing from the experiment catalog: {missing}")

    market = load_market_bundle(args.features)
    fold_rows = []
    for experiment in catalog:
        if experiment.experiment_id not in promoted_ids:
            continue
        for scheme in ("expanding", "rolling"):
            fold_rows.extend(validate_candidate(experiment, market, scheme))

    write_fold_results(args.output, fold_rows)
    print(
        f"Walk-forward complete: {len(promoted_ids)} Stage-1 survivors, "
        f"{len(fold_rows)} fold-segment rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
