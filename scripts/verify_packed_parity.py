from __future__ import annotations

import argparse
import gc
from pathlib import Path

from scripts.run_smoke import select_stratified_smoke_ids
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.single import run_experiment

REPRESENTATIVE_IDS = (
    "EXP1E2D424171EC",
    "EXP37889369D2F1",
    "EXPB62F70031929",
    "EXPB8942B9FF195",
    "EXPB8F12709D5FE",
    "EXPA6DA73D5B2BA",
    "EXPB1A10E7A4435",
    "EXP2053D4626356",
)


def _require_result(label, experiment_id, outcome):
    if not outcome.ok:
        raise RuntimeError(
            f"{label} failed for {experiment_id}: "
            f"{outcome.error_type}: {outcome.error_message}"
        )
    if outcome.master_result is None:
        raise RuntimeError(
            f"{label} returned no master result for {experiment_id}"
        )
    return outcome.master_result


def _differences(left, right):
    differences = []
    for field in sorted(set(left) | set(right)):
        left_present = field in left
        right_present = field in right
        left_value = left.get(field)
        right_value = right.get(field)

        if left_present != right_present or left_value != right_value:
            differences.append(
                (
                    field,
                    left_present,
                    left_value,
                    right_present,
                    right_value,
                )
            )
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/EXPERIMENT_CATALOG.csv"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--representatives", action="store_true")
    group.add_argument("--count", type=int)

    args = parser.parse_args()

    catalog = _read_catalog(args.catalog)
    by_id = {x.experiment_id: x for x in catalog}

    if args.representatives:
        selected_ids = list(REPRESENTATIVE_IDS)
    else:
        if args.count is None or args.count <= 0:
            parser.error("--count must be positive")
        selected_ids = select_stratified_smoke_ids(
            args.catalog,
            args.count,
        )

    missing = [x for x in selected_ids if x not in by_id]
    if missing:
        raise RuntimeError(
            "selected experiment IDs missing from catalog: "
            + ", ".join(missing)
        )

    market = load_market_bundle(args.features)

    passed = 0
    mismatches = 0

    for index, experiment_id in enumerate(selected_ids, 1):
        experiment = by_id[experiment_id]

        packed = run_experiment(
            experiment,
            market,
            include_trades=False,
        )

        detailed = run_experiment(
            experiment,
            market,
            include_trades=True,
        )

        left = _require_result(
            "packed",
            experiment_id,
            packed,
        )
        right = _require_result(
            "detailed",
            experiment_id,
            detailed,
        )

        differences = _differences(left, right)

        if differences:
            mismatches += 1

            for (
                field,
                left_present,
                left_value,
                right_present,
                right_value,
            ) in differences:
                print(
                    "MISMATCH",
                    experiment_id,
                    field,
                    f"packed_present={left_present}",
                    repr(left_value),
                    f"detailed_present={right_present}",
                    repr(right_value),
                )

            break

        passed += 1

        print(
            f"PARITY_PROGRESS={index}/{len(selected_ids)} "
            f"{experiment_id}",
            flush=True,
        )

        del packed
        del detailed
        gc.collect()

    print(f"PARITY_PASS={passed}")
    print(f"PARITY_MISMATCHES={mismatches}")

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
