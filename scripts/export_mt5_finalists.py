from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
from pathlib import Path

from xau_lab.io.finalist import export_finalist_package
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.runner.manifest import verify_manifest_artifacts
from xau_lab.runner.single import run_experiment
from xau_lab.validation.holdout import HoldoutRecord, freeze_holdout, load_holdout_registry


def _coerce(value: str) -> object:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if all(char not in text.lower() for char in (".", "e")):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze prospective holdouts and export research-only MT5 finalist packages"
    )
    parser.add_argument("--top-candidates", type=Path, default=Path("results/TOP_CANDIDATES.csv"))
    parser.add_argument("--master", type=Path, default=Path("results/MASTER_RESULTS.csv"))
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet")
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("results/RUN_MANIFEST.json"),
        help="Discovery RUN_MANIFEST.json used to verify feature/catalog hashes",
    )
    parser.add_argument("--registry", type=Path, default=Path("state/HOLDOUT_REGISTRY.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/finalists"))
    parser.add_argument(
        "--prospective-start",
        required=True,
        help="ISO date YYYY-MM-DD; must be after the freeze date for a new finalist",
    )
    args = parser.parse_args()

    try:
        requested_start = date.fromisoformat(args.prospective_start)
    except ValueError as exc:
        raise ValueError("--prospective-start must be YYYY-MM-DD") from exc

    manifest = verify_manifest_artifacts(args.run_manifest, args.features, args.catalog)
    source_sha = str(manifest["source_data_sha256"])

    top_rows = _read_rows(args.top_candidates)
    finalist_rows = [row for row in top_rows if row.get("verdict") == "ROBUST_CANDIDATE"]
    if len(finalist_rows) != len(top_rows):
        raise RuntimeError("TOP_CANDIDATES.csv contains a row not labeled ROBUST_CANDIDATE")

    master_by_id = {str(row["experiment_id"]): row for row in _read_rows(args.master)}
    catalog = _read_catalog(args.catalog)
    experiments = {experiment.experiment_id: experiment for experiment in catalog}
    market = load_market_bundle(args.features)

    missing_master = sorted(str(row["experiment_id"]) for row in finalist_rows if str(row["experiment_id"]) not in master_by_id)
    missing_catalog = sorted(str(row["experiment_id"]) for row in finalist_rows if str(row["experiment_id"]) not in experiments)
    if missing_master:
        raise RuntimeError(f"finalists missing from MASTER_RESULTS.csv: {missing_master}")
    if missing_catalog:
        raise RuntimeError(f"finalists missing from EXPERIMENT_CATALOG.csv: {missing_catalog}")

    registry_payload = load_holdout_registry(args.registry)
    existing_candidates = registry_payload["candidates"]
    exported = 0

    for robustness in finalist_rows:
        experiment_id = str(robustness["experiment_id"])
        experiment = experiments[experiment_id]
        existing = existing_candidates.get(experiment_id)

        if existing is None:
            freeze_time = datetime.now(timezone.utc)
            if requested_start <= freeze_time.date():
                raise ValueError(
                    "new prospective holdout must start after the freeze date; "
                    "previously inspected data must not be labeled pristine holdout"
                )
            record = HoldoutRecord(
                experiment_id=experiment_id,
                freeze_timestamp=freeze_time.isoformat().replace("+00:00", "Z"),
                source_data_sha256=source_sha,
                parameter_fingerprint=experiment.fingerprint,
                prospective_start=args.prospective_start,
                planned_months=3,
                status="frozen",
            )
            holdout = freeze_holdout(args.registry, record)
        else:
            if existing.get("source_data_sha256") != source_sha:
                raise ValueError(f"frozen source_data_sha256 changed for {experiment_id}")
            if existing.get("parameter_fingerprint") != experiment.fingerprint:
                raise ValueError(f"frozen parameter fingerprint changed for {experiment_id}")
            if existing.get("prospective_start") != args.prospective_start:
                raise ValueError(f"prospective_start is immutable for {experiment_id}")
            holdout = dict(existing)

        rerun = run_experiment(experiment, market, include_trades=True)
        if not rerun.ok or rerun.master_result is None:
            raise RuntimeError(rerun.error_message or f"finalist rerun failed for {experiment_id}")
        if not rerun.trades:
            raise RuntimeError(f"finalist rerun produced no trades for {experiment_id}")

        discovery = master_by_id[experiment_id]
        robustness_only = {
            key: value
            for key, value in robustness.items()
            if key not in discovery or key in {"verdict", "rejection_reason", "approval_scope", "final_score", "score_version"}
        }
        export_finalist_package(
            args.output_root,
            experiment,
            discovery_metrics=discovery,
            robustness_metrics=robustness_only,
            source_data_sha256=source_sha,
            trades=rerun.trades,
            holdout_record=holdout,
        )
        exported += 1

    print(f"Exported {exported} research-only MT5 finalist package(s) to {args.output_root}")
    print("No trades were placed. Separate real-money approval is required before any live use.")


if __name__ == "__main__":
    main()
