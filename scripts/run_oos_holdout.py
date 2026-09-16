from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.validate_oos_freeze import validate_oos_freeze
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.validation.oos_runner import append_oos_rows, load_holdout_plan, run_oos_experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_oos_snapshot(
    *,
    shortlist_config: str | Path,
    holdout_plan: str | Path,
    catalog_path: str | Path,
    canonical_feature_path: str | Path,
    oos_feature_path: str | Path,
    output_root: str | Path,
) -> dict[str, int]:
    shortlist_config = Path(shortlist_config)
    holdout_plan = Path(holdout_plan)
    catalog_path = Path(catalog_path)
    canonical_feature_path = Path(canonical_feature_path)
    oos_feature_path = Path(oos_feature_path)
    output_root = Path(output_root)

    if canonical_feature_path.resolve() == oos_feature_path.resolve():
        raise ValueError("OOS evaluation requires a separate snapshot feature artifact")

    validate_oos_freeze(shortlist_config, catalog_path, canonical_feature_path)
    plan = load_holdout_plan(holdout_plan)
    planned_shortlist = plan.get("shortlist_config")
    if planned_shortlist:
        planned_path = Path(str(planned_shortlist))
        if not planned_path.is_absolute():
            planned_path = Path.cwd() / planned_path
        if planned_path.resolve() != shortlist_config.resolve():
            raise ValueError("holdout plan shortlist_config does not match the supplied shortlist")

    shortlist = json.loads(shortlist_config.read_text(encoding="utf-8"))
    candidates = shortlist.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("frozen shortlist must contain candidates")

    catalog = _read_catalog(catalog_path)
    by_id = {item.experiment_id: item for item in catalog}
    market = load_market_bundle(oos_feature_path)
    start_epoch = int(plan["prospective_start_epoch"])
    end_epoch = int(plan["planned_end_exclusive_epoch"])
    snapshot_sha = _sha256(oos_feature_path)

    rows: list[dict[str, object]] = []
    observed_through: int | None = None
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("experiment"), dict):
            raise ValueError("invalid frozen candidate record")
        experiment_id = str(candidate["experiment"].get("experiment_id") or "")
        experiment = by_id.get(experiment_id)
        if experiment is None:
            raise ValueError(f"frozen experiment missing from catalog: {experiment_id}")
        outcome = run_oos_experiment(
            experiment,
            market,
            start_epoch=start_epoch,
            end_exclusive_epoch=end_epoch,
        )
        if not outcome.ok or outcome.master_result is None:
            raise RuntimeError(f"OOS evaluation failed for {experiment_id}")
        row = {
            "holdout_name": str(plan.get("name") or ""),
            "tier": str(candidate.get("tier") or ""),
            "snapshot_feature_sha256": snapshot_sha,
            **outcome.master_result,
        }
        rows.append(row)
        current_observed = int(outcome.master_result["observed_through_epoch"])
        observed_through = (
            current_observed
            if observed_through is None
            else max(observed_through, current_observed)
        )

    appended = append_oos_rows(output_root / "OOS_RESULTS.csv", rows)
    return {
        "candidate_count": len(rows),
        "appended_rows": appended,
        "observed_through_epoch": 0 if observed_through is None else observed_through,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen XAUUSD shortlist on a separate prospective OOS snapshot"
    )
    parser.add_argument(
        "--shortlist-config",
        type=Path,
        default=Path("config/oos_shortlist_v1.json"),
    )
    parser.add_argument(
        "--holdout-plan",
        type=Path,
        default=Path("config/oos_holdout_v1.json"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/EXPERIMENT_CATALOG.csv"),
    )
    parser.add_argument(
        "--canonical-features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument(
        "--oos-features",
        type=Path,
        default=Path("oos_snapshot/data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("oos_results"))
    args = parser.parse_args()

    result = run_oos_snapshot(
        shortlist_config=args.shortlist_config,
        holdout_plan=args.holdout_plan,
        catalog_path=args.catalog,
        canonical_feature_path=args.canonical_features,
        oos_feature_path=args.oos_features,
        output_root=args.output_root,
    )
    print("OOS_SNAPSHOT_VALID=true")
    print(f"CANDIDATES={result['candidate_count']}")
    print(f"APPENDED_ROWS={result['appended_rows']}")
    print(f"OBSERVED_THROUGH_EPOCH={result['observed_through_epoch']}")


if __name__ == "__main__":
    main()
