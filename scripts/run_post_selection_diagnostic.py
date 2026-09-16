from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.validate_oos_freeze import validate_oos_freeze
from xau_lab.runner.campaign import _read_catalog, load_market_bundle
from xau_lab.validation.oos_runner import append_oos_rows, load_holdout_plan, run_oos_experiment

EXPECTED_CLASSIFICATION = "post_selection_pre_freeze_diagnostic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: object, name: str) -> int:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    return int(parsed.astimezone(timezone.utc).timestamp())


def _load_diagnostic_plan(path: Path, *, true_oos_start_epoch: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic schema_version")
    if payload.get("classification") != EXPECTED_CLASSIFICATION:
        raise ValueError("diagnostic classification must remain post_selection_pre_freeze_diagnostic")
    if payload.get("is_pristine_prospective_oos") is not False:
        raise ValueError("diagnostic data must not be labeled pristine prospective OOS")

    selection_end = payload.get("selection_data_end_epoch")
    start_epoch = payload.get("diagnostic_start_epoch")
    end_epoch = payload.get("diagnostic_end_exclusive_epoch")
    if not all(isinstance(value, int) for value in (selection_end, start_epoch, end_epoch)):
        raise ValueError("diagnostic epoch values must be integers")
    if _parse_utc(payload.get("diagnostic_start_utc"), "diagnostic_start_utc") != start_epoch:
        raise ValueError("diagnostic_start_utc does not match diagnostic_start_epoch")
    if _parse_utc(payload.get("diagnostic_end_exclusive_utc"), "diagnostic_end_exclusive_utc") != end_epoch:
        raise ValueError("diagnostic_end_exclusive_utc does not match diagnostic_end_exclusive_epoch")
    if start_epoch <= selection_end:
        raise ValueError("diagnostic start must be after selection data end")
    if end_epoch <= start_epoch:
        raise ValueError("diagnostic end must be after diagnostic start")
    if end_epoch > true_oos_start_epoch:
        raise ValueError("diagnostic window must not overlap the prospective OOS window")
    return payload


def _resolve_reference(reference: object, base: Path) -> Path | None:
    if not reference:
        return None
    target = Path(str(reference))
    if not target.is_absolute():
        cwd_target = Path.cwd() / target
        base_target = base.parent / target
        target = cwd_target if cwd_target.exists() else base_target
    return target.resolve()


def run_diagnostic_snapshot(
    *,
    shortlist_config: str | Path,
    oos_holdout_plan: str | Path,
    diagnostic_plan: str | Path,
    catalog_path: str | Path,
    canonical_feature_path: str | Path,
    diagnostic_feature_path: str | Path,
    output_root: str | Path,
) -> dict[str, int]:
    shortlist_config = Path(shortlist_config)
    oos_holdout_plan = Path(oos_holdout_plan)
    diagnostic_plan = Path(diagnostic_plan)
    catalog_path = Path(catalog_path)
    canonical_feature_path = Path(canonical_feature_path)
    diagnostic_feature_path = Path(diagnostic_feature_path)
    output_root = Path(output_root)

    if canonical_feature_path.resolve() == diagnostic_feature_path.resolve():
        raise ValueError("diagnostic evaluation requires a separate snapshot feature artifact")

    validate_oos_freeze(shortlist_config, catalog_path, canonical_feature_path)
    oos_plan = load_holdout_plan(oos_holdout_plan)
    diag = _load_diagnostic_plan(
        diagnostic_plan,
        true_oos_start_epoch=int(oos_plan["prospective_start_epoch"]),
    )
    if int(diag["selection_data_end_epoch"]) != int(oos_plan["selection_data_end_epoch"]):
        raise ValueError("diagnostic selection-data endpoint differs from prospective holdout provenance")

    for plan_path, payload in ((oos_holdout_plan, oos_plan), (diagnostic_plan, diag)):
        planned_shortlist = _resolve_reference(payload.get("shortlist_config"), plan_path)
        if planned_shortlist is not None and planned_shortlist != shortlist_config.resolve():
            raise ValueError("plan shortlist_config does not match the supplied shortlist")

    shortlist = json.loads(shortlist_config.read_text(encoding="utf-8"))
    candidates = shortlist.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("frozen shortlist must contain candidates")

    catalog = _read_catalog(catalog_path)
    by_id = {item.experiment_id: item for item in catalog}
    market = load_market_bundle(diagnostic_feature_path)
    start_epoch = int(diag["diagnostic_start_epoch"])
    end_epoch = int(diag["diagnostic_end_exclusive_epoch"])
    snapshot_sha = _sha256(diagnostic_feature_path)

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
            raise RuntimeError(f"diagnostic evaluation failed for {experiment_id}")
        master = dict(outcome.master_result)
        master.pop("oos_start_epoch", None)
        master.pop("oos_end_exclusive_epoch", None)
        row = {
            "evaluation_kind": EXPECTED_CLASSIFICATION,
            "diagnostic_name": str(diag.get("name") or ""),
            "is_pristine_prospective_oos": False,
            "tier": str(candidate.get("tier") or ""),
            "snapshot_feature_sha256": snapshot_sha,
            "diagnostic_start_epoch": start_epoch,
            "diagnostic_end_exclusive_epoch": end_epoch,
            **master,
        }
        rows.append(row)
        current_observed = int(master["observed_through_epoch"])
        observed_through = current_observed if observed_through is None else max(observed_through, current_observed)

    appended = append_oos_rows(output_root / "DIAGNOSTIC_RESULTS.csv", rows)
    return {
        "candidate_count": len(rows),
        "appended_rows": appended,
        "observed_through_epoch": 0 if observed_through is None else observed_through,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen shortlist on the Sep 13-16 post-selection diagnostic window"
    )
    parser.add_argument("--shortlist-config", type=Path, default=Path("config/oos_shortlist_v1.json"))
    parser.add_argument("--oos-holdout-plan", type=Path, default=Path("config/oos_holdout_v1.json"))
    parser.add_argument(
        "--diagnostic-plan",
        type=Path,
        default=Path("config/post_selection_diagnostic_v1.json"),
    )
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument(
        "--canonical-features",
        type=Path,
        default=Path("data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument(
        "--diagnostic-features",
        type=Path,
        default=Path("diagnostic_snapshot/data/features/XAUUSD_M1_FEATURES.parquet"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("diagnostic_results"))
    args = parser.parse_args()

    result = run_diagnostic_snapshot(
        shortlist_config=args.shortlist_config,
        oos_holdout_plan=args.oos_holdout_plan,
        diagnostic_plan=args.diagnostic_plan,
        catalog_path=args.catalog,
        canonical_feature_path=args.canonical_features,
        diagnostic_feature_path=args.diagnostic_features,
        output_root=args.output_root,
    )
    print("POST_SELECTION_DIAGNOSTIC_VALID=true")
    print(f"CANDIDATES={result['candidate_count']}")
    print(f"APPENDED_ROWS={result['appended_rows']}")
    print(f"OBSERVED_THROUGH_EPOCH={result['observed_through_epoch']}")


if __name__ == "__main__":
    main()
