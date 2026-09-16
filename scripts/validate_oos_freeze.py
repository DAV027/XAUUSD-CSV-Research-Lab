from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FROZEN_EXPERIMENT_FIELDS = (
    "experiment_id",
    "fingerprint",
    "allocation_bucket",
    "family",
    "strategy_name",
    "canonical_parameters_json",
    "direction_mode",
    "stop_atr",
    "exit_type",
    "target_r",
    "time_exit_minutes",
    "atr_trail",
    "commission_round_trip_per_lot",
    "slippage_points_per_fill",
    "strategy_seed",
    "family_seed",
    "sampler_version",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex(value: object, length: int, name: str) -> str:
    text = str(value)
    if len(text) != length or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ValueError(f"{name} must be a {length}-character hexadecimal string")
    return text.lower()


def _load_catalog(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0].keys():
        raise ValueError("catalog must contain a header and at least one row")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        experiment_id = (row.get("experiment_id") or "").strip()
        if not experiment_id:
            raise ValueError("catalog row missing experiment_id")
        if experiment_id in by_id:
            raise ValueError(f"duplicate catalog experiment_id: {experiment_id}")
        by_id[experiment_id] = row
    return by_id


def validate_oos_freeze(
    config_path: str | Path,
    catalog_path: str | Path,
    feature_path: str | Path,
) -> dict[str, int]:
    config_path = Path(config_path)
    catalog_path = Path(catalog_path)
    feature_path = Path(feature_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported OOS freeze schema_version")

    source = config.get("selection_source")
    if not isinstance(source, dict):
        raise ValueError("selection_source must be an object")

    _require_hex(source.get("campaign_commit"), 40, "campaign_commit")
    expected_catalog_sha = _require_hex(source.get("catalog_sha256"), 64, "catalog_sha256")
    expected_feature_sha = _require_hex(source.get("feature_sha256"), 64, "feature_sha256")

    actual_catalog_sha = _sha256(catalog_path)
    if actual_catalog_sha != expected_catalog_sha:
        raise ValueError(
            f"catalog SHA256 mismatch: expected {expected_catalog_sha}, got {actual_catalog_sha}"
        )
    actual_feature_sha = _sha256(feature_path)
    if actual_feature_sha != expected_feature_sha:
        raise ValueError(
            f"feature SHA256 mismatch: expected {expected_feature_sha}, got {actual_feature_sha}"
        )

    data_end = source.get("campaign_data_end_epoch")
    boundary = source.get("oos_start_exclusive_epoch")
    if not isinstance(data_end, int) or not isinstance(boundary, int):
        raise ValueError("campaign/OOS epoch values must be integers")
    if boundary != data_end:
        raise ValueError("OOS boundary must equal the frozen campaign data end")
    expected_utc = datetime.fromtimestamp(data_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if source.get("campaign_data_end_utc") != expected_utc:
        raise ValueError("campaign_data_end_utc does not match campaign_data_end_epoch")
    if source.get("oos_rule") != "time_epoch > oos_start_exclusive_epoch":
        raise ValueError("OOS rule must remain strictly greater than the frozen boundary")

    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    expected_count = config.get("expected_candidate_count")
    if not isinstance(expected_count, int) or expected_count != len(candidates):
        raise ValueError("expected_candidate_count does not match candidates")

    catalog = _load_catalog(catalog_path)
    frozen_ids: set[str] = set()
    primary_count = diagnostic_count = 0

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        tier = candidate.get("tier")
        if tier == "primary":
            primary_count += 1
        elif tier == "diagnostic":
            diagnostic_count += 1
        else:
            raise ValueError(f"invalid candidate tier: {tier!r}")

        experiment = candidate.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("candidate experiment must be an object")
        if set(experiment) != set(FROZEN_EXPERIMENT_FIELDS):
            raise ValueError("candidate experiment fields differ from the frozen schema")
        if any(not isinstance(experiment[field], str) for field in FROZEN_EXPERIMENT_FIELDS):
            raise ValueError("all frozen experiment values must be strings")

        experiment_id = experiment["experiment_id"]
        if experiment_id in frozen_ids:
            raise ValueError(f"duplicate frozen experiment_id: {experiment_id}")
        frozen_ids.add(experiment_id)

        catalog_row = catalog.get(experiment_id)
        if catalog_row is None:
            raise ValueError(f"frozen experiment_id missing from catalog: {experiment_id}")
        for field in FROZEN_EXPERIMENT_FIELDS:
            actual = catalog_row.get(field, "")
            expected = experiment[field]
            if actual != expected:
                raise ValueError(
                    f"candidate drift for {experiment_id} field {field}: "
                    f"expected {expected!r}, got {actual!r}"
                )

    return {
        "candidate_count": len(candidates),
        "primary_count": primary_count,
        "diagnostic_count": diagnostic_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen XAUUSD OOS shortlist")
    parser.add_argument("--config", type=Path, default=Path("config/oos_shortlist_v1.json"))
    parser.add_argument("--catalog", type=Path, default=Path("results/EXPERIMENT_CATALOG.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/XAUUSD_M1_FEATURES.parquet"))
    args = parser.parse_args()

    result = validate_oos_freeze(args.config, args.catalog, args.features)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["selection_source"]
    print("OOS_FREEZE_VALID=true")
    print(f"CANDIDATES={result['candidate_count']}")
    print(f"PRIMARY={result['primary_count']}")
    print(f"DIAGNOSTIC={result['diagnostic_count']}")
    print(f"CAMPAIGN_COMMIT={source['campaign_commit']}")
    print(f"OOS_START_EXCLUSIVE_EPOCH={source['oos_start_exclusive_epoch']}")
    print(f"OOS_START_EXCLUSIVE_UTC={source['campaign_data_end_utc']}")


if __name__ == "__main__":
    main()
