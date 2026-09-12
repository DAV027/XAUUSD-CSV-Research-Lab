from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

REGISTRY_VERSION = 1
IMMUTABLE_FIELDS = (
    "experiment_id",
    "freeze_timestamp",
    "source_data_sha256",
    "parameter_fingerprint",
    "prospective_start",
    "planned_months",
    "status",
)


@dataclass(frozen=True)
class HoldoutRecord:
    experiment_id: str
    freeze_timestamp: str
    source_data_sha256: str
    parameter_fingerprint: str
    prospective_start: str
    planned_months: int = 3
    status: str = "frozen"
    observed_through: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.freeze_timestamp:
            raise ValueError("freeze_timestamp is required")
        digest = self.source_data_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("source_data_sha256 must be a 64-character SHA-256 hex digest")
        object.__setattr__(self, "source_data_sha256", digest)
        if not self.parameter_fingerprint:
            raise ValueError("parameter_fingerprint is required")
        _parse_date(self.prospective_start, "prospective_start")
        if self.planned_months != 3:
            raise ValueError("planned_months must be 3 for the approved prospective holdout")
        if not self.status:
            raise ValueError("status is required")
        if self.observed_through is not None:
            observed = _parse_date(self.observed_through, "observed_through")
            if observed < _parse_date(self.prospective_start, "prospective_start"):
                raise ValueError("observed_through cannot be before prospective_start")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD") from exc


def _empty_registry() -> dict[str, object]:
    return {"version": REGISTRY_VERSION, "candidates": {}}


def load_holdout_registry(path: str | Path) -> dict[str, object]:
    registry_path = Path(path)
    if not registry_path.exists():
        return _empty_registry()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid holdout registry: {registry_path}") from exc
    if payload.get("version") != REGISTRY_VERSION or not isinstance(payload.get("candidates"), dict):
        raise ValueError("invalid holdout registry schema")
    return payload


def _write_registry(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def freeze_holdout(path: str | Path, record: HoldoutRecord) -> dict[str, object]:
    registry_path = Path(path)
    payload = load_holdout_registry(registry_path)
    candidates = dict(payload["candidates"])
    new_row = record.to_dict()
    existing = candidates.get(record.experiment_id)

    if existing is not None:
        if not isinstance(existing, dict):
            raise ValueError("invalid candidate entry in holdout registry")
        old_start = _parse_date(str(existing["prospective_start"]), "prospective_start")
        new_start = _parse_date(record.prospective_start, "prospective_start")
        if new_start < old_start:
            raise ValueError("prospective_start cannot move backward after candidate freeze")
        changed = [field for field in IMMUTABLE_FIELDS if existing.get(field) != new_row.get(field)]
        if changed:
            raise ValueError(f"holdout fields are immutable after freeze: {changed}")
        if record.observed_through is not None and record.observed_through != existing.get("observed_through"):
            return update_observed_through(registry_path, record.experiment_id, record.observed_through)
        return dict(existing)

    candidates[record.experiment_id] = new_row
    payload = {"version": REGISTRY_VERSION, "candidates": candidates}
    _write_registry(registry_path, payload)
    return dict(new_row)


def update_observed_through(path: str | Path, experiment_id: str, observed_through: str) -> dict[str, object]:
    registry_path = Path(path)
    payload = load_holdout_registry(registry_path)
    candidates = dict(payload["candidates"])
    existing = candidates.get(experiment_id)
    if not isinstance(existing, dict):
        raise KeyError(f"candidate is not frozen: {experiment_id}")

    new_date = _parse_date(observed_through, "observed_through")
    prospective_start = _parse_date(str(existing["prospective_start"]), "prospective_start")
    current_text = existing.get("observed_through")
    current_date = _parse_date(str(current_text), "observed_through") if current_text else None
    floor = current_date if current_date is not None else prospective_start
    if new_date < floor:
        raise ValueError("observed_through cannot move backward")

    updated = dict(existing)
    updated["observed_through"] = observed_through
    candidates[experiment_id] = updated
    _write_registry(registry_path, {"version": REGISTRY_VERSION, "candidates": candidates})
    return updated


__all__ = [
    "HoldoutRecord",
    "IMMUTABLE_FIELDS",
    "REGISTRY_VERSION",
    "freeze_holdout",
    "load_holdout_registry",
    "update_observed_through",
]
