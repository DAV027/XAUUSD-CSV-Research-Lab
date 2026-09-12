from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("canonical experiment values must be finite")
        return float(format(value, ".10g"))
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_fingerprint(
    *,
    strategy_name: str,
    canonical_parameters: dict,
    direction_mode: str,
    stop_atr: float,
    exit_type: str,
    target_r: float | None,
    time_exit_minutes: int | None,
    atr_trail: float | None,
    commission_round_trip_per_lot: float,
    slippage_points_per_fill: float,
    strategy_seed: int,
    sampler_version: str,
) -> str:
    payload = {
        "strategy_name": strategy_name,
        "parameters": canonical_parameters,
        "direction_mode": direction_mode,
        "stop": {"type": "atr", "value": stop_atr},
        "exit": {
            "type": exit_type,
            "target_r": target_r,
            "time_exit_minutes": time_exit_minutes,
            "atr_trail": atr_trail,
        },
        "cost": {
            "commission_round_trip_per_lot": commission_round_trip_per_lot,
            "slippage_points_per_fill": slippage_points_per_fill,
        },
        "strategy_seed": int(strategy_seed),
        "sampler_version": sampler_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompleteExperiment:
    experiment_id: str
    fingerprint: str
    allocation_bucket: str
    family: str
    strategy_name: str
    parameters: dict
    canonical_parameters_json: str
    direction_mode: str
    stop_atr: float
    exit_type: str
    target_r: float | None
    time_exit_minutes: int | None
    atr_trail: float | None
    commission_round_trip_per_lot: float
    slippage_points_per_fill: float
    strategy_seed: int
    family_seed: int
    sampler_version: str = "v1"

    def __post_init__(self) -> None:
        if self.direction_mode not in {"long", "short", "combined"}:
            raise ValueError("invalid direction_mode")
        if self.exit_type not in {"target_r", "time", "atr_trail", "target_time"}:
            raise ValueError("invalid exit_type")
        if self.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive")

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "fingerprint": self.fingerprint,
            "allocation_bucket": self.allocation_bucket,
            "family": self.family,
            "strategy_name": self.strategy_name,
            "canonical_parameters_json": self.canonical_parameters_json,
            "direction_mode": self.direction_mode,
            "stop_atr": self.stop_atr,
            "exit_type": self.exit_type,
            "target_r": self.target_r,
            "time_exit_minutes": self.time_exit_minutes,
            "atr_trail": self.atr_trail,
            "commission_round_trip_per_lot": self.commission_round_trip_per_lot,
            "slippage_points_per_fill": self.slippage_points_per_fill,
            "strategy_seed": self.strategy_seed,
            "family_seed": self.family_seed,
            "sampler_version": self.sampler_version,
        }
