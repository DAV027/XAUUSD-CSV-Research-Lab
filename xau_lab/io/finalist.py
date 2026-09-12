from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping, Sequence

from xau_lab.backtest.models import Trade
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.validation.scoring import SCORE_VERSION

MT5_DELAYS_MS = (0, 100, 250, 500, 1000)
MT5_MODEL = "Every tick based on real ticks"
TRADE_FIELDS = tuple(field.name for field in fields(Trade))


def _validate_sha256(value: str) -> str:
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("source_data_sha256 must be a 64-character SHA-256 hex digest")
    return digest


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_trade_csv(path: Path, trades: Sequence[Trade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _candidate_payload(
    experiment: CompleteExperiment,
    discovery_metrics: Mapping[str, object],
    robustness_metrics: Mapping[str, object],
    source_data_sha256: str,
    holdout_record: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "family": experiment.family,
        "strategy_name": experiment.strategy_name,
        "parameters": dict(experiment.parameters),
        "direction_mode": experiment.direction_mode,
        "exit_spec": {
            "stop_atr": float(experiment.stop_atr),
            "exit_type": experiment.exit_type,
            "target_r": None if experiment.target_r is None else float(experiment.target_r),
            "time_exit_minutes": experiment.time_exit_minutes,
            "atr_trail": None if experiment.atr_trail is None else float(experiment.atr_trail),
        },
        "cost_spec": {
            "commission_round_trip_per_lot": float(experiment.commission_round_trip_per_lot),
            "slippage_points_per_fill": float(experiment.slippage_points_per_fill),
        },
        "fingerprint": experiment.fingerprint,
        "discovery_metrics": dict(discovery_metrics),
        "robustness_metrics": dict(robustness_metrics),
        "source_data_sha256": _validate_sha256(source_data_sha256),
        "score_version": SCORE_VERSION,
        "required_mt5_test_delays_ms": list(MT5_DELAYS_MS),
        "mt5_model": MT5_MODEL,
        "real_money_approval_required": True,
        "csv_evidence_tick_validated": False,
        "holdout": None if holdout_record is None else dict(holdout_record),
    }


def _checklist(payload: Mapping[str, object]) -> str:
    exit_spec = payload["exit_spec"]
    cost_spec = payload["cost_spec"]
    holdout = payload.get("holdout") or {}
    parameters_json = json.dumps(payload["parameters"], sort_keys=True, separators=(",", ":"))
    delays = ", ".join(str(value) for value in MT5_DELAYS_MS)
    lines = [
        "# MT5 Validation Checklist",
        "",
        f"- Experiment ID: `{payload['experiment_id']}`",
        f"- Fingerprint: `{payload['fingerprint']}`",
        f"- Strategy: `{payload['strategy_name']}`",
        f"- Frozen parameters: `{parameters_json}`",
        f"- Direction mode: `{payload['direction_mode']}`",
        f"- Exit specification: `{json.dumps(exit_spec, sort_keys=True)}`",
        f"- Cost specification: `{json.dumps(cost_spec, sort_keys=True)}`",
        f"- Source data SHA-256: `{payload['source_data_sha256']}`",
        f"- MT5 model: **{MT5_MODEL}**",
        f"- Required execution delays (ms): **{delays}**",
        "- Run each delay as a separate MT5 real-tick validation test.",
        "- CSV/bar evidence is **not tick-validated**.",
        "- Do not auto-place trades from this research package.",
        "- Separate **real-money approval required** before any live use.",
    ]
    if holdout:
        lines.extend(
            [
                f"- Prospective holdout start: `{holdout.get('prospective_start')}`",
                f"- Planned prospective months: `{holdout.get('planned_months')}`",
                f"- Holdout status: `{holdout.get('status')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def export_finalist_package(
    output_root: str | Path,
    experiment: CompleteExperiment,
    *,
    discovery_metrics: Mapping[str, object],
    robustness_metrics: Mapping[str, object],
    source_data_sha256: str,
    trades: Sequence[Trade],
    holdout_record: Mapping[str, object] | None = None,
) -> Path:
    if robustness_metrics.get("verdict") != "ROBUST_CANDIDATE":
        raise ValueError("only ROBUST_CANDIDATE rows may be exported as MT5 finalists")
    for trade in trades:
        if trade.experiment_id != experiment.experiment_id:
            raise ValueError("candidate trade ledger contains a different experiment_id")

    package = Path(output_root) / experiment.experiment_id
    payload = _candidate_payload(
        experiment,
        discovery_metrics,
        robustness_metrics,
        source_data_sha256,
        holdout_record,
    )
    _atomic_json(package / "candidate.json", payload)
    _atomic_trade_csv(package / "candidate_trades.csv", tuple(trades))
    _atomic_text(package / "MT5_VALIDATION_CHECKLIST.md", _checklist(payload))
    return package


__all__ = [
    "MT5_DELAYS_MS",
    "MT5_MODEL",
    "TRADE_FIELDS",
    "export_finalist_package",
]
