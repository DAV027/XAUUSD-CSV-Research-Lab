from __future__ import annotations

import csv
import json

import pytest

from xau_lab.backtest.models import Trade
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.io.finalist import MT5_DELAYS_MS, MT5_MODEL, export_finalist_package
from xau_lab.validation.holdout import HoldoutRecord, freeze_holdout, update_observed_through
from xau_lab.validation.scoring import SCORE_VERSION


def _experiment() -> CompleteExperiment:
    params = {"period": 20, "threshold": 1.1}
    return CompleteExperiment(
        experiment_id="EXP_FINAL",
        fingerprint="fingerprint-final",
        allocation_bucket="trend",
        family="trend",
        strategy_name="trend_pullback",
        parameters=params,
        canonical_parameters_json=canonical_json(params),
        direction_mode="combined",
        stop_atr=1.5,
        exit_type="target_r",
        target_r=2.0,
        time_exit_minutes=None,
        atr_trail=None,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        strategy_seed=123,
        family_seed=456,
    )


def _trade() -> Trade:
    return Trade(
        experiment_id="EXP_FINAL",
        parameter_set_id="fingerprint-final",
        fingerprint="fingerprint-final",
        signal_time=1,
        entry_time=2,
        direction=1,
        signal_price=2000.0,
        entry_price=2000.1,
        stop_price=1999.1,
        target_price=2002.1,
        exit_time=62,
        exit_price=2001.0,
        exit_reason="target",
        lot_size=0.01,
        planned_risk_usd=1.0,
        risk_R=1.0,
        gross_pnl=0.9,
        spread_cost=0.1,
        commission=0.06,
        slippage_cost=0.02,
        net_pnl=0.72,
        pnl_R=0.72,
        hold_minutes=1.0,
        broker_date="2026-09-11",
        broker_timezone="UTC",
    )


def _holdout(**overrides) -> HoldoutRecord:
    base = dict(
        experiment_id="EXP_FINAL",
        freeze_timestamp="2026-09-12T06:00:00Z",
        source_data_sha256="a" * 64,
        parameter_fingerprint="fingerprint-final",
        prospective_start="2026-09-13",
        planned_months=3,
        status="frozen",
        observed_through=None,
    )
    return HoldoutRecord(**(base | overrides))


def _robustness() -> dict[str, object]:
    return {
        "verdict": "ROBUST_CANDIDATE",
        "final_score": 72.5,
        "median_validation_pf": 1.12,
        "parameter_stability_pct": 80.0,
        "cost_stability_pct": 70.0,
    }


def test_holdout_registry_freezes_required_fields_and_only_observed_through_can_advance(tmp_path):
    registry = tmp_path / "state" / "HOLDOUT_REGISTRY.json"
    first = freeze_holdout(registry, _holdout())
    assert first["experiment_id"] == "EXP_FINAL"
    assert first["planned_months"] == 3

    updated = update_observed_through(registry, "EXP_FINAL", "2026-09-30")
    assert updated["observed_through"] == "2026-09-30"

    persisted = json.loads(registry.read_text(encoding="utf-8"))
    assert persisted["candidates"]["EXP_FINAL"]["prospective_start"] == "2026-09-13"
    assert persisted["candidates"]["EXP_FINAL"]["observed_through"] == "2026-09-30"


def test_new_holdout_must_start_strictly_after_its_freeze_date():
    with pytest.raises(ValueError, match="strictly after freeze"):
        _holdout(prospective_start="2026-09-12")
    with pytest.raises(ValueError, match="strictly after freeze"):
        _holdout(prospective_start="2026-09-11")


def test_holdout_requires_timezone_aware_freeze_timestamp():
    with pytest.raises(ValueError, match="freeze_timestamp"):
        _holdout(freeze_timestamp="2026-09-12T06:00:00")


def test_holdout_cannot_move_prospective_start_backward_or_change_frozen_identity(tmp_path):
    registry = tmp_path / "HOLDOUT_REGISTRY.json"
    freeze_holdout(registry, _holdout())

    with pytest.raises(ValueError, match="prospective_start"):
        freeze_holdout(
            registry,
            HoldoutRecord(
                experiment_id="EXP_FINAL",
                freeze_timestamp="2026-09-12T05:00:00Z",
                source_data_sha256="a" * 64,
                parameter_fingerprint="fingerprint-final",
                prospective_start="2026-09-12",
                planned_months=3,
                status="frozen",
            ),
        )

    with pytest.raises(ValueError, match="immutable"):
        freeze_holdout(registry, _holdout(source_data_sha256="b" * 64))

    with pytest.raises(ValueError, match="backward"):
        update_observed_through(registry, "EXP_FINAL", "2026-09-01")


def test_finalist_package_rejects_holdout_identity_or_source_mismatch(tmp_path):
    discovery = {"profit_factor": 1.24, "completed_trades": 800, "net_profit": 120.0}
    bad_holdouts = [
        _holdout(experiment_id="OTHER").to_dict(),
        _holdout(parameter_fingerprint="other-fingerprint").to_dict(),
        _holdout(source_data_sha256="b" * 64).to_dict(),
    ]
    for holdout in bad_holdouts:
        with pytest.raises(ValueError, match="holdout"):
            export_finalist_package(
                tmp_path / "results" / "finalists",
                _experiment(),
                discovery_metrics=discovery,
                robustness_metrics=_robustness(),
                source_data_sha256="a" * 64,
                trades=(_trade(),),
                holdout_record=holdout,
            )


def test_finalist_package_contains_exact_mt5_handoff_and_never_claims_tick_validation(tmp_path):
    discovery = {"profit_factor": 1.24, "completed_trades": 800, "net_profit": 120.0}
    robustness = _robustness()
    package = export_finalist_package(
        tmp_path / "results" / "finalists",
        _experiment(),
        discovery_metrics=discovery,
        robustness_metrics=robustness,
        source_data_sha256="a" * 64,
        trades=(_trade(),),
        holdout_record=_holdout().to_dict(),
    )

    candidate_path = package / "candidate.json"
    trades_path = package / "candidate_trades.csv"
    checklist_path = package / "MT5_VALIDATION_CHECKLIST.md"
    assert candidate_path.is_file()
    assert trades_path.is_file()
    assert checklist_path.is_file()

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["experiment_id"] == "EXP_FINAL"
    assert candidate["strategy_name"] == "trend_pullback"
    assert candidate["parameters"] == {"period": 20, "threshold": 1.1}
    assert candidate["direction_mode"] == "combined"
    assert candidate["exit_spec"]["stop_atr"] == 1.5
    assert candidate["exit_spec"]["target_r"] == 2.0
    assert candidate["cost_spec"]["commission_round_trip_per_lot"] == 6.0
    assert candidate["cost_spec"]["slippage_points_per_fill"] == 5.0
    assert candidate["fingerprint"] == "fingerprint-final"
    assert candidate["discovery_metrics"] == discovery
    assert candidate["robustness_metrics"] == robustness
    assert candidate["source_data_sha256"] == "a" * 64
    assert candidate["score_version"] == SCORE_VERSION
    assert candidate["required_mt5_test_delays_ms"] == [0, 100, 250, 500, 1000]
    assert candidate["mt5_model"] == "Every tick based on real ticks"
    assert candidate["real_money_approval_required"] is True
    assert candidate["csv_evidence_tick_validated"] is False
    assert tuple(candidate["required_mt5_test_delays_ms"]) == MT5_DELAYS_MS
    assert candidate["mt5_model"] == MT5_MODEL

    with trades_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment_id"] == "EXP_FINAL"
    assert rows[0]["net_pnl"] == "0.72"

    checklist = checklist_path.read_text(encoding="utf-8")
    assert "Every tick based on real ticks" in checklist
    assert "0, 100, 250, 500, 1000" in checklist
    assert "real-money approval required" in checklist.lower()
    assert "not tick-validated" in checklist.lower()
