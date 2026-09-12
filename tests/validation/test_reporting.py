from __future__ import annotations

import csv

from xau_lab.backtest.models import Trade
from xau_lab.validation.reporting import (
    RESEARCH_ONLY_SCOPE,
    build_robustness_evidence,
    classify_candidates,
    robust_candidate_decision,
    write_candidate_tables,
    write_promoted_trade_logs,
)
from xau_lab.validation.stress import StressReport, StressResult
from xau_lab.validation.walkforward import FoldResult


def _master(experiment_id: str, **overrides):
    base = dict(
        experiment_id=experiment_id,
        profit_factor=1.25,
        max_drawdown_pct=3.0,
        completed_trades=800,
        expectancy_usd=0.40,
        expectancy_R=0.08,
        median_profit_per_active_day=2.5,
        positive_year_fraction=0.75,
        positive_month_fraction=0.70,
        top_5_trade_profit_fraction=0.25,
        best_month_profit_fraction=0.30,
        active_months=18,
        long_PF=1.20,
        short_PF=1.15,
        long_trades=400,
        short_trades=400,
        integrity_ok=True,
    )
    return base | overrides


def _robust(**overrides):
    base = dict(
        expanding_validation_joint_stability_fraction=0.75,
        rolling_validation_joint_stability_fraction=0.70,
        median_validation_pf=1.12,
        parameter_stability_pct=80.0,
        cost_stability_pct=70.0,
        stress_max_drawdown_pct=4.0,
        top_five_removal_net_profit=8.0,
    )
    return base | overrides


def _trade(experiment_id: str, *, net_pnl: float = 0.22, broker_date: str = "2026-01-02") -> Trade:
    return Trade(
        experiment_id=experiment_id,
        parameter_set_id="p",
        fingerprint="f",
        signal_time=1,
        entry_time=2,
        direction=1,
        signal_price=2000.0,
        entry_price=2000.1,
        stop_price=1999.1,
        target_price=None,
        exit_time=62,
        exit_price=2000.5,
        exit_reason="time",
        lot_size=0.01,
        planned_risk_usd=1.0,
        risk_R=1.0,
        gross_pnl=0.4,
        spread_cost=0.1,
        commission=0.06,
        slippage_cost=0.02,
        net_pnl=net_pnl,
        pnl_R=net_pnl,
        hold_minutes=1.0,
        broker_date=broker_date,
        broker_timezone="UTC",
    )


def _fold(scheme: str, fold_id: str, segment: str, pf: float, expectancy: float) -> FoldResult:
    return FoldResult(
        experiment_id="robust",
        scheme=scheme,
        fold_id=fold_id,
        segment=segment,
        start="2024-01",
        end="2024-03",
        trades=100,
        pf=pf,
        expectancy_usd=expectancy,
        net_profit=10.0 if expectancy > 0 else -10.0,
        max_drawdown_pct=2.0,
        positive_day_fraction=0.60,
    )


def test_three_candidate_classes_land_in_exactly_one_output_table_each():
    master_rows = [
        _master("reject", profit_factor=1.05),
        _master("survivor"),
        _master("robust"),
    ]
    evidence = {
        "survivor": _robust(parameter_stability_pct=40.0),
        "robust": _robust(),
    }
    tables = classify_candidates(master_rows, evidence)

    assert [row["experiment_id"] for row in tables.rejected] == ["reject"]
    assert [row["experiment_id"] for row in tables.survivors] == ["survivor"]
    assert [row["experiment_id"] for row in tables.top_candidates] == ["robust"]

    assert tables.rejected[0]["verdict"] == "REJECTED"
    assert "pf_below_1_10" in tables.rejected[0]["rejection_reason"]
    assert tables.survivors[0]["verdict"] == "STAGE1_SURVIVOR"
    assert "parameter_stability_below_60_pct" in tables.survivors[0]["rejection_reason"]
    assert tables.top_candidates[0]["verdict"] == "ROBUST_CANDIDATE"
    assert tables.top_candidates[0]["rejection_reason"] == ""
    assert tables.top_candidates[0]["approval_scope"] == RESEARCH_ONLY_SCOPE


def test_robustness_component_uses_approved_walkforward_parameter_cost_mean():
    decision = robust_candidate_decision(_master("x"), _robust())
    expected_walkforward = (0.75 + 0.70) / 2.0
    expected_component = 15.0 * ((expected_walkforward + 0.80 + 0.70) / 3.0)
    assert decision.passed
    assert decision.walkforward_stability == expected_walkforward
    assert decision.robustness_component == expected_component


def test_each_robust_finalist_gate_can_fail_explicitly():
    cases = [
        ({"expanding_validation_joint_stability_fraction": 0.59}, "expanding_walkforward_below_60_pct"),
        ({"rolling_validation_joint_stability_fraction": 0.59}, "rolling_walkforward_below_60_pct"),
        ({"median_validation_pf": 1.049}, "median_validation_pf_below_1_05"),
        ({"parameter_stability_pct": 59.9}, "parameter_stability_below_60_pct"),
        ({"cost_stability_pct": 49.9}, "cost_stability_below_50_pct"),
        ({"stress_max_drawdown_pct": 5.01}, "stress_drawdown_above_5_pct"),
        ({"top_five_removal_net_profit": -0.01}, "top_five_removal_turns_negative"),
    ]
    for override, reason in cases:
        decision = robust_candidate_decision(_master("x"), _robust(**override))
        assert not decision.passed
        assert reason in decision.reasons

    baseline = robust_candidate_decision(_master("x", profit_factor=1.09), _robust())
    assert not baseline.passed
    assert "baseline_pf_below_1_10" in baseline.reasons


def test_missing_robustness_evidence_stays_stage1_survivor_with_null_final_score():
    tables = classify_candidates([_master("pending")], {})
    assert len(tables.survivors) == 1
    row = tables.survivors[0]
    assert row["verdict"] == "STAGE1_SURVIVOR"
    assert row["rejection_reason"] == "robustness_not_evaluated"
    assert row["robustness_component"] is None
    assert row["final_score"] is None
    assert row["final_score_pre_robustness"] is not None


def test_robust_candidate_has_completed_final_score_and_all_components():
    tables = classify_candidates([_master("robust")], {"robust": _robust()})
    row = tables.top_candidates[0]
    assert row["score_version"] == "v1"
    assert row["robustness_component"] is not None
    assert row["final_score"] is not None
    for field in (
        "pf_component",
        "expectancy_component",
        "median_daily_component",
        "stability_component",
        "sample_size_component",
        "drawdown_penalty",
        "concentration_penalty",
        "side_dependence_penalty",
    ):
        assert field in row


def test_build_robustness_evidence_combines_both_schemes_stress_and_seeded_resampling():
    expanding = [
        _fold("expanding", "E1", "research", 1.30, 0.20),
        _fold("expanding", "E1", "validation", 1.10, 0.10),
    ]
    rolling = [
        _fold("rolling", "R1", "research", 1.25, 0.15),
        _fold("rolling", "R1", "validation", 1.30, 0.12),
    ]
    stress = StressReport(
        results=(
            StressResult("robust", "parameter", "parameter:x:1", 1.1, 0.1, 5.0, 3.0),
            StressResult("robust", "cost", "slippage:20", 1.05, 0.05, 2.0, 4.0),
        ),
        parameter_stability_pct=80.0,
        cost_stability_pct=70.0,
        max_drawdown_pct=4.0,
        parameter_run_count=1,
        cost_run_count=1,
    )
    trades = (
        _trade("robust", net_pnl=10.0, broker_date="2026-01-02"),
        _trade("robust", net_pnl=8.0, broker_date="2026-01-02"),
        _trade("robust", net_pnl=5.0, broker_date="2026-01-03"),
        _trade("robust", net_pnl=-4.0, broker_date="2026-01-04"),
        _trade("robust", net_pnl=-3.0, broker_date="2026-01-05"),
        _trade("robust", net_pnl=2.0, broker_date="2026-01-06"),
        _trade("robust", net_pnl=-1.0, broker_date="2026-01-07"),
    )
    evidence = build_robustness_evidence(
        expanding,
        rolling,
        stress,
        trades,
        resample_n=100,
        seed=9_216_000,
    )
    assert evidence["expanding_validation_joint_stability_fraction"] == 1.0
    assert evidence["rolling_validation_joint_stability_fraction"] == 1.0
    assert evidence["median_validation_pf"] == 1.20
    assert evidence["parameter_stability_pct"] == 80.0
    assert evidence["cost_stability_pct"] == 70.0
    assert evidence["stress_max_drawdown_pct"] == 4.0
    assert evidence["top_five_removal_net_profit"] < 0.0
    assert evidence["bootstrap_seed"] == 9_216_000
    assert evidence["bootstrap_n"] == 100
    assert evidence["monte_carlo_seed"] == 9_216_000
    assert evidence["monte_carlo_label"] == "sequence_only_not_entry_edge_proof"


def test_reporting_writes_canonical_tables_and_only_promoted_trade_logs(tmp_path):
    master_rows = [
        _master("reject", profit_factor=1.05),
        _master("survivor"),
        _master("robust"),
    ]
    tables = classify_candidates(
        master_rows,
        {
            "survivor": _robust(parameter_stability_pct=40.0),
            "robust": _robust(),
        },
    )

    write_candidate_tables(tmp_path, tables)
    for name in ("REJECTED.csv", "SURVIVORS.csv", "TOP_CANDIDATES.csv"):
        assert (tmp_path / name).is_file()

    with (tmp_path / "TOP_CANDIDATES.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment_id"] == "robust"
    assert rows[0]["verdict"] == "ROBUST_CANDIDATE"

    write_promoted_trade_logs(
        tmp_path,
        tables,
        {
            "reject": (_trade("reject"),),
            "survivor": (_trade("survivor"),),
            "robust": (_trade("robust"),),
        },
    )
    log_root = tmp_path / "trade_logs"
    assert not (log_root / "reject.csv").exists()
    assert (log_root / "survivor.csv").is_file()
    assert (log_root / "robust.csv").is_file()

    with (log_root / "robust.csv").open(encoding="utf-8", newline="") as handle:
        trade_rows = list(csv.DictReader(handle))
    assert trade_rows[0]["experiment_id"] == "robust"
    assert trade_rows[0]["net_pnl"] == "0.22"
