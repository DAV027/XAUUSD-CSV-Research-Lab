from __future__ import annotations

from xau_lab.validation.reporting import (
    RESEARCH_ONLY_SCOPE,
    classify_candidates,
    robust_candidate_decision,
)


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
