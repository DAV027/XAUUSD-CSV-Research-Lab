from __future__ import annotations

from xau_lab.validation.promotion import stage1_decision


BASE = dict(
    profit_factor=1.20,
    max_drawdown_pct=4.0,
    completed_trades=500,
    expectancy_usd=0.5,
    positive_year_fraction=0.75,
    top_5_trade_profit_fraction=0.20,
    best_month_profit_fraction=0.25,
    active_months=12,
)


def test_good_candidate_passes():
    decision = stage1_decision(BASE)
    assert decision.passed
    assert decision.reasons == ()


def test_pf_below_1_10_fails():
    decision = stage1_decision(BASE | {"profit_factor": 1.099})
    assert not decision.passed
    assert "pf_below_1_10" in decision.reasons


def test_undefined_pf_fails():
    decision = stage1_decision(BASE | {"profit_factor": None})
    assert not decision.passed
    assert "pf_below_1_10" in decision.reasons


def test_drawdown_above_5_fails():
    decision = stage1_decision(BASE | {"max_drawdown_pct": 5.01})
    assert not decision.passed
    assert "drawdown_above_5_pct" in decision.reasons


def test_trade_count_below_300_fails():
    decision = stage1_decision(BASE | {"completed_trades": 299})
    assert not decision.passed
    assert "completed_trades_below_300" in decision.reasons


def test_nonpositive_expectancy_fails():
    decision = stage1_decision(BASE | {"expectancy_usd": 0.0})
    assert not decision.passed
    assert "expectancy_not_positive" in decision.reasons


def test_profitable_years_must_be_majority_when_available():
    decision = stage1_decision(BASE | {"positive_year_fraction": 0.50})
    assert not decision.passed
    assert "profitable_years_not_majority" in decision.reasons


def test_missing_year_fraction_does_not_invent_a_failure():
    decision = stage1_decision(BASE | {"positive_year_fraction": None})
    assert decision.passed


def test_top_five_profit_concentration_above_50_pct_fails():
    decision = stage1_decision(BASE | {"top_5_trade_profit_fraction": 0.5001})
    assert not decision.passed
    assert "top_five_profit_concentration_above_50_pct" in decision.reasons


def test_missing_top_five_concentration_evidence_fails_closed():
    decision = stage1_decision(BASE | {"top_5_trade_profit_fraction": None})
    assert not decision.passed
    assert "top_five_profit_concentration_missing" in decision.reasons


def test_best_month_concentration_above_60_pct_fails_with_six_active_months():
    decision = stage1_decision(
        BASE | {"best_month_profit_fraction": 0.6001, "active_months": 6}
    )
    assert not decision.passed
    assert "best_month_profit_concentration_above_60_pct" in decision.reasons


def test_missing_best_month_evidence_fails_closed_when_gate_applies():
    decision = stage1_decision(
        BASE | {"best_month_profit_fraction": None, "active_months": 6}
    )
    assert not decision.passed
    assert "best_month_profit_concentration_missing" in decision.reasons


def test_missing_active_month_count_fails_closed():
    decision = stage1_decision(BASE | {"active_months": None})
    assert not decision.passed
    assert "active_months_missing" in decision.reasons


def test_best_month_concentration_gate_is_not_applied_before_six_active_months():
    decision = stage1_decision(
        BASE | {"best_month_profit_fraction": 0.95, "active_months": 5}
    )
    assert decision.passed


def test_integrity_failure_rejects_candidate():
    decision = stage1_decision(BASE, integrity_ok=False)
    assert not decision.passed
    assert "integrity_failure" in decision.reasons
