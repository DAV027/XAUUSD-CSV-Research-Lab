from __future__ import annotations

from xau_lab.validation.scoring import SCORE_VERSION, score_candidate, score_survivor_set


def _metrics(**overrides):
    base = dict(
        experiment_id="A",
        profit_factor=1.30,
        expectancy_R=0.12,
        median_profit_per_active_day=3.0,
        median_profit_per_active_day_percentile=0.75,
        positive_year_fraction=0.80,
        positive_month_fraction=0.75,
        completed_trades=1000,
        max_drawdown_pct=3.0,
        top_5_trade_profit_fraction=0.20,
        best_month_profit_fraction=0.25,
        active_months=12,
        long_trades=500,
        short_trades=500,
        long_PF=1.20,
        short_PF=1.15,
    )
    return base | overrides


def test_score_version_is_v1_and_missing_robustness_is_not_faked_as_zero():
    out = score_candidate(_metrics())
    assert SCORE_VERSION == "v1"
    assert out["score_version"] == "v1"
    assert out["robustness_component"] is None
    assert out["final_score"] is None
    assert 0.0 <= out["final_score_pre_robustness"] <= 85.0


def test_component_endpoints_and_penalties_are_clipped():
    out = score_candidate(
        _metrics(
            profit_factor=2.0,
            expectancy_R=1.0,
            median_profit_per_active_day_percentile=2.0,
            positive_year_fraction=1.5,
            positive_month_fraction=1.5,
            completed_trades=30_000,
            max_drawdown_pct=10.0,
            top_5_trade_profit_fraction=2.0,
            best_month_profit_fraction=2.0,
            long_PF=0.0,
            short_PF=1.2,
        )
    )
    assert out["pf_component"] == 25.0
    assert out["expectancy_component"] == 20.0
    assert out["median_daily_component"] == 15.0
    assert out["stability_component"] == 15.0
    assert out["sample_size_component"] == 10.0
    assert out["drawdown_penalty"] == 15.0
    assert out["concentration_penalty"] == 15.0
    assert 0.0 <= out["side_dependence_penalty"] <= 10.0


def test_a_stable_candidate_ranks_above_higher_pf_but_concentrated_candidate_and_low_expectancy_candidate():
    a = _metrics(experiment_id="A")
    b = _metrics(
        experiment_id="B",
        profit_factor=1.35,
        max_drawdown_pct=4.9,
        top_5_trade_profit_fraction=0.49,
        best_month_profit_fraction=0.58,
        median_profit_per_active_day=3.2,
    )
    c = _metrics(
        experiment_id="C",
        profit_factor=1.18,
        expectancy_R=0.02,
        max_drawdown_pct=2.0,
        median_profit_per_active_day=2.0,
    )
    scored = score_survivor_set([a, b, c])
    by_id = {row["experiment_id"]: row for row in scored}
    assert by_id["A"]["final_score_pre_robustness"] > by_id["B"]["final_score_pre_robustness"]
    assert by_id["A"]["final_score_pre_robustness"] > by_id["C"]["final_score_pre_robustness"]


def test_survivor_set_percentile_ranking_is_deterministic_and_ties_share_rank():
    rows = [
        _metrics(experiment_id="low", median_profit_per_active_day=1.0),
        _metrics(experiment_id="tie1", median_profit_per_active_day=2.0),
        _metrics(experiment_id="tie2", median_profit_per_active_day=2.0),
        _metrics(experiment_id="high", median_profit_per_active_day=4.0),
    ]
    first = score_survivor_set(rows)
    second = score_survivor_set(list(reversed(rows)))
    p1 = {row["experiment_id"]: row["median_profit_per_active_day_percentile"] for row in first}
    p2 = {row["experiment_id"]: row["median_profit_per_active_day_percentile"] for row in second}
    assert p1 == p2
    assert p1["low"] == 0.0
    assert p1["high"] == 1.0
    assert p1["tie1"] == p1["tie2"]


def test_side_dependence_penalty_requires_at_least_100_trades_on_both_sides():
    penalized = score_candidate(_metrics(long_trades=100, short_trades=100, short_PF=0.45))
    unpenalized = score_candidate(_metrics(long_trades=100, short_trades=99, short_PF=0.45))
    assert penalized["side_dependence_penalty"] > 0.0
    assert unpenalized["side_dependence_penalty"] == 0.0
