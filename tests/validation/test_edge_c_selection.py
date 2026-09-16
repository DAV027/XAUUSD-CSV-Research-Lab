from xau_lab.validation.edge_c import select_edge_c_candidates


def _row(experiment_id, **overrides):
    row = {
        "experiment_id": experiment_id,
        "net_profit": 100.0,
        "profit_factor": 1.20,
        "expectancy_usd": 0.25,
        "max_drawdown_pct": 2.0,
        "final_score": 0.70,
        "trades_per_active_day": 3.0,
    }
    row.update(overrides)
    return row


def test_edge_c_accepts_inclusive_pf_and_drawdown_boundaries():
    selected = select_edge_c_candidates([
        _row("EXPBOUND", profit_factor=1.10, max_drawdown_pct=5.0),
    ])
    assert [row["experiment_id"] for row in selected] == ["EXPBOUND"]


def test_edge_c_requires_strictly_positive_net_and_expectancy():
    rows = [
        _row("GOOD"),
        _row("ZERO_NET", net_profit=0.0),
        _row("NEG_NET", net_profit=-1.0),
        _row("ZERO_EXP", expectancy_usd=0.0),
        _row("NEG_EXP", expectancy_usd=-0.01),
        _row("LOW_PF", profit_factor=1.0999),
        _row("HIGH_DD", max_drawdown_pct=5.0001),
    ]
    selected = select_edge_c_candidates(rows)
    assert [row["experiment_id"] for row in selected] == ["GOOD"]


def test_edge_c_trade_frequency_is_not_an_eligibility_condition():
    selected = select_edge_c_candidates([
        _row("SLOW", trades_per_active_day=0.2, final_score=0.8),
        _row("FAST", trades_per_active_day=500.0, final_score=0.7),
    ])
    assert [row["experiment_id"] for row in selected] == ["SLOW", "FAST"]


def test_edge_c_selection_is_deterministic_score_ordered_and_capped_at_six():
    rows = [_row(f"EXP{i}", final_score=float(i) / 10.0) for i in range(10)]
    selected = select_edge_c_candidates(rows)
    assert [row["experiment_id"] for row in selected] == [
        "EXP9", "EXP8", "EXP7", "EXP6", "EXP5", "EXP4"
    ]


def test_edge_c_rejects_malformed_metrics_and_allows_zero_survivors():
    rows = [
        _row("BAD_SCORE", final_score="bad"),
        _row("BAD_PF", profit_factor=None),
        _row(""),
    ]
    assert select_edge_c_candidates(rows) == []
