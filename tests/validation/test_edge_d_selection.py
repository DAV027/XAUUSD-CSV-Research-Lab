import math

from xau_lab.validation.edge_d import select_edge_d_candidates


def _row(experiment_id, **overrides):
    row = {
        "experiment_id": experiment_id,
        "net_profit": 100.0,
        "profit_factor": 1.20,
        "expectancy_usd": 0.25,
        "max_drawdown_pct": 2.0,
        "final_score": 0.70,
        "completed_trades": 300,
    }
    row.update(overrides)
    return row


def test_edge_d_accepts_inclusive_pf_and_drawdown_boundaries():
    selected = select_edge_d_candidates([
        _row("EXPBOUND", profit_factor=1.10, max_drawdown_pct=5.0),
    ])

    assert [row["experiment_id"] for row in selected] == ["EXPBOUND"]


def test_edge_d_requires_strictly_positive_net_and_expectancy():
    rows = [
        _row("GOOD"),
        _row("ZERO_NET", net_profit=0.0),
        _row("NEG_NET", net_profit=-1.0),
        _row("ZERO_EXP", expectancy_usd=0.0),
        _row("NEG_EXP", expectancy_usd=-0.01),
        _row("LOW_PF", profit_factor=1.0999),
        _row("HIGH_DD", max_drawdown_pct=5.0001),
    ]

    assert [row["experiment_id"] for row in select_edge_d_candidates(rows)] == ["GOOD"]


def test_edge_d_trade_frequency_is_not_an_eligibility_condition():
    selected = select_edge_d_candidates([
        _row("ONE", completed_trades=1, final_score=0.8),
        _row("MILLION", completed_trades=1_000_000, final_score=0.7),
    ])

    assert [row["experiment_id"] for row in selected] == ["ONE", "MILLION"]


def test_edge_d_selection_orders_by_score_then_id_and_caps_at_six():
    rows = [_row(f"EXP{i}", final_score=float(i) / 10.0) for i in range(10)]
    selected = select_edge_d_candidates(rows)
    assert [row["experiment_id"] for row in selected] == [
        "EXP9", "EXP8", "EXP7", "EXP6", "EXP5", "EXP4"
    ]

    tied = select_edge_d_candidates([
        _row("B", final_score=0.5),
        _row("A", final_score=0.5),
    ])
    assert [row["experiment_id"] for row in tied] == ["A", "B"]


def test_edge_d_rejects_missing_nonfinite_or_malformed_metrics_and_allows_zero_survivors():
    rows = [
        _row("BAD_SCORE", final_score="bad"),
        _row("NAN_SCORE", final_score=math.nan),
        _row("INF_SCORE", final_score=math.inf),
        _row("BAD_PF", profit_factor=None),
        _row("", final_score=0.9),
    ]

    assert select_edge_d_candidates(rows) == []


def test_edge_d_selector_rejects_negative_max_candidates():
    try:
        select_edge_d_candidates([_row("GOOD")], max_candidates=-1)
    except ValueError as exc:
        assert "max_candidates" in str(exc)
    else:
        raise AssertionError("negative max_candidates must fail closed")
