from xau_lab.validation.edge_b_v2 import select_edge_b_v2_candidates


def _row(exp_id, avg_day, median_day, score=1.0, trades_per_day=1.0):
    return {
        "experiment_id": exp_id,
        "profit_per_active_day": avg_day,
        "median_profit_per_active_day": median_day,
        "final_score": score,
        "trades_per_active_day": trades_per_day,
    }


def test_v2_selector_requires_50_average_daily_and_positive_median():
    rows = [
        _row("PASS", 50.0, 1.0, 2.0),
        _row("LOW_AVG", 49.99, 20.0, 5.0),
        _row("NONPOS_MEDIAN", 80.0, 0.0, 4.0),
    ]
    selected = select_edge_b_v2_candidates(rows)
    assert [row["experiment_id"] for row in selected] == ["PASS"]


def test_v2_selector_does_not_restrict_trade_frequency():
    rows = [
        _row("HIGH_FREQ", 75.0, 10.0, 3.0, trades_per_day=500.0),
        _row("LOW_FREQ", 60.0, 5.0, 2.0, trades_per_day=0.2),
    ]
    selected = select_edge_b_v2_candidates(rows)
    assert {row["experiment_id"] for row in selected} == {"HIGH_FREQ", "LOW_FREQ"}


def test_v2_selector_orders_by_existing_score_then_id_and_caps_six():
    rows = [_row(f"EXP{i:02d}", 60.0, 5.0, score=float(i)) for i in range(8)]
    selected = select_edge_b_v2_candidates(rows)
    assert [row["experiment_id"] for row in selected] == [
        "EXP07", "EXP06", "EXP05", "EXP04", "EXP03", "EXP02"
    ]
