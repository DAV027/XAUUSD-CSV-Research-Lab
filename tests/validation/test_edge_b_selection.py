import math

import pytest

from xau_lab.validation.edge_b import select_edge_b_candidates
from xau_lab.validation.reporting import RESEARCH_ONLY_SCOPE

WEEK = 7 * 24 * 60 * 60


def _row(experiment_id: str, trades: float, score: float, *, weeks: float = 1.0):
    return {
        "experiment_id": experiment_id,
        "data_start": 1_700_000_000,
        "data_end": 1_700_000_000 + weeks * WEEK,
        "completed_trades": trades,
        "final_score": score,
        "verdict": "ROBUST_CANDIDATE",
    }


def test_frequency_boundaries_are_inclusive_and_outside_rows_are_rejected():
    rows = [
        _row("EXP_LOW", 2.99, 99.0),
        _row("EXP_MIN", 3.0, 80.0),
        _row("EXP_MID", 6.0, 90.0),
        _row("EXP_MAX", 10.0, 70.0),
        _row("EXP_HIGH", 10.01, 100.0),
    ]

    selected = select_edge_b_candidates(rows)

    assert [row["experiment_id"] for row in selected] == ["EXP_MID", "EXP_MIN", "EXP_MAX"]
    assert [row["edge_b_frequency_trades_per_week"] for row in selected] == [6.0, 3.0, 10.0]
    assert all(row["approval_scope"] == RESEARCH_ONLY_SCOPE for row in selected)


def test_selector_truncates_to_six_by_score_then_experiment_id():
    rows = [
        _row("EXP_Z", 5.0, 80.0),
        _row("EXP_B", 5.0, 95.0),
        _row("EXP_A", 5.0, 95.0),
        _row("EXP_C", 5.0, 90.0),
        _row("EXP_D", 5.0, 85.0),
        _row("EXP_E", 5.0, 84.0),
        _row("EXP_F", 5.0, 83.0),
        _row("EXP_G", 5.0, 82.0),
    ]

    selected = select_edge_b_candidates(rows, max_candidates=6)

    assert [row["experiment_id"] for row in selected] == [
        "EXP_A",
        "EXP_B",
        "EXP_C",
        "EXP_D",
        "EXP_E",
        "EXP_F",
    ]


def test_selector_never_force_fills_when_fewer_than_three_qualify():
    rows = [
        _row("EXP_ONE", 5.0, 90.0),
        _row("EXP_TWO", 7.0, 80.0),
        _row("EXP_TOO_SLOW", 1.0, 100.0),
        _row("EXP_TOO_FAST", 12.0, 100.0),
    ]

    selected = select_edge_b_candidates(rows)

    assert [row["experiment_id"] for row in selected] == ["EXP_ONE", "EXP_TWO"]


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"data_end": 1_700_000_000}, "positive discovery span"),
        ({"completed_trades": ""}, "completed_trades"),
        ({"final_score": "nan"}, "final_score"),
        ({"experiment_id": ""}, "experiment_id"),
    ],
)
def test_selector_rejects_invalid_robust_rows(patch, match):
    row = _row("EXP_VALID", 5.0, 80.0)
    row.update(patch)

    with pytest.raises(ValueError, match=match):
        select_edge_b_candidates([row])


def test_selector_rejects_invalid_configuration():
    row = _row("EXP_VALID", 5.0, 80.0)

    with pytest.raises(ValueError, match="frequency bounds"):
        select_edge_b_candidates(row and [row], min_trades_per_week=10.0, max_trades_per_week=3.0)
    with pytest.raises(ValueError, match="max_candidates"):
        select_edge_b_candidates([row], max_candidates=0)
