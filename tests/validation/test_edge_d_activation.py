from copy import deepcopy

import pytest

from xau_lab.validation.edge_d_activation import (
    ACTIVATION_SAMPLE_SIZE,
    evaluate_activation,
    spread_sample_indices,
)


def _rows(*, at_least_50=205, at_least_300=64):
    rows = []
    for index in range(ACTIVATION_SAMPLE_SIZE):
        completed = 0
        if index < at_least_50:
            completed = 50
        if index < at_least_300:
            completed = 300
        rows.append(
            {
                "experiment_id": f"EXP{index:04d}",
                "completed_trades": completed,
                "risk_skip_count": 0,
                "end_of_data_exits": 0,
            }
        )
    return rows


def _five_percent_rows(*, exceed=False):
    rows = []
    for index in range(ACTIVATION_SAMPLE_SIZE):
        completed = 300 if index < 64 else 100
        end_of_data = 15 if index < 64 else 5
        if exceed and index == 0:
            end_of_data += 1
        rows.append(
            {
                "experiment_id": f"EXP{index:04d}",
                "completed_trades": completed,
                "risk_skip_count": 0,
                "end_of_data_exits": end_of_data,
            }
        )
    return rows


def test_spread_sample_indices_cover_full_default_catalog_deterministically():
    first = spread_sample_indices(10_000, 256)
    second = spread_sample_indices(10_000, 256)

    assert first == second
    assert len(first) == 256
    assert len(set(first)) == 256
    assert first[0] == 0
    assert first[-1] == 9_999
    assert all(left < right for left, right in zip(first, first[1:]))


def test_spread_sample_indices_reject_impossible_sample():
    with pytest.raises(ValueError):
        spread_sample_indices(100, 256)


def test_activation_passes_exact_frozen_205_and_64_boundaries():
    decision = evaluate_activation(_rows(at_least_50=205, at_least_300=64))

    assert decision.passed is True
    assert decision.at_least_50_count == 205
    assert decision.at_least_300_count == 64
    assert decision.zero_trade_count == 51
    assert decision.reasons == ()


def test_activation_fails_with_only_204_configs_at_50_trades():
    decision = evaluate_activation(_rows(at_least_50=204, at_least_300=64))

    assert decision.passed is False
    assert "configs_with_50_trades_below_205" in decision.reasons


def test_activation_fails_with_only_63_configs_at_300_trades():
    decision = evaluate_activation(_rows(at_least_50=205, at_least_300=63))

    assert decision.passed is False
    assert "configs_with_300_trades_below_64" in decision.reasons


def test_activation_accepts_exactly_five_percent_end_of_data_exits():
    decision = evaluate_activation(_five_percent_rows())

    assert decision.passed is True
    assert decision.total_completed_trades == 38_400
    assert decision.total_end_of_data_exits == 1_920
    assert decision.end_of_data_fraction == pytest.approx(0.05)


def test_activation_rejects_any_end_of_data_fraction_above_five_percent():
    decision = evaluate_activation(_five_percent_rows(exceed=True))

    assert decision.passed is False
    assert "end_of_data_fraction_above_5_pct" in decision.reasons


def test_activation_fails_when_total_completed_trades_is_zero():
    decision = evaluate_activation(_rows(at_least_50=0, at_least_300=0))

    assert decision.passed is False
    assert decision.zero_trade_count == 256
    assert decision.total_completed_trades == 0
    assert "no_completed_trades" in decision.reasons


def test_activation_fails_closed_on_partial_or_malformed_structural_rows():
    partial = _rows()
    partial.pop()
    decision = evaluate_activation(partial)
    assert decision.passed is False
    assert "sample_size_not_256" in decision.reasons

    malformed = _rows()
    malformed[0]["completed_trades"] = "not-a-number"
    decision = evaluate_activation(malformed)
    assert decision.passed is False
    assert "malformed_structural_row" in decision.reasons


def test_activation_fails_closed_when_end_of_data_exits_exceed_completed_trades():
    rows = _rows()
    rows[0]["end_of_data_exits"] = rows[0]["completed_trades"] + 1
    decision = evaluate_activation(rows)

    assert decision.passed is False
    assert "malformed_structural_row" in decision.reasons


def test_profitability_fields_cannot_change_activation_decision():
    base = _rows()
    profitable = deepcopy(base)
    disastrous = deepcopy(base)
    for row in profitable:
        row.update(
            net_profit=1_000_000.0,
            profit_factor=99.0,
            expectancy_usd=10_000.0,
            max_drawdown_pct=0.0,
            win_rate=1.0,
        )
    for row in disastrous:
        row.update(
            net_profit=-1_000_000.0,
            profit_factor=0.0,
            expectancy_usd=-10_000.0,
            max_drawdown_pct=99.0,
            win_rate=0.0,
        )

    assert evaluate_activation(base) == evaluate_activation(profitable)
    assert evaluate_activation(base) == evaluate_activation(disastrous)


def test_activation_report_payload_is_deterministic():
    first = evaluate_activation(_rows()).to_dict()
    second = evaluate_activation(_rows()).to_dict()

    assert first == second
    assert first["sample_size"] == 256
    assert first["at_least_50_count"] == 205
    assert first["at_least_300_count"] == 64
    assert first["passed"] is True
