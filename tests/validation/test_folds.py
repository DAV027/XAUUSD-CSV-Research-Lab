from __future__ import annotations

from datetime import datetime, timezone

from xau_lab.validation.folds import expanding_folds, rolling_folds


def _monthly_times(start_year: int, start_month: int, count: int) -> list[int]:
    out: list[int] = []
    year = start_year
    month = start_month
    for _ in range(count):
        out.append(int(datetime(year, month, 15, tzinfo=timezone.utc).timestamp()))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return out


def test_rolling_folds_are_12_month_research_3_month_validation_with_3_month_step():
    folds = rolling_folds(_monthly_times(2023, 1, 36))
    assert folds.status == "ok"
    assert len(folds) == 8

    first = folds[0]
    assert first.research_start == "2023-01"
    assert first.research_end == "2023-12"
    assert first.validation_start == "2024-01"
    assert first.validation_end == "2024-03"

    second = folds[1]
    assert second.research_start == "2023-04"
    assert second.research_end == "2024-03"
    assert second.validation_start == "2024-04"
    assert second.validation_end == "2024-06"


def test_expanding_folds_keep_research_start_fixed_and_expand_by_validation_step():
    folds = expanding_folds(_monthly_times(2023, 1, 24))
    assert len(folds) == 4
    assert folds[0].research_start == "2023-01"
    assert folds[0].research_end == "2023-12"
    assert folds[0].validation_start == "2024-01"
    assert folds[0].validation_end == "2024-03"
    assert folds[1].research_start == "2023-01"
    assert folds[1].research_end == "2024-03"
    assert folds[1].validation_start == "2024-04"
    assert folds[1].validation_end == "2024-06"


def test_research_and_validation_never_overlap_and_validation_is_strictly_later():
    for folds in (
        rolling_folds(_monthly_times(2023, 1, 36)),
        expanding_folds(_monthly_times(2023, 1, 36)),
    ):
        for fold in folds:
            assert set(fold.research_months).isdisjoint(fold.validation_months)
            assert fold.research_end < fold.validation_start


def test_short_history_uses_deterministic_80_20_fallback_and_records_rule():
    folds = rolling_folds(_monthly_times(2025, 1, 10))
    assert folds.status == "fallback_short_history"
    assert len(folds) == 1
    fold = folds[0]
    assert len(fold.research_months) == 8
    assert len(fold.validation_months) == 2
    assert fold.fallback_rule == "floor_80pct_min_6_validation_remainder_capped_3"
    assert fold.research_start == "2025-01"
    assert fold.research_end == "2025-08"
    assert fold.validation_start == "2025-09"
    assert fold.validation_end == "2025-10"


def test_seven_month_fallback_uses_minimum_six_research_months():
    folds = expanding_folds(_monthly_times(2025, 1, 7))
    assert len(folds) == 1
    assert len(folds[0].research_months) == 6
    assert len(folds[0].validation_months) == 1


def test_fewer_than_seven_months_returns_no_folds_with_insufficient_history_status():
    folds = rolling_folds(_monthly_times(2025, 1, 6))
    assert len(folds) == 0
    assert folds.status == "insufficient_history"
    assert folds.metadata["complete_months"] == 6


def test_duplicate_and_unsorted_timestamps_do_not_change_month_folds():
    times = _monthly_times(2023, 1, 18)
    shuffled = list(reversed(times + [times[0], times[-1]]))
    assert list(rolling_folds(shuffled)) == list(rolling_folds(times))
