import math

import polars as pl

from xau_lab.data import features as feature_module


def _bars(times: list[str]) -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "time": times,
            "open": [2000.0 + i for i in range(n)],
            "high": [2001.0 + i for i in range(n)],
            "low": [1999.0 + i for i in range(n)],
            "close": [2000.5 + i for i in range(n)],
            "tick_volume": [10] * n,
            "spread": [35] * n,
            "real_volume": [0] * n,
        }
    )


def test_research_window_starts_at_2017_and_preserves_source_order():
    frame = _bars(
        [
            "2016-12-30 23:59:00",
            "2017-01-02 00:00:00",
            "2017-01-02 00:01:00",
        ]
    )

    out = feature_module.filter_research_window(frame)

    assert feature_module.RESEARCH_START == "2017-01-01 00:00:00"
    assert out["time"].to_list() == ["2017-01-02 00:00:00", "2017-01-02 00:01:00"]


def test_scheduled_daily_break_is_flagged_but_does_not_reset_features():
    times = [f"2021-01-04 02:{45 + i:02d}:00" for i in range(10)] + [
        f"2021-01-04 04:{5 + i:02d}:00" for i in range(10)
    ]
    frame = _bars(times)

    out = feature_module.build_shared_features(frame)
    boundary = 10

    assert out["gap_seconds"][boundary] == 4260
    assert out["is_gap_after"][boundary]
    assert out["is_scheduled_break_after"][boundary]
    assert not out["is_irregular_gap_after"][boundary]
    assert out["entry_allowed"][boundary]
    assert math.isfinite(out["sma_5_lag1"][boundary])


def test_large_irregular_gap_resets_features_and_enforces_200_bar_cooldown():
    before = [f"2021-01-04 10:{i:02d}:00" for i in range(10)]
    after = ["2021-01-04 10:20:00"] + [f"2021-01-04 {10 + ((21 + i) // 60):02d}:{(21 + i) % 60:02d}:00" for i in range(205)]
    frame = _bars(before + after)

    out = feature_module.build_shared_features(frame)
    boundary = 10

    assert out["gap_seconds"][boundary] == 660
    assert out["is_gap_after"][boundary]
    assert not out["is_scheduled_break_after"][boundary]
    assert out["is_irregular_gap_after"][boundary]
    assert not out["entry_allowed"][boundary]
    assert not out["entry_allowed"][boundary + 199]
    assert out["entry_allowed"][boundary + 200]
    assert math.isnan(out["sma_5_lag1"][boundary + 4])
    assert math.isfinite(out["sma_5_lag1"][boundary + 5])


def test_two_minute_no_tick_gap_is_visible_but_not_treated_as_large_discontinuity():
    frame = _bars(
        [
            "2021-01-04 10:00:00",
            "2021-01-04 10:01:00",
            "2021-01-04 10:03:00",
            "2021-01-04 10:04:00",
        ]
    )

    out = feature_module.build_shared_features(frame)

    assert out["gap_seconds"][2] == 120
    assert out["is_gap_after"][2]
    assert not out["is_scheduled_break_after"][2]
    assert not out["is_irregular_gap_after"][2]
    assert out["entry_allowed"][2]
