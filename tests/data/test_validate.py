import polars as pl

from xau_lab.data.validate import validate_bars


def sample():
    return pl.DataFrame({
        "time": [
            "2026-01-05 10:00:00",
            "2026-01-05 10:01:00",
            "2026-01-05 10:01:00",
            "2026-01-05 10:03:00",
        ],
        "open": [2000.0, 2001.0, 2001.0, 2005.0],
        "high": [2002.0, 2000.0, 2002.0, 2006.0],
        "low": [1999.0, 2000.0, 2000.0, 2004.0],
        "close": [2001.0, 2001.5, 2001.5, 2005.0],
        "tick_volume": [10, 12, 12, -1],
        "spread": [35, -1, -1, 40],
        "real_volume": [0, 0, 0, 0],
    })


def test_validator_reports_duplicate_ohlc_spread_volume_and_gap_without_filling():
    result = validate_bars(sample())
    kinds = set(result.issues["issue_type"].to_list())
    assert {
        "duplicate_timestamp",
        "invalid_ohlc",
        "invalid_spread",
        "negative_volume",
        "unexpected_gap",
    } <= kinds
    assert result.clean.height < sample().height
    assert "2026-01-05 10:02:00" not in result.clean["time"].to_list()
    assert result.issues.columns == ["issue_type", "time", "severity", "details"]


def test_clean_output_is_sorted_and_keeps_last_duplicate_when_valid():
    frame = pl.DataFrame({
        "time": ["2026-01-05 10:01:00", "2026-01-05 10:00:00", "2026-01-05 10:01:00"],
        "open": [2001.0, 2000.0, 2002.0],
        "high": [2002.0, 2001.0, 2003.0],
        "low": [2000.0, 1999.0, 2001.0],
        "close": [2001.5, 2000.5, 2002.5],
        "tick_volume": [10, 10, 11],
        "spread": [35, 35, 34],
        "real_volume": [0, 0, 0],
    })

    result = validate_bars(frame)

    assert result.clean["time"].to_list() == ["2026-01-05 10:00:00", "2026-01-05 10:01:00"]
    assert result.clean.filter(pl.col("time") == "2026-01-05 10:01:00")["open"][0] == 2002.0
    assert result.summary["non_monotonic_input"] is True
