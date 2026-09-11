import polars as pl
import pytest

from xau_lab.data.sessions import SessionConfig, add_session_features


def test_sessions_are_derived_only_with_explicit_timezone():
    frame = pl.DataFrame({"time": ["2026-01-05 08:00:00", "2026-01-05 13:30:00"]})
    cfg = SessionConfig(
        broker_timezone="Etc/UTC",
        asia_start="00:00",
        asia_end="08:00",
        london_start="08:00",
        london_end="17:00",
        new_york_start="13:30",
        new_york_end="22:00",
    )
    out = add_session_features(frame, cfg)
    assert out["session_london"].to_list() == [True, True]
    assert out["session_new_york"].to_list() == [False, True]
    assert out["session_overlap"].to_list() == [False, True]
    assert out["weekday"].to_list() == [0, 0]


def test_session_research_fails_closed_without_broker_timezone():
    frame = pl.DataFrame({"time": ["2026-01-05 08:00:00"]})
    cfg = SessionConfig(
        broker_timezone=None,
        asia_start="00:00",
        asia_end="08:00",
        london_start="08:00",
        london_end="17:00",
        new_york_start="13:30",
        new_york_end="22:00",
    )
    with pytest.raises(ValueError, match="broker timezone must be explicit before session research"):
        add_session_features(frame, cfg)
