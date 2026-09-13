import polars as pl
import pytest

from xau_lab.data.sessions import SessionConfig, add_session_features


def _config(**overrides):
    values = {
        "broker_timezone": "EET",
        "asia_timezone": "Asia/Tokyo",
        "asia_start": "09:00",
        "asia_end": "18:00",
        "london_timezone": "Europe/London",
        "london_start": "08:00",
        "london_end": "17:00",
        "new_york_timezone": "America/New_York",
        "new_york_start": "08:00",
        "new_york_end": "17:00",
    }
    values.update(overrides)
    return SessionConfig(**values)


def test_sessions_follow_named_market_timezones_across_dst_transitions():
    frame = pl.DataFrame({
        "time": [
            "2026-01-05 02:00:00",  # 09:00 Tokyo in broker winter time
            "2026-07-06 03:00:00",  # 09:00 Tokyo in broker summer time
            "2026-03-16 14:00:00",  # 08:00 New York after US DST, before EU DST
            "2026-03-02 15:00:00",  # 08:00 New York before US DST
            "2026-01-05 10:00:00",  # 08:00 London in winter
            "2026-07-06 10:00:00",  # 08:00 London in summer
        ]
    })

    out = add_session_features(frame, _config())

    asia = out["session_asia"].to_list()
    assert asia[:2] == [True, True]
    assert asia[2:4] == [False, False]
    assert out["session_new_york"].to_list() == [False, False, True, True, False, False]
    assert out["session_london"].to_list()[-2:] == [True, True]
    assert out["broker_date"].to_list()[0] == "2026-01-05"


def test_session_research_fails_closed_without_broker_timezone():
    frame = pl.DataFrame({"time": ["2026-01-05 08:00:00"]})
    cfg = _config(broker_timezone=None)
    with pytest.raises(ValueError, match="broker timezone must be explicit before session research"):
        add_session_features(frame, cfg)


def test_session_research_rejects_invalid_market_timezone():
    frame = pl.DataFrame({"time": ["2026-01-05 08:00:00"]})
    cfg = _config(london_timezone="Not/AZone")
    with pytest.raises(ValueError, match="invalid IANA session timezone"):
        add_session_features(frame, cfg)
