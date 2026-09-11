from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SessionConfig:
    broker_timezone: str | None
    asia_start: str
    asia_end: str
    london_start: str
    london_end: str
    new_york_start: str
    new_york_end: str


def _clock_to_minutes(value: str) -> int:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError(f"invalid session clock {value!r}; expected HH:MM") from exc
    return parsed.hour * 60 + parsed.minute


def _in_window(minute_of_day: int, start: int, end: int) -> bool:
    if start == end:
        return True
    if start < end:
        return start <= minute_of_day < end
    return minute_of_day >= start or minute_of_day < end


def add_session_features(frame: pl.DataFrame, config: SessionConfig) -> pl.DataFrame:
    if config.broker_timezone is None:
        raise ValueError("broker timezone must be explicit before session research")
    try:
        zone = ZoneInfo(config.broker_timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid IANA broker timezone: {config.broker_timezone!r}") from exc

    asia_start = _clock_to_minutes(config.asia_start)
    asia_end = _clock_to_minutes(config.asia_end)
    london_start = _clock_to_minutes(config.london_start)
    london_end = _clock_to_minutes(config.london_end)
    new_york_start = _clock_to_minutes(config.new_york_start)
    new_york_end = _clock_to_minutes(config.new_york_end)

    broker_date: list[str] = []
    years: list[int] = []
    months: list[int] = []
    weekdays: list[int] = []
    hours: list[int] = []
    minutes: list[int] = []
    asia_flags: list[bool] = []
    london_flags: list[bool] = []
    new_york_flags: list[bool] = []
    overlap_flags: list[bool] = []

    for raw_time in frame.get_column("time").to_list():
        dt = datetime.strptime(str(raw_time), _TIME_FORMAT).replace(tzinfo=zone)
        minute_of_day = dt.hour * 60 + dt.minute
        asia = _in_window(minute_of_day, asia_start, asia_end)
        london = _in_window(minute_of_day, london_start, london_end)
        new_york = _in_window(minute_of_day, new_york_start, new_york_end)

        broker_date.append(dt.date().isoformat())
        years.append(dt.year)
        months.append(dt.month)
        weekdays.append(dt.weekday())
        hours.append(dt.hour)
        minutes.append(dt.minute)
        asia_flags.append(asia)
        london_flags.append(london)
        new_york_flags.append(new_york)
        overlap_flags.append(london and new_york)

    return frame.with_columns(
        pl.Series("broker_date", broker_date, dtype=pl.String),
        pl.Series("year", years, dtype=pl.Int32),
        pl.Series("month", months, dtype=pl.Int8),
        pl.Series("weekday", weekdays, dtype=pl.Int8),
        pl.Series("hour", hours, dtype=pl.Int8),
        pl.Series("minute", minutes, dtype=pl.Int8),
        pl.Series("session_asia", asia_flags, dtype=pl.Boolean),
        pl.Series("session_london", london_flags, dtype=pl.Boolean),
        pl.Series("session_new_york", new_york_flags, dtype=pl.Boolean),
        pl.Series("session_overlap", overlap_flags, dtype=pl.Boolean),
    )
