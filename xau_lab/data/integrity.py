from __future__ import annotations

import numpy as np
import polars as pl

RESEARCH_START = "2017-01-01 00:00:00"
IRREGULAR_GAP_MIN_SECONDS = 300
ENTRY_COOLDOWN_BARS = 200

# Empirically observed MetaQuotes-Demo XAUUSD daily maintenance/session break
# window in broker-local EET/EEST time across 2017-2026. The deliberately
# broad window accommodates the historical DST/server-clock shifts while
# rejecting unrelated daytime discontinuities.
_SESSION_BREAK_MIN_SECONDS = 45 * 60
_SESSION_BREAK_MAX_SECONDS = 3 * 60 * 60
_SESSION_PREVIOUS_MINUTE_MIN = 30       # 00:30
_SESSION_PREVIOUS_MINUTE_MAX = 3 * 60 + 5   # 03:05
_SESSION_CURRENT_MINUTE_MIN = 2 * 60         # 02:00
_SESSION_CURRENT_MINUTE_MAX = 4 * 60 + 10    # 04:10


def filter_research_window(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the canonical modern research window without mutating source data."""
    if "time" not in frame.columns:
        raise ValueError("research data requires a time column")
    return frame.filter(pl.col("time").cast(pl.String) >= RESEARCH_START)


def add_gap_provenance(frame: pl.DataFrame) -> pl.DataFrame:
    """Annotate M1 rows with gap provenance and a fail-closed entry guard."""
    if "time" not in frame.columns:
        raise ValueError("gap provenance requires a time column")
    n = frame.height
    if n == 0:
        return frame.with_columns(
            pl.Series("gap_seconds", [], dtype=pl.Int64),
            pl.Series("is_gap_after", [], dtype=pl.Boolean),
            pl.Series("is_scheduled_break_after", [], dtype=pl.Boolean),
            pl.Series("is_irregular_gap_after", [], dtype=pl.Boolean),
            pl.Series("entry_allowed", [], dtype=pl.Boolean),
        )

    parsed = frame.select(
        pl.col("time")
        .cast(pl.String)
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=True)
        .dt.epoch("s")
        .alias("epoch")
    )["epoch"].to_numpy()
    gap_seconds = np.full(n, 60, dtype=np.int64)
    if n > 1:
        gap_seconds[1:] = np.diff(parsed).astype(np.int64, copy=False)
    if np.any(gap_seconds <= 0):
        raise ValueError("feature input timestamps must be strictly increasing")

    clocks = frame.select(
        (
            pl.col("time").cast(pl.String).str.slice(11, 2).cast(pl.Int16) * 60
            + pl.col("time").cast(pl.String).str.slice(14, 2).cast(pl.Int16)
        ).alias("minute_of_day")
    )["minute_of_day"].to_numpy()
    previous_clocks = np.roll(clocks, 1)
    previous_clocks[0] = clocks[0]

    is_gap = gap_seconds > 60
    scheduled = (
        is_gap
        & (gap_seconds >= _SESSION_BREAK_MIN_SECONDS)
        & (gap_seconds <= _SESSION_BREAK_MAX_SECONDS)
        & (previous_clocks >= _SESSION_PREVIOUS_MINUTE_MIN)
        & (previous_clocks <= _SESSION_PREVIOUS_MINUTE_MAX)
        & (clocks >= _SESSION_CURRENT_MINUTE_MIN)
        & (clocks <= _SESSION_CURRENT_MINUTE_MAX)
    )
    irregular = is_gap & (gap_seconds > IRREGULAR_GAP_MIN_SECONDS) & ~scheduled

    entry_allowed = np.ones(n, dtype=np.bool_)
    for start in np.flatnonzero(irregular):
        entry_allowed[start : min(n, start + ENTRY_COOLDOWN_BARS)] = False

    return frame.with_columns(
        pl.Series("gap_seconds", gap_seconds, dtype=pl.Int64),
        pl.Series("is_gap_after", is_gap, dtype=pl.Boolean),
        pl.Series("is_scheduled_break_after", scheduled, dtype=pl.Boolean),
        pl.Series("is_irregular_gap_after", irregular, dtype=pl.Boolean),
        pl.Series("entry_allowed", entry_allowed, dtype=pl.Boolean),
    )


def research_segments(irregular_gap_after: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open slices that reset feature state at irregular gaps."""
    irregular = np.asarray(irregular_gap_after, dtype=np.bool_)
    n = len(irregular)
    if n == 0:
        return []
    starts = [0]
    starts.extend(int(i) for i in np.flatnonzero(irregular) if i > 0)
    starts = sorted(set(starts))
    stops = starts[1:] + [n]
    return list(zip(starts, stops))


__all__ = [
    "ENTRY_COOLDOWN_BARS",
    "IRREGULAR_GAP_MIN_SECONDS",
    "RESEARCH_START",
    "add_gap_provenance",
    "filter_research_window",
    "research_segments",
]
