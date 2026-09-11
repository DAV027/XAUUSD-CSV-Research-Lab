from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from xau_lab.data.schema import canonicalize_bar_frame

ISSUE_COLUMNS = ["issue_type", "time", "severity", "details"]
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class ValidationResult:
    clean: pl.DataFrame
    issues: pl.DataFrame
    summary: dict


def _invalid_ohlc_expr() -> pl.Expr:
    prices = [pl.col(name) for name in ("open", "high", "low", "close")]
    any_null = pl.any_horizontal([expr.is_null() for expr in prices])
    any_non_positive = pl.any_horizontal([expr <= 0 for expr in prices])
    return (
        any_null
        | any_non_positive
        | (pl.col("high") < pl.max_horizontal(pl.col("open"), pl.col("close")))
        | (pl.col("low") > pl.min_horizontal(pl.col("open"), pl.col("close")))
        | (pl.col("low") > pl.col("high"))
    ).fill_null(True)


def _invalid_spread_expr() -> pl.Expr:
    return (pl.col("spread").is_null() | (pl.col("spread") < 0)).fill_null(True)


def _negative_volume_expr() -> pl.Expr:
    return ((pl.col("tick_volume") < 0) | (pl.col("real_volume") < 0)).fill_null(False)


def _crosses_weekend(start: datetime, end: datetime) -> bool:
    day = start.date()
    last = end.date()
    while day <= last:
        if day.weekday() >= 5:
            return True
        day += timedelta(days=1)
    return False


def _issue_frame(rows: list[dict[str, str]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {name: [] for name in ISSUE_COLUMNS},
            schema={name: pl.String for name in ISSUE_COLUMNS},
        )
    return pl.DataFrame(rows).select(ISSUE_COLUMNS).with_columns(
        [pl.col(name).cast(pl.String) for name in ISSUE_COLUMNS]
    )


def _append_mask_issues(
    issues: list[dict[str, str]],
    frame: pl.DataFrame,
    mask: pl.Expr,
    issue_type: str,
    severity: str,
    details: str,
) -> None:
    for timestamp in frame.filter(mask).get_column("time").to_list():
        issues.append(
            {
                "issue_type": issue_type,
                "time": str(timestamp),
                "severity": severity,
                "details": details,
            }
        )


def _append_gap_issues(issues: list[dict[str, str]], frame: pl.DataFrame) -> None:
    times = frame.get_column("time").to_list()
    if len(times) < 2:
        return

    previous = datetime.strptime(str(times[0]), _TIME_FORMAT)
    for raw_time in times[1:]:
        current = datetime.strptime(str(raw_time), _TIME_FORMAT)
        gap_seconds = int((current - previous).total_seconds())
        if gap_seconds > 60:
            issue_type = "market_closed_gap" if _crosses_weekend(previous, current) else "unexpected_gap"
            issues.append(
                {
                    "issue_type": issue_type,
                    "time": current.strftime(_TIME_FORMAT),
                    "severity": "info" if issue_type == "market_closed_gap" else "warning",
                    "details": f"gap_seconds={gap_seconds}; previous={previous.strftime(_TIME_FORMAT)}",
                }
            )
        previous = current


def _append_repeated_bar_issues(issues: list[dict[str, str]], frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return

    fields = ["time", "open", "high", "low", "close", "spread"]
    run_start_time: str | None = None
    previous_signature = None
    run_length = 0

    def flush() -> None:
        if run_length >= 10 and run_start_time is not None:
            issues.append(
                {
                    "issue_type": "repeated_bar_run",
                    "time": run_start_time,
                    "severity": "warning",
                    "details": f"consecutive_identical_ohlc_spread={run_length}",
                }
            )

    for row in frame.select(fields).iter_rows():
        timestamp = str(row[0])
        signature = row[1:]
        if signature == previous_signature:
            run_length += 1
        else:
            flush()
            previous_signature = signature
            run_start_time = timestamp
            run_length = 1
    flush()


def validate_bars(frame: pl.DataFrame) -> ValidationResult:
    canonical = canonicalize_bar_frame(frame)
    input_times = [str(value) for value in canonical.get_column("time").to_list()]
    non_monotonic_input = input_times != sorted(input_times)
    issues: list[dict[str, str]] = []

    duplicate_mask = pl.col("time").is_duplicated()
    _append_mask_issues(
        issues,
        canonical,
        duplicate_mask,
        "duplicate_timestamp",
        "warning",
        "duplicate timestamp; clean output keeps the last provider row",
    )
    _append_mask_issues(
        issues,
        canonical,
        _invalid_ohlc_expr(),
        "invalid_ohlc",
        "error",
        "OHLC values violate positivity or bar consistency rules",
    )
    _append_mask_issues(
        issues,
        canonical,
        _invalid_spread_expr(),
        "invalid_spread",
        "error",
        "spread is null or negative",
    )
    _append_mask_issues(
        issues,
        canonical,
        _negative_volume_expr(),
        "negative_volume",
        "error",
        "tick_volume or real_volume is negative",
    )

    deduped = canonical.unique(subset=["time"], keep="last", maintain_order=True).sort("time")

    # Gap detection intentionally uses the timestamp stream after deduplication but
    # before row-value rejection. This preserves evidence of genuinely missing M1
    # timestamps even when a neighboring bar is itself invalid; no rows are filled.
    _append_gap_issues(issues, deduped)

    invalid_clean_row = _invalid_ohlc_expr() | _invalid_spread_expr() | _negative_volume_expr()
    clean = deduped.filter(~invalid_clean_row).sort("time")
    _append_repeated_bar_issues(issues, clean)

    issue_counts = Counter(row["issue_type"] for row in issues)
    summary = {
        "input_rows": canonical.height,
        "clean_rows": clean.height,
        "removed_rows": canonical.height - clean.height,
        "non_monotonic_input": non_monotonic_input,
        "issue_counts": dict(sorted(issue_counts.items())),
    }
    return ValidationResult(clean=clean, issues=_issue_frame(issues), summary=summary)
