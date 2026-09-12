from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from math import floor
from numbers import Real
from typing import Iterator, Mapping, Sequence


SHORT_HISTORY_RULE = "floor_80pct_min_6_validation_remainder_capped_3"


@dataclass(frozen=True)
class ValidationFold:
    scheme: str
    fold_id: str
    research_start: str
    research_end: str
    validation_start: str
    validation_end: str
    research_months: tuple[str, ...]
    validation_months: tuple[str, ...]
    fallback_rule: str | None = None


@dataclass(frozen=True)
class FoldSet(Sequence[ValidationFold]):
    folds: tuple[ValidationFold, ...]
    status: str = "ok"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.folds)

    def __iter__(self) -> Iterator[ValidationFold]:
        return iter(self.folds)

    def __getitem__(self, index):
        return self.folds[index]


def _month_label(value: object) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        return f"{value.year:04d}-{value.month:02d}"
    elif isinstance(value, Real):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if len(text) >= 7 and text[4] == "-":
            try:
                year = int(text[:4])
                month = int(text[5:7])
            except ValueError as exc:
                raise ValueError(f"unsupported time value: {value!r}") from exc
            if not 1 <= month <= 12:
                raise ValueError(f"unsupported time value: {value!r}")
            return f"{year:04d}-{month:02d}"
        raise ValueError(f"unsupported time value: {value!r}")
    else:
        # numpy integer/datetime-like scalars expose item(); recurse on the Python scalar.
        item = getattr(value, "item", None)
        if callable(item):
            converted = item()
            if converted is not value:
                return _month_label(converted)
        raise TypeError(f"unsupported time value type: {type(value).__name__}")
    return f"{dt.year:04d}-{dt.month:02d}"


def _complete_months(times: Sequence[object]) -> tuple[str, ...]:
    return tuple(sorted({_month_label(value) for value in times}))


def _fold(
    scheme: str,
    number: int,
    research: Sequence[str],
    validation: Sequence[str],
    *,
    fallback_rule: str | None = None,
) -> ValidationFold:
    if not research or not validation:
        raise ValueError("research and validation months must be nonempty")
    if research[-1] >= validation[0]:
        raise ValueError("validation must begin strictly after research")
    if set(research).intersection(validation):
        raise ValueError("research and validation months must not overlap")
    return ValidationFold(
        scheme=scheme,
        fold_id=f"{scheme}-{number:03d}",
        research_start=research[0],
        research_end=research[-1],
        validation_start=validation[0],
        validation_end=validation[-1],
        research_months=tuple(research),
        validation_months=tuple(validation),
        fallback_rule=fallback_rule,
    )


def _short_history(months: tuple[str, ...], scheme: str) -> FoldSet:
    count = len(months)
    metadata = {"complete_months": count, "scheme": scheme}
    if count < 7:
        return FoldSet((), status="insufficient_history", metadata=metadata)

    research_count = max(6, floor(0.80 * count))
    research_count = min(research_count, count - 1)
    validation_count = min(3, count - research_count)
    research = months[:research_count]
    validation = months[research_count : research_count + validation_count]
    metadata = metadata | {
        "fallback_rule": SHORT_HISTORY_RULE,
        "research_months": len(research),
        "validation_months": len(validation),
    }
    return FoldSet(
        (_fold(scheme, 1, research, validation, fallback_rule=SHORT_HISTORY_RULE),),
        status="fallback_short_history",
        metadata=metadata,
    )


def rolling_folds(
    times: Sequence[object],
    research_months: int = 12,
    validation_months: int = 3,
    step_months: int = 3,
) -> FoldSet:
    if research_months <= 0 or validation_months <= 0 or step_months <= 0:
        raise ValueError("fold month counts must be positive")
    months = _complete_months(times)
    if len(months) < research_months + validation_months:
        return _short_history(months, "rolling")

    folds: list[ValidationFold] = []
    start = 0
    number = 1
    while start + research_months + validation_months <= len(months):
        research = months[start : start + research_months]
        validation = months[
            start + research_months : start + research_months + validation_months
        ]
        folds.append(_fold("rolling", number, research, validation))
        start += step_months
        number += 1
    return FoldSet(
        tuple(folds),
        status="ok",
        metadata={
            "complete_months": len(months),
            "research_months": research_months,
            "validation_months": validation_months,
            "step_months": step_months,
        },
    )


def expanding_folds(
    times: Sequence[object],
    validation_months: int = 3,
    min_research_months: int = 12,
) -> FoldSet:
    if validation_months <= 0 or min_research_months <= 0:
        raise ValueError("fold month counts must be positive")
    months = _complete_months(times)
    if len(months) < min_research_months + validation_months:
        return _short_history(months, "expanding")

    folds: list[ValidationFold] = []
    research_end = min_research_months
    number = 1
    while research_end + validation_months <= len(months):
        research = months[:research_end]
        validation = months[research_end : research_end + validation_months]
        folds.append(_fold("expanding", number, research, validation))
        research_end += validation_months
        number += 1
    return FoldSet(
        tuple(folds),
        status="ok",
        metadata={
            "complete_months": len(months),
            "min_research_months": min_research_months,
            "validation_months": validation_months,
            "step_months": validation_months,
        },
    )


__all__ = [
    "FoldSet",
    "SHORT_HISTORY_RULE",
    "ValidationFold",
    "expanding_folds",
    "rolling_folds",
]
