from __future__ import annotations

from typing import Iterable, Mapping

EDGE_B_V2_MIN_PROFIT_PER_ACTIVE_DAY = 50.0
EDGE_B_V2_MAX_CANDIDATES = 6


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_edge_b_v2_candidates(
    rows: Iterable[Mapping[str, object]],
    *,
    min_profit_per_active_day: float = EDGE_B_V2_MIN_PROFIT_PER_ACTIVE_DAY,
    max_candidates: int = EDGE_B_V2_MAX_CANDIDATES,
) -> list[dict[str, object]]:
    if min_profit_per_active_day <= 0.0:
        raise ValueError("min_profit_per_active_day must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    eligible: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        avg_day = _number(row, "profit_per_active_day")
        median_day = _number(row, "median_profit_per_active_day")
        score = _number(row, "final_score")
        experiment_id = str(row.get("experiment_id", ""))

        if not experiment_id or score is None:
            continue
        if avg_day is None or avg_day < min_profit_per_active_day:
            continue
        if median_day is None or median_day <= 0.0:
            continue
        eligible.append(row)

    eligible.sort(
        key=lambda row: (
            -float(row["final_score"]),
            str(row["experiment_id"]),
        )
    )
    return eligible[:max_candidates]


__all__ = [
    "EDGE_B_V2_MAX_CANDIDATES",
    "EDGE_B_V2_MIN_PROFIT_PER_ACTIVE_DAY",
    "select_edge_b_v2_candidates",
]
