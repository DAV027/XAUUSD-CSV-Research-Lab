from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from xau_lab.validation.reporting import RESEARCH_ONLY_SCOPE

SECONDS_PER_WEEK = 7.0 * 24.0 * 60.0 * 60.0


def _finite_number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Edge B robust row requires {key}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Edge B robust row requires finite {key}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Edge B robust row requires finite {key}")
    return parsed


def select_edge_b_candidates(
    rows: Iterable[Mapping[str, object]],
    *,
    min_trades_per_week: float = 3.0,
    max_trades_per_week: float = 10.0,
    max_candidates: int = 6,
) -> tuple[dict[str, object], ...]:
    if (
        not math.isfinite(float(min_trades_per_week))
        or not math.isfinite(float(max_trades_per_week))
        or min_trades_per_week < 0.0
        or max_trades_per_week < min_trades_per_week
    ):
        raise ValueError("invalid Edge B frequency bounds")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    eligible: list[tuple[float, str, dict[str, object]]] = []
    for source in rows:
        experiment_id = str(source.get("experiment_id") or "").strip()
        if not experiment_id:
            raise ValueError("Edge B robust row requires experiment_id")

        data_start = _finite_number(source, "data_start")
        data_end = _finite_number(source, "data_end")
        completed_trades = _finite_number(source, "completed_trades")
        final_score = _finite_number(source, "final_score")
        if completed_trades < 0.0:
            raise ValueError("completed_trades must be nonnegative")

        span_seconds = data_end - data_start
        if span_seconds <= 0.0:
            raise ValueError("Edge B robust row requires a positive discovery span")
        span_weeks = span_seconds / SECONDS_PER_WEEK
        trades_per_week = completed_trades / span_weeks

        if not (min_trades_per_week <= trades_per_week <= max_trades_per_week):
            continue

        row = dict(source)
        row["edge_b_frequency_trades_per_week"] = trades_per_week
        row["approval_scope"] = RESEARCH_ONLY_SCOPE
        eligible.append((final_score, experiment_id, row))

    eligible.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in eligible[:max_candidates])


__all__ = ["select_edge_b_candidates"]
