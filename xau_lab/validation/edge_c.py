from __future__ import annotations

import math
from typing import Iterable, Mapping

EDGE_C_MIN_PROFIT_FACTOR = 1.10
EDGE_C_MAX_DRAWDOWN_PCT = 5.0
EDGE_C_MAX_CANDIDATES = 6


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def select_edge_c_candidates(
    rows: Iterable[Mapping[str, object]],
    *,
    max_candidates: int = EDGE_C_MAX_CANDIDATES,
) -> list[dict[str, object]]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    eligible: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        experiment_id = str(row.get("experiment_id", ""))
        net_profit = _number(row, "net_profit")
        profit_factor = _number(row, "profit_factor")
        expectancy = _number(row, "expectancy_usd")
        drawdown = _number(row, "max_drawdown_pct")
        score = _number(row, "final_score")

        if not experiment_id or None in (net_profit, profit_factor, expectancy, drawdown, score):
            continue
        if net_profit <= 0.0:
            continue
        if profit_factor < EDGE_C_MIN_PROFIT_FACTOR:
            continue
        if expectancy <= 0.0:
            continue
        if drawdown > EDGE_C_MAX_DRAWDOWN_PCT:
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
    "EDGE_C_MAX_CANDIDATES",
    "EDGE_C_MAX_DRAWDOWN_PCT",
    "EDGE_C_MIN_PROFIT_FACTOR",
    "select_edge_c_candidates",
]
