from __future__ import annotations

import math
from typing import Mapping, Sequence

EDGE_C_V2_PF_MIN = 1.10
EDGE_C_V2_MAX_DRAWDOWN_PCT = 5.0
EDGE_C_V2_MAX_CANDIDATES = 6


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def select_edge_c_v2_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    max_candidates: int = EDGE_C_V2_MAX_CANDIDATES,
) -> list[dict[str, object]]:
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")

    eligible: list[dict[str, object]] = []
    for source in rows:
        experiment_id = source.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        net = _number(source, "net_profit")
        pf = _number(source, "profit_factor")
        expectancy = _number(source, "expectancy_usd")
        drawdown = _number(source, "max_drawdown_pct")
        score = _number(source, "final_score")
        if None in (net, pf, expectancy, drawdown, score):
            continue
        if net <= 0.0:
            continue
        if pf < EDGE_C_V2_PF_MIN:
            continue
        if expectancy <= 0.0:
            continue
        if drawdown > EDGE_C_V2_MAX_DRAWDOWN_PCT:
            continue
        eligible.append(dict(source))

    eligible.sort(
        key=lambda row: (
            -float(row["final_score"]),
            str(row["experiment_id"]),
        )
    )
    return eligible[:max_candidates]


__all__ = [
    "EDGE_C_V2_MAX_CANDIDATES",
    "EDGE_C_V2_MAX_DRAWDOWN_PCT",
    "EDGE_C_V2_PF_MIN",
    "select_edge_c_v2_candidates",
]
