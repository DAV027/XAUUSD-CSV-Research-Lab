from __future__ import annotations

from math import log
from typing import Iterable, Mapping

SCORE_VERSION = "v1"

SCORE_MANIFEST = {
    "score_version": SCORE_VERSION,
    "components": {
        "profit_factor": {"weight": 25.0, "min": 1.10, "max": 1.50},
        "expectancy_R": {"weight": 20.0, "min": 0.0, "max": 0.20},
        "median_profit_per_active_day_percentile": {"weight": 15.0, "min": 0.0, "max": 1.0},
        "chronological_stability": {"weight": 15.0, "min": 0.0, "max": 1.0},
        "sample_size": {"weight": 10.0, "min_trades": 300, "max_trades": 3000, "scale": "log"},
        "robustness": {"weight": 15.0, "status_before_validation": None},
    },
    "penalties": {
        "drawdown": {"max_points": 15.0, "max_drawdown_pct": 5.0},
        "concentration": {
            "max_points": 15.0,
            "top_five_profit_fraction_max": 0.50,
            "best_month_profit_fraction_max": 0.60,
            "best_month_min_active_months": 6,
        },
        "side_dependence": {
            "max_points": 10.0,
            "min_trades_per_side": 100,
            "weak_side_pf": 0.90,
        },
    },
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _number(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _linear_component(value: float | None, low: float, high: float, weight: float) -> float:
    if value is None or high <= low:
        return 0.0
    return weight * _clip((value - low) / (high - low))


def _stability_component(metrics: Mapping[str, object]) -> float:
    available = [
        value
        for value in (
            _number(metrics, "positive_year_fraction"),
            _number(metrics, "positive_month_fraction"),
        )
        if value is not None
    ]
    if not available:
        return 0.0
    stability = sum(_clip(value) for value in available) / len(available)
    return 15.0 * stability


def _sample_size_component(completed_trades: float | None) -> float:
    if completed_trades is None or completed_trades <= 300.0:
        return 0.0
    ratio = log(completed_trades / 300.0) / log(3000.0 / 300.0)
    return 10.0 * _clip(ratio)


def _drawdown_penalty(max_drawdown_pct: float | None) -> float:
    if max_drawdown_pct is None:
        return 15.0
    return 15.0 * _clip(max_drawdown_pct / 5.0)


def _concentration_penalty(metrics: Mapping[str, object]) -> float:
    top_five = _number(metrics, "top_5_trade_profit_fraction")
    best_month = _number(metrics, "best_month_profit_fraction")
    active_months = _number(metrics, "active_months")

    ratios: list[float] = []
    if top_five is not None:
        ratios.append(top_five / 0.50)
    if best_month is not None and active_months is not None and active_months >= 6.0:
        ratios.append(best_month / 0.60)
    if not ratios:
        return 0.0
    return 15.0 * _clip(max(ratios))


def _side_dependence_penalty(metrics: Mapping[str, object]) -> float:
    long_trades = _number(metrics, "long_trades")
    short_trades = _number(metrics, "short_trades")
    if long_trades is None or short_trades is None or long_trades < 100.0 or short_trades < 100.0:
        return 0.0

    numeric_pfs = [
        value
        for value in (_number(metrics, "long_PF"), _number(metrics, "short_PF"))
        if value is not None
    ]
    if len(numeric_pfs) != 2:
        return 0.0
    weak_pf = min(numeric_pfs)
    if weak_pf >= 0.90:
        return 0.0
    return 10.0 * _clip((0.90 - weak_pf) / 0.90)


def score_candidate(metrics: Mapping[str, object]) -> dict[str, object]:
    pf_component = _linear_component(_number(metrics, "profit_factor"), 1.10, 1.50, 25.0)
    expectancy_component = _linear_component(_number(metrics, "expectancy_R"), 0.0, 0.20, 20.0)
    median_daily_component = 15.0 * _clip(
        _number(metrics, "median_profit_per_active_day_percentile") or 0.0
    )
    stability_component = _stability_component(metrics)
    sample_size_component = _sample_size_component(_number(metrics, "completed_trades"))

    drawdown_penalty = _drawdown_penalty(_number(metrics, "max_drawdown_pct"))
    concentration_penalty = _concentration_penalty(metrics)
    side_dependence_penalty = _side_dependence_penalty(metrics)

    pre_robustness = _clip(
        pf_component
        + expectancy_component
        + median_daily_component
        + stability_component
        + sample_size_component
        - drawdown_penalty
        - concentration_penalty
        - side_dependence_penalty,
        0.0,
        85.0,
    )

    robustness_raw = _number(metrics, "robustness_component")
    robustness_component = None if robustness_raw is None else _clip(robustness_raw, 0.0, 15.0)
    final_score = (
        None
        if robustness_component is None
        else _clip(pre_robustness + robustness_component, 0.0, 100.0)
    )

    result: dict[str, object] = {
        "score_version": SCORE_VERSION,
        "pf_component": float(pf_component),
        "expectancy_component": float(expectancy_component),
        "median_daily_component": float(median_daily_component),
        "stability_component": float(stability_component),
        "sample_size_component": float(sample_size_component),
        "robustness_component": robustness_component,
        "drawdown_penalty": float(drawdown_penalty),
        "concentration_penalty": float(concentration_penalty),
        "side_dependence_penalty": float(side_dependence_penalty),
        "final_score_pre_robustness": float(pre_robustness),
        "final_score": final_score,
    }
    for key in ("experiment_id", "median_profit_per_active_day_percentile"):
        if key in metrics:
            result[key] = metrics[key]
    return result


def _percentile_map(rows: list[Mapping[str, object]]) -> dict[object, float]:
    values: list[tuple[object, float]] = []
    for index, row in enumerate(rows):
        key = row.get("experiment_id", index)
        value = _number(row, "median_profit_per_active_day")
        if value is None:
            raise ValueError("median_profit_per_active_day is required for survivor-set scoring")
        values.append((key, value))

    if not values:
        return {}
    if len(values) == 1:
        return {values[0][0]: 1.0}

    ordered = sorted(values, key=lambda item: (item[1], str(item[0])))
    positions: dict[float, list[int]] = {}
    for position, (_, value) in enumerate(ordered):
        positions.setdefault(value, []).append(position)
    percentile_by_value = {
        value: (sum(group) / len(group)) / (len(ordered) - 1)
        for value, group in positions.items()
    }
    return {key: float(percentile_by_value[value]) for key, value in values}


def score_survivor_set(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    row_list = [dict(row) for row in rows]
    percentiles = _percentile_map(row_list)
    scored: list[dict[str, object]] = []
    for index, row in enumerate(row_list):
        key = row.get("experiment_id", index)
        enriched = dict(row)
        enriched["median_profit_per_active_day_percentile"] = percentiles[key]
        result = dict(enriched)
        result.update(score_candidate(enriched))
        scored.append(result)
    return scored


__all__ = ["SCORE_MANIFEST", "SCORE_VERSION", "score_candidate", "score_survivor_set"]
