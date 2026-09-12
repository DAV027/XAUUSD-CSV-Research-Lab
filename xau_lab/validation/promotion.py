from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

PF_MIN = 1.10
MAX_DRAWDOWN_PCT = 5.0
MIN_COMPLETED_TRADES = 300
TOP_FIVE_PROFIT_FRACTION_MAX = 0.50
BEST_MONTH_PROFIT_FRACTION_MAX = 0.60
BEST_MONTH_MIN_ACTIVE_MONTHS = 6


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: Sequence[str]


def _number(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stage1_decision(
    metrics: Mapping[str, object],
    integrity_ok: bool = True,
) -> PromotionDecision:
    reasons: list[str] = []

    pf = _number(metrics, "profit_factor")
    if pf is None or pf < PF_MIN:
        reasons.append("pf_below_1_10")

    max_dd = _number(metrics, "max_drawdown_pct")
    if max_dd is None or max_dd > MAX_DRAWDOWN_PCT:
        reasons.append("drawdown_above_5_pct")

    completed = _number(metrics, "completed_trades")
    if completed is None or completed < MIN_COMPLETED_TRADES:
        reasons.append("completed_trades_below_300")

    expectancy = _number(metrics, "expectancy_usd")
    if expectancy is None or expectancy <= 0.0:
        reasons.append("expectancy_not_positive")

    positive_year_fraction = _number(metrics, "positive_year_fraction")
    if positive_year_fraction is not None and positive_year_fraction <= 0.50:
        reasons.append("profitable_years_not_majority")

    top_five = _number(metrics, "top_5_trade_profit_fraction")
    if top_five is None:
        reasons.append("top_five_profit_concentration_missing")
    elif top_five > TOP_FIVE_PROFIT_FRACTION_MAX:
        reasons.append("top_five_profit_concentration_above_50_pct")

    best_month = _number(metrics, "best_month_profit_fraction")
    active_months = _number(metrics, "active_months")
    if active_months is None:
        reasons.append("active_months_missing")
    elif active_months >= BEST_MONTH_MIN_ACTIVE_MONTHS:
        if best_month is None:
            reasons.append("best_month_profit_concentration_missing")
        elif best_month > BEST_MONTH_PROFIT_FRACTION_MAX:
            reasons.append("best_month_profit_concentration_above_60_pct")

    if not integrity_ok:
        reasons.append("integrity_failure")

    return PromotionDecision(passed=not reasons, reasons=tuple(reasons))


__all__ = [
    "BEST_MONTH_MIN_ACTIVE_MONTHS",
    "BEST_MONTH_PROFIT_FRACTION_MAX",
    "MAX_DRAWDOWN_PCT",
    "MIN_COMPLETED_TRADES",
    "PF_MIN",
    "PromotionDecision",
    "TOP_FIVE_PROFIT_FRACTION_MAX",
    "stage1_decision",
]
