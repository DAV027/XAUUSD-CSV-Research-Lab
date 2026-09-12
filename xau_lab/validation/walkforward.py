from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable

import numpy as np

from xau_lab.backtest.models import MarketBars
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.runner.single import ExperimentOutcome, MarketBundle, run_experiment
from xau_lab.validation.folds import ValidationFold, expanding_folds, rolling_folds


@dataclass(frozen=True)
class FoldResult:
    experiment_id: str
    scheme: str
    fold_id: str
    segment: str
    start: str
    end: str
    trades: int
    pf: float | None
    expectancy_usd: float | None
    net_profit: float
    max_drawdown_pct: float
    positive_day_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _slice_market(market: MarketBundle, months: tuple[str, ...]) -> MarketBundle:
    month_set = set(months)
    mask = np.asarray([str(value)[:7] in month_set for value in market.broker_date], dtype=bool)
    if not bool(mask.any()):
        raise ValueError("fold contains no market bars")
    bars = market.bars
    sliced_bars = MarketBars(
        time_epoch=bars.time_epoch[mask],
        open=bars.open[mask],
        high=bars.high[mask],
        low=bars.low[mask],
        close=bars.close[mask],
        spread=bars.spread[mask],
        atr=bars.atr[mask],
        broker_timezone=bars.broker_timezone,
    )
    return MarketBundle(
        bars=sliced_bars,
        symbol=market.symbol,
        broker_date=market.broker_date[mask],
        features={name: values[mask] for name, values in market.features.items()},
    )


def _positive_day_fraction(outcome: ExperimentOutcome) -> float | None:
    if not outcome.trades:
        return None
    daily: dict[str, float] = defaultdict(float)
    for trade in outcome.trades:
        daily[trade.broker_date] += float(trade.net_pnl)
    if not daily:
        return None
    return float(sum(value > 0.0 for value in daily.values()) / len(daily))


def _evaluate_segment(
    experiment: CompleteExperiment,
    market: MarketBundle,
    fold: ValidationFold,
    segment: str,
) -> FoldResult:
    if segment == "research":
        months = fold.research_months
        start, end = fold.research_start, fold.research_end
    elif segment == "validation":
        months = fold.validation_months
        start, end = fold.validation_start, fold.validation_end
    else:
        raise ValueError(f"unknown fold segment: {segment}")

    # The experiment object is reused unchanged in every fold. No parameter
    # selection or optimization is allowed inside this validation layer.
    outcome = run_experiment(experiment, _slice_market(market, months), include_trades=True)
    if not outcome.ok or outcome.master_result is None:
        raise RuntimeError(outcome.error_message or "walk-forward experiment failed")
    metrics = outcome.master_result
    pf = metrics.get("profit_factor")
    expectancy = metrics.get("expectancy_usd")
    return FoldResult(
        experiment_id=experiment.experiment_id,
        scheme=fold.scheme,
        fold_id=fold.fold_id,
        segment=segment,
        start=start,
        end=end,
        trades=int(metrics.get("completed_trades") or 0),
        pf=None if pf is None else float(pf),
        expectancy_usd=None if expectancy is None else float(expectancy),
        net_profit=float(metrics.get("net_profit") or 0.0),
        max_drawdown_pct=float(metrics.get("max_drawdown_pct") or 0.0),
        positive_day_fraction=_positive_day_fraction(outcome),
    )


def validate_candidate(
    experiment: CompleteExperiment,
    market_bundle: MarketBundle,
    scheme: str,
) -> list[FoldResult]:
    if scheme == "expanding":
        folds = expanding_folds(market_bundle.broker_date)
    elif scheme == "rolling":
        folds = rolling_folds(market_bundle.broker_date)
    else:
        raise ValueError("scheme must be 'expanding' or 'rolling'")

    results: list[FoldResult] = []
    for fold in folds:
        results.append(_evaluate_segment(experiment, market_bundle, fold, "research"))
        results.append(_evaluate_segment(experiment, market_bundle, fold, "validation"))
    return results


def _median_numeric(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(median(numeric)) if numeric else None


def summarize_fold_results(results: Iterable[FoldResult]) -> dict[str, float | int | None]:
    rows = list(results)
    validation = [row for row in rows if row.segment == "validation"]
    research = [row for row in rows if row.segment == "research"]
    count = len(validation)

    validation_pf_median = _median_numeric(row.pf for row in validation)
    research_pf_median = _median_numeric(row.pf for row in research)
    pf_above = sum(row.pf is not None and row.pf > 1.0 for row in validation)
    expectancy_positive = sum(
        row.expectancy_usd is not None and row.expectancy_usd > 0.0 for row in validation
    )
    joint = sum(
        row.pf is not None
        and row.pf > 1.0
        and row.expectancy_usd is not None
        and row.expectancy_usd > 0.0
        for row in validation
    )

    degradation = None
    if (
        research_pf_median is not None
        and research_pf_median > 0.0
        and validation_pf_median is not None
    ):
        degradation = float(validation_pf_median / research_pf_median)

    return {
        "validation_fold_count": count,
        "median_validation_pf": validation_pf_median,
        "validation_pf_above_1_fraction": (pf_above / count) if count else None,
        "validation_positive_expectancy_fraction": (
            expectancy_positive / count if count else None
        ),
        "validation_joint_stability_fraction": (joint / count) if count else None,
        "worst_validation_net_profit": (
            float(min(row.net_profit for row in validation)) if validation else None
        ),
        "research_to_validation_pf_degradation_ratio": degradation,
    }


__all__ = ["FoldResult", "summarize_fold_results", "validate_candidate"]
