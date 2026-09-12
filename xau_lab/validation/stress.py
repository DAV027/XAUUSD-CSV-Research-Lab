from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from xau_lab.backtest.models import MarketBars
from xau_lab.experiments.spec import CompleteExperiment, canonical_json
from xau_lab.runner.single import MarketBundle, run_experiment
from xau_lab.strategies.registry import get_strategy


@dataclass(frozen=True)
class StressConfig:
    slippage_points: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    commission_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5)
    spread_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5)
    parameter_shifts: tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20)


@dataclass(frozen=True)
class StressResult:
    experiment_id: str
    stress_type: str
    label: str
    pf: float | None
    expectancy_usd: float | None
    net_profit: float
    max_drawdown_pct: float
    parameter_name: str | None = None
    parameter_value: object | None = None
    slippage_points: float | None = None
    commission_multiplier: float | None = None
    spread_multiplier: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


STRESS_RESULT_FIELDS = tuple(field.name for field in fields(StressResult))


@dataclass(frozen=True)
class StressReport:
    results: tuple[StressResult, ...]
    parameter_stability_pct: float
    cost_stability_pct: float
    max_drawdown_pct: float
    parameter_run_count: int
    cost_run_count: int


def _is_numeric_domain(domain: object) -> bool:
    return (
        isinstance(domain, tuple)
        and len(domain) == 2
        and all(isinstance(value, Real) and not isinstance(value, bool) for value in domain)
    )


def parameter_neighbors(
    experiment_or_params: CompleteExperiment | Mapping[str, object],
    domains: Mapping[str, object] | None = None,
) -> list[dict]:
    if isinstance(experiment_or_params, CompleteExperiment):
        if domains is not None:
            raise ValueError("domains must be omitted when passing a CompleteExperiment")
        params: Mapping[str, object] = experiment_or_params.parameters
        domains = get_strategy(experiment_or_params.strategy_name).parameter_domain
    else:
        params = experiment_or_params
        if domains is None:
            raise ValueError("domains are required when passing a parameter mapping")

    rows: list[dict] = []
    seen: set[str] = set()
    for key in sorted(params):
        if key not in domains or not _is_numeric_domain(domains[key]):
            continue
        current = params[key]
        if not isinstance(current, Real) or isinstance(current, bool):
            continue
        low, high = domains[key]
        integer_parameter = isinstance(current, int) and not isinstance(current, bool)
        for shift in (-0.20, -0.10, 0.10, 0.20):
            raw = float(current) * (1.0 + shift)
            clipped = min(float(high), max(float(low), raw))
            value: object
            if integer_parameter:
                value = int(round(clipped))
                value = min(int(high), max(int(low), int(value)))
            else:
                value = float(round(clipped, 12))
            if value == current:
                continue
            row = dict(params)
            row[key] = value
            marker = canonical_json(row)
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(row)
    return rows


def _format_number(value: float) -> str:
    return f"{float(value):g}"


def _spread_market(market: MarketBundle, multiplier: float) -> MarketBundle:
    bars = market.bars
    spread = np.rint(bars.spread.astype(np.float64) * float(multiplier)).astype(np.int64)
    stressed = MarketBars(
        time_epoch=bars.time_epoch,
        open=bars.open,
        high=bars.high,
        low=bars.low,
        close=bars.close,
        spread=spread,
        atr=bars.atr,
        broker_timezone=bars.broker_timezone,
    )
    return MarketBundle(
        bars=stressed,
        symbol=market.symbol,
        broker_date=market.broker_date,
        features=market.features,
    )


def _evaluate(
    experiment: CompleteExperiment,
    market: MarketBundle,
    *,
    stress_type: str,
    label: str,
    parameter_name: str | None = None,
    parameter_value: object | None = None,
    slippage_points: float | None = None,
    commission_multiplier: float | None = None,
    spread_multiplier: float | None = None,
) -> StressResult:
    outcome = run_experiment(experiment, market)
    if not outcome.ok or outcome.master_result is None:
        raise RuntimeError(outcome.error_message or f"stress run failed: {label}")
    metrics = outcome.master_result
    pf = metrics.get("profit_factor")
    expectancy = metrics.get("expectancy_usd")
    return StressResult(
        experiment_id=experiment.experiment_id,
        stress_type=stress_type,
        label=label,
        pf=None if pf is None else float(pf),
        expectancy_usd=None if expectancy is None else float(expectancy),
        net_profit=float(metrics.get("net_profit") or 0.0),
        max_drawdown_pct=float(metrics.get("max_drawdown_pct") or 0.0),
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        slippage_points=slippage_points,
        commission_multiplier=commission_multiplier,
        spread_multiplier=spread_multiplier,
    )


def stress_candidate(
    candidate: CompleteExperiment,
    data: MarketBundle,
    config: StressConfig,
) -> StressReport:
    results: list[StressResult] = []

    neighbors = parameter_neighbors(candidate)
    for params in neighbors:
        changed = [key for key in candidate.parameters if params.get(key) != candidate.parameters.get(key)]
        if len(changed) != 1:
            raise RuntimeError("parameter stress must change exactly one parameter")
        key = changed[0]
        stressed = replace(
            candidate,
            parameters=params,
            canonical_parameters_json=canonical_json(params),
        )
        value = params[key]
        results.append(
            _evaluate(
                stressed,
                data,
                stress_type="parameter",
                label=f"parameter:{key}:{value}",
                parameter_name=key,
                parameter_value=value,
            )
        )

    for slippage in config.slippage_points:
        stressed = replace(candidate, slippage_points_per_fill=float(slippage))
        results.append(
            _evaluate(
                stressed,
                data,
                stress_type="cost",
                label=f"slippage:{_format_number(slippage)}",
                slippage_points=float(slippage),
            )
        )

    for multiplier in config.commission_multipliers:
        stressed = replace(
            candidate,
            commission_round_trip_per_lot=(
                float(candidate.commission_round_trip_per_lot) * float(multiplier)
            ),
        )
        results.append(
            _evaluate(
                stressed,
                data,
                stress_type="cost",
                label=f"commission:{_format_number(multiplier)}",
                commission_multiplier=float(multiplier),
            )
        )

    for multiplier in config.spread_multipliers:
        results.append(
            _evaluate(
                candidate,
                _spread_market(data, float(multiplier)),
                stress_type="cost",
                label=f"spread:{_format_number(multiplier)}",
                spread_multiplier=float(multiplier),
            )
        )

    parameter_rows = [row for row in results if row.stress_type == "parameter"]
    cost_rows = [row for row in results if row.stress_type == "cost"]
    parameter_passes = sum(
        row.pf is not None
        and row.pf > 1.0
        and row.expectancy_usd is not None
        and row.expectancy_usd > 0.0
        for row in parameter_rows
    )
    cost_passes = sum(row.pf is not None and row.pf > 1.0 for row in cost_rows)
    parameter_stability = (
        100.0 * parameter_passes / len(parameter_rows) if parameter_rows else 100.0
    )
    cost_stability = 100.0 * cost_passes / len(cost_rows) if cost_rows else 100.0
    max_drawdown = max((row.max_drawdown_pct for row in results), default=0.0)

    return StressReport(
        results=tuple(results),
        parameter_stability_pct=float(parameter_stability),
        cost_stability_pct=float(cost_stability),
        max_drawdown_pct=float(max_drawdown),
        parameter_run_count=len(parameter_rows),
        cost_run_count=len(cost_rows),
    )


def run_stress_suite(
    experiment: CompleteExperiment,
    market_bundle: MarketBundle,
    config: StressConfig | None = None,
) -> StressReport:
    return stress_candidate(experiment, market_bundle, config or StressConfig())


def write_stress_results(path: str | Path, results: Iterable[StressResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    rows = list(results)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRESS_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output)


__all__ = [
    "STRESS_RESULT_FIELDS",
    "StressConfig",
    "StressReport",
    "StressResult",
    "parameter_neighbors",
    "run_stress_suite",
    "stress_candidate",
    "write_stress_results",
]
