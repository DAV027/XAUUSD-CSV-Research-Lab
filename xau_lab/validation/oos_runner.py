from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.backtest.models import CostModel, MarketBars, RiskModel
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.metrics.performance import summarize_trades
from xau_lab.runner.single import (
    ExperimentOutcome,
    MarketBundle,
    _annotate_trade,
    _direction_filter,
    _exit_spec,
)
from xau_lab.strategies.registry import get_strategy


def _parse_utc_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp")
    return parsed.astimezone(timezone.utc)


def load_holdout_plan(path: str | Path) -> dict[str, object]:
    plan_path = Path(path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported OOS holdout schema_version")
    if payload.get("planned_months") != 3:
        raise ValueError("planned_months must remain 3")

    freeze = _parse_utc_timestamp(payload.get("freeze_timestamp"), "freeze_timestamp")
    start = _parse_utc_timestamp(payload.get("prospective_start_utc"), "prospective_start_utc")
    end = _parse_utc_timestamp(
        payload.get("planned_end_exclusive_utc"), "planned_end_exclusive_utc"
    )
    start_epoch = payload.get("prospective_start_epoch")
    end_epoch = payload.get("planned_end_exclusive_epoch")
    selection_end = payload.get("selection_data_end_epoch")
    if not all(isinstance(value, int) for value in (start_epoch, end_epoch, selection_end)):
        raise ValueError("holdout epoch values must be integers")
    if int(start.timestamp()) != start_epoch:
        raise ValueError("prospective_start_utc does not match prospective_start_epoch")
    if int(end.timestamp()) != end_epoch:
        raise ValueError("planned_end_exclusive_utc does not match planned_end_exclusive_epoch")
    if start <= freeze:
        raise ValueError("prospective start must be after freeze")
    if start_epoch <= selection_end:
        raise ValueError("prospective start must be after selection data end")
    if end <= start:
        raise ValueError("planned holdout end must be after prospective start")
    return payload


def _market_before_end(market: MarketBundle, end_exclusive_epoch: int) -> MarketBundle:
    epochs = market.bars.time_epoch
    stop = int(np.searchsorted(epochs, end_exclusive_epoch, side="left"))
    if stop == len(epochs):
        return market
    bars = MarketBars(
        time_epoch=market.bars.time_epoch[:stop],
        open=market.bars.open[:stop],
        high=market.bars.high[:stop],
        low=market.bars.low[:stop],
        close=market.bars.close[:stop],
        spread=market.bars.spread[:stop],
        atr=market.bars.atr[:stop],
        broker_timezone=market.bars.broker_timezone,
    )
    return MarketBundle(
        bars=bars,
        symbol=market.symbol,
        broker_date=market.broker_date[:stop],
        features={name: values[:stop] for name, values in market.features.items()},
        broker_day_id=market.broker_day_id[:stop],
        broker_month_id=market.broker_month_id[:stop],
        broker_year_id=market.broker_year_id[:stop],
    )


def build_holdout_signals(
    experiment: CompleteExperiment,
    market: MarketBundle,
    *,
    start_epoch: int,
    end_exclusive_epoch: int,
) -> np.ndarray:
    if start_epoch >= end_exclusive_epoch:
        raise ValueError("OOS start must be before OOS end")
    epochs = market.bars.time_epoch
    if len(epochs) and np.any(epochs[1:] <= epochs[:-1]):
        raise ValueError("market timestamps must be strictly increasing")
    definition = get_strategy(experiment.strategy_name)
    signals = definition.generate(market.strategy_context(), experiment.parameters)
    signals = _direction_filter(signals, experiment.direction_mode)
    masked = np.ascontiguousarray(signals, dtype=np.int8).copy()
    outside = (epochs < start_epoch) | (epochs >= end_exclusive_epoch)
    masked[outside] = 0
    return masked


def run_oos_experiment(
    experiment: CompleteExperiment,
    market: MarketBundle,
    *,
    start_epoch: int,
    end_exclusive_epoch: int,
    risk_model: RiskModel | None = None,
) -> ExperimentOutcome:
    if start_epoch >= end_exclusive_epoch:
        raise ValueError("OOS start must be before OOS end")
    bounded_market = _market_before_end(market, end_exclusive_epoch)
    epochs = bounded_market.bars.time_epoch
    if not np.any(epochs >= start_epoch):
        raise ValueError("no prospective OOS rows are available in the snapshot")

    signals = build_holdout_signals(
        experiment,
        bounded_market,
        start_epoch=start_epoch,
        end_exclusive_epoch=end_exclusive_epoch,
    )
    cost = CostModel(
        commission_round_trip_per_lot=float(experiment.commission_round_trip_per_lot),
        slippage_points_per_fill=float(experiment.slippage_points_per_fill),
    )
    risk = risk_model or RiskModel()
    result = run_fast_backtest(
        bounded_market.bars,
        signals,
        bounded_market.symbol,
        cost,
        risk,
        _exit_spec(experiment),
    )
    trades = tuple(
        _annotate_trade(trade, experiment, bounded_market) for trade in result.trades
    )
    if any(
        trade.entry_time < start_epoch or trade.entry_time >= end_exclusive_epoch
        for trade in trades
    ):
        raise RuntimeError("OOS backtest produced a trade outside the frozen holdout window")
    metrics = summarize_trades(trades, starting_equity=risk.account_equity)
    observed_through = int(epochs[-1])
    master = {
        **experiment.to_dict(),
        "history_data_start": int(epochs[0]),
        "oos_start_epoch": int(start_epoch),
        "oos_end_exclusive_epoch": int(end_exclusive_epoch),
        "observed_through_epoch": observed_through,
        "risk_skip_count": int(result.risk_skip_count),
        **metrics,
    }
    return ExperimentOutcome(
        experiment_id=experiment.experiment_id,
        master_result=master,
        trades=trades,
    )


def _normalize_row(row: Mapping[str, object], fieldnames: list[str]) -> dict[str, str]:
    if set(row) != set(fieldnames):
        raise ValueError("OOS result row schema differs from the existing report")
    return {name: "" if row.get(name) is None else str(row.get(name)) for name in fieldnames}


def append_oos_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    report_path = Path(path)
    materialized = list(rows)
    if not materialized:
        return 0
    fieldnames = list(materialized[0].keys())
    if "experiment_id" not in fieldnames or "observed_through_epoch" not in fieldnames:
        raise ValueError("OOS rows require experiment_id and observed_through_epoch")
    normalized = [_normalize_row(row, fieldnames) for row in materialized]

    existing_rows: list[dict[str, str]] = []
    if report_path.exists() and report_path.stat().st_size:
        with report_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fieldnames:
                raise ValueError("OOS result row schema differs from the existing report")
            existing_rows = list(reader)

    existing = {
        (row["experiment_id"], row["observed_through_epoch"]): row
        for row in existing_rows
    }
    pending: list[dict[str, str]] = []
    for row in normalized:
        key = (row["experiment_id"], row["observed_through_epoch"])
        prior = existing.get(key)
        if prior is not None:
            if prior != row:
                raise ValueError(f"immutable OOS snapshot drift for {key[0]} at {key[1]}")
            continue
        if key in {(item["experiment_id"], item["observed_through_epoch"]) for item in pending}:
            raise ValueError(f"duplicate OOS snapshot row for {key[0]} at {key[1]}")
        pending.append(row)

    if not pending:
        return 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not report_path.exists() or report_path.stat().st_size == 0
    with report_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(pending)
        handle.flush()
        os.fsync(handle.fileno())
    return len(pending)


__all__ = [
    "append_oos_rows",
    "build_holdout_signals",
    "load_holdout_plan",
    "run_oos_experiment",
]
