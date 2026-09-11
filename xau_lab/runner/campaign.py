from __future__ import annotations

import csv
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from xau_lab.backtest.models import MarketBars, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.io.checkpoint import CheckpointStore
from xau_lab.io.results import ResultStore
from xau_lab.runner.single import ExperimentOutcome, MarketBundle, run_experiment

_WORKER_MARKET: MarketBundle | None = None


def _optional_number(value: str | None, cast):
    if value is None or value == "":
        return None
    return cast(value)


def _experiment_from_row(row: dict[str, str]) -> CompleteExperiment:
    parameters = json.loads(row["canonical_parameters_json"])
    return CompleteExperiment(
        experiment_id=row["experiment_id"],
        fingerprint=row["fingerprint"],
        allocation_bucket=row["allocation_bucket"],
        family=row["family"],
        strategy_name=row["strategy_name"],
        parameters=parameters,
        canonical_parameters_json=row["canonical_parameters_json"],
        direction_mode=row["direction_mode"],
        stop_atr=float(row["stop_atr"]),
        exit_type=row["exit_type"],
        target_r=_optional_number(row.get("target_r"), float),
        time_exit_minutes=_optional_number(row.get("time_exit_minutes"), int),
        atr_trail=_optional_number(row.get("atr_trail"), float),
        commission_round_trip_per_lot=float(row["commission_round_trip_per_lot"]),
        slippage_points_per_fill=float(row["slippage_points_per_fill"]),
        strategy_seed=int(row["strategy_seed"]),
        family_seed=int(row["family_seed"]),
        sampler_version=row.get("sampler_version") or "v1",
    )


def _read_catalog(path: Path) -> list[CompleteExperiment]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    catalog = [_experiment_from_row(row) for row in rows]
    ids = [item.experiment_id for item in catalog]
    if len(ids) != len(set(ids)):
        raise ValueError("catalog contains duplicate experiment_id values")
    return catalog


def _metadata_candidates(feature_path: Path) -> list[Path]:
    return [
        feature_path.with_suffix(".metadata.json"),
        feature_path.parent / "DATA_METADATA.json",
        feature_path.parent.parent.parent / "results" / "DATA_METADATA.json",
    ]


def _load_metadata(feature_path: Path) -> dict:
    for candidate in _metadata_candidates(feature_path):
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"symbol metadata must be a JSON object: {candidate}")
            return value
    raise FileNotFoundError(
        "symbol metadata is required; provide <feature>.metadata.json or canonical results/DATA_METADATA.json"
    )


def _column(frame: pl.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return frame[name].to_numpy()
    raise ValueError(f"feature dataset is missing required column; expected one of {names}")


def _time_epoch(frame: pl.DataFrame) -> np.ndarray:
    if "time_epoch" in frame.columns:
        return np.asarray(frame["time_epoch"].to_numpy(), dtype=np.int64)
    if "time" not in frame.columns:
        raise ValueError("feature dataset requires time_epoch or time")
    parsed = frame.select(pl.col("time").str.to_datetime(strict=True).dt.epoch("s").alias("epoch"))
    return np.asarray(parsed["epoch"].to_numpy(), dtype=np.int64)


def load_market_bundle(feature_path: str | Path) -> MarketBundle:
    feature_path = Path(feature_path)
    frame = pl.read_parquet(feature_path)
    metadata = _load_metadata(feature_path)
    required_meta = ["digits", "point", "trade_contract_size", "volume_min", "volume_step"]
    missing = [key for key in required_meta if metadata.get(key) is None]
    if missing:
        raise ValueError(f"symbol metadata missing required fields: {missing}")
    timezone = metadata.get("server_timezone")
    if not timezone:
        raise ValueError("symbol metadata requires explicit server_timezone")

    symbol = SymbolSpec(
        point=float(metadata["point"]),
        digits=int(metadata["digits"]),
        contract_size=float(metadata["trade_contract_size"]),
        volume_min=float(metadata["volume_min"]),
        volume_step=float(metadata["volume_step"]),
        volume_max=float(metadata.get("volume_max", 100.0)),
    )
    bars = MarketBars(
        time_epoch=_time_epoch(frame),
        open=_column(frame, "open"),
        high=_column(frame, "high"),
        low=_column(frame, "low"),
        close=_column(frame, "close"),
        spread=_column(frame, "spread"),
        atr=_column(frame, "atr_14_lag1", "atr14_lag1", "atr14"),
        broker_timezone=str(timezone),
    )
    if "broker_date" in frame.columns:
        broker_date = frame["broker_date"].cast(pl.String).to_numpy()
    else:
        broker_date = frame.select(
            pl.from_epoch(pl.Series("t", bars.time_epoch), time_unit="s")
            .dt.convert_time_zone("UTC")
            .dt.convert_time_zone(str(timezone))
            .dt.date()
            .cast(pl.String)
            .alias("broker_date")
        )["broker_date"].to_numpy()

    base = {"time_epoch", "time", "open", "high", "low", "close", "spread", "broker_date"}
    features = {name: frame[name].to_numpy() for name in frame.columns if name not in base}
    for name in ("session_asia", "session_london", "session_new_york", "session_overlap"):
        if name in frame.columns:
            features[name] = frame[name].to_numpy()
    return MarketBundle(bars=bars, symbol=symbol, broker_date=broker_date, features=features)


def _init_worker(feature_path: str) -> None:
    global _WORKER_MARKET
    _WORKER_MARKET = load_market_bundle(feature_path)


def _run_worker(experiment: CompleteExperiment) -> ExperimentOutcome:
    try:
        if _WORKER_MARKET is None:
            raise RuntimeError("worker market was not initialized")
        return run_experiment(experiment, _WORKER_MARKET)
    except Exception as exc:  # isolate experiment failures by contract
        return ExperimentOutcome(
            experiment_id=experiment.experiment_id,
            master_result=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
        )


def run_campaign(
    catalog_path: Path,
    feature_path: Path,
    result_root: Path,
    workers: int | None = None,
    limit: int | None = None,
) -> None:
    catalog_path = Path(catalog_path)
    feature_path = Path(feature_path)
    result_root = Path(result_root)
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    catalog = _read_catalog(catalog_path)
    catalog_ids = [item.experiment_id for item in catalog]
    checkpoint = CheckpointStore(result_root.parent / "state" / "CHECKPOINT.json")
    store = ResultStore(result_root, catalog_ids=catalog_ids, checkpoint=checkpoint)
    pending_set = set(store.pending_ids())
    pending = [item for item in catalog if item.experiment_id in pending_set]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return

    # Parent is the only writer. executor.map preserves input order, so CSV output
    # remains deterministic across worker counts.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(feature_path),),
    ) as executor:
        for outcome in executor.map(_run_worker, pending):
            if outcome.ok:
                assert outcome.master_result is not None
                store.append_result(outcome.master_result)
            else:
                store.append_error(
                    experiment_id=outcome.experiment_id,
                    exception_type=outcome.error_type or "Exception",
                    message=outcome.error_message or "",
                    traceback_text=outcome.traceback_text or "",
                )


__all__ = ["load_market_bundle", "run_campaign"]
