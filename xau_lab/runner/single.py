from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

# Import strategy modules so the registry is populated in worker processes.
from xau_lab.strategies import breakout as _breakout  # noqa: F401
from xau_lab.strategies import mean_reversion as _mean_reversion  # noqa: F401
from xau_lab.strategies import price_action as _price_action  # noqa: F401
from xau_lab.strategies import session as _session  # noqa: F401
from xau_lab.strategies import statistical as _statistical  # noqa: F401
from xau_lab.strategies import trend as _trend  # noqa: F401
from xau_lab.strategies import volatility as _volatility  # noqa: F401
from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec, Trade
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.metrics.performance import summarize_trades
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy


@dataclass(frozen=True)
class MarketBundle:
    bars: MarketBars
    symbol: SymbolSpec
    broker_date: np.ndarray
    features: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        broker_date = np.asarray(self.broker_date, dtype=object)
        if len(broker_date) != len(self.bars):
            raise ValueError("broker_date length must match market bars")
        broker_date.setflags(write=False)
        object.__setattr__(self, "broker_date", broker_date)
        feature_map: dict[str, np.ndarray] = {}
        for name, values in self.features.items():
            array = np.ascontiguousarray(values)
            if len(array) != len(self.bars):
                raise ValueError(f"feature {name!r} length must match market bars")
            array.setflags(write=False)
            feature_map[str(name)] = array
        object.__setattr__(self, "features", MappingProxyType(feature_map))

    def strategy_context(self) -> StrategyContext:
        return StrategyContext(
            open=self.bars.open,
            high=self.bars.high,
            low=self.bars.low,
            close=self.bars.close,
            spread=self.bars.spread,
            atr14=self.bars.atr,
            time_epoch=self.bars.time_epoch,
            features=self.features,
        )


@dataclass(frozen=True)
class ExperimentOutcome:
    experiment_id: str
    master_result: dict | None
    trades: tuple[Trade, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_type is None


def _direction_filter(signals: np.ndarray, direction_mode: str) -> np.ndarray:
    filtered = np.ascontiguousarray(signals, dtype=np.int8).copy()
    if direction_mode == "long":
        filtered[filtered < 0] = 0
    elif direction_mode == "short":
        filtered[filtered > 0] = 0
    elif direction_mode != "combined":
        raise ValueError(f"unknown direction mode: {direction_mode}")
    return filtered


def _exit_spec(experiment: CompleteExperiment) -> ExitSpec:
    return ExitSpec(
        stop_atr=float(experiment.stop_atr),
        target_r=None if experiment.target_r is None else float(experiment.target_r),
        time_exit_minutes=(
            None if experiment.time_exit_minutes is None else int(experiment.time_exit_minutes)
        ),
        atr_trail=None if experiment.atr_trail is None else float(experiment.atr_trail),
    )


def _annotate_trade(trade: Trade, experiment: CompleteExperiment, market: MarketBundle) -> Trade:
    entry_index = trade.entry_index
    broker_date = str(market.broker_date[entry_index]) if entry_index >= 0 else trade.broker_date
    features = market.features

    def flag(name: str) -> bool | None:
        values = features.get(name)
        if values is None or entry_index < 0:
            return None
        return bool(values[entry_index])

    return replace(
        trade,
        experiment_id=experiment.experiment_id,
        parameter_set_id=experiment.fingerprint,
        fingerprint=experiment.fingerprint,
        broker_date=broker_date,
        session_asia=flag("session_asia"),
        session_london=flag("session_london"),
        session_new_york=flag("session_new_york"),
        session_overlap=flag("session_overlap"),
        broker_timezone=market.bars.broker_timezone,
    )


def run_experiment(
    experiment: CompleteExperiment,
    market_bundle: MarketBundle,
    *,
    include_trades: bool = False,
    risk_model: RiskModel | None = None,
) -> ExperimentOutcome:
    definition = get_strategy(experiment.strategy_name)
    signals = definition.generate(market_bundle.strategy_context(), experiment.parameters)
    signals = _direction_filter(signals, experiment.direction_mode)
    cost = CostModel(
        commission_round_trip_per_lot=float(experiment.commission_round_trip_per_lot),
        slippage_points_per_fill=float(experiment.slippage_points_per_fill),
    )
    risk = risk_model or RiskModel()
    result = run_fast_backtest(
        market_bundle.bars,
        signals,
        market_bundle.symbol,
        cost,
        risk,
        _exit_spec(experiment),
    )
    if include_trades:
        trades = tuple(_annotate_trade(trade, experiment, market_bundle) for trade in result.trades)
        metrics = summarize_trades(trades, starting_equity=risk.account_equity)
    else:
        trades = ()
        broker_dates = [str(market_bundle.broker_date[trade.entry_index])
                        if trade.entry_index >= 0 else trade.broker_date
                        for trade in result.trades]
        metrics = summarize_trades(result.trades, starting_equity=risk.account_equity,
                                   broker_dates=broker_dates)

    master = {
        **experiment.to_dict(),
        "data_start": int(market_bundle.bars.time_epoch[0]) if len(market_bundle.bars) else None,
        "data_end": int(market_bundle.bars.time_epoch[-1]) if len(market_bundle.bars) else None,
        "risk_skip_count": int(result.risk_skip_count),
        **metrics,
    }
    return ExperimentOutcome(
        experiment_id=experiment.experiment_id,
        master_result=master,
        trades=trades if include_trades else (),
    )


__all__ = ["ExperimentOutcome", "MarketBundle", "run_experiment"]
