from __future__ import annotations
from dataclasses import dataclass
from . import __init__  # noqa
from xau_lab.backtest.models import MarketBars,SymbolSpec,RiskModel,CostModel,ExitSpec
from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.registry import get_strategy
import xau_lab.strategies.all  # register
from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.metrics.performance import summarize_trades

@dataclass(frozen=True)
class MarketBundle:
    bars: MarketBars
    context: StrategyContext
    symbol: SymbolSpec
    risk: RiskModel

@dataclass(frozen=True)
class ExperimentOutcome:
    result_row: dict
    trades: tuple=()
    error: dict|None=None

def _direction_filter(signals,mode):
    s=signals.copy()
    if mode=='long': s[s<0]=0
    elif mode=='short': s[s>0]=0
    elif mode!='combined': raise ValueError(f'unknown direction mode {mode}')
    return s

def run_experiment(experiment,market_bundle: MarketBundle,capture_trades=False):
    d=get_strategy(experiment.strategy_name)
    signals=d.signal(market_bundle.context,experiment.parameters)
    signals=_direction_filter(signals,experiment.direction_mode)
    exit_spec=ExitSpec(experiment.stop_atr,experiment.target_r,experiment.time_exit_minutes,experiment.atr_trail)
    cost=CostModel(experiment.commission_round_trip_per_lot,experiment.slippage_points_per_fill)
    bt=run_fast_backtest(market_bundle.bars,signals,market_bundle.symbol,cost,market_bundle.risk,exit_spec)
    metrics=summarize_trades(bt.trades,market_bundle.risk.account_equity)
    row={
        'experiment_id':experiment.experiment_id,'fingerprint':experiment.fingerprint,
        'strategy_family':experiment.strategy_family,'strategy_name':experiment.strategy_name,
        'parameters_json':__import__('xau_lab.experiments.spec',fromlist=['canonical_json']).canonical_json(experiment.parameters),
        'direction_mode':experiment.direction_mode,'stop_atr':experiment.stop_atr,'exit_mode':experiment.exit_mode,
        'target_r':experiment.target_r,'time_exit_minutes':experiment.time_exit_minutes,'atr_trail':experiment.atr_trail,
        **metrics,'risk_skip_count':bt.risk_skip_count,
    }
    return ExperimentOutcome(row,bt.trades if capture_trades else ())
