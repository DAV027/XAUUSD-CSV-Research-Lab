from __future__ import annotations
from dataclasses import dataclass
from statistics import median
import numpy as np
from xau_lab.validation.folds import expanding_folds,rolling_folds
from xau_lab.backtest.models import MarketBars,CostModel,ExitSpec
from xau_lab.backtest.reference import run_reference_backtest
from xau_lab.metrics.performance import summarize_trades
from xau_lab.strategies.registry import get_strategy
import xau_lab.strategies.all
from xau_lab.strategies.base import StrategyContext

@dataclass(frozen=True)
class FoldResult:
    experiment_id:str; scheme:str; fold_id:str; segment:str; start:int; end:int; trades:int; pf:float|None; expectancy_usd:float|None; net_profit:float; max_drawdown_pct:float; positive_day_fraction:float|None

def _window_backtest(experiment,market,start,end,warmup=250):
    t=market.bars.time_epoch
    lo=int(np.searchsorted(t,start,'left')); hi=int(np.searchsorted(t,end,'right'))
    ws=max(0,lo-warmup)
    if hi<=lo:return summarize_trades((),market.risk.account_equity)
    idx=slice(ws,hi); b=market.bars
    sb=MarketBars(b.time_epoch[idx],b.open[idx],b.high[idx],b.low[idx],b.close[idx],b.spread[idx],b.atr[idx])
    c=market.context
    sf={k:np.asarray(v)[idx] for k,v in c.features.items()}
    sc=StrategyContext(c.open[idx],c.high[idx],c.low[idx],c.close[idx],c.spread[idx],c.atr14[idx],c.time_epoch[idx],sf)
    sig=get_strategy(experiment.strategy_name).signal(sc,experiment.parameters)
    actual_start=lo-ws
    sig[:actual_start]=0
    if experiment.direction_mode=='long':sig[sig<0]=0
    elif experiment.direction_mode=='short':sig[sig>0]=0
    exit_spec=ExitSpec(experiment.stop_atr,experiment.target_r,experiment.time_exit_minutes,experiment.atr_trail)
    cost=CostModel(experiment.commission_round_trip_per_lot,experiment.slippage_points_per_fill)
    bt=run_reference_backtest(sb,sig,market.symbol,cost,market.risk,exit_spec)
    return summarize_trades(bt.trades,market.risk.account_equity)

def validate_candidate(experiment,market_bundle,scheme='expanding'):
    folds=expanding_folds(market_bundle.bars.time_epoch) if scheme=='expanding' else rolling_folds(market_bundle.bars.time_epoch)
    out=[]
    for f in folds:
        for seg,start,end in [('research',f.research_start,f.research_end),('validation',f.validation_start,f.validation_end)]:
            m=_window_backtest(experiment,market_bundle,start,end)
            out.append(FoldResult(experiment.experiment_id,scheme,f.fold_id,seg,start,end,int(m.get('completed_trades') or 0),m.get('profit_factor'),m.get('expectancy_usd'),float(m.get('net_profit') or 0),float(m.get('max_drawdown_pct') or 0),m.get('positive_day_fraction')))
    return out

def summarize_fold_results(rows):
    vals=[r for r in rows if r.segment=='validation']
    pfs=[r.pf for r in vals if r.pf is not None and np.isfinite(r.pf)]
    good=[r for r in vals if r.pf is not None and r.pf>1 and r.expectancy_usd is not None and r.expectancy_usd>0]
    return {'validation_fold_count':len(vals),'median_validation_pf':median(pfs) if pfs else None,'validation_pass_fraction':len(good)/len(vals) if vals else None,'positive_expectancy_fraction':sum((r.expectancy_usd or 0)>0 for r in vals)/len(vals) if vals else None,'worst_validation_net_profit':min((r.net_profit for r in vals),default=None)}
