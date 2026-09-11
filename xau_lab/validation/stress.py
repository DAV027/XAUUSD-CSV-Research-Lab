from __future__ import annotations
from dataclasses import replace
import numpy as np
from xau_lab.strategies.registry import get_strategy
import xau_lab.strategies.all  # register
from xau_lab.runner.single import run_experiment,MarketBundle
from xau_lab.backtest.models import MarketBars

def numeric_neighbors(value,kind,lo,hi):
    vals=[]
    for f in (.8,.9,1.1,1.2):
        x=max(lo,min(hi,value*f)); x=int(round(x)) if kind=='int' else float(round(x,10))
        if x!=value and x not in vals: vals.append(x)
    return vals

def parameter_neighbors(experiment):
    d=get_strategy(experiment.strategy_name); out=[]
    for k,v in experiment.parameters.items():
        spec=d.parameter_domains.get(k)
        if not spec or spec[0] not in ('int','float') or not isinstance(v,(int,float)): continue
        for nv in numeric_neighbors(v,*spec):
            p=dict(experiment.parameters); p[k]=nv; out.append(replace(experiment,parameters=p,fingerprint='',experiment_id=''))
    return out

def _spread_market(market,mult):
    b=market.bars
    nb=MarketBars(b.time_epoch,b.open,b.high,b.low,b.close,b.spread*mult,b.atr)
    c=market.context; nc=type(c)(c.open,c.high,c.low,c.close,c.spread*mult,c.atr14,c.time_epoch,c.features)
    return MarketBundle(nb,nc,market.symbol,market.risk)

def run_stress_suite(experiment,market_bundle):
    rows=[]
    # cost stresses
    for slip in (0.,5.,10.,20.):
        e=replace(experiment,slippage_points_per_fill=slip,fingerprint='',experiment_id='')
        o=run_experiment(e,market_bundle); rows.append({'kind':'slippage','value':slip,**o.result_row})
    for mult in (1.,1.25,1.5):
        e=replace(experiment,commission_round_trip_per_lot=6.0*mult,fingerprint='',experiment_id='')
        o=run_experiment(e,market_bundle); rows.append({'kind':'commission_mult','value':mult,**o.result_row})
    for mult in (1.,1.25,1.5):
        o=run_experiment(experiment,_spread_market(market_bundle,mult)); rows.append({'kind':'spread_mult','value':mult,**o.result_row})
    param_rows=[]
    for e in parameter_neighbors(experiment):
        o=run_experiment(e,market_bundle); r={'kind':'parameter_neighbor','value':e.parameters,**o.result_row}; rows.append(r); param_rows.append(r)
    cost_rows=[r for r in rows if r['kind']!='parameter_neighbor']
    def good(r):
        pf=r.get('profit_factor'); ex=r.get('expectancy_usd'); return pf is not None and float(pf)>1 and ex is not None and float(ex)>0
    return {'rows':rows,'parameter_stability_score':100*sum(good(r) for r in param_rows)/len(param_rows) if param_rows else None,'cost_stability_score':100*sum(good(r) for r in cost_rows)/len(cost_rows) if cost_rows else None,'stress_max_drawdown_pct':max((float(r.get('max_drawdown_pct') or 0) for r in rows),default=0)}
