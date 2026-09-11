from __future__ import annotations
from typing import Sequence
from .models import MarketBars,SymbolSpec,CostModel,RiskModel,ExitSpec,Trade,BacktestResult
from .pricing import buy_entry_price,sell_entry_price,fill_exit_price
from .sizing import size_for_stop

def _make_trade(bars, symbol, cost, direction, entry_i, exit_i, entry, stop, target, lot, initial_risk, raw_exit, reason):
    spread_pts=float(bars.spread[entry_i])
    exit_price=fill_exit_price(float(raw_exit),direction,float(bars.spread[exit_i]),symbol,cost)
    gross=(exit_price-entry)*symbol.contract_size*lot*direction
    commission=cost.commission_round_trip_per_lot*lot
    spread_cost=spread_pts*symbol.point*symbol.contract_size*lot if direction==1 else float(bars.spread[exit_i])*symbol.point*symbol.contract_size*lot
    slippage_cost=2*cost.slippage_points_per_fill*symbol.point*symbol.contract_size*lot
    net=gross-commission
    r=net/initial_risk if initial_risk>0 else None
    return Trade(entry_i,exit_i,int(bars.time_epoch[entry_i]),int(bars.time_epoch[exit_i]),direction,
                 float(bars.open[entry_i]),entry,stop,target,exit_price,reason,lot,initial_risk,
                 gross,spread_cost,commission,slippage_cost,net,r,(int(bars.time_epoch[exit_i])-int(bars.time_epoch[entry_i]))/60.0)

def run_reference_backtest(bars: MarketBars, signals: Sequence[int], symbol: SymbolSpec, cost: CostModel, risk: RiskModel, exit_spec: ExitSpec) -> BacktestResult:
    if len(signals)!=len(bars.open): raise ValueError("signals length mismatch")
    n=len(signals); trades=[]; risk_skips=0; pos=None
    for i in range(1,n):
        if pos is None:
            sig=int(signals[i-1])
            if sig not in (-1,1): continue
            atr=float(bars.atr[i-1])
            if not (atr>0): continue
            entry=buy_entry_price(float(bars.open[i]),float(bars.spread[i]),symbol,cost) if sig==1 else sell_entry_price(float(bars.open[i]),float(bars.spread[i]),symbol,cost)
            dist=exit_spec.stop_atr*atr
            stop=entry-dist if sig==1 else entry+dist
            target=None if exit_spec.target_r is None else entry+sig*dist*exit_spec.target_r
            sized=size_for_stop(entry,stop,symbol,risk)
            if not sized.feasible:
                risk_skips+=1; continue
            pos={'direction':sig,'entry_i':i,'entry':entry,'stop':stop,'target':target,'lot':sized.lot,'risk':sized.stop_risk_usd}
        if pos is not None:
            d=pos['direction']; stop=pos['stop']; target=pos['target']
            if d==1:
                hit_stop=float(bars.low[i])<=stop
                hit_target=target is not None and float(bars.high[i])>=target
            else:
                ask_high=float(bars.high[i])+float(bars.spread[i])*symbol.point
                ask_low=float(bars.low[i])+float(bars.spread[i])*symbol.point
                hit_stop=ask_high>=stop
                hit_target=target is not None and ask_low<=target
            if hit_stop:
                reason='stop_same_bar_ambiguous' if hit_target else 'stop'
                raw_stop=stop if d==1 else stop-float(bars.spread[i])*symbol.point
                trades.append(_make_trade(bars,symbol,cost,d,pos['entry_i'],i,pos['entry'],stop,target,pos['lot'],pos['risk'],raw_stop,reason)); pos=None; continue
            if hit_target:
                raw_target=target if d==1 else target-float(bars.spread[i])*symbol.point
                trades.append(_make_trade(bars,symbol,cost,d,pos['entry_i'],i,pos['entry'],stop,target,pos['lot'],pos['risk'],raw_target,'target')); pos=None; continue
            age=(int(bars.time_epoch[i])-int(bars.time_epoch[pos['entry_i']]))/60.0
            if exit_spec.time_exit_minutes is not None and age>=exit_spec.time_exit_minutes:
                trades.append(_make_trade(bars,symbol,cost,d,pos['entry_i'],i,pos['entry'],stop,target,pos['lot'],pos['risk'],float(bars.close[i]),'time')); pos=None; continue
            if exit_spec.atr_trail is not None:
                atr_i=float(bars.atr[i])
                if atr_i>0:
                    nxt=float(bars.close[i])-exit_spec.atr_trail*atr_i if d==1 else float(bars.close[i])+exit_spec.atr_trail*atr_i
                    pos['stop']=max(stop,nxt) if d==1 else min(stop,nxt)
    if pos is not None:
        i=n-1; d=pos['direction']
        trades.append(_make_trade(bars,symbol,cost,d,pos['entry_i'],i,pos['entry'],pos['stop'],pos['target'],pos['lot'],pos['risk'],float(bars.close[i]),'end_of_data'))
    return BacktestResult(tuple(trades),risk_skips)
