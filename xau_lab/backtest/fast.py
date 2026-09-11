from __future__ import annotations
import math
import numpy as np
from numba import njit
from .models import MarketBars,SymbolSpec,CostModel,RiskModel,ExitSpec,Trade,BacktestResult

@njit(cache=True)
def _kernel(time_epoch,op,hi,lo,cl,spread,atr,signals,
            point,contract,vol_min,vol_step,vol_max,
            commission,slip_pts,preferred_risk,hard_risk,max_lot,
            stop_atr,target_r,time_exit,atr_trail):
    n=len(op)
    maxtr=n
    entry_i=np.empty(maxtr,np.int64); exit_i=np.empty(maxtr,np.int64); direction=np.empty(maxtr,np.int8)
    entry_price=np.empty(maxtr,np.float64); stop_price=np.empty(maxtr,np.float64); target_price=np.empty(maxtr,np.float64)
    exit_price=np.empty(maxtr,np.float64); reason=np.empty(maxtr,np.int8); lot_arr=np.empty(maxtr,np.float64); risk_arr=np.empty(maxtr,np.float64); net_arr=np.empty(maxtr,np.float64)
    count=0; risk_skips=0
    pos=0; pe=0.; ps=0.; pt=np.nan; plot=0.; prisk=0.; pei=-1
    for i in range(1,n):
        if pos==0:
            sig=int(signals[i-1])
            if sig!=1 and sig!=-1:
                continue
            a=atr[i-1]
            if not (a>0):
                continue
            if sig==1:
                e=op[i]+(spread[i]+slip_pts)*point
            else:
                e=op[i]-slip_pts*point
            dist=stop_atr*a
            st=e-dist if sig==1 else e+dist
            tg=np.nan if target_r<0 else e+sig*dist*target_r
            per_lot=abs(e-st)*contract
            if per_lot<=0:
                risk_skips+=1; continue
            pref=preferred_risk/per_lot
            lot=math.floor((pref+1e-12)/vol_step)*vol_step
            if lot>max_lot: lot=max_lot
            if lot>vol_max: lot=vol_max
            feasible=True
            if lot < vol_min-1e-12:
                minrisk=vol_min*per_lot
                if minrisk <= hard_risk+1e-12:
                    lot=vol_min; arisk=minrisk
                else:
                    feasible=False; arisk=minrisk
            else:
                if lot<vol_min: lot=vol_min
                arisk=lot*per_lot
                if arisk>hard_risk+1e-9: feasible=False
            if not feasible:
                risk_skips+=1; continue
            pos=sig; pe=e; ps=st; pt=tg; plot=lot; prisk=arisk; pei=i
        d=pos
        hit_stop=False; hit_target=False
        if d==1:
            hit_stop=lo[i]<=ps
            if target_r>=0: hit_target=hi[i]>=pt
        else:
            ask_hi=hi[i]+spread[i]*point
            ask_lo=lo[i]+spread[i]*point
            hit_stop=ask_hi>=ps
            if target_r>=0: hit_target=ask_lo<=pt
        raw_exit=0.; rc=0
        if hit_stop:
            raw_exit=ps if d==1 else ps-spread[i]*point; rc=2 if hit_target else 1
        elif hit_target:
            raw_exit=pt if d==1 else pt-spread[i]*point; rc=3
        else:
            age=(time_epoch[i]-time_epoch[pei])/60.0
            if time_exit>=0 and age>=time_exit:
                raw_exit=cl[i]; rc=4
        if rc!=0:
            xp=raw_exit-slip_pts*point if d==1 else raw_exit+spread[i]*point+slip_pts*point
            gross=(xp-pe)*contract*plot*d
            net=gross-commission*plot
            entry_i[count]=pei; exit_i[count]=i; direction[count]=d; entry_price[count]=pe; stop_price[count]=ps; target_price[count]=pt; exit_price[count]=xp; reason[count]=rc; lot_arr[count]=plot; risk_arr[count]=prisk; net_arr[count]=net
            count+=1; pos=0
            continue
        if atr_trail>=0:
            ai=atr[i]
            if ai>0:
                nxt=cl[i]-atr_trail*ai if d==1 else cl[i]+atr_trail*ai
                if d==1:
                    if nxt>ps: ps=nxt
                else:
                    if nxt<ps: ps=nxt
    if pos!=0:
        i=n-1; d=pos; raw_exit=cl[i]
        xp=raw_exit-slip_pts*point if d==1 else raw_exit+spread[i]*point+slip_pts*point
        gross=(xp-pe)*contract*plot*d; net=gross-commission*plot
        entry_i[count]=pei; exit_i[count]=i; direction[count]=d; entry_price[count]=pe; stop_price[count]=ps; target_price[count]=pt; exit_price[count]=xp; reason[count]=5; lot_arr[count]=plot; risk_arr[count]=prisk; net_arr[count]=net
        count+=1
    return (entry_i[:count],exit_i[:count],direction[:count],entry_price[:count],stop_price[:count],target_price[:count],exit_price[:count],reason[:count],lot_arr[:count],risk_arr[:count],net_arr[:count],risk_skips)

_REASON={1:'stop',2:'stop_same_bar_ambiguous',3:'target',4:'time',5:'end_of_data'}

def run_fast_backtest(bars:MarketBars,signals,symbol:SymbolSpec,cost:CostModel,risk:RiskModel,exit_spec:ExitSpec)->BacktestResult:
    arr=np.ascontiguousarray(np.asarray(signals,dtype=np.int8))
    tr=-1.0 if exit_spec.target_r is None else float(exit_spec.target_r)
    te=-1.0 if exit_spec.time_exit_minutes is None else float(exit_spec.time_exit_minutes)
    at=-1.0 if exit_spec.atr_trail is None else float(exit_spec.atr_trail)
    vals=_kernel(np.ascontiguousarray(bars.time_epoch,dtype=np.int64),np.ascontiguousarray(bars.open,dtype=float),np.ascontiguousarray(bars.high,dtype=float),np.ascontiguousarray(bars.low,dtype=float),np.ascontiguousarray(bars.close,dtype=float),np.ascontiguousarray(bars.spread,dtype=float),np.ascontiguousarray(bars.atr,dtype=float),arr,
                 symbol.point,symbol.contract_size,symbol.volume_min,symbol.volume_step,symbol.volume_max,
                 cost.commission_round_trip_per_lot,cost.slippage_points_per_fill,risk.preferred_risk_usd,risk.hard_risk_usd,risk.max_lot,
                 exit_spec.stop_atr,tr,te,at)
    ei,xi,ds,eps,sps,tps,xps,rs,lots,risks,nets,risk_skips=vals
    trades=[]
    for k in range(len(ei)):
        d=int(ds[k]); e=int(ei[k]); x=int(xi[k]); lot=float(lots[k]); comm=cost.commission_round_trip_per_lot*lot
        spcost=(float(bars.spread[e]) if d==1 else float(bars.spread[x]))*symbol.point*symbol.contract_size*lot
        slipcost=2*cost.slippage_points_per_fill*symbol.point*symbol.contract_size*lot
        gross=float(nets[k])+comm
        target=None if np.isnan(tps[k]) else float(tps[k])
        pr=float(risks[k]); net=float(nets[k]); rr=net/pr if pr>0 else None
        trades.append(Trade(e,x,int(bars.time_epoch[e]),int(bars.time_epoch[x]),d,float(bars.open[e]),float(eps[k]),float(sps[k]),target,float(xps[k]),_REASON[int(rs[k])],lot,pr,gross,spcost,comm,slipcost,net,rr,(int(bars.time_epoch[x])-int(bars.time_epoch[e]))/60.0))
    return BacktestResult(tuple(trades),int(risk_skips))
