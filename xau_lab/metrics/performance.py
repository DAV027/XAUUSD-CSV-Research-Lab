from __future__ import annotations
from collections import defaultdict
import math, statistics

def _pf(values):
    gp=sum(x for x in values if x>0); gl=-sum(x for x in values if x<0)
    if gl==0: return None if gp==0 else float('inf')
    return gp/gl

def summarize_pnl(pnl, starting_equity=5000.0):
    p=[float(x) for x in pnl]
    gp=sum(x for x in p if x>0); gl=sum(x for x in p if x<0)
    eq=starting_equity; peak=eq; maxdd=0.0; maxddpct=0.0; streak=mxstreak=0
    for x in p:
        eq+=x; peak=max(peak,eq); dd=peak-eq
        if dd>maxdd: maxdd=dd
        if peak>0: maxddpct=max(maxddpct,dd/peak*100)
        if x<0: streak+=1; mxstreak=max(mxstreak,streak)
        else: streak=0
    return {'gross_profit':gp,'gross_loss':gl,'profit_factor':_pf(p),'expectancy_usd':sum(p)/len(p) if p else None,
            'max_loss_streak':mxstreak,'max_drawdown_usd':maxdd,'max_drawdown_pct':maxddpct,
            'net_profit':sum(p),'completed_trades':len(p),'wins':sum(x>0 for x in p),'losses':sum(x<0 for x in p)}

def summarize_trades(trades, starting_equity=5000.0):
    base=summarize_pnl([t.net_pnl for t in trades],starting_equity)
    n=len(trades); base['win_rate']=base['wins']/n if n else None
    base['after_cost_profit']=base['net_profit']
    base['expectancy_R']=sum(t.pnl_R for t in trades if t.pnl_R is not None)/sum(t.pnl_R is not None for t in trades) if any(t.pnl_R is not None for t in trades) else None
    base['median_hold_minutes']=statistics.median([t.hold_minutes for t in trades]) if trades else None
    longs=[t.net_pnl for t in trades if t.direction==1]; shorts=[t.net_pnl for t in trades if t.direction==-1]
    base['long_PF']=_pf(longs); base['short_PF']=_pf(shorts); base['long_trades']=len(longs); base['short_trades']=len(shorts)
    from datetime import datetime, timezone
    daily=defaultdict(float); monthly=defaultdict(float); yearly=defaultdict(float)
    for t in trades:
        dt=datetime.fromtimestamp(t.exit_time,timezone.utc)
        daily[dt.date().isoformat()]+=t.net_pnl; monthly[dt.strftime('%Y-%m')]+=t.net_pnl; yearly[str(dt.year)]+=t.net_pnl
    vals=list(daily.values())
    base['profit_per_active_day']=sum(vals)/len(vals) if vals else None
    base['median_profit_per_active_day']=statistics.median(vals) if vals else None
    base['trades_per_active_day']=n/len(daily) if daily else None
    base['positive_day_fraction']=sum(v>0 for v in daily.values())/len(daily) if daily else None
    base['positive_month_fraction']=sum(v>0 for v in monthly.values())/len(monthly) if monthly else None
    base['positive_year_fraction']=sum(v>0 for v in yearly.values())/len(yearly) if yearly else None
    base['active_years']=len(yearly)
    base['worst_month']=min(monthly.values()) if monthly else None
    profits=sorted((max(0,t.net_pnl) for t in trades),reverse=True); totalpos=sum(profits)
    base['top_5_trade_profit_fraction']=sum(profits[:5])/totalpos if totalpos>0 else None
    bestm=max(monthly.values()) if monthly else 0
    base['best_month_profit_fraction']=bestm/sum(v for v in monthly.values() if v>0) if sum(v for v in monthly.values() if v>0)>0 else None
    base['active_months']=len(monthly)
    return base
