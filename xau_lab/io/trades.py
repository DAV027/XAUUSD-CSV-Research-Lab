from __future__ import annotations
from pathlib import Path
import csv
from datetime import datetime,timezone
TRADE_FIELDS=['entry_time','direction','signal_price','entry_price','stop_price','target_price','exit_time','exit_price','exit_reason','lot_size','initial_risk_usd','gross_pnl','spread_cost','commission','slippage_cost','net_pnl','pnl_R','hold_minutes','entry_index','exit_index','risk_infeasible']
def _iso(ts): return datetime.fromtimestamp(int(ts),timezone.utc).isoformat()
def trades_to_rows(trades):
    out=[]
    for t in trades:
        d={k:getattr(t,k,None) for k in TRADE_FIELDS};d['entry_time']=_iso(t.entry_time);d['exit_time']=_iso(t.exit_time);out.append(d)
    return out
def trades_to_frame(trades):
    import polars as pl
    return pl.DataFrame(trades_to_rows(trades))
def write_trades_csv(trades,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=TRADE_FIELDS);w.writeheader();w.writerows(trades_to_rows(trades))
    return path
