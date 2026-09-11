from __future__ import annotations
from pathlib import Path
import csv,os,json

MASTER_FIELDS=[
'experiment_id','fingerprint','strategy_family','strategy_name','parameters_json','direction_mode','stop_atr','exit_mode','target_r','time_exit_minutes','atr_trail',
'completed_trades','wins','losses','win_rate','gross_profit','gross_loss','profit_factor','net_profit','after_cost_profit','expectancy_usd','expectancy_R','max_drawdown_usd','max_drawdown_pct','max_loss_streak','profit_per_active_day','median_profit_per_active_day','positive_year_fraction','positive_month_fraction','worst_month','long_PF','short_PF','long_trades','short_trades','active_years','positive_day_fraction','trades_per_active_day','median_hold_minutes','top_5_trade_profit_fraction','best_month_profit_fraction','active_months','risk_skip_count','runtime_seconds']

class ResultStore:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.path=self.root/'MASTER_RESULTS.csv'; self.completed=set()
        if self.path.exists():
            with self.path.open(newline='',encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('experiment_id'): self.completed.add(r['experiment_id'])
    def append(self,row):
        eid=str(row['experiment_id'])
        if eid in self.completed: raise ValueError(f'duplicate experiment_id {eid}')
        exists=self.path.exists() and self.path.stat().st_size>0
        payload={k:row.get(k,'') for k in MASTER_FIELDS}
        # preserve custom fields as JSON sidecar-style field if a caller only needs a minimal test row; canonical fields are fixed.
        with self.path.open('a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=MASTER_FIELDS)
            if not exists:w.writeheader()
            w.writerow(payload); f.flush(); os.fsync(f.fileno())
        self.completed.add(eid)
    def pending_ids(self,ids): return [x for x in ids if x not in self.completed]
    def read_rows(self):
        if not self.path.exists(): return []
        with self.path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

class ErrorStore:
    FIELDS=['experiment_id','exception_type','message','traceback_file']
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); (self.root/'errors').mkdir(exist_ok=True); self.path=self.root/'ERRORS.csv'
    def append(self,eid,exc,traceback_text):
        tb=self.root/'errors'/f'{eid}.txt'; tb.write_text(traceback_text,encoding='utf-8')
        exists=self.path.exists() and self.path.stat().st_size>0
        with self.path.open('a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=self.FIELDS)
            if not exists:w.writeheader()
            w.writerow({'experiment_id':eid,'exception_type':type(exc).__name__,'message':str(exc),'traceback_file':str(tb)})
            f.flush();os.fsync(f.fileno())
