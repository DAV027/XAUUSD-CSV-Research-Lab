from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import os,time,traceback
from .single import run_experiment
from xau_lab.io.results import ResultStore,ErrorStore
from xau_lab.io.checkpoint import CheckpointStore

_WORKER_MARKET=None
def _init_worker(market):
    global _WORKER_MARKET; _WORKER_MARKET=market

def _worker(exp):
    t=time.perf_counter()
    out=run_experiment(exp,_WORKER_MARKET)
    row=dict(out.result_row); row['runtime_seconds']=time.perf_counter()-t
    return row

def run_campaign_experiments(experiments,market,result_root,workers=None,limit=None):
    root=Path(result_root); root.mkdir(parents=True,exist_ok=True)
    rs=ResultStore(root); es=ErrorStore(root); cp=CheckpointStore(root)
    pending=[e for e in experiments if e.experiment_id not in rs.completed]
    if limit is not None: pending=pending[:int(limit)]
    if not pending:
        cp.write(rs.completed); return
    workers=max(1,(os.cpu_count() or 2)-1) if workers is None else int(workers)
    if workers<1: raise ValueError('workers must be >=1')
    if workers==1:
        _init_worker(market)
        for e in pending:
            try:
                row=_worker(e); rs.append(row); cp.write(rs.completed)
            except Exception as exc:
                es.append(e.experiment_id,exc,traceback.format_exc()); cp.write(rs.completed)
        return
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(market,)) as pool:
        fut={pool.submit(_worker,e):e for e in pending}
        for f in as_completed(fut):
            e=fut[f]
            try:
                rs.append(f.result())
            except Exception as exc:
                es.append(e.experiment_id,exc,traceback.format_exc())
            cp.write(rs.completed)
