from __future__ import annotations
from pathlib import Path
import csv
from .spec import CompleteExperiment,canonical_json
FIELDS=['experiment_id','fingerprint','strategy_family','strategy_name','parameters_json','direction_mode','stop_atr','exit_mode','target_r','time_exit_minutes','atr_trail','commission_round_trip_per_lot','slippage_points_per_fill','sampler_version','seed']
def write_catalog(experiments,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for e in experiments:
            d=e.to_dict();w.writerow({k:(canonical_json(e.parameters) if k=='parameters_json' else d.get(k,'')) for k in FIELDS})
    return path
def read_catalog(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return [CompleteExperiment.from_dict(r) for r in csv.DictReader(f)]
