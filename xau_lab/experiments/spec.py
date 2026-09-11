from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json

def _norm(v):
    if isinstance(v,float): return float(f'{v:.10g}')
    if isinstance(v,dict): return {k:_norm(v[k]) for k in sorted(v)}
    if isinstance(v,(list,tuple)): return [_norm(x) for x in v]
    return v

def canonical_json(obj)->str:
    return json.dumps(_norm(obj),sort_keys=True,separators=(',',':'),allow_nan=False)

def canonical_fingerprint(payload: dict)->str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

@dataclass(frozen=True)
class CompleteExperiment:
    strategy_family: str
    strategy_name: str
    parameters: dict
    direction_mode: str
    stop_atr: float
    exit_mode: str
    target_r: float|None
    time_exit_minutes: int|None
    atr_trail: float|None
    commission_round_trip_per_lot: float=6.0
    slippage_points_per_fill: float=5.0
    sampler_version: str='v1'
    seed: int=9215000
    fingerprint: str=''
    experiment_id: str=''
    def __post_init__(self):
        payload={k:v for k,v in asdict(self).items() if k not in ('fingerprint','experiment_id')}
        fp=canonical_fingerprint(payload)
        object.__setattr__(self,'fingerprint',fp)
        object.__setattr__(self,'experiment_id','EXP'+fp[:12].upper())
    def to_dict(self):
        d=asdict(self); d['parameters_json']=canonical_json(self.parameters); return d
    @classmethod
    def from_dict(cls,d):
        p=d.get('parameters')
        if p is None:
            p=json.loads(d.get('parameters_json','{}'))
        def opt_float(k):
            v=d.get(k); return None if v in (None,'','None') else float(v)
        def opt_int(k):
            v=d.get(k); return None if v in (None,'','None') else int(float(v))
        return cls(
            strategy_family=str(d['strategy_family']),strategy_name=str(d['strategy_name']),parameters=p,
            direction_mode=str(d['direction_mode']),stop_atr=float(d['stop_atr']),exit_mode=str(d['exit_mode']),
            target_r=opt_float('target_r'),time_exit_minutes=opt_int('time_exit_minutes'),atr_trail=opt_float('atr_trail'),
            commission_round_trip_per_lot=float(d.get('commission_round_trip_per_lot',6.0)),
            slippage_points_per_fill=float(d.get('slippage_points_per_fill',5.0)),
            sampler_version=str(d.get('sampler_version','v1')),seed=int(float(d.get('seed',9215000))),
        )
