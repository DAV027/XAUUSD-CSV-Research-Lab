from __future__ import annotations
from collections import Counter
import hashlib
import numpy as np
from scipy.stats import qmc
from .spec import CompleteExperiment
from xau_lab.strategies.registry import list_strategies,get_strategy
import xau_lab.strategies.all  # noqa: F401

FAMILY_WEIGHTS={'trend':.16,'breakout':.16,'mean_reversion':.16,'price_action':.14,'session':.10,'volatility':.10,'statistical':.10,'exit_execution':.08}
STOP_ATR=(.5,.75,1.,1.5,2.)
TARGET_R=(1.,1.5,2.,3.)
TIME_EXITS=(5,15,30,60,120)
DIRECTIONS=('long','short','combined')

def _map_domain(u, spec):
    typ=spec[0]
    if typ=='cat':
        vals=spec[1]; return vals[min(len(vals)-1,int(u*len(vals)))]
    _,lo,hi=spec
    if typ=='int': return int(lo+np.floor(u*(hi-lo+1))) if u<1 else int(hi)
    return float(lo+u*(hi-lo))

def generate_catalog(total_budget=50000,seed=9215000):
    if total_budget<=0:return []
    names=list_strategies(); defs=[get_strategy(n) for n in names]
    fams=sorted(set(d.family for d in defs))
    fam_weight=np.array([FAMILY_WEIGHTS.get(f,.08) for f in fams],float); fam_weight/=fam_weight.sum()
    rng=np.random.default_rng(seed)
    selected_fams=rng.choice(fams,size=total_budget,p=fam_weight)
    fam_to_defs={f:[d for d in defs if d.family==f] for f in fams}
    counts=Counter(selected_fams); streams={}
    for f,c in counts.items():
        ds=fam_to_defs[f]
        pick=rng.integers(0,len(ds),size=c)
        streams[f]=(pick,ds)
    fam_cursor=Counter(); lhs_cache={}
    strategy_counts=Counter()
    for f,(pick,ds) in streams.items():
        for j in pick: strategy_counts[ds[int(j)].name]+=1
    for d in defs:
        c=strategy_counts[d.name]
        dims=max(1,len(d.parameter_domains))
        stable=int.from_bytes(hashlib.sha256(d.name.encode()).digest()[:4],'big')%100000
        lhs_cache[d.name]=qmc.LatinHypercube(d=dims,seed=seed+stable).random(c) if c else np.empty((0,dims))
    strategy_cursor=Counter(); fam_cursor=Counter(); out=[]; seen=set()
    for idx,f in enumerate(selected_fams):
        pick,ds=streams[f]; j=fam_cursor[f]; fam_cursor[f]+=1; d=ds[int(pick[j])]
        row=lhs_cache[d.name][strategy_cursor[d.name]]; strategy_cursor[d.name]+=1
        params={}
        for p_i,k in enumerate(sorted(d.parameter_domains)):
            params[k]=_map_domain(float(row[p_i]),d.parameter_domains[k])
        direction=DIRECTIONS[int(rng.integers(0,len(DIRECTIONS)))]
        stop=float(STOP_ATR[int(rng.integers(0,len(STOP_ATR)))])
        mode_roll=float(rng.random())
        if mode_roll < .65:
            exit_mode='target'; target=float(TARGET_R[int(rng.integers(0,len(TARGET_R)))]); time_exit=None; trail=None
        elif mode_roll < .90:
            exit_mode='time'; target=None; time_exit=int(TIME_EXITS[int(rng.integers(0,len(TIME_EXITS)))]); trail=None
        else:
            exit_mode='atr_trail'; target=None; time_exit=None; trail=float(STOP_ATR[int(rng.integers(0,len(STOP_ATR)))])
        for retry in range(100):
            exp=CompleteExperiment(d.family,d.name,params,direction,stop,exit_mode,target,time_exit,trail,seed=seed)
            semantic=(d.name,tuple(sorted(params.items())),direction,stop,exit_mode,target,time_exit,trail)
            if semantic not in seen: break
            direction=DIRECTIONS[(DIRECTIONS.index(direction)+1)%3]
            if retry%3==2: stop=STOP_ATR[(STOP_ATR.index(stop)+1)%len(STOP_ATR)]
        else: raise RuntimeError(f'unable to create unique complete experiment for {d.name} at index {idx}, params={params}')
        seen.add(semantic); out.append(exp)
    return out
