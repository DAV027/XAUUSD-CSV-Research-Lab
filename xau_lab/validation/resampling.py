from __future__ import annotations
import numpy as np

def _pct(a):
    q=np.percentile(np.asarray(a,float),[5,50,95]); return {'p05':float(q[0]),'p50':float(q[1]),'p95':float(q[2])}
def block_bootstrap_daily(daily_pnl,iterations=5000,seed=9216000):
    vals=np.asarray(list(daily_pnl.values()),float)
    if len(vals)==0:return {'mean_daily':None,'total':None,'iterations':iterations}
    rng=np.random.default_rng(seed); means=[]; totals=[]
    for _ in range(iterations):
        s=rng.choice(vals,size=len(vals),replace=True); means.append(float(s.mean())); totals.append(float(s.sum()))
    return {'mean_daily':_pct(means),'total':_pct(totals),'iterations':iterations,'seed':seed}
def _maxdd(seq):
    eq=0.;peak=0.;dd=0.
    for x in seq: eq+=x;peak=max(peak,eq);dd=max(dd,peak-eq)
    return dd
def trade_order_monte_carlo(pnl,iterations=5000,seed=9216000):
    a=np.asarray(pnl,float); rng=np.random.default_rng(seed); vals=[]
    for _ in range(iterations): vals.append(_maxdd(rng.permutation(a)))
    return {'max_drawdown':_pct(vals),'iterations':iterations,'seed':seed,'label':'sequence_only_not_entry_edge_proof'}
def _pf(a):
    gp=sum(x for x in a if x>0);gl=-sum(x for x in a if x<0);return None if gl==0 else gp/gl
def top_trade_removal(pnl):
    a=list(map(float,pnl)); base=sum(a); out={'baseline_net':base,'baseline_pf':_pf(a)}
    idx=sorted(range(len(a)),key=lambda i:a[i],reverse=True)
    for k in (1,5,10):
        rem=set(idx[:min(k,len(idx))]); b=[x for i,x in enumerate(a) if i not in rem];out[f'remove_top_{k}_net']=sum(b);out[f'remove_top_{k}_pf']=_pf(b)
    return out
