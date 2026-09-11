from pathlib import Path
import argparse,csv,json,math
from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.scoring import score_candidate

def _num(v):
    if v in ('',None,'None'):return None
    try:return float(v)
    except:return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();master=a.root/'results/MASTER_RESULTS.csv'
    with master.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    parsed=[]
    for r in rows:
        m={k:_num(v) for k,v in r.items()};m['completed_trades']=int(float(m.get('completed_trades') or 0));m['active_months']=int(float(m.get('active_months') or 0));parsed.append((r,m))
    survivors=[];rejects=[]
    for raw,m in parsed:
        d=stage1_decision(m)
        if d.passed: survivors.append((raw,m))
        else: rejects.append({**raw,'verdict':'REJECTED','rejection_reason':';'.join(d.reasons)})
    vals=sorted(float(m.get('median_profit_per_active_day') or 0) for _,m in survivors)
    scored=[]
    for raw,m in survivors:
        x=float(m.get('median_profit_per_active_day') or 0);pct=(sum(v<=x for v in vals)-.5)/len(vals) if vals else .5;sc=score_candidate(m,max(0,min(1,pct)));scored.append({**raw,**sc,'verdict':'SURVIVOR','rejection_reason':''})
    fields=sorted(set().union(*(r.keys() for r in rejects+scored))) if rejects or scored else []
    for name,data in [('REJECTED.csv',rejects),('SURVIVORS.csv',scored)]:
        path=a.root/'results'/name
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    print(f'rejected={len(rejects)} survivors={len(scored)}')
if __name__=='__main__':main()
