from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
import polars as pl
@dataclass(frozen=True)
class ValidationResult:
    clean:pl.DataFrame; issues:pl.DataFrame; summary:dict

def validate_bars(frame:pl.DataFrame)->ValidationResult:
    rows=frame.to_dicts(); issues=[]
    parsed=[]
    for r in rows:
        try: parsed.append(datetime.strptime(str(r['time']),'%Y-%m-%d %H:%M:%S'))
        except Exception: parsed.append(None)
    if any(parsed[i] and parsed[i-1] and parsed[i]<parsed[i-1] for i in range(1,len(parsed))):
        issues.append({'issue_type':'non_monotonic_input','time':'','severity':'warning','details':'input timestamps not ascending'})
    counts={}
    for r in rows: counts[str(r['time'])]=counts.get(str(r['time']),0)+1
    for t,c in counts.items():
        if c>1: issues.append({'issue_type':'duplicate_timestamp','time':t,'severity':'warning','details':f'{c} rows; keeping last'})
    dedup={str(r['time']):r for r in rows}; valid=[]
    for t,r in dedup.items():
        reasons=[]
        try:
            o,h,l,c=map(float,(r['open'],r['high'],r['low'],r['close']))
            if min(o,h,l,c)<=0 or h<max(o,c) or l>min(o,c) or l>h: reasons.append('invalid_ohlc')
        except Exception: reasons.append('invalid_ohlc')
        try:
            if r['spread'] is None or float(r['spread'])<0: reasons.append('invalid_spread')
        except Exception: reasons.append('invalid_spread')
        try:
            if float(r['tick_volume'])<0 or float(r['real_volume'])<0: reasons.append('negative_volume')
        except Exception: reasons.append('negative_volume')
        for reason in sorted(set(reasons)): issues.append({'issue_type':reason,'time':t,'severity':'error','details':'row removed'})
        if not reasons: valid.append(r)
    valid.sort(key=lambda r:str(r['time']))
    run=1
    for i in range(1,len(valid)):
        a,b=valid[i-1],valid[i]; same=all(a[k]==b[k] for k in ('open','high','low','close','spread'))
        run=run+1 if same else 1
        if run==10: issues.append({'issue_type':'repeated_bar_run','time':str(b['time']),'severity':'warning','details':'at least 10 identical OHLC+spread rows'})
    for i in range(1,len(valid)):
        a=datetime.strptime(str(valid[i-1]['time']),'%Y-%m-%d %H:%M:%S'); b=datetime.strptime(str(valid[i]['time']),'%Y-%m-%d %H:%M:%S'); sec=(b-a).total_seconds()
        if sec>60:
            weekend=any((a+timedelta(days=d)).weekday()>=5 for d in range(0,max(1,(b-a).days+1)))
            issues.append({'issue_type':'market_closed_gap' if weekend else 'unexpected_gap','time':str(valid[i]['time']),'severity':'info' if weekend else 'warning','details':f'{int(sec)} second gap'})
    clean=pl.DataFrame(valid,schema=frame.schema) if valid else frame.head(0)
    issue_cols=['issue_type','time','severity','details']; issue_frame=pl.DataFrame(issues,schema={c:pl.String for c in issue_cols}) if issues else pl.DataFrame({c:[] for c in issue_cols},schema={c:pl.String for c in issue_cols})
    summary={'input_rows':frame.height,'clean_rows':clean.height,'removed_rows':frame.height-clean.height,'issue_count':issue_frame.height}
    return ValidationResult(clean,issue_frame,summary)
