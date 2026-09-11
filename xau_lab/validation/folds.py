from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
import calendar
@dataclass(frozen=True)
class Fold:
    fold_id: str
    research_start: int
    research_end: int
    validation_start: int
    validation_end: int
    metadata: dict

def _month_start(ts):
    d=datetime.fromtimestamp(int(ts),timezone.utc); return (d.year,d.month)
def _add_month(y,m,k):
    idx=y*12+(m-1)+k; return idx//12,idx%12+1
def _epoch_start(y,m): return int(datetime(y,m,1,tzinfo=timezone.utc).timestamp())
def _epoch_end(y,m):
    ny,nm=_add_month(y,m,1); return _epoch_start(ny,nm)-1
def _bounds(times):
    months=sorted(set(_month_start(t) for t in times)); return months

def rolling_folds(times,research_months=12,validation_months=3,step_months=3):
    ms=_bounds(times)
    if len(ms)<7:return []
    if len(ms)<research_months+validation_months:
        research_months=max(6,int(len(ms)*.8)); validation_months=min(3,len(ms)-research_months)
        if validation_months<=0:return []
    out=[]; i=0
    while i+research_months+validation_months<=len(ms):
        rs=ms[i]; re=ms[i+research_months-1]; vs=ms[i+research_months]; ve=ms[i+research_months+validation_months-1]
        out.append(Fold(f'R{len(out)+1:03d}',_epoch_start(*rs),_epoch_end(*re),_epoch_start(*vs),_epoch_end(*ve),{'scheme':'rolling'})); i+=step_months
    return out

def expanding_folds(times,validation_months=3,min_research_months=12):
    ms=_bounds(times)
    if len(ms)<7:return []
    if len(ms)<min_research_months+validation_months:
        min_research_months=max(6,int(len(ms)*.8)); validation_months=min(3,len(ms)-min_research_months)
        if validation_months<=0:return []
    out=[]; cut=min_research_months
    while cut+validation_months<=len(ms):
        rs=ms[0]; re=ms[cut-1]; vs=ms[cut]; ve=ms[cut+validation_months-1]
        out.append(Fold(f'E{len(out)+1:03d}',_epoch_start(*rs),_epoch_end(*re),_epoch_start(*vs),_epoch_end(*ve),{'scheme':'expanding'})); cut+=validation_months
    return out
