from __future__ import annotations
import numpy as np
import polars as pl
from .sessions import add_session_features,SessionConfig

def _lag(a): return np.r_[np.nan,np.asarray(a,float)[:-1]]
def _roll(a,w,fn):
    a=np.asarray(a,float); o=np.full(len(a),np.nan)
    for i in range(w-1,len(a)): o[i]=fn(a[i-w+1:i+1])
    return o
def _ema(a,w):
    a=np.asarray(a,float); o=np.full(len(a),np.nan); alpha=2/(w+1); e=a[0]
    for i,x in enumerate(a): e=x if i==0 else alpha*x+(1-alpha)*e; o[i]=e if i>=w-1 else np.nan
    return o
def _atr(h,l,c,w):
    prev=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(np.abs(h-prev),np.abs(l-prev))); return _roll(tr,w,np.mean)
def _rsi(c,w):
    d=np.diff(c,prepend=c[0]); up=np.maximum(d,0); dn=np.maximum(-d,0); o=np.full(len(c),np.nan)
    for i in range(w,len(c)):
        au=up[i-w+1:i+1].mean(); ad=dn[i-w+1:i+1].mean(); o[i]=100 if ad==0 else 100-100/(1+au/ad)
    return o
def _slope(c,w):
    x=np.arange(w,dtype=float); xc=x-x.mean(); den=(xc*xc).sum(); o=np.full(len(c),np.nan)
    for i in range(w-1,len(c)):
        y=c[i-w+1:i+1]; o[i]=((y-y.mean())*xc).sum()/den
    return o
def _er(c,w):
    o=np.full(len(c),np.nan)
    for i in range(w,len(c)):
        x=c[i-w:i+1]; path=np.abs(np.diff(x)).sum(); o[i]=abs(x[-1]-x[0])/path if path else 0
    return o

def build_shared_features(frame:pl.DataFrame,session_config:SessionConfig|None=None)->pl.DataFrame:
    out=add_session_features(frame,session_config) if session_config is not None else frame
    o=np.asarray(frame['open'],float); h=np.asarray(frame['high'],float); l=np.asarray(frame['low'],float); c=np.asarray(frame['close'],float)
    ret=np.r_[np.nan,c[1:]/c[:-1]-1]; logret=np.r_[np.nan,np.log(c[1:]/c[:-1])]
    feats={'return_1_lag1':_lag(ret),'log_return_1_lag1':_lag(logret),'range_lag1':_lag(h-l),'body_lag1':_lag(c-o),'upper_wick_lag1':_lag(h-np.maximum(o,c)),'lower_wick_lag1':_lag(np.minimum(o,c)-l)}
    for w in (5,9,21,50,100,200): feats[f'sma_{w}_lag1']=_lag(_roll(c,w,np.mean))
    for w in (9,21,50,100,200): feats[f'ema_{w}_lag1']=_lag(_ema(c,w))
    for w in (7,14,28): feats[f'atr_{w}_lag1']=_lag(_atr(h,l,c,w))
    for w in (7,14,21): feats[f'rsi_{w}_lag1']=_lag(_rsi(c,w))
    mid=_roll(c,20,np.mean); sd=_roll(c,20,np.std); upper=mid+2*sd; lower=mid-2*sd
    for n,a in [('bb_mid_20_2',mid),('bb_upper_20_2',upper),('bb_lower_20_2',lower),('bb_bandwidth_20_2',(upper-lower)/mid),('bb_zscore_20_2',(c-mid)/sd)]: feats[n+'_lag1']=_lag(a)
    for w in (5,10,20,50):
        rh=_roll(h,w,np.max); rl=_roll(l,w,np.min); feats[f'rolling_high_{w}_lag1']=_lag(rh); feats[f'rolling_low_{w}_lag1']=_lag(rl); feats[f'range_position_{w}_lag1']=_lag((c-rl)/(rh-rl))
    for w in (10,20,50): feats[f'realized_vol_{w}_lag1']=_lag(_roll(np.nan_to_num(logret,nan=0.0),w,np.std)); feats[f'regression_slope_{w}_lag1']=_lag(_slope(c,w)); feats[f'efficiency_ratio_{w}_lag1']=_lag(_er(c,w))
    return out.with_columns([pl.Series(k,v) for k,v in feats.items()])
