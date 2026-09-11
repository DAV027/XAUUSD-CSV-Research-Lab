from __future__ import annotations
import numpy as np
from .base import StrategyDefinition,StrategyContext
from .registry import register_strategy

def _sig(n): return np.zeros(n,dtype=np.int8)
def _rolling_mean(a,w):
    out=np.full(len(a),np.nan); cs=np.cumsum(np.insert(a,0,0.0)); out[w-1:]=(cs[w:]-cs[:-w])/w; return out
def _rolling_std(a,w):
    out=np.full(len(a),np.nan)
    for i in range(w-1,len(a)): out[i]=np.std(a[i-w+1:i+1])
    return out
def _rsi(a,w):
    out=np.full(len(a),np.nan); d=np.diff(a,prepend=a[0]); up=np.maximum(d,0); dn=np.maximum(-d,0)
    for i in range(w,len(a)):
        au=up[i-w+1:i+1].mean(); ad=dn[i-w+1:i+1].mean(); out[i]=100 if ad==0 else 100-100/(1+au/ad)
    return out

def regression_slope(ctx,p):
    w=int(p.get('lookback',20)); th=float(p.get('threshold_atr',0)); s=_sig(len(ctx)); x=np.arange(w,dtype=float); xc=x-x.mean(); den=(xc*xc).sum()
    for i in range(w-1,len(ctx)):
        y=ctx.close[i-w+1:i+1]; slope=((y-y.mean())*xc).sum()/den; atr=ctx.atr14[i]
        z=slope/atr if atr>0 else 0
        if z>th:s[i]=1
        elif z<-th:s[i]=-1
    return s

def roc_momentum(ctx,p):
    w=int(p.get('lookback',10)); th=float(p.get('threshold_pct',0.1))/100; s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        r=ctx.close[i]/ctx.close[i-w]-1
        if r>th:s[i]=1
        elif r<-th:s[i]=-1
    return s

def sma_slope(ctx,p):
    w=int(p.get('lookback',20)); h=int(p.get('horizon',3)); th=float(p.get('threshold_atr',0)); ma=_rolling_mean(ctx.close,w); s=_sig(len(ctx))
    for i in range(w-1+h,len(ctx)):
        z=(ma[i]-ma[i-h])/max(ctx.atr14[i],1e-12)
        if z>th:s[i]=1
        elif z<-th:s[i]=-1
    return s

def ema_slope(ctx,p):
    w=int(p.get('lookback',20)); h=int(p.get('horizon',3)); alpha=2/(w+1); ema=np.full(len(ctx),np.nan); ema[0]=ctx.close[0]
    for i in range(1,len(ctx)): ema[i]=alpha*ctx.close[i]+(1-alpha)*ema[i-1]
    s=_sig(len(ctx)); th=float(p.get('threshold_atr',0))
    for i in range(max(w,h),len(ctx)):
        z=(ema[i]-ema[i-h])/max(ctx.atr14[i],1e-12)
        if z>th:s[i]=1
        elif z<-th:s[i]=-1
    return s

def efficiency_trend(ctx,p):
    w=int(p.get('lookback',20)); th=float(p.get('er_threshold',.3)); s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        net=ctx.close[i]-ctx.close[i-w]; path=np.abs(np.diff(ctx.close[i-w:i+1])).sum(); er=abs(net)/path if path else 0
        if er>=th:s[i]=1 if net>0 else -1 if net<0 else 0
    return s

def nbar_breakout(ctx,p):
    w=int(p.get('lookback',20)); s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        hi=np.max(ctx.high[i-w:i]); lo=np.min(ctx.low[i-w:i])
        if ctx.close[i]>hi:s[i]=1
        elif ctx.close[i]<lo:s[i]=-1
    return s

def nbar_failed_breakout(ctx,p):
    w=int(p.get('lookback',20)); s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        hi=np.max(ctx.high[i-w:i]); lo=np.min(ctx.low[i-w:i])
        if ctx.high[i]>hi and ctx.close[i]<hi:s[i]=-1
        elif ctx.low[i]<lo and ctx.close[i]>lo:s[i]=1
    return s

def range_expansion(ctx,p):
    w=int(p.get('lookback',20)); mult=float(p.get('multiplier',1.5)); s=_sig(len(ctx)); tr=ctx.high-ctx.low
    for i in range(w,len(ctx)):
        med=np.median(tr[i-w:i])
        if med>0 and tr[i]>=mult*med:
            s[i]=1 if ctx.close[i]>ctx.open[i] else -1 if ctx.close[i]<ctx.open[i] else 0
    return s

def zscore_reversion(ctx,p):
    w=int(p.get('lookback',30)); th=float(p.get('threshold',2)); s=_sig(len(ctx)); ma=_rolling_mean(ctx.close,w); sd=_rolling_std(ctx.close,w)
    for i in range(w-1,len(ctx)):
        if sd[i]>0:
            z=(ctx.close[i]-ma[i])/sd[i]
            if z>=th:s[i]=-1
            elif z<=-th:s[i]=1
    return s

def bollinger_reversion(ctx,p):
    return zscore_reversion(ctx,{'lookback':int(p.get('lookback',20)),'threshold':float(p.get('std_mult',2))})
def rsi_extreme(ctx,p):
    w=int(p.get('lookback',14)); low=float(p.get('lower',30)); high=100-low; rr=_rsi(ctx.close,w); s=_sig(len(ctx))
    s[rr<low]=1; s[rr>high]=-1; return s

def stochastic_extreme(ctx,p):
    w=int(p.get('lookback',14)); low=float(p.get('lower',20)); high=100-low; s=_sig(len(ctx))
    for i in range(w-1,len(ctx)):
        lo=np.min(ctx.low[i-w+1:i+1]); hi=np.max(ctx.high[i-w+1:i+1]); k=100*(ctx.close[i]-lo)/(hi-lo) if hi>lo else 50
        if k<low:s[i]=1
        elif k>high:s[i]=-1
    return s

def cci_extreme(ctx,p):
    w=int(p.get('lookback',20)); th=float(p.get('threshold',100)); tp=(ctx.high+ctx.low+ctx.close)/3; s=_sig(len(ctx))
    for i in range(w-1,len(ctx)):
        x=tp[i-w+1:i+1]; ma=x.mean(); md=np.mean(np.abs(x-ma)); cci=(tp[i]-ma)/(.015*md) if md>0 else 0
        if cci>th:s[i]=-1
        elif cci<-th:s[i]=1
    return s

def williams_r_extreme(ctx,p):
    w=int(p.get('lookback',14)); band=float(p.get('band',20)); s=_sig(len(ctx))
    for i in range(w-1,len(ctx)):
        lo=np.min(ctx.low[i-w+1:i+1]); hi=np.max(ctx.high[i-w+1:i+1]); wr=-100*(hi-ctx.close[i])/(hi-lo) if hi>lo else -50
        if wr<-100+band:s[i]=1
        elif wr>-band:s[i]=-1
    return s

def engulfing(ctx,p):
    s=_sig(len(ctx)); min_body=float(p.get('min_body_atr',0.0))
    for i in range(1,len(ctx)):
        if abs(ctx.close[i]-ctx.open[i]) < min_body*max(ctx.atr14[i],1e-12): continue
        if ctx.close[i]>ctx.open[i] and ctx.close[i-1]<ctx.open[i-1] and ctx.open[i]<=ctx.close[i-1] and ctx.close[i]>=ctx.open[i-1]: s[i]=1
        elif ctx.close[i]<ctx.open[i] and ctx.close[i-1]>ctx.open[i-1] and ctx.open[i]>=ctx.close[i-1] and ctx.close[i]<=ctx.open[i-1]: s[i]=-1
    return s

def rejection_candle(ctx,p):
    ratio=float(p.get('wick_body_ratio',2)); s=_sig(len(ctx))
    for i in range(len(ctx)):
        body=abs(ctx.close[i]-ctx.open[i]); body=max(body,1e-12); up=ctx.high[i]-max(ctx.open[i],ctx.close[i]); dn=min(ctx.open[i],ctx.close[i])-ctx.low[i]
        if dn/body>=ratio and up<body:s[i]=1
        elif up/body>=ratio and dn<body:s[i]=-1
    return s

def inside_bar_break(ctx,p):
    s=_sig(len(ctx)); min_range=float(p.get('mother_range_atr_min',0.0))
    for i in range(2,len(ctx)):
        if (ctx.high[i-2]-ctx.low[i-2]) < min_range*max(ctx.atr14[i-2],1e-12): continue
        if ctx.high[i-1]<ctx.high[i-2] and ctx.low[i-1]>ctx.low[i-2]:
            if ctx.close[i]>ctx.high[i-1]:s[i]=1
            elif ctx.close[i]<ctx.low[i-1]:s[i]=-1
    return s

def outside_bar(ctx,p):
    s=_sig(len(ctx)); min_range=float(p.get('min_range_atr',0.0))
    for i in range(1,len(ctx)):
        if (ctx.high[i]-ctx.low[i]) < min_range*max(ctx.atr14[i],1e-12): continue
        if ctx.high[i]>ctx.high[i-1] and ctx.low[i]<ctx.low[i-1]: s[i]=1 if ctx.close[i]>ctx.open[i] else -1
    return s

def level_sweep_reclaim(ctx,p): return nbar_failed_breakout(ctx,{'lookback':int(p.get('lookback',20))})
def return_continuation(ctx,p):
    w=int(p.get('lookback',1)); th=float(p.get('threshold_pct',0.0))/100; s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        r=ctx.close[i]/ctx.close[i-w]-1
        if r>th:s[i]=1
        elif r<-th:s[i]=-1
    return s
def return_reversal(ctx,p): return -return_continuation(ctx,p)
def range_position_reversal(ctx,p):
    w=int(p.get('lookback',20)); edge=float(p.get('edge',.1)); s=_sig(len(ctx))
    for i in range(w-1,len(ctx)):
        lo=np.min(ctx.low[i-w+1:i+1]); hi=np.max(ctx.high[i-w+1:i+1]); q=(ctx.close[i]-lo)/(hi-lo) if hi>lo else .5
        if q<edge:s[i]=1
        elif q>1-edge:s[i]=-1
    return s
def standardized_return_signal(ctx,p):
    w=int(p.get('lookback',30)); th=float(p.get('threshold',1.5)); ret=np.diff(ctx.close,prepend=ctx.close[0]); s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        x=ret[i-w:i]; sd=x.std(); z=ret[i]/sd if sd>0 else 0
        if z>th:s[i]=1
        elif z<-th:s[i]=-1
    return s
def rolling_autocorr_direction(ctx,p):
    w=int(p.get('lookback',30)); s=_sig(len(ctx)); ret=np.diff(ctx.close,prepend=ctx.close[0])
    for i in range(w,len(ctx)):
        x=ret[i-w+1:i+1]; a=x[:-1]; b=x[1:]; ac=np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else 0
        if ac>0:s[i]=1 if ret[i]>0 else -1 if ret[i]<0 else 0
        elif ac<0:s[i]=-1 if ret[i]>0 else 1 if ret[i]<0 else 0
    return s

def atr_percentile_regime(ctx,p):
    w=int(p.get('lookback',100)); q=float(p.get('percentile',.7)); s=_sig(len(ctx))
    for i in range(w,len(ctx)):
        if ctx.atr14[i]>=np.quantile(ctx.atr14[i-w:i],q): s[i]=1 if ctx.close[i]>ctx.open[i] else -1
    return s
def volatility_expansion_direction(ctx,p): return range_expansion(ctx,{'lookback':int(p.get('lookback',20)),'multiplier':float(p.get('multiplier',1.5))})
def volatility_contraction_reversion(ctx,p):
    w=int(p.get('lookback',20)); q=float(p.get('percentile',.2)); s=_sig(len(ctx)); tr=ctx.high-ctx.low
    for i in range(w,len(ctx)):
        if tr[i]<=np.quantile(tr[i-w:i],q): s[i]= -1 if ctx.close[i]>ctx.open[i] else 1 if ctx.close[i]<ctx.open[i] else 0
    return s

def _register(name,family,fn,domains):
    try: register_strategy(StrategyDefinition(family,name,fn,domains))
    except ValueError: pass

_register('sma_slope','trend',sma_slope,{'lookback':('int',5,100),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('ema_slope','trend',ema_slope,{'lookback':('int',5,100),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('regression_slope','trend',regression_slope,{'lookback':('int',5,120),'threshold_atr':('float',0,.5)})
_register('roc_momentum','trend',roc_momentum,{'lookback':('int',2,60),'threshold_pct':('float',0,1.5)})
_register('efficiency_trend','trend',efficiency_trend,{'lookback':('int',5,80),'er_threshold':('float',.1,.8)})
_register('nbar_breakout','breakout',nbar_breakout,{'lookback':('int',2,100)})
_register('nbar_failed_breakout','breakout',nbar_failed_breakout,{'lookback':('int',2,100)})
_register('range_expansion','breakout',range_expansion,{'lookback':('int',5,100),'multiplier':('float',1.1,3)})
_register('zscore_reversion','mean_reversion',zscore_reversion,{'lookback':('int',10,200),'threshold':('float',1,3.5)})
_register('bollinger_reversion','mean_reversion',bollinger_reversion,{'lookback':('int',10,100),'std_mult':('float',1,3.5)})
_register('rsi_extreme','mean_reversion',rsi_extreme,{'lookback':('int',5,40),'lower':('float',10,40)})
_register('stochastic_extreme','mean_reversion',stochastic_extreme,{'lookback':('int',5,40),'lower':('float',5,35)})
_register('cci_extreme','mean_reversion',cci_extreme,{'lookback':('int',5,60),'threshold':('float',50,250)})
_register('williams_r_extreme','mean_reversion',williams_r_extreme,{'lookback':('int',5,40),'band':('float',5,35)})
_register('engulfing','price_action',engulfing,{'min_body_atr':('float',0.0,1.5)})
_register('rejection_candle','price_action',rejection_candle,{'wick_body_ratio':('float',1.5,4)})
_register('inside_bar_break','price_action',inside_bar_break,{'mother_range_atr_min':('float',0.0,2.0)})
_register('outside_bar','price_action',outside_bar,{'min_range_atr':('float',0.2,3.0)})
_register('level_sweep_reclaim','price_action',level_sweep_reclaim,{'lookback':('int',5,100)})
_register('return_continuation','statistical',return_continuation,{'lookback':('int',1,30),'threshold_pct':('float',0,1)})
_register('return_reversal','statistical',return_reversal,{'lookback':('int',1,30),'threshold_pct':('float',0,1)})
_register('range_position_reversal','statistical',range_position_reversal,{'lookback':('int',5,100),'edge':('float',.05,.3)})
_register('standardized_return_signal','statistical',standardized_return_signal,{'lookback':('int',10,100),'threshold':('float',.5,3)})
_register('rolling_autocorr_direction','statistical',rolling_autocorr_direction,{'lookback':('int',10,100)})
_register('atr_percentile_regime','volatility',atr_percentile_regime,{'lookback':('int',20,200),'percentile':('float',.5,.95)})
_register('volatility_expansion_direction','volatility',volatility_expansion_direction,{'lookback':('int',5,100),'multiplier':('float',1.1,3)})
_register('volatility_contraction_reversion','volatility',volatility_contraction_reversion,{'lookback':('int',10,100),'percentile':('float',.05,.4)})

def bollinger_expansion(ctx,p):
    w=int(p.get('lookback',20)); mult=float(p.get('std_mult',2)); pct=float(p.get('bandwidth_percentile',.75)); s=_sig(len(ctx)); ma=_rolling_mean(ctx.close,w); sd=_rolling_std(ctx.close,w); bw=2*mult*sd/np.maximum(np.abs(ma),1e-12)
    for i in range(2*w-2,len(ctx)):
        hist=bw[i-w+1:i+1]
        if np.isfinite(bw[i]) and bw[i]>=np.nanquantile(hist,pct):
            if ctx.close[i]>ma[i]+mult*sd[i]:s[i]=1
            elif ctx.close[i]<ma[i]-mult*sd[i]:s[i]=-1
    return s

def compression_breakout(ctx,p):
    cw=int(p.get('compression_lookback',30)); cp=float(p.get('compression_percentile',.2)); bw=int(p.get('breakout_lookback',10)); s=_sig(len(ctx)); tr=ctx.high-ctx.low
    start=max(cw,bw)+1
    for i in range(start,len(ctx)):
        compressed=tr[i-1] <= np.quantile(tr[i-cw:i],cp)
        if not compressed: continue
        hi=np.max(ctx.high[i-bw:i]); lo=np.min(ctx.low[i-bw:i])
        if ctx.close[i]>hi:s[i]=1
        elif ctx.close[i]<lo:s[i]=-1
    return s

def _session_flag(ctx,name):
    key='session_'+name
    if key not in ctx.features: raise ValueError(f'missing required session feature {key}')
    return np.asarray(ctx.features[key],bool)

def session_open_momentum(ctx,p):
    name=str(p.get('session','london')); flag=_session_flag(ctx,name); bars=int(p.get('opening_bars',5)); s=_sig(len(ctx)); n=len(ctx); i=0
    while i<n:
        if not flag[i]: i+=1; continue
        start=i
        while i<n and flag[i]: i+=1
        end=i
        j=min(end,start+bars)-1
        if j>=start:
            move=ctx.close[j]-ctx.open[start]
            if move>0:s[j]=1
            elif move<0:s[j]=-1
    return s

def opening_range_breakout(ctx,p):
    name=str(p.get('session','london')); flag=_session_flag(ctx,name); minutes=int(p.get('range_minutes',15)); s=_sig(len(ctx)); n=len(ctx); i=0
    while i<n:
        if not flag[i]: i+=1; continue
        start=i
        while i<n and flag[i]: i+=1
        end=i; cut=min(end,start+minutes)
        if cut<=start: continue
        hi=np.max(ctx.high[start:cut]); lo=np.min(ctx.low[start:cut])
        for j in range(cut,end):
            if ctx.close[j]>hi:s[j]=1
            elif ctx.close[j]<lo:s[j]=-1
    return s

def prior_session_high_low_break(ctx,p):
    name=str(p.get('session','london')); buffer_atr=float(p.get('buffer_atr',0.0)); flag=_session_flag(ctx,name); s=_sig(len(ctx)); blocks=[]; n=len(ctx); i=0
    while i<n:
        if not flag[i]: i+=1; continue
        st=i
        while i<n and flag[i]:i+=1
        blocks.append((st,i))
    for bi in range(1,len(blocks)):
        pst,pen=blocks[bi-1]; st,en=blocks[bi]; hi=np.max(ctx.high[pst:pen]);lo=np.min(ctx.low[pst:pen])
        for j in range(st,en):
            buf=buffer_atr*max(ctx.atr14[j],1e-12)
            if ctx.close[j]>hi+buf:s[j]=1
            elif ctx.close[j]<lo-buf:s[j]=-1
    return s

_register('bollinger_expansion','breakout',bollinger_expansion,{'lookback':('int',10,80),'std_mult':('float',1,3),'bandwidth_percentile':('float',.5,.95)})
_register('compression_breakout','breakout',compression_breakout,{'compression_lookback':('int',10,100),'compression_percentile':('float',.05,.4),'breakout_lookback':('int',2,50)})
_register('session_open_momentum','session',session_open_momentum,{'session':('cat',('london','new_york','asia')),'opening_bars':('int',1,30)})
_register('opening_range_breakout','session',opening_range_breakout,{'session':('cat',('london','new_york','asia')),'range_minutes':('int',5,60)})
_register('prior_session_high_low_break','session',prior_session_high_low_break,{'session':('cat',('london','new_york','asia')),'buffer_atr':('float',0.0,0.5)})

# --- Additional standalone indicator baselines ---------------------------------
# These are intentionally independent signals. They are not combined with other
# filters during Stage 1; the common backtest engine supplies exits and costs.

def _ema_series(a,w):
    a=np.asarray(a,float); out=np.full(len(a),np.nan)
    if len(a)==0:return out
    alpha=2.0/(int(w)+1.0); out[0]=a[0]
    for i in range(1,len(a)): out[i]=alpha*a[i]+(1.0-alpha)*out[i-1]
    return out

def _wma_series(a,w):
    a=np.asarray(a,float); w=max(1,int(w)); out=np.full(len(a),np.nan); weights=np.arange(1,w+1,dtype=float); den=weights.sum()
    for i in range(w-1,len(a)):
        x=a[i-w+1:i+1]
        if np.all(np.isfinite(x)): out[i]=float(np.dot(x,weights)/den)
    return out

def _atr_series(ctx,w):
    w=max(1,int(w)); prev=np.r_[ctx.close[0],ctx.close[:-1]]; tr=np.maximum(ctx.high-ctx.low,np.maximum(np.abs(ctx.high-prev),np.abs(ctx.low-prev)))
    return _rolling_mean(tr,w)

def _slope_signal(series,ctx,horizon,threshold_atr):
    h=max(1,int(horizon)); th=float(threshold_atr); s=_sig(len(ctx))
    for i in range(h,len(ctx)):
        if not np.isfinite(series[i]) or not np.isfinite(series[i-h]): continue
        z=(series[i]-series[i-h])/max(ctx.atr14[i],1e-12)
        if z>th:s[i]=1
        elif z<-th:s[i]=-1
    return s

def wma_slope(ctx,p):
    w=int(p.get('lookback',20)); return _slope_signal(_wma_series(ctx.close,w),ctx,p.get('horizon',3),p.get('threshold_atr',0.0))

def hma_slope(ctx,p):
    w=max(4,int(p.get('lookback',20))); half=max(2,w//2); root=max(2,int(round(np.sqrt(w))))
    a=_wma_series(ctx.close,half); b=_wma_series(ctx.close,w); raw=2.0*a-b; hma=_wma_series(raw,root)
    return _slope_signal(hma,ctx,p.get('horizon',3),p.get('threshold_atr',0.0))

def kama_slope(ctx,p):
    w=max(2,int(p.get('lookback',20))); fast=max(2,int(p.get('fast',2))); slow=max(fast+1,int(p.get('slow',30)))
    a=np.asarray(ctx.close,float); out=np.full(len(a),np.nan)
    if len(a)<=w:return _sig(len(ctx))
    out[w-1]=a[w-1]; fast_sc=2.0/(fast+1.0); slow_sc=2.0/(slow+1.0)
    for i in range(w,len(a)):
        change=abs(a[i]-a[i-w]); volatility=float(np.abs(np.diff(a[i-w:i+1])).sum()); er=change/volatility if volatility>0 else 0.0
        sc=(er*(fast_sc-slow_sc)+slow_sc)**2; out[i]=out[i-1]+sc*(a[i]-out[i-1])
    return _slope_signal(out,ctx,p.get('horizon',3),p.get('threshold_atr',0.0))

def dema_slope(ctx,p):
    w=int(p.get('lookback',20)); e1=_ema_series(ctx.close,w); e2=_ema_series(e1,w); return _slope_signal(2.0*e1-e2,ctx,p.get('horizon',3),p.get('threshold_atr',0.0))

def tema_slope(ctx,p):
    w=int(p.get('lookback',20)); e1=_ema_series(ctx.close,w); e2=_ema_series(e1,w); e3=_ema_series(e2,w); return _slope_signal(3.0*e1-3.0*e2+e3,ctx,p.get('horizon',3),p.get('threshold_atr',0.0))

def macd_momentum(ctx,p):
    fast=max(2,int(p.get('fast',12))); slow=max(fast+1,int(p.get('slow',26))); sigw=max(2,int(p.get('signal',9))); th=float(p.get('min_hist_atr',0.0))
    ef=_ema_series(ctx.close,fast); es=_ema_series(ctx.close,slow); macd=ef-es; signal=_ema_series(macd,sigw); hist=macd-signal; s=_sig(len(ctx))
    warm=slow+sigw
    for i in range(min(warm,len(ctx)),len(ctx)):
        limit=th*max(ctx.atr14[i],1e-12)
        if macd[i]>0 and hist[i]>=limit:s[i]=1
        elif macd[i]<0 and hist[i]<=-limit:s[i]=-1
    return s

def adx_dmi_trend(ctx,p):
    w=max(2,int(p.get('lookback',14))); threshold=float(p.get('adx_threshold',20.0)); s=_sig(len(ctx)); n=len(ctx)
    up=np.zeros(n); dn=np.zeros(n); tr=np.zeros(n); prev_close=ctx.close[0]
    for i in range(1,n):
        u=ctx.high[i]-ctx.high[i-1]; d=ctx.low[i-1]-ctx.low[i]; up[i]=u if u>d and u>0 else 0.0; dn[i]=d if d>u and d>0 else 0.0
        tr[i]=max(ctx.high[i]-ctx.low[i],abs(ctx.high[i]-prev_close),abs(ctx.low[i]-prev_close)); prev_close=ctx.close[i]
    tr[0]=ctx.high[0]-ctx.low[0]; atr=_rolling_mean(tr,w); pu=_rolling_mean(up,w); pd=_rolling_mean(dn,w); plus=np.full(n,np.nan); minus=np.full(n,np.nan); dx=np.full(n,np.nan)
    valid=atr>0; plus[valid]=100*pu[valid]/atr[valid]; minus[valid]=100*pd[valid]/atr[valid]
    den=plus+minus; good=den>0; dx[good]=100*np.abs(plus[good]-minus[good])/den[good]; adx=_rolling_mean(np.nan_to_num(dx,nan=0.0),w)
    for i in range(2*w-2,n):
        if adx[i] < threshold: continue
        if plus[i]>minus[i]:s[i]=1
        elif minus[i]>plus[i]:s[i]=-1
    return s

def supertrend_direction(ctx,p):
    w=max(2,int(p.get('atr_period',14))); mult=float(p.get('multiplier',3.0)); atr=_atr_series(ctx,w); n=len(ctx); s=_sig(n); upper=np.full(n,np.nan); lower=np.full(n,np.nan); trend=1
    for i in range(w-1,n):
        mid=(ctx.high[i]+ctx.low[i])/2.0; bu=mid+mult*atr[i]; bl=mid-mult*atr[i]
        if i==w-1:
            upper[i]=bu; lower[i]=bl; trend=1 if ctx.close[i]>=mid else -1
        else:
            upper[i]=bu if bu<upper[i-1] or ctx.close[i-1]>upper[i-1] else upper[i-1]
            lower[i]=bl if bl>lower[i-1] or ctx.close[i-1]<lower[i-1] else lower[i-1]
            if trend<0 and ctx.close[i]>upper[i-1]:trend=1
            elif trend>0 and ctx.close[i]<lower[i-1]:trend=-1
        s[i]=trend
    return s

def ichimoku_cloud(ctx,p):
    conv=max(2,int(p.get('conversion',9))); base=max(conv+1,int(p.get('base',26))); span=max(base+1,int(p.get('span_b',52))); s=_sig(len(ctx))
    for i in range(span-1,len(ctx)):
        tenkan=(np.max(ctx.high[i-conv+1:i+1])+np.min(ctx.low[i-conv+1:i+1]))/2.0
        kijun=(np.max(ctx.high[i-base+1:i+1])+np.min(ctx.low[i-base+1:i+1]))/2.0
        sa=(tenkan+kijun)/2.0; sb=(np.max(ctx.high[i-span+1:i+1])+np.min(ctx.low[i-span+1:i+1]))/2.0
        top=max(sa,sb); bottom=min(sa,sb)
        if ctx.close[i]>top and tenkan>kijun:s[i]=1
        elif ctx.close[i]<bottom and tenkan<kijun:s[i]=-1
    return s

def keltner_reversion(ctx,p):
    w=max(2,int(p.get('lookback',20))); mult=float(p.get('atr_mult',1.5)); ema=_ema_series(ctx.close,w); atr=_atr_series(ctx,w); s=_sig(len(ctx))
    for i in range(w-1,len(ctx)):
        if ctx.close[i] > ema[i]+mult*atr[i]:s[i]=-1
        elif ctx.close[i] < ema[i]-mult*atr[i]:s[i]=1
    return s

def choppiness_trend(ctx,p):
    w=max(2,int(p.get('lookback',20))); threshold=float(p.get('max_chop',45.0)); s=_sig(len(ctx)); prev=np.r_[ctx.close[0],ctx.close[:-1]]; tr=np.maximum(ctx.high-ctx.low,np.maximum(np.abs(ctx.high-prev),np.abs(ctx.low-prev)))
    for i in range(w-1,len(ctx)):
        hi=np.max(ctx.high[i-w+1:i+1]); lo=np.min(ctx.low[i-w+1:i+1]); span=hi-lo
        if span<=0:continue
        chop=100.0*np.log10(max(np.sum(tr[i-w+1:i+1]),1e-12)/span)/np.log10(w)
        if chop<=threshold:
            net=ctx.close[i]-ctx.close[i-w+1]
            if net>0:s[i]=1
            elif net<0:s[i]=-1
    return s

_register('wma_slope','trend',wma_slope,{'lookback':('int',5,120),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('hma_slope','trend',hma_slope,{'lookback':('int',6,120),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('kama_slope','trend',kama_slope,{'lookback':('int',5,80),'fast':('int',2,5),'slow':('int',20,60),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('dema_slope','trend',dema_slope,{'lookback':('int',5,120),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('tema_slope','trend',tema_slope,{'lookback':('int',5,120),'horizon':('int',1,10),'threshold_atr':('float',0,.5)})
_register('macd_momentum','trend',macd_momentum,{'fast':('int',5,20),'slow':('int',21,60),'signal':('int',3,18),'min_hist_atr':('float',0,.15)})
_register('adx_dmi_trend','trend',adx_dmi_trend,{'lookback':('int',7,35),'adx_threshold':('float',10,40)})
_register('supertrend_direction','trend',supertrend_direction,{'atr_period':('int',7,35),'multiplier':('float',1,5)})
_register('ichimoku_cloud','trend',ichimoku_cloud,{'conversion':('int',5,15),'base':('int',20,35),'span_b':('int',40,70)})
_register('keltner_reversion','mean_reversion',keltner_reversion,{'lookback':('int',10,80),'atr_mult':('float',1,3.5)})
_register('choppiness_trend','volatility',choppiness_trend,{'lookback':('int',10,100),'max_chop':('float',30,55)})

def previous_day_high_low_break(ctx,p):
    if 'broker_date' not in ctx.features: raise ValueError('missing required session feature broker_date')
    dates=np.asarray(ctx.features['broker_date']); buffer_atr=float(p.get('buffer_atr',0.0)); s=_sig(len(ctx))
    if len(ctx)==0:return s
    # Work only with completed prior broker dates. Current-day high/low never enters its own threshold.
    starts=[0]
    for i in range(1,len(ctx)):
        if dates[i]!=dates[i-1]: starts.append(i)
    starts.append(len(ctx))
    for di in range(1,len(starts)-1):
        pst,pen=starts[di-1],starts[di]; st,en=starts[di],starts[di+1]
        hi=float(np.max(ctx.high[pst:pen])); lo=float(np.min(ctx.low[pst:pen]))
        for j in range(st,en):
            buf=buffer_atr*max(ctx.atr14[j],1e-12)
            if ctx.close[j]>hi+buf:s[j]=1
            elif ctx.close[j]<lo-buf:s[j]=-1
    return s

_register('previous_day_high_low_break','session',previous_day_high_low_break,{'buffer_atr':('float',0.0,0.75)})
