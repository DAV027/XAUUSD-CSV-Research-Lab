from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from xau_lab.backtest.models import MarketBars,SymbolSpec,RiskModel
from xau_lab.strategies.base import StrategyContext
from xau_lab.runner.single import MarketBundle

def _atr14(h,l,c):
    prev=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(np.abs(h-prev),np.abs(l-prev))); out=np.full(len(c),np.nan)
    for i in range(13,len(c)): out[i]=tr[i-13:i+1].mean()
    if len(c)>=14: out[:13]=out[13]
    else: out[:]=np.nanmean(tr) if len(tr) else 0
    return out

def load_market_bundle(feature_path,symbol=None,risk=None,metadata_path=None):
    import polars as pl
    feature_path=Path(feature_path)
    df=pl.read_parquet(feature_path)
    time_col=df['time']
    from datetime import datetime,timezone
    t=np.array([int(datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()) for x in time_col.to_list()],dtype=np.int64)
    o=np.asarray(df['open'],float);h=np.asarray(df['high'],float);l=np.asarray(df['low'],float);c=np.asarray(df['close'],float);sp=np.asarray(df['spread'],float);atr=_atr14(h,l,c)
    bars=MarketBars(t,o,h,l,c,sp,atr)
    base={'time','open','high','low','close','tick_volume','spread','real_volume'}
    feats={col:np.asarray(df[col]) for col in df.columns if col not in base}
    ctx=StrategyContext(o,h,l,c,sp,atr,t,feats)
    if symbol is None:
        if metadata_path is None:
            root=feature_path.parents[2] if len(feature_path.parents)>=3 else Path.cwd()
            candidate=root/'results/DATA_METADATA.json'
            metadata_path=candidate if candidate.exists() else None
        if metadata_path is not None and Path(metadata_path).exists():
            md=json.loads(Path(metadata_path).read_text())
            symbol=SymbolSpec(float(md['point']),int(md['digits']),float(md['trade_contract_size']),float(md['volume_min']),float(md['volume_step']),float(md.get('volume_max',100.0)))
        else:
            symbol=SymbolSpec(.01,2,100,.01,.01)
    return MarketBundle(bars,ctx,symbol,risk or RiskModel())
