from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime,timezone
import json,os
import polars as pl
from .schema import DataPaths,SymbolMetadata,RAW_COLUMNS,canonicalize_bar_frame

def _atomic_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8'); os.replace(tmp,path)

def export_all_m1(provider,symbol:str,paths:DataPaths,server_timezone:str,chunk_size:int=100_000,overwrite:bool=False)->SymbolMetadata:
    tz=ZoneInfo(server_timezone)
    if paths.raw_csv.exists() and not overwrite: raise FileExistsError(paths.raw_csv)
    if not provider.initialize(): raise RuntimeError('MetaTrader5 initialize() failed')
    try:
        info=provider.symbol_info(symbol)
        if info is None: raise RuntimeError(f'symbol_info({symbol!r}) returned None')
        chunks=[]; pos=0
        while True:
            arr=provider.copy_rates_from_pos(symbol,provider.TIMEFRAME_M1,pos,chunk_size)
            if arr is None or len(arr)==0: break
            chunks.append(arr); pos+=len(arr)
            if len(arr)<chunk_size: break
        if not chunks: raise RuntimeError('no M1 history returned')
        rows=[]
        for arr in chunks:
            for r in arr:
                epoch=int(r['time']); dt=datetime.fromtimestamp(epoch,timezone.utc).astimezone(tz)
                rows.append({'time':dt.strftime('%Y-%m-%d %H:%M:%S'),'_epoch':epoch,'open':float(r['open']),'high':float(r['high']),'low':float(r['low']),'close':float(r['close']),'tick_volume':int(r['tick_volume']),'spread':int(r['spread']),'real_volume':int(r['real_volume'])})
        by_epoch={r['_epoch']:r for r in rows}; ordered=[by_epoch[k] for k in sorted(by_epoch)]
        frame=pl.DataFrame([{k:v for k,v in r.items() if k!='_epoch'} for r in ordered])
        frame=canonicalize_bar_frame(frame)
        paths.raw_csv.parent.mkdir(parents=True,exist_ok=True); frame.write_csv(paths.raw_csv)
        meta=SymbolMetadata(symbol,'M1',int(info.digits),float(info.point),float(info.trade_contract_size),float(info.volume_min),float(info.volume_max),float(info.volume_step),str(info.currency_profit),server_timezone,frame.height,str(frame['time'][0]),str(frame['time'][-1]))
        _atomic_json(paths.metadata_json,meta.to_dict()); return meta
    finally:
        provider.shutdown()
