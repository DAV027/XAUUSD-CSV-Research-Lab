from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import polars as pl
@dataclass(frozen=True)
class SessionConfig:
    broker_timezone:str|None
    asia_start:str='00:00'; asia_end:str='08:00'; london_start:str='08:00'; london_end:str='17:00'; new_york_start:str='13:30'; new_york_end:str='22:00'

def _mins(s): h,m=map(int,s.split(':')); return h*60+m
def _inside(x,a,b): return a<=x<b if a<=b else (x>=a or x<b)
def add_session_features(frame:pl.DataFrame,config:SessionConfig)->pl.DataFrame:
    if config.broker_timezone is None: raise ValueError('broker timezone must be explicit before session research')
    ZoneInfo(config.broker_timezone)
    vals=[datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') for x in frame['time'].to_list()]
    asia=(_mins(config.asia_start),_mins(config.asia_end)); london=(_mins(config.london_start),_mins(config.london_end)); ny=(_mins(config.new_york_start),_mins(config.new_york_end))
    mins=[d.hour*60+d.minute for d in vals]; sa=[_inside(x,*asia) for x in mins]; sl=[_inside(x,*london) for x in mins]; sn=[_inside(x,*ny) for x in mins]
    return frame.with_columns(
        pl.Series('broker_date',[d.date().isoformat() for d in vals]),pl.Series('year',[d.year for d in vals]),pl.Series('month',[d.month for d in vals]),pl.Series('weekday',[d.weekday() for d in vals]),pl.Series('hour',[d.hour for d in vals]),pl.Series('minute',[d.minute for d in vals]),pl.Series('session_asia',sa),pl.Series('session_london',sl),pl.Series('session_new_york',sn),pl.Series('session_overlap',[a and b for a,b in zip(sl,sn)])
    )
