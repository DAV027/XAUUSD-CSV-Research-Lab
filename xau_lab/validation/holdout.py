from __future__ import annotations
from pathlib import Path
import json,os
class HoldoutRegistry:
    def __init__(self,path): self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self): return json.loads(self.path.read_text()) if self.path.exists() else {}
    def _save(self,d):
        t=self.path.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True,indent=2));os.replace(t,self.path)
    def freeze(self,experiment_id,freeze_timestamp,source_data_sha256,parameter_fingerprint,prospective_start,planned_months=3):
        d=self._load()
        row={'experiment_id':experiment_id,'freeze_timestamp':freeze_timestamp,'source_data_sha256':source_data_sha256,'parameter_fingerprint':parameter_fingerprint,'prospective_start':prospective_start,'planned_months':planned_months,'status':'frozen','observed_through':None}
        if experiment_id in d:
            old=d[experiment_id]
            for k in ('freeze_timestamp','source_data_sha256','parameter_fingerprint','prospective_start','planned_months'):
                if old.get(k)!=row.get(k): raise ValueError('frozen holdout fields are immutable')
            return old
        d[experiment_id]=row;self._save(d);return row
    def update_observed_through(self,experiment_id,date):
        d=self._load();r=d[experiment_id]
        if date<r['prospective_start']: raise ValueError('observed_through cannot precede prospective_start')
        if r.get('observed_through') and date<r['observed_through']: raise ValueError('observed_through cannot move backward')
        r['observed_through']=date;d[experiment_id]=r;self._save(d);return r
