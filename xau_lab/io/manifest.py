from __future__ import annotations
from pathlib import Path
import hashlib,json,os,subprocess
from datetime import datetime,timezone

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def git_commit(root):
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None
def write_run_manifest(root,feature_path,catalog_path,workers,seed=9215000):
    root=Path(root); out=root/'results/RUN_MANIFEST.json';out.parent.mkdir(parents=True,exist_ok=True)
    payload={'source_data_path':str(Path(feature_path).resolve()),'source_data_sha256':sha256_file(feature_path),'catalog_path':str(Path(catalog_path).resolve()),'catalog_sha256':sha256_file(catalog_path),'git_commit':git_commit(root),'workers':workers,'cost_model':{'commission_round_trip_per_lot':6.0,'slippage_points_per_fill':5.0,'spread':'recorded_M1'},'account_risk_model':{'equity':5000.0,'preferred_risk_usd':2.0,'hard_risk_usd':5.0,'max_drawdown_promotion_pct':5.0},'campaign_seed':seed,'start_timestamp_utc':datetime.now(timezone.utc).isoformat()}
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(payload,sort_keys=True,indent=2));os.replace(tmp,out);return payload
