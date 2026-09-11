from __future__ import annotations
from pathlib import Path
import csv,json
from dataclasses import asdict

def export_finalist(experiment,metrics,robustness,source_data_sha256,result_root,trades=()):
    root=Path(result_root)/'finalists'/experiment.experiment_id; root.mkdir(parents=True,exist_ok=True)
    payload={'experiment':experiment.to_dict(),'discovery_metrics':metrics,'robustness':robustness,'source_data_sha256':source_data_sha256,'score_version':'v1','required_mt5_delays_ms':[0,100,250,500,1000],'mt5_model':'Every tick based on real ticks','real_money_approval_required':True}
    (root/'candidate.json').write_text(json.dumps(payload,sort_keys=True,indent=2,default=str))
    fields=['entry_time','exit_time','direction','entry_price','stop_price','target_price','exit_price','exit_reason','lot_size','net_pnl','pnl_R','hold_minutes']
    with (root/'candidate_trades.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t in trades:w.writerow({k:getattr(t,k,'') for k in fields})
    (root/'MT5_VALIDATION_CHECKLIST.md').write_text('# MT5 Finalist Validation\n\n- Model: Every tick based on real ticks\n- Execution delays: 0, 100, 250, 500, 1000 ms\n- Reproduce exact frozen strategy/parameters.\n- Verify costs, spread, slippage, fill-based risk, and broker symbol.\n- Do not use real money without explicit approval.\n')
    return root
