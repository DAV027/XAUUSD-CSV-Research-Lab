from pathlib import Path
import argparse,csv,json,statistics
from datetime import datetime,timezone
from xau_lab.experiments.catalog import read_catalog
from xau_lab.runner.market import load_market_bundle
from xau_lab.runner.single import run_experiment
from xau_lab.validation.walkforward import validate_candidate,summarize_fold_results
from xau_lab.validation.stress import run_stress_suite
from xau_lab.validation.resampling import block_bootstrap_daily,trade_order_monte_carlo,top_trade_removal
from xau_lab.validation.reporting import robust_candidate_gate
from xau_lab.validation.scoring import score_candidate
from xau_lab.io.trades import write_trades_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--limit',type=int);a=p.parse_args();
    with (a.root/'results/SURVIVORS.csv').open(newline='',encoding='utf-8') as f: survivors=list(csv.DictReader(f))
    if a.limit is not None: survivors=survivors[:a.limit]
    catalog={e.experiment_id:e for e in read_catalog(a.root/'results/EXPERIMENT_CATALOG.csv')};market=load_market_bundle(a.root/'data/features/XAUUSD_M1_FEATURES.parquet');summaries={};fold_rows=[];stress_rows=[];top=[]
    for row in survivors:
        eid=row['experiment_id'];e=catalog[eid];exp_rows=validate_candidate(e,market,'expanding');roll_rows=validate_candidate(e,market,'rolling');fold_rows += exp_rows+roll_rows; exs=summarize_fold_results(exp_rows);ros=summarize_fold_results(roll_rows);stress=run_stress_suite(e,market);stress_rows += [{'experiment_id':eid,**r} for r in stress['rows']]
        full=run_experiment(e,market,capture_trades=True);trades=full.trades
        daily={}
        for t in trades:
            d=datetime.fromtimestamp(t.exit_time,timezone.utc).date().isoformat();daily[d]=daily.get(d,0)+t.net_pnl
        removal=top_trade_removal([t.net_pnl for t in trades]);wf_good=[]
        for x in (exs,ros): wf_good.append(float(x.get('validation_pass_fraction') or 0))
        robust={'expanding_pass_fraction':wf_good[0],'rolling_pass_fraction':wf_good[1],'median_validation_pf':statistics.median([x for x in [exs.get('median_validation_pf'),ros.get('median_validation_pf')] if x is not None]) if any(x is not None for x in [exs.get('median_validation_pf'),ros.get('median_validation_pf')]) else None,'parameter_stability_score':stress['parameter_stability_score'],'cost_stability_score':stress['cost_stability_score'],'stress_max_drawdown_pct':stress['stress_max_drawdown_pct'],'top5_removed_net_profit':removal['remove_top_5_net'],'bootstrap':block_bootstrap_daily(daily),'trade_order_monte_carlo':trade_order_monte_carlo([t.net_pnl for t in trades]),'top_trade_removal':removal}
        base={k:(float(v) if isinstance(v,str) and v not in ('','None') and _isnum(v) else v) for k,v in row.items()}
        gate=robust_candidate_gate({**base,**robust}); robust_component=15*((wf_good[0]+wf_good[1])/2 + (stress['parameter_stability_score'] or 0)/100 + (stress['cost_stability_score'] or 0)/100)/3; score=score_candidate(base,robustness_component=robust_component);summary={**robust,'robustness_component':robust_component,'gate':gate,'score':score};summaries[eid]=summary;write_trades_csv(trades,a.root/'results/trade_logs'/f'{eid}.csv')
        if gate['passed']:top.append({**row,**score,'verdict':'ROBUST_CANDIDATE'})
    if fold_rows:
        fields=list(fold_rows[0].__dataclass_fields__)
        with (a.root/'results/FOLD_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([x.__dict__ for x in fold_rows])
    if stress_rows:
        fields=sorted(set().union(*(r.keys() for r in stress_rows)))
        with (a.root/'results/STRESS_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(stress_rows)
    (a.root/'results/ROBUSTNESS_SUMMARY.json').write_text(json.dumps(summaries,indent=2,sort_keys=True,default=str))
    if top:
        fields=sorted(set().union(*(r.keys() for r in top)))
        with (a.root/'results/TOP_CANDIDATES.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(top)
    else:(a.root/'results/TOP_CANDIDATES.csv').write_text('experiment_id,verdict\n')
    print(f'robust candidates={len(top)} of {len(survivors)} survivors')

def _isnum(v):
    try:float(v);return True
    except:return False
if __name__=='__main__':main()
