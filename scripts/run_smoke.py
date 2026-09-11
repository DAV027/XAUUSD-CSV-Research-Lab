from pathlib import Path
import argparse,json,os,time
from xau_lab.experiments.catalog import read_catalog
from xau_lab.runner.market import load_market_bundle
from xau_lab.runner.campaign import run_campaign_experiments
from xau_lab.runner.preflight import validate_campaign_inputs
from xau_lab.io.manifest import write_run_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--count',type=int,default=100);p.add_argument('--workers',type=int);a=p.parse_args();workers=a.workers or max(1,(os.cpu_count() or 2)-1)
    cat=a.root/'results/EXPERIMENT_CATALOG.csv';feat=a.root/'data/features/XAUUSD_M1_FEATURES.parquet';exps=read_catalog(cat);market=load_market_bundle(feat);report=validate_campaign_inputs(market,exps,expected_catalog_size=50_000);print(f'PREFLIGHT OK | market_rows={report.market_rows} | catalog_rows={report.catalog_rows}');write_run_manifest(a.root,feat,cat,workers);st=time.perf_counter();run_campaign_experiments(exps,market,a.root/'results',workers=workers,limit=a.count);elapsed=time.perf_counter()-st
    benchmark={'experiments':a.count,'elapsed_seconds':elapsed,'experiments_per_minute':a.count/elapsed*60 if elapsed else None,'estimated_50000_hours':elapsed/max(1,a.count)*50000/3600,'workers':workers,'cpu_count':os.cpu_count(),'data_rows':len(market.bars.open)}
    path=a.root/'results/THROUGHPUT_BENCHMARK.json';path.write_text(json.dumps(benchmark,indent=2,sort_keys=True));print(json.dumps(benchmark,indent=2));
    if benchmark['estimated_50000_hours'] and benchmark['estimated_50000_hours']>24: print('STOP: projected 50,000-run time exceeds 24 hours; profile before full campaign.')
if __name__=='__main__':main()
