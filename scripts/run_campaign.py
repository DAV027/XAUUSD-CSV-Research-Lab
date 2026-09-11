from pathlib import Path
import argparse,os
from xau_lab.experiments.catalog import read_catalog
from xau_lab.runner.market import load_market_bundle
from xau_lab.runner.campaign import run_campaign_experiments
from xau_lab.runner.preflight import validate_campaign_inputs
from xau_lab.io.manifest import write_run_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--workers',type=int);p.add_argument('--limit',type=int);a=p.parse_args();workers=a.workers or max(1,(os.cpu_count() or 2)-1)
    cat=a.root/'results/EXPERIMENT_CATALOG.csv'; feat=a.root/'data/features/XAUUSD_M1_FEATURES.parquet'; exps=read_catalog(cat); market=load_market_bundle(feat); report=validate_campaign_inputs(market,exps,expected_catalog_size=50_000); print(f'PREFLIGHT OK | market_rows={report.market_rows} | catalog_rows={report.catalog_rows}'); print(write_run_manifest(a.root,feat,cat,workers));run_campaign_experiments(exps,market,a.root/'results',workers=workers,limit=a.limit)
if __name__=='__main__':main()
