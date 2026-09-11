from pathlib import Path
import argparse,csv
from xau_lab.experiments.catalog import read_catalog
from xau_lab.runner.market import load_market_bundle
from xau_lab.validation.walkforward import validate_candidate

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--experiment-id',required=True);p.add_argument('--scheme',choices=['expanding','rolling','both'],default='both');a=p.parse_args()
    exps={e.experiment_id:e for e in read_catalog(a.root/'results/EXPERIMENT_CATALOG.csv')};e=exps[a.experiment_id];m=load_market_bundle(a.root/'data/features/XAUUSD_M1_FEATURES.parquet');schemes=['expanding','rolling'] if a.scheme=='both' else [a.scheme];rows=[]
    for s in schemes: rows.extend(validate_candidate(e,m,s))
    path=a.root/'results/FOLD_RESULTS.csv'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',newline='',encoding='utf-8') as f:
        fields=list(rows[0].__dataclass_fields__) if rows else [];w=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0:w.writeheader()
        for r in rows:w.writerow(r.__dict__)
    print(f'wrote {len(rows)} fold rows to {path}')
if __name__=='__main__':main()
