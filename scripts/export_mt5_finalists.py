from pathlib import Path
import argparse,csv,json
from xau_lab.experiments.catalog import read_catalog
from xau_lab.io.finalist import export_finalist

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--experiment-id',required=True);p.add_argument('--source-sha256',required=True);a=p.parse_args(); exps={e.experiment_id:e for e in read_catalog(a.root/'results/EXPERIMENT_CATALOG.csv')}; e=exps[a.experiment_id]
    rows=[]
    with (a.root/'results/MASTER_RESULTS.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    metric=next(r for r in rows if r['experiment_id']==a.experiment_id); robust={}
    rpath=a.root/'results/ROBUSTNESS_SUMMARY.json'
    if rpath.exists(): robust=json.loads(rpath.read_text()).get(a.experiment_id,{})
    out=export_finalist(e,metric,robust,a.source_sha256,a.root/'results');print(out)
if __name__=='__main__':main()
