from pathlib import Path
import argparse,time
from xau_lab.experiments.sampler import generate_catalog
from xau_lab.experiments.catalog import write_catalog
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--budget',type=int,default=50000);p.add_argument('--seed',type=int,default=9215000);a=p.parse_args();t=time.perf_counter();cat=generate_catalog(a.budget,a.seed);path=write_catalog(cat,a.root/'results/EXPERIMENT_CATALOG.csv');print(f'wrote {len(cat)} unique complete experiments to {path} in {time.perf_counter()-t:.2f}s')
if __name__=='__main__':main()
