from pathlib import Path
import argparse
from xau_lab.experiments.history import import_historical_checkpoint
def main():
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();rows=import_historical_checkpoint(a.zip,a.root/'results');print(f'imported {len(rows)} historical experiment rows')
if __name__=='__main__':main()
