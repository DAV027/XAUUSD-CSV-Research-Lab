from pathlib import Path
import argparse
from xau_lab.data.schema import DataPaths
from xau_lab.data.mt5_export import export_all_m1
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path.cwd()); p.add_argument('--symbol',default='XAUUSD'); p.add_argument('--server-timezone',required=True); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
    import MetaTrader5 as mt5
    print(export_all_m1(mt5,a.symbol,DataPaths(a.root),a.server_timezone,overwrite=a.overwrite))
if __name__=='__main__':main()
