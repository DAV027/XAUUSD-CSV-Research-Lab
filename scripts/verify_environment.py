from __future__ import annotations
import importlib,sys,platform
from pathlib import Path
REQ=['numpy','scipy','numba','polars','pyarrow','MetaTrader5']
def main():
    print('Python:',sys.version.replace('\n',' '));print('Platform:',platform.platform());ok=True
    for name in REQ:
        try:
            m=importlib.import_module(name);print(f'[OK] {name} {getattr(m,"__version__","")}')
        except Exception as e:
            ok=False;print(f'[MISSING] {name}: {e}')
    print('[OK] project package' if Path('xau_lab').exists() else '[MISSING] run from project root')
    raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
