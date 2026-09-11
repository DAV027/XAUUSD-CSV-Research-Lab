from pathlib import Path
import argparse,json,os,polars as pl
from xau_lab.data.schema import DataPaths
from xau_lab.data.features import build_shared_features
from xau_lab.data.sessions import SessionConfig
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--session-config',type=Path);a=p.parse_args(); paths=DataPaths(a.root); cfg=SessionConfig(**json.loads(a.session_config.read_text())) if a.session_config else None; out=build_shared_features(pl.read_csv(paths.clean_csv),cfg); paths.features_parquet.parent.mkdir(parents=True,exist_ok=True); tmp=paths.features_parquet.with_suffix('.tmp.parquet');out.write_parquet(tmp,compression='zstd');os.replace(tmp,paths.features_parquet)
if __name__=='__main__':main()
