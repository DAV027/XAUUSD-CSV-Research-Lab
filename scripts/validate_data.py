from pathlib import Path
import argparse,json,os,polars as pl
from xau_lab.data.schema import DataPaths
from xau_lab.data.validate import validate_bars
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args(); paths=DataPaths(a.root); r=validate_bars(pl.read_csv(paths.raw_csv)); paths.clean_csv.parent.mkdir(parents=True,exist_ok=True); paths.quality_csv.parent.mkdir(parents=True,exist_ok=True); r.clean.write_csv(paths.clean_csv); r.issues.write_csv(paths.quality_csv)
    meta={};
    if paths.metadata_json.exists(): meta=json.loads(paths.metadata_json.read_text())
    meta.update(r.summary); tmp=paths.metadata_json.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2,sort_keys=True));os.replace(tmp,paths.metadata_json)
if __name__=='__main__':main()
