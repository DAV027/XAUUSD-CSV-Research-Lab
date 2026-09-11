from __future__ import annotations
from pathlib import Path
import csv,hashlib,json,re,zipfile

FIELDS=['historical_id','origin','strategy_family','strategy_name','status','net_profit','profit_factor','after_cost_profit','after_cost_pf','trades','max_drawdown_pct','source_file','manifest_file','results_file','evidence_notes']

def _norm_name(s): return ''.join(re.findall(r'[a-z0-9]+',str(s).lower()))

def import_historical_checkpoint(zip_path,result_root):
    zip_path=Path(zip_path); result_root=Path(result_root); result_root.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as z:
        names=sorted(z.namelist())
        ledger=[]
        if 'outputs/experiment_ledger.jsonl' in names:
            for line in z.read('outputs/experiment_ledger.jsonl').decode('utf-8',errors='replace').splitlines():
                line=line.strip()
                if line:
                    try: ledger.append(json.loads(line))
                    except json.JSONDecodeError: pass
        by_id={}
        for row in ledger:
            hid=str(row.get('experiment_id') or row.get('id') or row.get('historical_id') or '')
            if hid: by_id[hid]=row
        lab_ids=sorted(set(re.findall(r'(LAB\d+[A-Z]?)_(?:results|manifest)\.json', '\n'.join(names))))
        for hid in lab_ids: by_id.setdefault(hid,{})
        rows=[]
        for hid in sorted(by_id):
            base=by_id[hid]
            rname=next((n for n in names if n.endswith(f'{hid}_results.json')),None)
            mname=next((n for n in names if n.endswith(f'{hid}_manifest.json')),None)
            results={}; manifest={}
            if rname:
                try: results=json.loads(z.read(rname))
                except Exception: results={}
            if mname:
                try: manifest=json.loads(z.read(mname))
                except Exception: manifest={}
            def pick(*keys):
                for src in (results,base,manifest):
                    for k in keys:
                        if k in src and src[k] is not None: return src[k]
                return ''
            rows.append({
                'historical_id':hid,'origin':'historical_mt5',
                'strategy_family':pick('strategy_family','family'),
                'strategy_name':pick('strategy_name','strategy','name','label'),
                'status':pick('status','verdict'),'net_profit':pick('net_profit','net_pnl'),
                'profit_factor':pick('profit_factor','pf'),'after_cost_profit':pick('after_cost_profit','net_after_cost'),
                'after_cost_pf':pick('after_cost_pf','pf_after_cost'),'trades':pick('trades','completed_trades'),
                'max_drawdown_pct':pick('max_drawdown_pct','drawdown_pct'),
                'source_file':pick('source_file','mq5_file'),'manifest_file':mname or '', 'results_file':rname or '',
                'evidence_notes':pick('notes','evidence_notes'),
            })
    out=result_root/'HISTORICAL_EVIDENCE.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    manifest_path=result_root/'HISTORICAL_IMPORT_MANIFEST.json'
    manifest_path.write_text(json.dumps({'source_zip':str(zip_path),'source_zip_sha256':sha,'imported_members':names,'row_count':len(rows)},indent=2,sort_keys=True),encoding='utf-8')
    return rows

def historical_matches(strategy_name,canonical_parameters,evidence):
    if hasattr(evidence,'to_dicts'): rows=evidence.to_dicts()
    elif hasattr(evidence,'to_dict'): rows=evidence.to_dict('records')
    else: rows=list(evidence)
    target=_norm_name(strategy_name); out=[]
    for r in rows:
        n=_norm_name(r.get('strategy_name',''))
        if not n: continue
        if n==target or n in target or target in n:
            out.append(str(r.get('historical_id','')))
    return sorted(set(x for x in out if x))
