from pathlib import Path
import json,os,tempfile
class CheckpointStore:
    def __init__(self,root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.path=self.root/'CHECKPOINT.json'
    def write(self,completed_ids):
        payload={'completed_ids':sorted(set(completed_ids)),'completed_count':len(set(completed_ids))}
        tmp=self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,sort_keys=True,indent=2),encoding='utf-8')
        with tmp.open('r+') as f: f.flush(); os.fsync(f.fileno())
        os.replace(tmp,self.path)
    def read(self):
        if not self.path.exists(): return {'completed_ids':[],'completed_count':0}
        return json.loads(self.path.read_text(encoding='utf-8'))
