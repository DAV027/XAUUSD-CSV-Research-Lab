from xau_lab.io.results import ResultStore
from xau_lab.io.checkpoint import CheckpointStore

def test_resume_and_duplicate_rejection(tmp_path):
    ids=[f'EXP{i}' for i in range(1,11)]
    r=ResultStore(tmp_path)
    c=CheckpointStore(tmp_path)
    for x in ids[:4]: r.append({'experiment_id':x,'net_profit':0.0})
    c.write(ids[:4])
    r2=ResultStore(tmp_path)
    assert r2.pending_ids(ids)==ids[4:]
    try:
        r2.append({'experiment_id':'EXP4','net_profit':0.0})
        assert False
    except ValueError: pass
