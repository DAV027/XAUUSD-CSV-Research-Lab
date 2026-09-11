from xau_lab.experiments.sampler import generate_catalog

def test_catalog_deterministic_unique_budgeted():
    a=generate_catalog(1000,9215000)
    b=generate_catalog(1000,9215000)
    assert a==b
    assert len(a)==1000
    assert len({x.fingerprint for x in a})==1000
    assert len({x.experiment_id for x in a})==1000

def test_catalog_is_stable_across_python_hash_seeds(tmp_path):
    import os, subprocess, sys
    code="from xau_lab.experiments.sampler import generate_catalog; print('|'.join(x.fingerprint for x in generate_catalog(50,9215000)))"
    env=dict(os.environ); env['PYTHONPATH']=os.getcwd()
    env['PYTHONHASHSEED']='1'; a=subprocess.check_output([sys.executable,'-c',code],env=env,text=True)
    env['PYTHONHASHSEED']='999'; b=subprocess.check_output([sys.executable,'-c',code],env=env,text=True)
    assert a==b
