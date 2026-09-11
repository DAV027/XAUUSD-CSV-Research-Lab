import numpy as np
from xau_lab.backtest.models import MarketBars, SymbolSpec, RiskModel
from xau_lab.strategies.base import StrategyContext
from xau_lab.experiments.sampler import generate_catalog
from xau_lab.runner.single import MarketBundle, run_experiment
from xau_lab.runner.campaign import run_campaign_experiments
from xau_lab.io.results import ResultStore

def market(n=600):
    rng=np.random.default_rng(123); close=2000+np.cumsum(rng.normal(0,.4,n)); op=np.r_[close[0],close[:-1]]
    hi=np.maximum(op,close)+.2; lo=np.minimum(op,close)-.2; atr=np.full(n,1.0); sp=np.full(n,35.0); t=np.arange(n,dtype=np.int64)*60+1700000000
    bars=MarketBars(t,op,hi,lo,close,sp,atr)
    ctx=StrategyContext(op,hi,lo,close,sp,atr,t,{})
    return MarketBundle(bars,ctx,SymbolSpec(.01,2,100,.01,.01),RiskModel())

def test_single_experiment_returns_canonical_result():
    e=generate_catalog(1,9215000)[0]
    o=run_experiment(e,market())
    assert o.result_row['experiment_id']==e.experiment_id
    assert 'profit_factor' in o.result_row

def test_campaign_resume_and_worker_parity(tmp_path):
    exps=generate_catalog(20,9215000)
    m=market()
    one=tmp_path/'one'; two=tmp_path/'two'
    run_campaign_experiments(exps,m,one,workers=1)
    run_campaign_experiments(exps[:10],m,two,workers=1)
    run_campaign_experiments(exps,m,two,workers=2)
    a=sorted(ResultStore(one).read_rows(),key=lambda r:r['experiment_id'])
    b=sorted(ResultStore(two).read_rows(),key=lambda r:r['experiment_id'])
    for rows in (a,b):
        for r in rows:r.pop('runtime_seconds',None)
    assert len(a)==len(b)==20
    assert a==b
