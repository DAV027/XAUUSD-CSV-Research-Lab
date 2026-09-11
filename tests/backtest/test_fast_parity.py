import numpy as np
from xau_lab.backtest.models import MarketBars,SymbolSpec,CostModel,RiskModel,ExitSpec
from xau_lab.backtest.reference import run_reference_backtest
from xau_lab.backtest.fast import run_fast_backtest

def test_fast_matches_reference_randomized():
    rng=np.random.default_rng(9212000);n=2000
    close=2000+np.cumsum(rng.normal(0,.35,n));op=np.r_[close[0],close[:-1]];span=np.abs(rng.normal(.4,.15,n))+0.05;hi=np.maximum(op,close)+span;lo=np.minimum(op,close)-span
    b=MarketBars(np.arange(n,dtype=np.int64)*60,op,hi,lo,close,rng.integers(20,50,n).astype(float),np.full(n,1.0))
    sig=rng.choice(np.array([-1,0,1],dtype=np.int8),n,p=[.05,.9,.05])
    sym=SymbolSpec(.01,2,100,.01,.01);cost=CostModel(6,5);risk=RiskModel()
    for ex in [ExitSpec(1.0,target_r=1.5),ExitSpec(1.0,time_exit_minutes=15),ExitSpec(1.0,atr_trail=1.0)]:
        a=run_reference_backtest(b,sig,sym,cost,risk,ex);f=run_fast_backtest(b,sig,sym,cost,risk,ex)
        assert a.risk_skip_count==f.risk_skip_count
        assert len(a.trades)==len(f.trades)
        for x,y in zip(a.trades,f.trades):
            assert (x.entry_index,x.exit_index,x.direction,x.exit_reason)==(y.entry_index,y.exit_index,y.direction,y.exit_reason)
            assert abs(x.net_pnl-y.net_pnl)<1e-9
