import numpy as np
import pytest
from xau_lab.backtest.models import SymbolSpec, CostModel, RiskModel, ExitSpec, MarketBars
from xau_lab.backtest.pricing import buy_entry_price, sell_entry_price
from xau_lab.backtest.sizing import size_for_stop
from xau_lab.backtest.reference import run_reference_backtest

SPEC=SymbolSpec(point=0.01,digits=2,contract_size=100.0,volume_min=0.01,volume_step=0.01)
RISK=RiskModel(account_equity=5000.0,preferred_risk_usd=2.0,hard_risk_usd=5.0)

def bars(o,h,l,c,sp=None,atr=None):
    n=len(o)
    return MarketBars(
        time_epoch=np.arange(n,dtype=np.int64)*60,
        open=np.array(o,float), high=np.array(h,float), low=np.array(l,float), close=np.array(c,float),
        spread=np.array(sp if sp is not None else [0]*n,float),
        atr=np.array(atr if atr is not None else [1]*n,float),
    )

def test_models_and_pricing_contract():
    with pytest.raises(ValueError):
        SymbolSpec(point=0,digits=2,contract_size=100,volume_min=.01,volume_step=.01)
    assert CostModel().commission_round_trip_per_lot == 6.0
    assert CostModel().slippage_points_per_fill == 5.0
    assert buy_entry_price(2000,35,SPEC,CostModel(slippage_points_per_fill=5)) == pytest.approx(2000.40)
    assert sell_entry_price(2000,35,SPEC,CostModel(slippage_points_per_fill=5)) == pytest.approx(1999.95)

def test_minlot_hard_risk_skip():
    s=size_for_stop(2000,1994,SPEC,RISK)
    assert not s.feasible and s.lot == 0

def test_next_bar_long_target_and_same_bar_stop_first():
    b=bars([100,100,100],[101,100.5,103],[99,99.5,99.5],[100,100,102])
    r=run_reference_backtest(b,[1,0,0],SPEC,CostModel(0,0),RISK,ExitSpec(stop_atr=1,target_r=2))
    assert len(r.trades)==1 and r.trades[0].entry_index==1 and r.trades[0].exit_reason=='target'
    b2=bars([100,100,100],[101,100.2,103],[99,99.8,97],[100,100,100])
    r2=run_reference_backtest(b2,[1,0,0],SPEC,CostModel(0,0),RISK,ExitSpec(stop_atr=1,target_r=2))
    assert r2.trades[0].exit_reason=='stop_same_bar_ambiguous' and r2.trades[0].net_pnl<0

def test_time_exit_and_one_position():
    b=bars([100]*8,[100.1]*8,[99.9]*8,[100]*8,atr=[1]*8)
    r=run_reference_backtest(b,[1,-1,1,1,1,1,0,0],SPEC,CostModel(0,0),RISK,ExitSpec(stop_atr=1,time_exit_minutes=5))
    assert len(r.trades)==1
    assert r.trades[0].exit_reason=='time'
    assert r.trades[0].exit_index-r.trades[0].entry_index==5

def test_short_stop_trigger_uses_ask_side_spread():
    # Short enters at 100.00 bid. Stop is 101.00. On bar 2 bid-high is only 100.95,
    # but with a 10-point (0.10) spread ask-high is 101.05, so stop must trigger.
    b=bars([100,100,100],[100.2,100.2,100.95],[99.8,99.8,99.7],[100,100,100],sp=[10,10,10],atr=[1,1,1])
    r=run_reference_backtest(b,[-1,0,0],SPEC,CostModel(0,0),RISK,ExitSpec(stop_atr=1,target_r=1))
    assert len(r.trades)==1
    assert r.trades[0].exit_reason=='stop'
    assert r.trades[0].exit_index==2
