from datetime import datetime, timezone
import numpy as np
from xau_lab.validation.promotion import stage1_gate
from xau_lab.validation.folds import rolling_folds, expanding_folds
from xau_lab.validation.scoring import score_candidate

def months(start_year=2023, n=36):
    out=[]
    y=start_year; m=1
    for _ in range(n):
        out.append(int(datetime(y,m,1,tzinfo=timezone.utc).timestamp()))
        m+=1
        if m==13: m=1; y+=1
    return np.array(out,dtype=np.int64)

def test_stage1_gate_balanced_thresholds():
    ok={'profit_factor':1.2,'max_drawdown_pct':4,'completed_trades':400,'expectancy_usd':0.1,'positive_year_fraction':0.67,'top_5_trade_profit_fraction':0.2,'best_month_profit_fraction':0.3,'active_months':12}
    assert stage1_gate(ok)['passed']
    bad=dict(ok,profit_factor=1.05)
    assert not stage1_gate(bad)['passed']

def test_rolling_and_expanding_are_chronological():
    t=months()
    r=rolling_folds(t,12,3,3)
    e=expanding_folds(t,3,12)
    assert r and e
    for f in r+e:
        assert f.research_end < f.validation_start

def test_score_rewards_stability_over_tiny_pf_advantage():
    a={'profit_factor':1.30,'expectancy_R':0.12,'median_profit_per_active_day':2,'positive_year_fraction':.8,'positive_month_fraction':.7,'completed_trades':1000,'max_drawdown_pct':3,'top_5_trade_profit_fraction':.2,'best_month_profit_fraction':.3}
    b=dict(a,profit_factor=1.35,max_drawdown_pct=4.9,top_5_trade_profit_fraction=.5,best_month_profit_fraction=.6)
    assert score_candidate(a)['final_score_pre_robustness'] > score_candidate(b)['final_score_pre_robustness']
