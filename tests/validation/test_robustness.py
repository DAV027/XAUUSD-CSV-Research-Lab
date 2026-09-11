import numpy as np, json
from xau_lab.experiments.sampler import generate_catalog
from xau_lab.validation.stress import numeric_neighbors
from xau_lab.validation.resampling import block_bootstrap_daily, trade_order_monte_carlo, top_trade_removal
from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.reporting import robust_candidate_gate
from xau_lab.validation.holdout import HoldoutRegistry

def test_numeric_neighbors_exact():
    assert numeric_neighbors(20,'int',5,100)==[16,18,22,24]
    assert numeric_neighbors(1.0,'float',0,10)==[.8,.9,1.1,1.2]

def test_resampling_seed_reproducible():
    daily={'2026-01-01':1.0,'2026-01-02':-2.0,'2026-01-03':3.0,'2026-01-04':1.5}
    assert block_bootstrap_daily(daily,iterations=100,seed=9216000)==block_bootstrap_daily(daily,iterations=100,seed=9216000)
    pnl=[1,-1,2,-.5,3,-2]
    assert trade_order_monte_carlo(pnl,100,seed=9216000)==trade_order_monte_carlo(pnl,100,seed=9216000)
    assert top_trade_removal(pnl)['remove_top_1_net'] < sum(pnl)

def test_stage1_decision_integrity_fails_closed():
    m=dict(profit_factor=1.2,max_drawdown_pct=2,completed_trades=500,expectancy_usd=.2,positive_year_fraction=.7,top_5_trade_profit_fraction=.2,best_month_profit_fraction=.2,active_months=12)
    assert stage1_decision(m,integrity_ok=True).passed
    assert not stage1_decision(m,integrity_ok=False).passed

def test_robust_gate_requires_both_schemes():
    m={'profit_factor':1.2,'expanding_pass_fraction':.7,'rolling_pass_fraction':.7,'median_validation_pf':1.1,'parameter_stability_score':70,'cost_stability_score':60,'stress_max_drawdown_pct':4,'top5_removed_net_profit':1}
    assert robust_candidate_gate(m)['passed']
    assert not robust_candidate_gate(dict(m,rolling_pass_fraction=.5))['passed']

def test_holdout_cannot_move_start_backward(tmp_path):
    r=HoldoutRegistry(tmp_path/'HOLDOUT_REGISTRY.json')
    r.freeze('EXP1','2026-09-11T00:00:00Z','abc','fp','2026-09-12',3)
    try:
        r.update_observed_through('EXP1','2026-09-10')
        assert False
    except ValueError: pass
