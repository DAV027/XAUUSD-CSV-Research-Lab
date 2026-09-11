from xau_lab.metrics.performance import summarize_pnl

def test_summary_known_sequence():
    o=summarize_pnl([10,-5,-5,20,-10],100)
    assert o['gross_profit']==30
    assert o['gross_loss']==-20
    assert o['profit_factor']==1.5
    assert o['expectancy_usd']==2
    assert o['max_loss_streak']==2
    assert o['max_drawdown_usd']==10
