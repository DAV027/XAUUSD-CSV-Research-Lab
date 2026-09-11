import math
SCORE_VERSION='v1'
def _clip(x,a,b): return max(a,min(b,x))
def score_candidate(m,median_profit_percentile=.5,robustness_component=None):
    pf=float(m.get('profit_factor') or 0); er=float(m.get('expectancy_R') or 0); n=max(1,int(m.get('completed_trades') or 0)); dd=float(m.get('max_drawdown_pct') or 0)
    pfscore=25*_clip((pf-1.10)/.40,0,1); erscore=20*_clip(er/.20,0,1); dayscore=15*_clip(median_profit_percentile,0,1)
    stability=15*_clip(((m.get('positive_year_fraction') or 0)+(m.get('positive_month_fraction') or 0))/2,0,1)
    samplescore=10*_clip(math.log(max(n,300)/300)/math.log(10),0,1)
    ddpen=15*_clip(dd/5,0,1)
    concentration=max(float(m.get('top_5_trade_profit_fraction') or 0),float(m.get('best_month_profit_fraction') or 0))
    conpen=15*_clip(concentration/.60,0,1)
    sidepen=0
    lp=m.get('long_PF'); sp=m.get('short_PF')
    lt=int(m.get('long_trades') or 0); st=int(m.get('short_trades') or 0)
    if lp is not None and sp is not None and lt>=100 and st>=100 and min(float(lp),float(sp))<.90: sidepen=10
    pre=_clip(pfscore+erscore+dayscore+stability+samplescore-ddpen-conpen-sidepen,0,85)
    out={'score_version':SCORE_VERSION,'pf_component':pfscore,'expectancy_component':erscore,'daily_component':dayscore,'stability_component':stability,'sample_component':samplescore,'drawdown_penalty':ddpen,'concentration_penalty':conpen,'side_penalty':sidepen,'robustness_component':robustness_component,'final_score_pre_robustness':pre}
    out['final_score']=pre if robustness_component is None else _clip(pre+robustness_component,0,100)
    return out
