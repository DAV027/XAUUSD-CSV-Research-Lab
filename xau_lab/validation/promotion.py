from __future__ import annotations
from dataclasses import dataclass
PF_MIN=1.10; DD_MAX=5.0; TRADES_MIN=300; TOP5_MAX=.50; BEST_MONTH_MAX=.60
@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str,...]

def stage1_decision(m,integrity_ok=True):
    reasons=[]
    if not integrity_ok: reasons.append('integrity_failure')
    pf=m.get('profit_factor'); dd=m.get('max_drawdown_pct'); n=m.get('completed_trades',0); exp=m.get('expectancy_usd'); py=m.get('positive_year_fraction')
    if pf is None or float(pf)<PF_MIN: reasons.append('pf_below_1_10')
    if dd is None or float(dd)>DD_MAX: reasons.append('drawdown_above_5pct')
    if int(n or 0)<TRADES_MIN: reasons.append('fewer_than_300_trades')
    if exp is None or float(exp)<=0: reasons.append('nonpositive_expectancy')
    active_years=int(m.get('active_years') or 0)
    if active_years>=2 and py is not None and float(py)<=.5: reasons.append('majority_years_not_positive')
    top=m.get('top_5_trade_profit_fraction'); bm=m.get('best_month_profit_fraction'); am=int(m.get('active_months') or 0)
    if top is not None and float(top)>TOP5_MAX: reasons.append('top5_profit_concentration')
    if am>=6 and bm is not None and float(bm)>BEST_MONTH_MAX: reasons.append('best_month_profit_concentration')
    return PromotionDecision(not reasons,tuple(reasons))
def stage1_gate(m,integrity_ok=True):
    d=stage1_decision(m,integrity_ok); return {'passed':d.passed,'reasons':list(d.reasons)}
