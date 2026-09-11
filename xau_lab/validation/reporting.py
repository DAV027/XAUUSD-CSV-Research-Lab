from __future__ import annotations
from .promotion import stage1_decision

def robust_candidate_gate(m):
    reasons=[]
    if float(m.get('profit_factor') or 0)<1.10: reasons.append('baseline_pf')
    if float(m.get('expanding_pass_fraction') or 0)<.60: reasons.append('expanding_walkforward')
    if float(m.get('rolling_pass_fraction') or 0)<.60: reasons.append('rolling_walkforward')
    if float(m.get('median_validation_pf') or 0)<1.05: reasons.append('median_validation_pf')
    if float(m.get('parameter_stability_score') or 0)<60: reasons.append('parameter_stability')
    if float(m.get('cost_stability_score') or 0)<50: reasons.append('cost_stability')
    if float(m.get('stress_max_drawdown_pct') or 999)>5: reasons.append('stress_drawdown')
    if float(m.get('top5_removed_net_profit') or 0)<=0: reasons.append('top5_removal')
    return {'passed':not reasons,'reasons':reasons}

def classify_candidate(metrics,robustness=None,integrity_ok=True):
    s=stage1_decision(metrics,integrity_ok)
    if not s.passed:return {'verdict':'REJECTED','rejection_reason':';'.join(s.reasons)}
    if robustness is None:return {'verdict':'SURVIVOR','rejection_reason':''}
    merged={**metrics,**robustness}; r=robust_candidate_gate(merged)
    return {'verdict':'ROBUST_CANDIDATE' if r['passed'] else 'SURVIVOR','rejection_reason':'' if r['passed'] else ';'.join(r['reasons'])}
