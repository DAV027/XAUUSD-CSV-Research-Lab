from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.scoring import score_candidate, score_survivor_set

RESEARCH_ONLY_SCOPE = "research_candidate_only_not_live_trading_approval"


@dataclass(frozen=True)
class RobustCandidateDecision:
    passed: bool
    reasons: Sequence[str]
    walkforward_stability: float | None
    robustness_component: float | None


@dataclass(frozen=True)
class CandidateTables:
    rejected: tuple[dict[str, object], ...]
    survivors: tuple[dict[str, object], ...]
    top_candidates: tuple[dict[str, object], ...]


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integrity_ok(row: Mapping[str, object]) -> bool:
    value = row.get("integrity_ok", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "fail", "failed"}
    return bool(value)


def _robustness_component(evidence: Mapping[str, object]) -> tuple[float | None, float | None]:
    expanding = _number(evidence, "expanding_validation_joint_stability_fraction")
    rolling = _number(evidence, "rolling_validation_joint_stability_fraction")
    parameter = _number(evidence, "parameter_stability_pct")
    cost = _number(evidence, "cost_stability_pct")
    if None in (expanding, rolling, parameter, cost):
        return None, None

    walkforward = (float(expanding) + float(rolling)) / 2.0
    component = 15.0 * ((walkforward + float(parameter) / 100.0 + float(cost) / 100.0) / 3.0)
    component = min(15.0, max(0.0, component))
    return walkforward, component


def robust_candidate_decision(
    master: Mapping[str, object],
    evidence: Mapping[str, object],
) -> RobustCandidateDecision:
    reasons: list[str] = []

    expanding = _number(evidence, "expanding_validation_joint_stability_fraction")
    rolling = _number(evidence, "rolling_validation_joint_stability_fraction")
    median_pf = _number(evidence, "median_validation_pf")
    parameter = _number(evidence, "parameter_stability_pct")
    cost = _number(evidence, "cost_stability_pct")
    stress_dd = _number(evidence, "stress_max_drawdown_pct")
    top_five_net = _number(evidence, "top_five_removal_net_profit")
    baseline_pf = _number(master, "profit_factor")

    if expanding is None or expanding < 0.60:
        reasons.append("expanding_walkforward_below_60_pct")
    if rolling is None or rolling < 0.60:
        reasons.append("rolling_walkforward_below_60_pct")
    if median_pf is None or median_pf < 1.05:
        reasons.append("median_validation_pf_below_1_05")
    if parameter is None or parameter < 60.0:
        reasons.append("parameter_stability_below_60_pct")
    if cost is None or cost < 50.0:
        reasons.append("cost_stability_below_50_pct")
    if stress_dd is None or stress_dd > 5.0:
        reasons.append("stress_drawdown_above_5_pct")
    if top_five_net is None or top_five_net < 0.0:
        reasons.append("top_five_removal_turns_negative")
    if baseline_pf is None or baseline_pf < 1.10:
        reasons.append("baseline_pf_below_1_10")

    walkforward, component = _robustness_component(evidence)
    return RobustCandidateDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        walkforward_stability=walkforward,
        robustness_component=component,
    )


def _reason_text(reasons: Sequence[str]) -> str:
    return ";".join(str(reason) for reason in reasons)


def classify_candidates(
    master_rows: Sequence[Mapping[str, object]],
    robustness_evidence: Mapping[str, Mapping[str, object]],
) -> CandidateTables:
    rejected: list[dict[str, object]] = []
    stage1_rows: list[dict[str, object]] = []

    for source in master_rows:
        row = dict(source)
        experiment_id = str(row.get("experiment_id", ""))
        if not experiment_id:
            raise ValueError("every master row requires experiment_id")
        decision = stage1_decision(row, integrity_ok=_integrity_ok(row))
        if not decision.passed:
            row.update(
                {
                    "verdict": "REJECTED",
                    "rejection_reason": _reason_text(decision.reasons),
                    "approval_scope": RESEARCH_ONLY_SCOPE,
                }
            )
            rejected.append(row)
        else:
            stage1_rows.append(row)

    scored_stage1 = score_survivor_set(stage1_rows) if stage1_rows else []
    survivors: list[dict[str, object]] = []
    top_candidates: list[dict[str, object]] = []

    for row in scored_stage1:
        experiment_id = str(row["experiment_id"])
        evidence = robustness_evidence.get(experiment_id)
        if evidence is None:
            row.update(
                {
                    "verdict": "STAGE1_SURVIVOR",
                    "rejection_reason": "robustness_not_evaluated",
                    "approval_scope": RESEARCH_ONLY_SCOPE,
                }
            )
            survivors.append(row)
            continue

        robust = robust_candidate_decision(row, evidence)
        enriched = dict(row)
        enriched.update(dict(evidence))
        enriched["walkforward_stability"] = robust.walkforward_stability
        enriched["robustness_component"] = robust.robustness_component
        rescored = dict(enriched)
        rescored.update(score_candidate(enriched))
        rescored["approval_scope"] = RESEARCH_ONLY_SCOPE

        if robust.passed:
            rescored["verdict"] = "ROBUST_CANDIDATE"
            rescored["rejection_reason"] = ""
            top_candidates.append(rescored)
        else:
            rescored["verdict"] = "STAGE1_SURVIVOR"
            rescored["rejection_reason"] = _reason_text(robust.reasons)
            survivors.append(rescored)

    return CandidateTables(
        rejected=tuple(rejected),
        survivors=tuple(survivors),
        top_candidates=tuple(top_candidates),
    )


__all__ = [
    "CandidateTables",
    "RESEARCH_ONLY_SCOPE",
    "RobustCandidateDecision",
    "classify_candidates",
    "robust_candidate_decision",
]
