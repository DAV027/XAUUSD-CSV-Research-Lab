from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

from xau_lab.backtest.models import Trade
from xau_lab.validation.promotion import stage1_decision
from xau_lab.validation.resampling import (
    DEFAULT_RESAMPLES,
    RESAMPLING_SEED,
    block_bootstrap_daily,
    top_trade_removal,
    trade_order_monte_carlo,
)
from xau_lab.validation.scoring import score_candidate, score_survivor_set
from xau_lab.validation.stress import StressReport
from xau_lab.validation.walkforward import FoldResult, summarize_fold_results

RESEARCH_ONLY_SCOPE = "research_candidate_only_not_live_trading_approval"
TRADE_LOG_FIELDS = tuple(field.name for field in fields(Trade))
_BASE_TABLE_FIELDS = ("experiment_id", "verdict", "rejection_reason", "approval_scope")


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


def build_robustness_evidence(
    expanding_results: Sequence[FoldResult],
    rolling_results: Sequence[FoldResult],
    stress_report: StressReport,
    trades: Sequence[Trade],
    *,
    resample_n: int = DEFAULT_RESAMPLES,
    seed: int = RESAMPLING_SEED,
) -> dict[str, object]:
    if not trades:
        raise ValueError("robustness evidence requires a nonempty baseline trade ledger")

    expanding_summary = summarize_fold_results(expanding_results)
    rolling_summary = summarize_fold_results(rolling_results)
    validation_pfs = [
        float(row.pf)
        for row in (*expanding_results, *rolling_results)
        if row.segment == "validation" and row.pf is not None
    ]
    combined_median_pf = float(median(validation_pfs)) if validation_pfs else None

    bootstrap = block_bootstrap_daily(trades, n=resample_n, seed=seed)
    pnl = [float(trade.net_pnl) for trade in trades]
    monte_carlo = trade_order_monte_carlo(pnl, n=resample_n, seed=seed)
    concentration = top_trade_removal(pnl)
    top_five = next(
        (row for row in concentration["scenarios"] if row["requested_count"] == 5),
        None,
    )
    if top_five is None:
        raise RuntimeError("top-five trade-removal scenario is required")

    evidence: dict[str, object] = {}
    evidence.update({f"expanding_{key}": value for key, value in expanding_summary.items()})
    evidence.update({f"rolling_{key}": value for key, value in rolling_summary.items()})
    evidence.update(
        {
            "median_validation_pf": combined_median_pf,
            "parameter_stability_pct": stress_report.parameter_stability_pct,
            "cost_stability_pct": stress_report.cost_stability_pct,
            "stress_max_drawdown_pct": stress_report.max_drawdown_pct,
            "stress_parameter_run_count": stress_report.parameter_run_count,
            "stress_cost_run_count": stress_report.cost_run_count,
            "top_five_removal_net_profit": top_five["net_profit"],
            "top_five_removal_profit_factor": top_five["profit_factor"],
            "top_five_removal_net_profit_sensitivity_pct": top_five[
                "net_profit_sensitivity_pct"
            ],
            "top_five_removal_profit_factor_sensitivity_pct": top_five[
                "profit_factor_sensitivity_pct"
            ],
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_n": bootstrap["n"],
            "bootstrap_mean_daily_p5": bootstrap["mean_daily_p5"],
            "bootstrap_mean_daily_p50": bootstrap["mean_daily_p50"],
            "bootstrap_mean_daily_p95": bootstrap["mean_daily_p95"],
            "bootstrap_total_pnl_p5": bootstrap["total_pnl_p5"],
            "bootstrap_total_pnl_p50": bootstrap["total_pnl_p50"],
            "bootstrap_total_pnl_p95": bootstrap["total_pnl_p95"],
            "monte_carlo_seed": monte_carlo["seed"],
            "monte_carlo_n": monte_carlo["n"],
            "monte_carlo_label": monte_carlo["label"],
            "monte_carlo_max_drawdown_p5": monte_carlo["max_drawdown_p5"],
            "monte_carlo_max_drawdown_p50": monte_carlo["max_drawdown_p50"],
            "monte_carlo_max_drawdown_p95": monte_carlo["max_drawdown_p95"],
        }
    )
    return evidence


def _robustness_component(evidence: Mapping[str, object]) -> tuple[float | None, float | None]:
    expanding = _number(evidence, "expanding_validation_joint_stability_fraction")
    rolling = _number(evidence, "rolling_validation_joint_stability_fraction")
    parameter = _number(evidence, "parameter_stability_pct")
    cost = _number(evidence, "cost_stability_pct")
    if None in (expanding, rolling, parameter, cost):
        return None, None
    walkforward = (float(expanding) + float(rolling)) / 2.0
    component = 15.0 * ((walkforward + float(parameter) / 100.0 + float(cost) / 100.0) / 3.0)
    return walkforward, min(15.0, max(0.0, component))


def robust_candidate_decision(master: Mapping[str, object], evidence: Mapping[str, object]) -> RobustCandidateDecision:
    reasons: list[str] = []
    expanding = _number(evidence, "expanding_validation_joint_stability_fraction")
    rolling = _number(evidence, "rolling_validation_joint_stability_fraction")
    median_pf = _number(evidence, "median_validation_pf")
    parameter = _number(evidence, "parameter_stability_pct")
    cost = _number(evidence, "cost_stability_pct")
    stress_dd = _number(evidence, "stress_max_drawdown_pct")
    top_five_net = _number(evidence, "top_five_removal_net_profit")
    baseline_pf = _number(master, "profit_factor")
    if expanding is None or expanding < 0.60: reasons.append("expanding_walkforward_below_60_pct")
    if rolling is None or rolling < 0.60: reasons.append("rolling_walkforward_below_60_pct")
    if median_pf is None or median_pf < 1.05: reasons.append("median_validation_pf_below_1_05")
    if parameter is None or parameter < 60.0: reasons.append("parameter_stability_below_60_pct")
    if cost is None or cost < 50.0: reasons.append("cost_stability_below_50_pct")
    if stress_dd is None or stress_dd > 5.0: reasons.append("stress_drawdown_above_5_pct")
    if top_five_net is None or top_five_net < 0.0: reasons.append("top_five_removal_turns_negative")
    if baseline_pf is None or baseline_pf < 1.10: reasons.append("baseline_pf_below_1_10")
    walkforward, component = _robustness_component(evidence)
    return RobustCandidateDecision(not reasons, tuple(reasons), walkforward, component)


def _reason_text(reasons: Sequence[str]) -> str:
    return ";".join(str(reason) for reason in reasons)


def classify_candidates(master_rows: Sequence[Mapping[str, object]], robustness_evidence: Mapping[str, Mapping[str, object]]) -> CandidateTables:
    rejected: list[dict[str, object]] = []
    stage1_rows: list[dict[str, object]] = []
    for source in master_rows:
        row = dict(source)
        experiment_id = str(row.get("experiment_id", ""))
        if not experiment_id: raise ValueError("every master row requires experiment_id")
        decision = stage1_decision(row, integrity_ok=_integrity_ok(row))
        if not decision.passed:
            row.update({"verdict":"REJECTED","rejection_reason":_reason_text(decision.reasons),"approval_scope":RESEARCH_ONLY_SCOPE})
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
            row.update({"verdict":"STAGE1_SURVIVOR","rejection_reason":"robustness_not_evaluated","approval_scope":RESEARCH_ONLY_SCOPE})
            survivors.append(row); continue
        robust = robust_candidate_decision(row, evidence)
        enriched = dict(row); enriched.update(dict(evidence))
        enriched["walkforward_stability"] = robust.walkforward_stability
        enriched["robustness_component"] = robust.robustness_component
        rescored = dict(enriched); rescored.update(score_candidate(enriched)); rescored["approval_scope"] = RESEARCH_ONLY_SCOPE
        if robust.passed:
            rescored["verdict"]="ROBUST_CANDIDATE"; rescored["rejection_reason"]=""; top_candidates.append(rescored)
        else:
            rescored["verdict"]="STAGE1_SURVIVOR"; rescored["rejection_reason"]=_reason_text(robust.reasons); survivors.append(rescored)
    return CandidateTables(tuple(rejected), tuple(survivors), tuple(top_candidates))


def _field_order(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    order=list(_BASE_TABLE_FIELDS); seen=set(order)
    for row in rows:
        for key in row:
            if key not in seen: order.append(key); seen.add(key)
    return tuple(order)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames,extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({key:row.get(key) for key in fieldnames})
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp,path)


def write_candidate_tables(result_root: str | Path, tables: CandidateTables) -> None:
    root=Path(result_root)
    for filename,rows in (("REJECTED.csv",tables.rejected),("SURVIVORS.csv",tables.survivors),("TOP_CANDIDATES.csv",tables.top_candidates)):
        _atomic_csv(root/filename,_field_order(rows),rows)


def _promoted_ids(tables: CandidateTables) -> tuple[str,...]:
    return tuple(str(row["experiment_id"]) for row in (*tables.survivors,*tables.top_candidates))


def write_promoted_trade_logs(result_root: str | Path, tables: CandidateTables, trade_logs: Mapping[str, Sequence[Trade]]) -> None:
    root=Path(result_root)/"trade_logs"; root.mkdir(parents=True,exist_ok=True)
    promoted_ids=_promoted_ids(tables); promoted_set=set(promoted_ids)
    missing=[eid for eid in promoted_ids if eid not in trade_logs]
    if missing: raise ValueError(f"missing captured trade logs for promoted experiments: {missing}")
    for path in sorted(root.glob("*.csv")):
        if path.stem not in promoted_set: path.unlink()
    for experiment_id in promoted_ids:
        trades=tuple(trade_logs[experiment_id])
        for trade in trades:
            if trade.experiment_id != experiment_id: raise ValueError(f"trade log experiment mismatch: expected {experiment_id}, got {trade.experiment_id}")
        _atomic_csv(root/f"{experiment_id}.csv",TRADE_LOG_FIELDS,[asdict(trade) for trade in trades])


__all__=["CandidateTables","RESEARCH_ONLY_SCOPE","RobustCandidateDecision","TRADE_LOG_FIELDS","build_robustness_evidence","classify_candidates","robust_candidate_decision","write_candidate_tables","write_promoted_trade_logs"]
