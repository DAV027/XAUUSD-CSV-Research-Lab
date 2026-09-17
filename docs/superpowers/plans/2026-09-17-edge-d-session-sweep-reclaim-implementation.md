# Edge D v1 Session Sweep Reclaim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Edge D v1 XAUUSD M1 research family that trades causal prior-session liquidity sweep/reclaim reversals and cannot run a full 10,000-experiment campaign until a frozen structural activation gate passes.

**Architecture:** Follow the existing Edge C v2 isolation pattern: a dedicated stateful strategy module, independent deterministic sampler/catalog, pure activation evaluator plus CLI report, guarded campaign wrapper, pure post-Stage-1 viability selector, tests, and runbook. Reuse the repository backtester, manifest, promotion, cost, risk, and checkpoint infrastructure without changing historical families or canonical data/results.

**Tech Stack:** Python 3.12, NumPy, Numba, SciPy LatinHypercube sampling, pytest, existing `xau_lab` backtest/runner/validation modules.

**Spec:** `docs/superpowers/specs/2026-09-17-edge-d-session-sweep-reclaim-design.md`

## Global Constraints

- family: `edge_d_session_sweep_reclaim`
- strategy: `prior_session_sweep_reclaim`
- sampler: `edge_d_v1`
- seed: `9216400`
- budget: `10000`
- workers: `2`
- result root: `edge_d/results/`
- direction mode: `combined` only
- exactly five strategy dimensions: transition, entry_window_bars, sweep_atr, max_reclaim_bars, reclaim_depth_atr
- transitions only: `asia_to_london`, `london_pre_ny_to_new_york`
- stop ATR values: 0.75, 1.0, 1.5, 2.0
- target-R values: 1.0, 1.5, 2.0, 3.0
- time exits: 30, 60, 120, 240 minutes
- no ATR trail
- activation sample size: 256 deterministic catalog-spread rows
- activation PASS: >=205 configs with >=50 trades; >=64 with >=300; EOD exits <=5%; total trades >0
- activation is structural only; economic fields may not influence PASS/FAIL
- unchanged Stage-1 promotion rules; zero survivors is valid
- no modification to Edge A/OOS, Edge B, Edge C, canonical data, original results, or original sampler behavior

---

### Task 1: Stateful Edge D strategy

**Files:**
- Create: `xau_lab/strategies/edge_d.py`
- Modify: `xau_lab/strategies/__init__.py`
- Test: `tests/strategies/test_edge_d.py`

**Interfaces:**
- Consumes `StrategyContext.open/high/low/close/atr14` and boolean features `session_asia`, `session_london`, `session_new_york`.
- Produces registered `StrategyDefinition("edge_d_session_sweep_reclaim", "prior_session_sweep_reclaim", ...)` returning int8 signals in {-1,0,+1}.

- [ ] **Step 1: Write RED tests** covering Asia→London reference construction, London-pre-NY freeze excluding first NY bar, strictly lagged frozen ATR, upper-sweep bearish reclaim SHORT, lower-sweep bullish reclaim LONG, no-reclaim, reclaim timeout, same-bar reclaim when `max_reclaim_bars=0`, same-bar/sequential double-sweep invalidation, one-signal lock, entry-window expiry, transition reset, NaN/invalid ATR safety, and exact registry domain/identity.
- [ ] **Step 2: Run `python -m pytest tests/strategies/test_edge_d.py -v`** and require failure because `xau_lab.strategies.edge_d` is missing.
- [ ] **Step 3: Implement minimal Numba state machine** with explicit reference building, target-start freeze, frozen lagged ATR, sweep direction/state, ambiguity lockout, reclaim age, and one signal per transition. `max_reclaim_bars=0` permits same-bar reclaim only when that bar does not sweep both sides.
- [ ] **Step 4: Register Edge D** with domain `{transition:("asia_to_london","london_pre_ny_to_new_york"), entry_window_bars:(30,180), sweep_atr:(0.10,1.00), max_reclaim_bars:(0,10), reclaim_depth_atr:(0.00,0.50)}` and import it once in `strategies/__init__.py`.
- [ ] **Step 5: Run targeted tests, then full `python -m pytest -v`** and require green before Task 2.

### Task 2: Independent deterministic sampler and catalog CLI

**Files:**
- Create: `xau_lab/experiments/edge_d_sampler.py`
- Create: `scripts/create_edge_d_catalog.py`
- Test: `tests/experiments/test_edge_d_sampler.py`
- Test: `tests/integration/test_edge_d_cli.py`

**Interfaces:**
- Produces `generate_edge_d_catalog(total_budget=10000, seed=9216400) -> list[CompleteExperiment]`.
- Catalog rows use combined direction only, original discovery costs, no ATR trail, target-R or time exit only.

- [ ] **Step 1: Write RED tests** for deterministic equality, 10,000 unique fingerprints/IDs, exact five-dimensional domain bounds, both transitions, combined-only direction, stop/exit sets, canonical JSON/fingerprint consistency, and proof original samplers remain unchanged.
- [ ] **Step 2: Run targeted tests** and require missing-module/CLI failure.
- [ ] **Step 3: Implement LatinHypercube(d=5)** with transition sampled from one dimension, integer/continuous frozen mappings for the remaining dimensions, dedicated exit RNG salt, deterministic strategy seeds, duplicate-fingerprint rejection, target-R/time exit isolation, and original discovery commission/slippage constants.
- [ ] **Step 4: Implement `scripts.create_edge_d_catalog`** using atomic temp-write + fsync + replace and defaults `edge_d/results/EXPERIMENT_CATALOG.csv`, budget 10000, seed 9216400.
- [ ] **Step 5: Run targeted tests and full suite**; require green before Task 3.

### Task 3: Structural activation evaluator and CLI

**Files:**
- Create: `xau_lab/validation/edge_d_activation.py`
- Create: `scripts/run_edge_d_activation.py`
- Extend: `tests/validation/test_edge_d_activation.py`
- Extend: `tests/integration/test_edge_d_cli.py`

**Interfaces:**
- Pure evaluator accepts diagnostic rows with only experiment_id, completed_trades, risk_skip_count, end_of_data_exits.
- CLI writes `edge_d/results/ACTIVATION_REPORT.json` bound to family/strategy/sampler/seed/budget/catalog SHA/feature SHA.

- [ ] **Step 1: Write RED tests** for deterministic 256 spread indices over a 10,000 catalog; exact 205/256 and 64/256 passing boundaries; just-below failures; exactly 5% EOD pass and above-5% fail; zero-total-trade fail; zero-trade accounting; profit-field neutrality; deterministic report serialization.
- [ ] **Step 2: Run targeted tests** and require missing activation module/CLI failure.
- [ ] **Step 3: Implement spread-sampling and pure evaluator** returning status, reasons, structural distributions and frozen thresholds; do not import/use profit, PF, win-rate, expectancy, DD, or active-day P/L for the decision.
- [ ] **Step 4: Implement activation CLI** that reads catalog/features, runs exactly 256 experiments with `include_trades=True`, counts `trade.exit_reason == "end_of_data"`, writes a deterministic JSON report, and prints PASS/FAIL status.
- [ ] **Step 5: Run targeted tests and full suite**; require green before Task 4.

### Task 4: Guarded campaign and post-robustness selector

**Files:**
- Create: `scripts/run_edge_d_campaign.py`
- Create: `xau_lab/validation/edge_d.py`
- Create: `scripts/select_edge_d_candidates.py`
- Test: `tests/validation/test_edge_d_selection.py`
- Extend: `tests/integration/test_edge_d_cli.py`

**Interfaces:**
- Campaign wrapper validates a PASS activation report plus exact family/strategy/sampler/catalog SHA/feature SHA before manifest/campaign execution.
- Selector `select_edge_d_candidates(rows, max_candidates=6)` applies only net_profit>0, PF>=1.10, expectancy_usd>0, max_drawdown_pct<=5, finite final_score; no frequency criterion.

- [ ] **Step 1: Write RED tests** for campaign refusal when report missing/FAIL/wrong identity/wrong hashes and acceptance of a matching PASS report; selector threshold boundaries, deterministic score+ID ordering, frequency neutrality, max six, and zero-survivor behavior.
- [ ] **Step 2: Run targeted tests** and require missing wrapper/selector failure.
- [ ] **Step 3: Implement guarded runner** reusing `_guard_slow_campaign`, manifest helpers, and `run_campaign`; defaults: catalog `edge_d/results/EXPERIMENT_CATALOG.csv`, features canonical parquet, result root `edge_d/results`, workers 2, seed 9216400.
- [ ] **Step 4: Implement pure selector + CLI** with default robust input `edge_d/results/TOP_CANDIDATES.csv`, output `edge_d/results/EDGE_D_SHORTLIST_CANDIDATES.csv`, and approval_scope `edge_d_v1_viability`.
- [ ] **Step 5: Run targeted tests and full suite**; require green before documentation.

### Task 5: Operator runbook and final verification

**Files:**
- Create: `docs/edge_d_runbook.md`

- [ ] **Step 1: Write exact commands** for catalog creation, activation diagnostic, PASS/FAIL branching, 20-row smoke, full resume with `--allow-slow`, promotion, Edge D selector, and preservation rules.
- [ ] **Step 2: State explicit stop rules**: activation FAIL stops v1; completed campaign with zero Stage-1 survivors stops v1; do not retune around near-misses.
- [ ] **Step 3: Run fresh full `python -m pytest -v`** on the final branch head and record exact pass count/time.
- [ ] **Step 4: Compare branch vs `main`** and verify only Edge D docs/code/tests plus one registration import changed; no historical/canonical artifacts changed.
- [ ] **Step 5: Open/update PR** with frozen identity, activation gate, TDD RED/GREEN evidence, final CI evidence, diff audit, and after-merge commands; mark ready for review only after fresh verification.
