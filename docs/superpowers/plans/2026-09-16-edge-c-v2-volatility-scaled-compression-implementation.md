# Edge C v2 Volatility-Scaled Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Edge C v2 as an isolated XAUUSD M1 compression-breakout-retest research family using `ATR * sqrt(lookback)` compression normalization and a deterministic 256-experiment activation gate that must pass before the 10,000-experiment campaign is allowed.

**Architecture:** Preserve Edge C v1 byte-for-byte and add versioned v2 strategy, sampler, activation evaluator, campaign wrapper, selector, tests, and runbook. The activation path runs detailed backtests only for a deterministic catalog-spread 256-row sample and records structural counts only; it never ranks by profit. The full campaign reuses the existing campaign/manifest engine and is guarded by a persisted activation PASS report.

**Tech Stack:** Python 3.12, NumPy, Numba, SciPy Latin Hypercube sampler, existing xau_lab runner/backtest/metrics infrastructure, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-edge-c-v2-volatility-scaled-compression-design.md`

## Global Constraints

- Family: `edge_c_breakout_retest_v2`.
- Strategy: `compression_breakout_retest_v2`.
- Sampler: `edge_c_v2`.
- Default budget: 10,000.
- Default seed: 9,216,300.
- Default workers: 2.
- Result root: `edge_c_v2/results/`.
- Direction: `combined` only.
- Compression score: `(range_high - range_low) / (atr14[i-1] * sqrt(compression_lookback))`.
- Strategy parameter domains exactly match the frozen spec.
- Exit domains exactly match the frozen spec; no time exits.
- Edge C v1 source/results remain unchanged.
- Activation uses exactly 256 deterministic catalog-spread experiments at the default 10,000 budget.
- Activation PASS requires at least 205 sample configs with >=10 completed trades, at least 26 with >=300 completed trades, total completed trades >0, and end-of-data exits <=5% of all completed diagnostic trades.
- Activation decision must not inspect P/L, PF, expectancy, drawdown, win rate, or return fields.
- Full 10,000 campaign is blocked unless the persisted activation report says PASS for the same catalog/features identity.
- `$50/day` is not a strategy or activation objective.

---

### Task 1: Versioned Edge C v2 state machine

**Files:**
- Create: `xau_lab/strategies/edge_c_v2.py`
- Modify: `xau_lab/strategies/__init__.py`
- Create: `tests/strategies/test_edge_c_v2.py`

**Interfaces:**
- Consumes: `StrategyContext`, `StrategyDefinition`, strategy registry.
- Produces: `compression_breakout_retest_v2(ctx: StrategyContext, params: dict) -> np.ndarray` and registered strategy name `compression_breakout_retest_v2`.

- [ ] **Step 1: Write RED tests**

Add tests that construct compact synthetic OHLC/ATR arrays and assert:

```python
score = (range_high - range_low) / (lagged_atr * np.sqrt(lookback))
```

is used with an inclusive `<= compression_atr_ratio` boundary. Cover strict-prior range construction, fixed episode ATR, long and short breakout/retest/confirmation, invalidation, expiry, breakout bar not self-retesting, same later bar allowed to retest+confirm, one signal per episode, leave-before-rearm, and invalid ATR/NaN handling. Add one regression test proving a range that v1 rejects solely because `range/ATR > ratio` can qualify under the v2 `sqrt(L)` denominator.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tests/strategies/test_edge_c_v2.py -v
```

Expected: collection/import failure because `xau_lab.strategies.edge_c_v2` does not exist.

- [ ] **Step 3: Implement the minimal v2 strategy**

Copy the causal state-machine structure, not the source identity, from Edge C v1 into a new module. The only core hypothesis change is:

```python
volatility_scale = candidate_atr * np.sqrt(float(compression_lookback))
compression = width / volatility_scale <= compression_atr_ratio
```

Register domains exactly as:

```python
{
    "compression_lookback": (20, 120),
    "compression_atr_ratio": (0.50, 1.10),
    "breakout_buffer_atr": (0.25, 1.50),
    "breakout_body_atr": (0.25, 1.50),
    "retest_window": (2, 20),
    "retest_tolerance_atr": (0.10, 0.75),
    "confirmation_atr": (0.10, 1.00),
}
```

Add one import in `xau_lab/strategies/__init__.py`; do not edit `edge_c.py`.

- [ ] **Step 4: Run focused and full tests**

```bash
python -m pytest tests/strategies/test_edge_c_v2.py -v
python -m pytest -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xau_lab/strategies/edge_c_v2.py xau_lab/strategies/__init__.py tests/strategies/test_edge_c_v2.py
git commit -m "feat: add Edge C v2 volatility-scaled strategy"
```

---

### Task 2: Deterministic Edge C v2 catalog

**Files:**
- Create: `xau_lab/experiments/edge_c_v2_sampler.py`
- Create: `scripts/create_edge_c_v2_catalog.py`
- Create: `tests/experiments/test_edge_c_v2_sampler.py`
- Create/extend: `tests/integration/test_edge_c_v2_cli.py`

**Interfaces:**
- Produces constants `EDGE_C_V2_DEFAULT_BUDGET`, `EDGE_C_V2_DEFAULT_SEED`, `EDGE_C_V2_SAMPLER_VERSION`, `EDGE_C_V2_FAMILY`, `EDGE_C_V2_STRATEGY` and `generate_edge_c_v2_catalog(total_budget=10000, seed=9216300)`.
- CLI writes `edge_c_v2/results/EXPERIMENT_CATALOG.csv` atomically.

- [ ] **Step 1: Write RED sampler tests**

Assert exact identity/defaults, deterministic repeated generation, 10,000 unique IDs and fingerprints, seven frozen domains, `combined` direction only, commission/slippage unchanged, only frozen target-R/ATR-trail exits, no time exits, and that v1 sampler constants/output identity remain unchanged.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/experiments/test_edge_c_v2_sampler.py tests/integration/test_edge_c_v2_cli.py -v
```

Expected: missing v2 sampler/CLI imports.

- [ ] **Step 3: Implement sampler and catalog CLI**

Use `scipy.stats.qmc.LatinHypercube(d=7, seed=seed)` and the same deterministic seed/fingerprint conventions as existing dedicated samplers. Use a v2-specific exit RNG salt. Do not import or mutate v1 sampler internals.

- [ ] **Step 4: Run focused and full tests**

```bash
python -m pytest tests/experiments/test_edge_c_v2_sampler.py tests/integration/test_edge_c_v2_cli.py -v
python -m pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add xau_lab/experiments/edge_c_v2_sampler.py scripts/create_edge_c_v2_catalog.py tests/experiments/test_edge_c_v2_sampler.py tests/integration/test_edge_c_v2_cli.py
git commit -m "feat: add Edge C v2 deterministic catalog"
```

---

### Task 3: Activation evaluator and structural report

**Files:**
- Create: `xau_lab/validation/edge_c_v2_activation.py`
- Create: `scripts/run_edge_c_v2_activation.py`
- Create: `tests/validation/test_edge_c_v2_activation.py`
- Extend: `tests/integration/test_edge_c_v2_cli.py`

**Interfaces:**
- `spread_sample_indices(budget: int, sample_size: int = 256) -> tuple[int, ...]`.
- `evaluate_activation(rows: Sequence[Mapping[str, object]]) -> ActivationDecision` where rows contain only structural diagnostic fields.
- CLI reads the v2 catalog, selects the deterministic spread sample, loads canonical market data, runs each sampled experiment with `run_experiment(..., include_trades=True)`, counts `trade.exit_reason == "end_of_data"`, and writes `edge_c_v2/results/ACTIVATION_REPORT.json` atomically.

- [ ] **Step 1: Write RED unit tests**

Required assertions:

```python
indices = spread_sample_indices(10_000, 256)
assert len(indices) == 256
assert len(set(indices)) == 256
assert indices[0] == 0
assert indices[-1] == 9_999
```

Construct synthetic structural rows to prove exact boundary behavior: 205 rows at >=10 trades is enough for criterion 1; 204 is not. 26 rows at >=300 is enough for criterion 2; 25 is not. Exactly 5% end-of-data exits passes; any value above 5% fails. Zero total completed trades fails. Missing/non-numeric structural fields fail closed. Add a test where rows contain absurd profitability fields and prove the activation decision is unchanged.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/validation/test_edge_c_v2_activation.py -v
```

Expected: missing activation module.

- [ ] **Step 3: Implement pure activation logic**

Use `np.rint(np.linspace(0, budget - 1, sample_size)).astype(int)` and explicitly reject duplicate indices. Define a frozen dataclass decision/report payload with counts/fractions/distribution and a `passed` boolean. Only read these fields from each diagnostic row:

```text
experiment_id
completed_trades
risk_skip_count
end_of_data_exits
```

No economic fields are permitted in decision logic.

- [ ] **Step 4: Implement activation CLI**

Load catalog with the existing catalog reader, verify it is the frozen Edge C v2 identity and default budget unless explicitly running a test fixture, select the spread indices, load market data once, run detailed experiments, collect only structural rows, evaluate, and atomically write deterministic JSON. Print PASS/FAIL and all gate counts. Exit nonzero on malformed input/runtime failure, but activation FAIL itself should be a valid completed research outcome with a clear message.

- [ ] **Step 5: Run focused/full tests**

```bash
python -m pytest tests/validation/test_edge_c_v2_activation.py tests/integration/test_edge_c_v2_cli.py -v
python -m pytest -v
```

- [ ] **Step 6: Commit**

```bash
git add xau_lab/validation/edge_c_v2_activation.py scripts/run_edge_c_v2_activation.py tests/validation/test_edge_c_v2_activation.py tests/integration/test_edge_c_v2_cli.py
git commit -m "feat: add Edge C v2 activation gate"
```

---

### Task 4: Gated campaign and post-robustness selector

**Files:**
- Create: `scripts/run_edge_c_v2_campaign.py`
- Create: `xau_lab/validation/edge_c_v2.py`
- Create: `scripts/select_edge_c_v2_candidates.py`
- Create: `tests/validation/test_edge_c_v2_selection.py`
- Extend: `tests/integration/test_edge_c_v2_cli.py`

**Interfaces:**
- Campaign defaults to v2 catalog/result root/workers/seed and requires `ACTIVATION_REPORT.json` with `passed: true` before calling the existing campaign engine.
- `select_edge_c_v2_candidates(rows, max_candidates=6)` applies only post-robustness economic gates: net >0, PF >=1.10, expectancy >0, DD <=5%, finite final score; trade frequency is ignored.

- [ ] **Step 1: Write RED tests**

Prove campaign refuses missing, malformed, or failed activation reports and accepts a valid PASS report. Prove selector accepts economically identical rows regardless of completed trade count, rejects missing/non-finite fields, sorts by final score then ID, and returns zero candidates without error.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/validation/test_edge_c_v2_selection.py tests/integration/test_edge_c_v2_cli.py -v
```

- [ ] **Step 3: Implement campaign guard and selector**

The campaign guard must validate that the activation report matches the v2 sampler version/seed/catalog identity recorded by the diagnostic. Reuse existing manifest and `run_campaign`; do not fork backtest execution logic.

- [ ] **Step 4: Run focused/full tests**

```bash
python -m pytest tests/validation/test_edge_c_v2_selection.py tests/integration/test_edge_c_v2_cli.py -v
python -m pytest -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_edge_c_v2_campaign.py xau_lab/validation/edge_c_v2.py scripts/select_edge_c_v2_candidates.py tests/validation/test_edge_c_v2_selection.py tests/integration/test_edge_c_v2_cli.py
git commit -m "feat: gate Edge C v2 campaign on activation"
```

---

### Task 5: Operator runbook and final verification

**Files:**
- Create: `docs/edge_c_v2_runbook.md`
- Verify all files from Tasks 1-4.

**Interfaces:**
- Runbook gives exact commands for catalog creation, activation, conditional campaign execution, unchanged robustness promotion, and v2 selection.

- [ ] **Step 1: Write the runbook**

Document this exact operator order:

```powershell
python -m scripts.create_edge_c_v2_catalog --budget 10000 --seed 9216300
python -m scripts.run_edge_c_v2_activation
```

If activation prints FAIL, stop and preserve the report. If PASS:

```powershell
python -m scripts.run_edge_c_v2_campaign --workers 2 --limit 20
python -m scripts.run_edge_c_v2_campaign --workers 2 --allow-slow
```

Then use the unchanged `scripts.promote_candidates` with v2 catalog/master/result-root paths, followed by `scripts.select_edge_c_v2_candidates`. State explicitly that activation PASS is not evidence of profitability and that no v2 domains may be changed after seeing the diagnostic.

- [ ] **Step 2: Fresh full verification**

```bash
python -m pytest -v
```

Expected: zero failures.

- [ ] **Step 3: Diff audit**

Compare branch to `main`. Confirm Edge C v1 files, Edge A configs/OOS, Edge B, canonical `data/`, original `results/`, and diagnostics are absent from the diff except the one expected strategy-registration import.

- [ ] **Step 4: Commit runbook**

```bash
git add docs/edge_c_v2_runbook.md
git commit -m "docs: add Edge C v2 activation-first runbook"
```

- [ ] **Step 5: Fresh final CI / PR gate**

Run the repository GitHub Actions test workflow on the final head, read the completed test output, and only then mark the PR ready for review.
