# Edge C Breakout-Retest v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an isolated, deterministic Edge C compression-breakout-retest research campaign that discovers robust positive after-cost XAUUSD M1 breakout-continuation candidates without modifying Edge A or Edge B research artifacts.

**Architecture:** Add one stateful Numba-backed strategy that emits at most one signal per compression-breakout episode, one dedicated Latin-hypercube sampler/version, thin catalog/campaign CLIs over existing infrastructure, and a post-robustness economic selector. All discovery artifacts live under `edge_c/results/`; the original sampler-v1 and frozen OOS paths remain unchanged.

**Tech Stack:** Python 3.12, NumPy, Numba, SciPy `qmc.LatinHypercube`, pytest, existing `xau_lab` campaign/backtest/manifest infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-16-edge-c-breakout-retest-design.md`

## Global Constraints

- Family: `edge_c_breakout_retest`.
- Strategy: `compression_breakout_retest`.
- Sampler version: `edge_c_v1`.
- Default budget: 10,000; default seed: 9,216,200; workers: exactly 2 by default.
- Direction mode is `combined` only in v1.
- Frozen strategy parameters: `compression_lookback` 20–120 integer, `compression_atr_ratio` 0.35–0.80, `breakout_buffer_atr` 0.25–1.50, `breakout_body_atr` 0.25–1.50, `retest_window` 2–20 integer, `retest_tolerance_atr` 0.10–0.75, `confirmation_atr` 0.10–1.00.
- Episode ATR reference is `atr14[i-1]` at arm time and remains fixed for the episode.
- Same-bar retest+confirmation after a breakout is allowed only on bars strictly after the breakout bar; the breakout bar itself cannot count as the retest.
- Rearm requires at least one later bar where the compression condition is false before another compression can arm.
- Exit domain: stop ATR {1.0, 1.5, 2.0, 3.0}; target-R {1.5, 2.0, 3.0, 4.0}; ATR trail {1.0, 1.5, 2.0}; no time exits.
- Existing commission/slippage, backtester, risk sizing, entry-integrity mask, and promotion pipeline remain authoritative.
- Viability minimums: net profit > 0, PF >= 1.10, expectancy USD > 0, max DD <= 5%; trade frequency is neither capped nor rewarded.
- The `$50/day` objective is evaluated only after robustness; it is not a sampler knob or strategy acceptance shortcut.
- No changes to Edge A frozen configs/OOS, canonical `data/`, original `results/`, `edge_b/`, `edge_b_v2/`, or original sampler-v1 allocation/hashes.

---

### Task 1: Stateful compression-breakout-retest strategy

**Files:**
- Create: `tests/strategies/test_edge_c.py`
- Create: `xau_lab/strategies/edge_c.py`
- Modify: `xau_lab/strategies/__init__.py`

**Interfaces:**
- Consumes: `StrategyContext(open, high, low, close, atr14, ...)` and the seven frozen parameters.
- Produces: `compression_breakout_retest(ctx: StrategyContext, params: dict) -> np.ndarray` and registered `StrategyDefinition("edge_c_breakout_retest", "compression_breakout_retest", ...)`.

- [ ] **Step 1: Write failing state-machine tests**

Create synthetic contexts and tests covering: prior-bars-only compression; wick-only breakout rejection; valid long/short breakout progression; invalidation on close through tolerance; expiry with no retest; retest+confirmation emits exactly one signal; repeated retests do not emit twice; rearm requires a non-compressed bar; NaN ATR/price blocks state changes; registration exposes exactly seven frozen parameters.

Core expectations should include patterns like:

```python
signals = compression_breakout_retest(ctx, params)
assert np.flatnonzero(signals).tolist() == [expected_confirmation_bar]
assert signals[expected_confirmation_bar] == 1
assert int(np.count_nonzero(signals)) == 1
```

and for rearm:

```python
assert int(np.count_nonzero(signals[first_episode_end:second_compression_start])) == 0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/strategies/test_edge_c.py -v
```

Expected: collection/import failure because `xau_lab.strategies.edge_c` does not exist.

- [ ] **Step 3: Implement the minimal causal state machine**

Use a Numba kernel with explicit integer states: `SEEK=0`, `ARMED=1`, `RETEST=2`, plus `rearm_ready` boolean. At each bar:

```python
compression = range_width / atr_candidate <= compression_atr_ratio
```

Arm only when `state == SEEK and rearm_ready and compression`. Freeze `range_high`, `range_low`, and `episode_atr = atr14[i-1]`. After breakout, remember direction, broken level, breakout index, and `retest_seen=False`. Only bars `i > breakout_index` may mark retest. In RETEST, invalidate on tolerance close, expire after `retest_window`, mark retest from intrabar touch, then confirm on close displacement plus matching body direction. After signal/invalidation/expiry, set `state=SEEK` and `rearm_ready=False`; set it true only after a later non-compressed bar.

Expose exactly:

```python
{
    "compression_lookback": (20, 120),
    "compression_atr_ratio": (0.35, 0.80),
    "breakout_buffer_atr": (0.25, 1.50),
    "breakout_body_atr": (0.25, 1.50),
    "retest_window": (2, 20),
    "retest_tolerance_atr": (0.10, 0.75),
    "confirmation_atr": (0.10, 1.00),
}
```

- [ ] **Step 4: Register Edge C without altering sampler-v1 eligibility**

Add only a package import in `xau_lab/strategies/__init__.py`, mirroring Edge B research registration. Do not modify original sampler family budgets.

- [ ] **Step 5: Run focused and registry tests**

Run:

```bash
python -m pytest tests/strategies/test_edge_c.py tests/strategies/test_registry.py tests/experiments/test_sampler.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/strategies/test_edge_c.py xau_lab/strategies/edge_c.py xau_lab/strategies/__init__.py
git commit -m "feat: add Edge C breakout-retest strategy"
```

### Task 2: Deterministic isolated Edge C catalog

**Files:**
- Create: `tests/experiments/test_edge_c_sampler.py`
- Create: `xau_lab/experiments/edge_c_sampler.py`
- Create: `scripts/create_edge_c_catalog.py`

**Interfaces:**
- Produces constants `EDGE_C_DEFAULT_BUDGET=10_000`, `EDGE_C_DEFAULT_SEED=9_216_200`, `EDGE_C_SAMPLER_VERSION="edge_c_v1"`, `EDGE_C_FAMILY="edge_c_breakout_retest"`, `EDGE_C_STRATEGY="compression_breakout_retest"`.
- Produces `generate_edge_c_catalog(total_budget: int = 10_000, seed: int = 9_216_200) -> list[CompleteExperiment]`.

- [ ] **Step 1: Write failing sampler tests**

Tests must assert deterministic equality for equal seed/budget, unique experiment IDs/fingerprints, exact family/strategy/version/direction fields, all seven parameter bounds, exit-domain validity, unchanged discovery commission/slippage, and that `generate_catalog(..., sampler_version="v1")` still never allocates `edge_c_breakout_retest`.

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest tests/experiments/test_edge_c_sampler.py -v
```

Expected: import failure for `xau_lab.experiments.edge_c_sampler`.

- [ ] **Step 3: Implement deterministic LHS sampler**

Use `qmc.LatinHypercube(d=7, seed=int(seed))`, canonical JSON normalization, existing `canonical_fingerprint`, and a dedicated RNG for exits. Sample the seven strategy dimensions once per experiment. Exit selection must choose either `target_r` or `atr_trail`; `time_exit_minutes` is always `None`.

Use frozen exit sets:

```python
EDGE_C_STOP_ATR_VALUES = (1.0, 1.5, 2.0, 3.0)
EDGE_C_TARGET_R_VALUES = (1.5, 2.0, 3.0, 4.0)
EDGE_C_ATR_TRAIL_VALUES = (1.0, 1.5, 2.0)
```

Raise if a duplicate complete fingerprint is generated.

- [ ] **Step 4: Add atomic catalog CLI**

`scripts/create_edge_c_catalog.py` writes `edge_c/results/EXPERIMENT_CATALOG.csv` via temp file + `os.replace`, with CLI options `--budget`, `--seed`, `--output`.

- [ ] **Step 5: Run sampler tests plus original sampler regression**

```bash
python -m pytest tests/experiments/test_edge_c_sampler.py tests/experiments/test_sampler.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/experiments/test_edge_c_sampler.py xau_lab/experiments/edge_c_sampler.py scripts/create_edge_c_catalog.py
git commit -m "feat: add isolated Edge C catalog"
```

### Task 3: Resumable Edge C campaign wrapper

**Files:**
- Create: `tests/integration/test_edge_c_cli.py`
- Create: `scripts/run_edge_c_campaign.py`

**Interfaces:**
- Produces `build_parser()` and `run_edge_c_campaign(...) -> None`.
- Delegates to existing `_guard_slow_campaign`, `build_run_manifest`, `write_or_validate_run_manifest`, `run_campaign`, and `resolved_workers`.

- [ ] **Step 1: Write failing CLI wiring tests**

Assert defaults exactly:

```python
catalog == Path("edge_c/results/EXPERIMENT_CATALOG.csv")
features == Path("data/features/XAUUSD_M1_FEATURES.parquet")
result_root == Path("edge_c/results")
seed == 9_216_200
workers == 2
limit is None
allow_slow is False
```

Monkeypatch engine calls and assert manifest seed/workers/paths and campaign `limit` wiring.

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest tests/integration/test_edge_c_cli.py -v
```

Expected: missing `scripts.run_edge_c_campaign`.

- [ ] **Step 3: Implement thin runner**

Mirror the existing Edge B v2 wrapper but point exclusively at `edge_c/results`. Do not add a new execution engine.

- [ ] **Step 4: Run focused integration tests**

```bash
python -m pytest tests/integration/test_edge_c_cli.py tests/runner/test_manifest.py tests/runner/test_campaign.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_edge_c_cli.py scripts/run_edge_c_campaign.py
git commit -m "feat: add Edge C campaign runner"
```

### Task 4: Post-robustness viability selector

**Files:**
- Create: `tests/validation/test_edge_c_selection.py`
- Create: `xau_lab/validation/edge_c.py`
- Create: `scripts/select_edge_c_candidates.py`

**Interfaces:**
- Produces `select_edge_c_candidates(rows, *, max_candidates=6) -> list[dict[str, object]]`.
- Input is robust/promotion output rows; selector must still defensively enforce Edge C economic minimums and must never impose a trade-frequency condition.

- [ ] **Step 1: Write failing selector tests**

Test inclusive PF/DD boundaries, strict positive net/expectancy, deterministic score ordering, max-six cap, malformed metrics rejected, a 1-trade/week row accepted when otherwise valid, a 500-trades/week row accepted when otherwise valid, and zero qualifiers returns `[]` without changing thresholds.

Example valid boundary row:

```python
{
    "experiment_id": "EXP1",
    "net_profit": 1.0,
    "profit_factor": 1.10,
    "expectancy_usd": 0.01,
    "max_drawdown_pct": 5.0,
    "final_score": 0.8,
    "trades_per_active_day": 500.0,
}
```

- [ ] **Step 2: Run and verify RED**

```bash
python -m pytest tests/validation/test_edge_c_selection.py -v
```

Expected: missing selector module.

- [ ] **Step 3: Implement pure defensive selector**

Eligibility:

```python
net_profit > 0.0
profit_factor >= 1.10
expectancy_usd > 0.0
max_drawdown_pct <= 5.0
final_score is finite/present
```

Sort by descending `final_score`, tie-break by `experiment_id`; return at most six. Do not read any trade-frequency field for eligibility.

- [ ] **Step 4: Add atomic selector CLI**

Default input `edge_c/results/TOP_CANDIDATES.csv`, output `edge_c/results/EDGE_C_SHORTLIST_CANDIDATES.csv`. Preserve source fields and add `approval_scope=edge_c_v1_viability` if absent. Print robust input count, selected count, and a status that explicitly treats zero survivors as valid.

- [ ] **Step 5: Run selector and promotion regressions**

```bash
python -m pytest tests/validation/test_edge_c_selection.py tests/validation/test_promotion.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/validation/test_edge_c_selection.py xau_lab/validation/edge_c.py scripts/select_edge_c_candidates.py
git commit -m "feat: add Edge C viability selector"
```

### Task 5: Operator runbook and final verification

**Files:**
- Create: `docs/edge_c_runbook.md`

**Interfaces:**
- Documents catalog creation, 20-experiment smoke run, resumable 10k run, promotion, Edge C selection, negative-result handling, and later coverage/$50-day analysis.

- [ ] **Step 1: Write runbook**

Document commands:

```powershell
python -m scripts.create_edge_c_catalog --budget 10000 --seed 9216200
python -m scripts.run_edge_c_campaign --workers 2 --limit 20
python -m scripts.run_edge_c_campaign --workers 2 --allow-slow
```

Then document the existing promotion command as currently supported by the repo, followed by:

```powershell
python -m scripts.select_edge_c_candidates
```

State explicitly: do not delete `edge_c/results` between smoke/full run; do not relax thresholds if zero survive; do not use the `$50/day` target to tune discovery parameters.

- [ ] **Step 2: Run complete test suite**

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Diff audit against main**

Verify changed files are limited to Edge C strategy/sampler/scripts/tests/docs plus one strategy registration import. Confirm no changes under frozen Edge A configs, canonical `data/`, original `results/`, `edge_b/`, `edge_b_v2/`, or OOS/diagnostic artifacts.

- [ ] **Step 4: Commit runbook**

```bash
git add docs/edge_c_runbook.md
git commit -m "docs: add Edge C research runbook"
```

- [ ] **Step 5: Open/update PR with TDD evidence**

Include initial RED workflows, final full-suite GREEN workflow, exact head SHA, diff audit, frozen domains, and the statement that zero robust candidates is a valid outcome.
