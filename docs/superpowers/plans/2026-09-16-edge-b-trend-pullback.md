# Edge B Trend/Pullback Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, deterministic Edge B trend/pullback/recovery research campaign without altering Edge A or original sampler-v1 behavior.

**Architecture:** Add one new strategy family registered with the existing strategy registry, a dedicated Edge B catalog generator with its own seed/version, a thin campaign CLI that reuses the existing runner/backtester, and a post-promotion frequency selector. The original global sampler continues to include only families named by `FAMILY_BUDGET_WEIGHTS`, so Edge B registration cannot enter the original 50k catalog.

**Tech Stack:** Python 3.12, NumPy, Numba, SciPy LatinHypercube, existing XAU Lab backtest/runner/validation modules, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-edge-b-trend-pullback-design.md`

## Global Constraints

- Do not modify any frozen Edge A config or OOS/diagnostic result contract.
- Do not modify canonical `data/` artifacts or original `results/` outputs.
- Original `generate_catalog()` sampler version `v1` must produce identical experiments before and after Edge B registration.
- Edge B strategy has exactly four tunable signal parameters and fixed `direction_mode="combined"`.
- Default Edge B campaign budget is 10,000 and seed is `9_216_000`.
- Existing commission `$6.0` round trip per lot and slippage `5.0` points per fill are reused.
- Frequency eligibility is 3-10 completed trades per 7 calendar days, inclusive; never force-fill below three qualifiers.

---

### Task 1: Edge B strategy kernel

**Files:**
- Create: `xau_lab/strategies/edge_b.py`
- Modify: `xau_lab/runner/single.py`
- Create: `tests/strategies/test_edge_b.py`

**Interfaces:**
- Produces strategy registry entry `trend_pullback_recovery` in family `edge_b_trend_pullback`.
- Parameters: `trend_lookback`, `trend_threshold_atr`, `pullback_lookback`, `pullback_threshold_atr`.

- [ ] **Step 1: Write failing strategy tests**

```python
from xau_lab.strategies.edge_b import trend_pullback_recovery


def test_long_signal_occurs_only_on_first_recovery_turn():
    close = np.array([100,101,102,103,104,105,106,105,104,103,104,105], dtype=float)
    ctx = context(close, atr=1.0)
    sig = trend_pullback_recovery(ctx, {
        "trend_lookback": 6,
        "trend_threshold_atr": 2.0,
        "pullback_lookback": 3,
        "pullback_threshold_atr": 1.0,
    })
    assert sig[10] == 1
    assert sig[11] == 0


def test_short_signal_is_mirror_image():
    close = np.array([106,105,104,103,102,101,100,101,102,103,102,101], dtype=float)
    ctx = context(close, atr=1.0)
    sig = trend_pullback_recovery(ctx, {...same numeric params...})
    assert sig[10] == -1
    assert sig[11] == 0
```

Also test warmup, continuing pullback/no recovery, NaN/invalid ATR, and registry parameter domains.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/strategies/test_edge_b.py -v`
Expected: import/module failure because `xau_lab.strategies.edge_b` does not exist.

- [ ] **Step 3: Implement minimal Numba kernel**

Core logic:

```python
@njit(cache=True)
def _trend_pullback_recovery_kernel(close, atr14, trend_lookback,
                                    trend_threshold_atr, pullback_lookback,
                                    pullback_threshold_atr):
    out = np.zeros(len(close), dtype=np.int8)
    warmup = max(trend_lookback, pullback_lookback) + 1
    for i in range(warmup, len(close)):
        atr = atr14[i - 1]
        if not np.isfinite(atr) or atr <= 0.0:
            continue
        values = (close[i], close[i-1], close[i-2],
                  close[i-1-trend_lookback], close[i-1-pullback_lookback])
        if not all(np.isfinite(v) for v in values):
            continue
        trend = close[i-1] - close[i-1-trend_lookback]
        pullback = close[i-1] - close[i-1-pullback_lookback]
        if trend >= trend_threshold_atr * atr:
            if (pullback <= -pullback_threshold_atr * atr
                    and close[i-1] < close[i-2]
                    and close[i] > close[i-1]):
                out[i] = 1
        elif trend <= -trend_threshold_atr * atr:
            if (pullback >= pullback_threshold_atr * atr
                    and close[i-1] > close[i-2]
                    and close[i] < close[i-1]):
                out[i] = -1
    return out
```

Register with exactly the four domains from the spec.

Import `edge_b` in `xau_lab/runner/single.py` only for worker registry population.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/strategies/test_edge_b.py tests/strategies/test_entry_integrity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat: add Edge B trend pullback recovery strategy`

---

### Task 2: Dedicated deterministic Edge B catalog

**Files:**
- Create: `xau_lab/experiments/edge_b_sampler.py`
- Create: `scripts/create_edge_b_catalog.py`
- Create: `tests/experiments/test_edge_b_sampler.py`

**Interfaces:**
- `generate_edge_b_catalog(total_budget: int = 10_000, seed: int = 9_216_000) -> list[CompleteExperiment]`
- `write_edge_b_catalog(path: Path, *, budget: int, seed: int) -> int`

- [ ] **Step 1: Write failing catalog tests**

Test that:

```python
catalog = generate_edge_b_catalog(total_budget=500, seed=9_216_000)
assert len(catalog) == 500
assert len({x.fingerprint for x in catalog}) == 500
assert all(x.family == "edge_b_trend_pullback" for x in catalog)
assert all(x.strategy_name == "trend_pullback_recovery" for x in catalog)
assert all(x.direction_mode == "combined" for x in catalog)
assert all(x.sampler_version == "edge_b_v1" for x in catalog)
assert catalog == generate_edge_b_catalog(total_budget=500, seed=9_216_000)
```

Add original-sampler isolation test:

```python
baseline = generate_catalog(total_budget=500, seed=9_215_000)
import xau_lab.strategies.edge_b  # registry side effect
again = generate_catalog(total_budget=500, seed=9_215_000)
assert again == baseline
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/experiments/test_edge_b_sampler.py -v`
Expected: module/function missing.

- [ ] **Step 3: Implement generator**

Use `scipy.stats.qmc.LatinHypercube(d=4, seed=seed)` for the four signal dimensions. Reuse existing stop/exit/cost constants. Sample exit choices deterministically from an RNG seeded from the Edge B seed. Set family/allocation bucket to `edge_b_trend_pullback`, direction `combined`, and sampler version `edge_b_v1`. Use `canonical_fingerprint()` and `canonical_json()` for IDs/parameters.

- [ ] **Step 4: Implement atomic catalog CLI**

Defaults:

```python
--budget 10000
--seed 9216000
--output edge_b/results/EXPERIMENT_CATALOG.csv
```

Write via temp file + `os.replace`, matching `scripts/create_catalog.py`.

- [ ] **Step 5: Run GREEN**

Run: `python -m pytest tests/experiments/test_edge_b_sampler.py tests/experiments/test_sampler.py -v`
Expected: PASS, proving original sampler behavior remains intact under Edge B registration.

- [ ] **Step 6: Commit**

Commit: `feat: add isolated Edge B experiment catalog`

---

### Task 3: Edge B campaign CLI using existing engine

**Files:**
- Create: `scripts/run_edge_b_campaign.py`
- Create: `tests/integration/test_edge_b_cli.py`

**Interfaces:**
- CLI calls existing `build_run_manifest`, `write_or_validate_run_manifest`, and `run_campaign`.

- [ ] **Step 1: Write failing CLI default/isolation test**

Test parser/helper behavior so defaults resolve to:

```text
catalog=edge_b/results/EXPERIMENT_CATALOG.csv
features=data/features/XAUUSD_M1_FEATURES.parquet
result_root=edge_b/results
seed=9216000
```

Use a monkeypatched `run_campaign`/manifest writer to assert the wrapper never targets canonical `results/` by default.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/integration/test_edge_b_cli.py -v`
Expected: module missing.

- [ ] **Step 3: Implement thin wrapper**

Reuse the existing runner and manifest functions. Support `--workers`, `--limit`, `--allow-slow`; default local production guidance remains 2 workers but do not hard-code a machine-wide restriction.

Import/register `xau_lab.strategies.edge_b` before running the campaign so spawned workers resolve the strategy through `runner.single`.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/integration/test_edge_b_cli.py tests/integration/test_smoke_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat: add isolated Edge B campaign CLI`

---

### Task 4: Frequency-eligible Edge B shortlist selector

**Files:**
- Create: `xau_lab/validation/edge_b.py`
- Create: `scripts/select_edge_b_candidates.py`
- Create: `tests/validation/test_edge_b_selection.py`

**Interfaces:**
- `select_edge_b_candidates(rows, *, min_trades_per_week=3.0, max_trades_per_week=10.0, max_candidates=6) -> tuple[dict, ...]`
- Input is robust rows from `edge_b/results/TOP_CANDIDATES.csv`.
- Output default: `edge_b/results/EDGE_B_SHORTLIST_CANDIDATES.csv`.

- [ ] **Step 1: Write failing selection tests**

Use synthetic rows with `data_start`, `data_end`, `completed_trades`, `final_score`.

Compute:

```python
span_weeks = ((data_end - data_start) / 86400.0) / 7.0
trades_per_week = completed_trades / span_weeks
```

Assert:
- below 3/week rejected.
- above 10/week rejected.
- boundaries 3 and 10 included.
- more than six eligible rows sort by `final_score` descending then `experiment_id` ascending and truncate to six.
- two eligible rows return two; selector does not force-fill to three.
- every output row gets `edge_b_frequency_trades_per_week` and research-only scope.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/validation/test_edge_b_selection.py -v`
Expected: module missing.

- [ ] **Step 3: Implement selector and atomic CSV CLI**

Reject rows with nonpositive span, missing/nonfinite score, or missing completed-trade count with explicit `ValueError` rather than guessing.

The CLI reads only `TOP_CANDIDATES.csv`; it does not relax robustness rules or inspect rejected candidates.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/validation/test_edge_b_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat: add Edge B frequency shortlist selector`

---

### Task 5: Full regression verification and operator instructions

**Files:**
- Modify if needed: `README.md` or add `docs/edge_b_v1_runbook.md`

- [ ] **Step 1: Add runbook commands**

Document exactly:

```powershell
python -m scripts.create_edge_b_catalog --budget 10000 --seed 9216000
python -m scripts.run_edge_b_campaign --workers 2
python -m scripts.promote_candidates `
  --master edge_b/results/MASTER_RESULTS.csv `
  --catalog edge_b/results/EXPERIMENT_CATALOG.csv `
  --features data/features/XAUUSD_M1_FEATURES.parquet `
  --result-root edge_b/results
python -m scripts.select_edge_b_candidates
```

State that Edge B data through the shortlist-freeze date is discovery data and that a separate prospective OOS config must be created only after shortlist freeze.

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 3: Inspect branch diff**

Verify no changes under frozen Edge A config files and no changes to canonical output artifacts.

- [ ] **Step 4: Commit documentation**

Commit: `docs: add Edge B v1 research runbook`

- [ ] **Step 5: Open PR**

PR title: `feat: add isolated Edge B trend pullback research track`

PR body must state TDD RED/GREEN evidence, exact changed files, full-suite test count, and explicitly state that Edge A frozen configs and original sampler-v1 outputs are unchanged.
