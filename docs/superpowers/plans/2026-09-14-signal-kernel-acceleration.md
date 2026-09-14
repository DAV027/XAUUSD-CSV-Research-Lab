# Signal Kernel Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce canonical 50,000-experiment discovery runtime to a projected 24 hours or less by replacing Python-per-bar signal loops with exact Numba kernels while preserving every strategy signal and campaign result.

**Architecture:** Keep the public `strategy(ctx: StrategyContext, params: dict) -> np.ndarray` API unchanged. Add a focused private `xau_lab/strategies/_kernels.py` module for reusable exact rolling primitives, then convert hot strategy families to thin Python wrappers around `@njit(cache=True)` kernels in benchmark order. Re-verify signal equality, result parity, full CI, the eight-experiment profiler, and a fresh 100-smoke before permitting the 50,000 campaign.

**Tech Stack:** Python 3.12, NumPy, Numba, Polars, PyArrow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-signal-kernel-acceleration-design.md`

## Global Constraints

- Preserve `results/EXPERIMENT_CATALOG.csv` exactly, including all 50,000 experiment IDs and fingerprints.
- Preserve canonical feature semantics and the approved 72-column feature artifact.
- Preserve strategy outputs exactly for identical inputs/parameters.
- Preserve warmups, window endpoints, comparisons, threshold conventions, NaN/Inf handling, zero-division behavior, and fail-closed behavior.
- Preserve direction filtering, `entry_allowed`, cost/risk models, exits, backtest ordering, summary metrics, and result schema.
- Do not approximate quantiles or correlations.
- Do not add shared lookback caches in Stage 1.
- Keep PR #16 draft and unmerged until the runtime gate is re-verified.
- Do not launch the full 50,000 campaign while projected runtime exceeds 24 hours or any correctness/provenance gate fails.

---

## File Structure

- Create `xau_lab/strategies/_kernels.py` — private Numba rolling/statistical primitives shared by strategy modules.
- Create `tests/strategies/test_kernel_equivalence.py` — exact primitive-level equivalence against NumPy reference behavior.
- Modify `xau_lab/strategies/volatility.py` — compiled volatility strategy kernels.
- Modify `xau_lab/strategies/mean_reversion.py` — compiled rolling mean/std/min/max/RSI-family kernels.
- Modify `xau_lab/strategies/statistical.py` — compiled standardized-return, autocorrelation, range-position, continuation/reversal kernels.
- Modify `xau_lab/strategies/trend.py` — compiled regression/SMA/EMA/efficiency/ROC kernels.
- Modify `xau_lab/strategies/breakout.py` — retain exact expanding quantile helper and compile remaining signal loops.
- Modify `xau_lab/strategies/price_action.py` — compile candle/level loops.
- Modify `xau_lab/strategies/session.py` — compile session signal loops where applicable.
- Extend existing family test files with literal pre-optimization reference implementations and randomized/edge-case equality tests.

---

### Task 1: Exact reusable Numba rolling primitives

**Files:**
- Create: `xau_lab/strategies/_kernels.py`
- Create: `tests/strategies/test_kernel_equivalence.py`

**Interfaces:**
- Consumes: contiguous one-dimensional `np.ndarray[np.float64]` values plus integer lookbacks/scalar parameters.
- Produces: private Numba helpers including `_window_all_finite`, `_rolling_mean_std_at`, `_rolling_min_max_at`, `_linear_quantile_sorted`, `_rolling_quantile_at`, `_regression_slope_at`, and `_rolling_autocorr_at`.

- [ ] **Step 1: Write failing primitive-equivalence tests**

Add tests that compare the new helpers to literal NumPy calculations on deterministic and randomized windows:

```python
@pytest.mark.parametrize("q", [0.05, 0.236588625, 0.4, 0.5, 0.95])
def test_rolling_quantile_matches_numpy_linear(q):
    rng = np.random.default_rng(9215000)
    values = rng.normal(size=128).astype(np.float64)
    for end in range(7, len(values)):
        lookback = 7
        expected = np.quantile(values[end-lookback:end], q)
        actual = _rolling_quantile_at(values, end, lookback, q, include_current=False)
        assert actual == expected
```

Also cover repeated values, one-element rank interpolation boundaries, flat windows, NaN/Inf, zero standard deviation, and inclusive-current versus strict-prior windows.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests/strategies/test_kernel_equivalence.py -v
```

Expected: import/function failures because `_kernels.py` does not yet exist.

- [ ] **Step 3: Implement minimal exact helpers**

Create `xau_lab/strategies/_kernels.py` with `@njit(cache=True)` helpers. For exact rolling quantiles, copy the requested small window into a scratch array, reject any non-finite member, `scratch.sort()`, compute `h=(n-1)*q`, and reproduce NumPy linear interpolation using the same stable branch already used by `_expanding_linear_quantiles_kernel`.

Required signatures:

```python
@njit(cache=True)
def _window_all_finite(values, start, stop): ...

@njit(cache=True)
def _rolling_mean_std_at(values, start, stop):
    # returns (valid: bool, mean: float, std: float)

@njit(cache=True)
def _rolling_min_max_at(values, start, stop):
    # returns (valid: bool, minimum: float, maximum: float)

@njit(cache=True)
def _linear_quantile_sorted(sorted_values, q): ...

@njit(cache=True)
def _rolling_quantile_at(values, i, lookback, q, include_current): ...

@njit(cache=True)
def _regression_slope_at(values, i, lookback): ...

@njit(cache=True)
def _rolling_autocorr_at(close, i, lookback): ...
```

Do not silently coerce non-finite windows into valid outputs.

- [ ] **Step 4: Run primitive tests to verify GREEN**

Run:

```powershell
python -m pytest tests/strategies/test_kernel_equivalence.py -v
```

Expected: all primitive-equivalence tests pass exactly.

- [ ] **Step 5: Commit**

```powershell
git add xau_lab/strategies/_kernels.py tests/strategies/test_kernel_equivalence.py
git commit -m "perf: add exact compiled rolling primitives"
```

---

### Task 2: Compile the volatility family first

**Files:**
- Modify: `tests/strategies/test_misc_families.py`
- Modify: `xau_lab/strategies/volatility.py`

**Interfaces:**
- Consumes: existing `StrategyContext` and unchanged volatility parameter dictionaries.
- Produces: identical `np.int8` signal arrays for `atr_percentile_regime`, `volatility_expansion_direction`, and `volatility_contraction_reversion`.

- [ ] **Step 1: Add literal reference implementations**

In tests, copy the current Python algorithms into private `_reference_*` functions before replacing production code. Include the exact slicing semantics:

```python
prior = ranges[i-lookback:i]
threshold = np.quantile(prior, percentile)
```

- [ ] **Step 2: Add randomized and edge-case equality tests**

For each volatility strategy, generate finite random OHLC/ATR arrays plus fixtures containing NaN, Inf, repeated ranges, flat candles, minimum/maximum lookbacks, percentile boundaries, and equality-at-threshold cases. Assert:

```python
np.testing.assert_array_equal(reference(ctx, params), production(ctx, params))
```

- [ ] **Step 3: Verify baseline oracle tests pass before production edits**

Run:

```powershell
python -m pytest tests/strategies/test_misc_families.py -k "volatility" -v
```

Expected: PASS against current Python implementation.

- [ ] **Step 4: Replace Python-per-bar volatility loops with private `@njit(cache=True)` kernels**

Keep public wrappers unchanged in name/signature. `volatility_contraction_reversion` and `atr_percentile_regime` must call `_rolling_quantile_at`; `volatility_expansion_direction` must preserve `np.median(prior)` semantics by exact 0.5 linear quantile.

- [ ] **Step 5: Verify equality after optimization**

Run the same volatility tests plus:

```powershell
python -m pytest tests/strategies/test_kernel_equivalence.py tests/strategies/test_entry_integrity.py -v
```

Expected: exact signal equality, no integrity regressions.

- [ ] **Step 6: Micro-benchmark the canonical worst offender**

Run the existing canonical profile only for `EXP A6DA73D5B2BA`/`volatility_contraction_reversion` and record signal time. Require exact result equality versus the already recorded canonical row; do not set a hard speed threshold yet, but the signal time must materially drop from 189.002s.

- [ ] **Step 7: Commit**

```powershell
git add tests/strategies/test_misc_families.py xau_lab/strategies/volatility.py
git commit -m "perf: compile volatility signal generation"
```

---

### Task 3: Compile mean-reversion and exit-execution hot paths

**Files:**
- Modify: `tests/strategies/test_mean_reversion.py`
- Modify: `xau_lab/strategies/mean_reversion.py`

**Interfaces:**
- Produces exact signals for `zscore_reversion`, `bollinger_reversion`, `rsi_extreme`, `stochastic_extreme`, `cci_extreme`, and `williams_r_extreme`.

- [ ] **Step 1: Add literal reference versions of all six strategies in tests**

Retain current `_finite_window`, NumPy `mean/std/diff/max/min`, RSI zero-gain/zero-loss branches, CCI mean-deviation behavior, and inclusive-current window boundaries exactly.

- [ ] **Step 2: Add randomized/edge-case exact equality tests**

Exercise minimum/maximum lookbacks, flat windows, zero std, zero range, NaN/Inf, RSI all-up/all-down/flat, threshold equality, and repeated values.

- [ ] **Step 3: Verify baseline tests pass**

```powershell
python -m pytest tests/strategies/test_mean_reversion.py -v
```

- [ ] **Step 4: Implement Numba kernels and thin wrappers**

Use reusable rolling mean/std/min/max helpers where semantics match. For RSI and CCI, keep direct compiled loops when that is clearer than over-generalizing helpers.

- [ ] **Step 5: Verify GREEN and representative result parity**

```powershell
python -m pytest tests/strategies/test_mean_reversion.py tests/strategies/test_entry_integrity.py tests/backtest/test_fast_parity.py -v
```

Then rerun `EXPB62F70031929` and `EXP2053D4626356` in isolated result roots and compare all 51 stored fields against the current canonical outputs.

- [ ] **Step 6: Commit**

```powershell
git add tests/strategies/test_mean_reversion.py xau_lab/strategies/mean_reversion.py
git commit -m "perf: compile mean reversion signal kernels"
```

---

### Task 4: Compile statistical and trend/momentum families

**Files:**
- Modify: `tests/strategies/test_misc_families.py`
- Modify: `tests/strategies/test_trend.py`
- Modify: `xau_lab/strategies/statistical.py`
- Modify: `xau_lab/strategies/trend.py`

**Interfaces:**
- Produces exact signals for statistical strategies and all trend/momentum strategies.

- [ ] **Step 1: Add reference-oracle coverage for statistical strategies**

Cover `return_continuation`, `return_reversal`, `rolling_autocorr_direction`, `range_position_reversal`, and `standardized_return_signal`. Preserve `np.std(..., ddof=0)`, `np.corrcoef` semantics, minimum return-count checks, zero denominators, and current/prior window boundaries.

- [ ] **Step 2: Add reference-oracle coverage for trend strategies**

Cover `sma_slope`, `ema_slope`, `regression_slope`, `roc_momentum`, and `efficiency_trend`. Preserve the current EMA-of-window definition rather than substituting a global EMA series.

- [ ] **Step 3: Verify oracle tests pass before production edits**

```powershell
python -m pytest tests/strategies/test_misc_families.py tests/strategies/test_trend.py -v
```

- [ ] **Step 4: Implement compiled statistical kernels**

Use `_rolling_mean_std_at`, `_rolling_min_max_at`, and `_rolling_autocorr_at` only where exact behavior matches the oracle. Keep return arrays and direction logic identical.

- [ ] **Step 5: Implement compiled trend kernels**

For regression slope, compute the exact current formula over each inclusive-current window. For EMA slope, implement the same per-window recurrence from the first element of each window; do not replace it with a persistent EMA.

- [ ] **Step 6: Verify GREEN**

```powershell
python -m pytest tests/strategies/test_misc_families.py tests/strategies/test_trend.py tests/strategies/test_entry_integrity.py tests/backtest/test_fast_parity.py -v
```

- [ ] **Step 7: Rerun canonical representatives**

Rerun `EXPB1A10E7A4435` and `EXP1E2D424171EC`; require full stored-result equality against the current canonical outputs.

- [ ] **Step 8: Commit**

```powershell
git add tests/strategies/test_misc_families.py tests/strategies/test_trend.py xau_lab/strategies/statistical.py xau_lab/strategies/trend.py
git commit -m "perf: compile statistical and trend signal kernels"
```

---

### Task 5: Compile remaining breakout, price-action, and session loops

**Files:**
- Modify: `tests/strategies/test_breakout.py`
- Modify: `tests/strategies/test_misc_families.py`
- Modify: `xau_lab/strategies/breakout.py`
- Modify: `xau_lab/strategies/price_action.py`
- Modify: `xau_lab/strategies/session.py`

**Interfaces:**
- Keeps `_expanding_linear_quantiles` semantics unchanged while removing remaining Python-per-bar loops.

- [ ] **Step 1: Extend breakout reference coverage**

Keep existing compression-breakout semantic oracle and add references for `nbar_breakout`, `nbar_failed_breakout`, `range_expansion`, and `bollinger_expansion`, including strict-prior windows and equality boundaries.

- [ ] **Step 2: Add price-action/session reference coverage**

Cover all candle strategies and every session strategy with deterministic and randomized exact signal equality tests, including session-overlap and warmup boundaries.

- [ ] **Step 3: Verify baseline oracles pass**

```powershell
python -m pytest tests/strategies/test_breakout.py tests/strategies/test_misc_families.py -v
```

- [ ] **Step 4: Compile breakout post-processing loops**

Retain `_expanding_linear_quantiles` unchanged unless a test-proven internal refactor is necessary. Move the `compression_breakout` warmup/compressed-mean/breakout-high-low loop into Numba. Compile other breakout loops with exact rolling helpers.

- [ ] **Step 5: Compile price-action and session loops**

Avoid per-bar temporary `np.array(...)` allocations; read scalar inputs directly inside Numba while preserving non-finite checks and signal precedence exactly.

- [ ] **Step 6: Verify GREEN**

```powershell
python -m pytest tests/strategies/test_breakout.py tests/strategies/test_misc_families.py tests/strategies/test_entry_integrity.py tests/backtest/test_fast_parity.py -v
```

- [ ] **Step 7: Rerun canonical representatives**

Rerun `EXP37889369D2F1`, `EXPB8942B9FF195`, and `EXPB8F12709D5FE`; require full stored-result equality against current canonical outputs.

- [ ] **Step 8: Commit**

```powershell
git add tests/strategies/test_breakout.py tests/strategies/test_misc_families.py xau_lab/strategies/breakout.py xau_lab/strategies/price_action.py xau_lab/strategies/session.py
git commit -m "perf: compile remaining signal families"
```

---

### Task 6: Full semantic regression and eight-experiment profiler gate

**Files:**
- No production changes unless a failing equivalence test exposes a semantic bug.

**Interfaces:**
- Consumes: all compiled Stage-1 strategy implementations.
- Produces: verified signal/result parity and measured post-optimization family timings.

- [ ] **Step 1: Run the full test suite**

```powershell
python -m pytest -v
```

Expected: all tests pass; test count may exceed 191 because new equivalence tests were added.

- [ ] **Step 2: Verify canonical hashes remain unchanged**

```powershell
Get-FileHash .\results\EXPERIMENT_CATALOG.csv -Algorithm SHA256
Get-FileHash .\data\features\XAUUSD_M1_FEATURES.parquet -Algorithm SHA256
```

Require catalog SHA `AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9` and feature SHA `5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF`.

- [ ] **Step 3: Re-run the same eight-experiment profiler**

Use `multi_probe/PROBE_CATALOG.csv` and the exact profiler script already used to establish the baseline. Record market-load time, per-family signal/backtest/metrics/total times, and shares.

Baseline to beat:

```text
SIGNAL_TOTAL=473.883
BACKTEST_TOTAL=17.916
METRICS_TOTAL=5.497
PROFILE_TOTAL=497.320
SIGNAL_SHARE=95.3%
```

- [ ] **Step 4: Compare all eight stored result rows against the pre-optimization canonical rows**

Run each in a fresh isolated namespace and require zero field differences for all shared result columns.

- [ ] **Step 5: Decide whether Stage 1 has a plausible path to the 24-hour gate**

If aggregate profile time is still so high that even two-worker extrapolation cannot plausibly approach <=24 hours, stop and design Stage 2 caching before running another 100-smoke. Do not weaken the gate.

- [ ] **Step 6: Commit any test-only benchmark harness changes, if created**

Commit message:

```text
test: verify signal acceleration parity and profile gate
```

---

### Task 7: Fresh canonical 100-smoke and final runtime decision

**Files:**
- No production changes unless the smoke exposes a reproducible correctness bug.

**Interfaces:**
- Produces: authoritative Stage-1 campaign correctness, provenance, memory, and throughput evidence.

- [ ] **Step 1: Confirm a clean tracked worktree and record HEAD**

```powershell
git status --short --untracked-files=no
git rev-parse HEAD
```

Tracked status must be empty.

- [ ] **Step 2: Run a fresh 100-smoke in a new namespace**

```powershell
$smoke = ".\results\SMOKE_100_SIGNAL_KERNELS_V1"
Remove-Item -Recurse -Force $smoke -ErrorAction SilentlyContinue

Measure-Command {
    python -m scripts.run_smoke `
        --catalog .\results\EXPERIMENT_CATALOG.csv `
        --features .\data\features\XAUUSD_M1_FEATURES.parquet `
        --result-root $smoke `
        --count 100 `
        --workers 2 `
        --seed 9215000
}
```

- [ ] **Step 3: Verify smoke completeness and quotas**

Require 100 completed, 100 unique IDs, exact quotas 16/8/16/14/10/10/16/10 across the eight buckets, and `NO ERRORS`.

- [ ] **Step 4: Verify manifest provenance**

Require the current git commit plus the canonical catalog/feature SHA256 values above, seed `9215000`, and workers `2`.

- [ ] **Step 5: Verify resource stability**

Record C:/E: free space and pagefile usage. Reject any MemoryError or sustained resource growth indicating the old memory failure has returned.

- [ ] **Step 6: Apply the hard runtime gate**

Read `THROUGHPUT_BENCHMARK.json`.

Pass condition:

```text
projected_50000_hours <= 24
requires_profiling_before_full_run == false
```

If both are true, the branch is eligible for final review before any full-campaign launch. If not, the 50,000 campaign remains blocked and Stage 2 caching requires a separate design/spec approval.

- [ ] **Step 7: Update PR #16 evidence without merging**

Update the PR body/comment with fresh test count, exact head SHA, profiler results, smoke completeness, hashes, resource measurements, and projected runtime. Keep PR draft unless all success criteria are satisfied; never merge without explicit user instruction.
