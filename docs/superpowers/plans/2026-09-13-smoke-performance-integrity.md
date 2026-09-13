# Smoke Performance Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical 100-experiment smoke campaign complete without memory failures or pathological multi-hour experiments while preserving the exact semantics of the existing 50,000-experiment catalog.

**Architecture:** Optimize the existing pipeline in place without changing experiment definitions. First replace the quadratic expanding percentile in `compression_breakout` with an exact equivalent order-statistics implementation; then remove avoidable signal copies, bound backtest trade storage by actual trade capacity rather than bar count, and avoid annotation duplication in discovery-only runs. All semantic changes are guarded by exact-reference tests.

**Tech Stack:** Python 3.12, NumPy, Numba, Polars, pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-smoke-performance-integrity-design.md`

## Global Constraints

- Preserve `results/EXPERIMENT_CATALOG.csv` and every experiment ID/fingerprint.
- Preserve canonical feature data and research window.
- Preserve strategy signals exactly for identical inputs.
- Preserve fast/reference backtest trade semantics and summary metrics.
- Do not weaken `entry_allowed`, signal-domain, cost, risk, or provenance checks.
- Do not merge the PR without explicit user instruction.
- Do not launch the full 50,000 campaign until the fresh 100-smoke passes and projected runtime is <=24 hours.

---

### Task 1: Exact `compression_breakout` equivalence and scalable expanding quantile

**Files:**
- Modify: `tests/strategies/test_breakout.py`
- Modify: `xau_lab/strategies/breakout.py`

**Interfaces:**
- Consumes: `StrategyContext`, current `compression_breakout(ctx, params)` contract.
- Produces: the same `np.ndarray[int8]` signal vector, with the expanding percentile matching NumPy default linear quantile exactly.

- [ ] **Step 1: Add a test-only reference implementation of current behavior**

Add `_compression_breakout_reference(ctx, params)` to `tests/strategies/test_breakout.py` by copying the current algorithm literally: `ranges[:i]`, `np.quantile(..., q)`, same warmup, finite checks, breakout checks, and comparisons.

- [ ] **Step 2: Add exact-equivalence cases**

Test deterministic repeated values, randomized finite arrays, percentile values `0.05`, `0.236588625`, and `0.40`, varying lookbacks, and NaN/Inf cases. Assert `np.array_equal(reference, compression_breakout(...))`.

- [ ] **Step 3: Add a structural regression test for the optimized helper**

Expose a private helper such as `_expanding_linear_quantile(values, q)` and assert it matches `np.quantile(values[:i], q)` for every prefix of small deterministic fixtures. This is an exactness test, not a wall-clock test.

- [ ] **Step 4: Verify RED**

Run locally: `python -m pytest tests/strategies/test_breakout.py -v`.

Expected before implementation: new helper/import test fails because the scalable helper does not exist.

- [ ] **Step 5: Implement exact scalable quantile**

In `xau_lab/strategies/breakout.py`, implement an exact expanding rank structure. Preferred design: coordinate-compress finite `ranges`, maintain a Fenwick tree of counts, and select floor/ceil order statistics for NumPy linear interpolation `h=(n-1)*q`. Track whether any non-finite value has appeared in the expanding history so behavior remains fail-closed thereafter, matching `_finite(history)`.

Do not change `compression_lookback`, `breakout_lookback`, or signal comparisons.

- [ ] **Step 6: Verify GREEN**

Run `python -m pytest tests/strategies/test_breakout.py -v` and confirm all cases pass.

- [ ] **Step 7: Commit**

Commit message: `perf: make compression breakout expanding quantile scalable`

---

### Task 2: Remove signal-array amplification while preserving validation

**Files:**
- Modify: `tests/strategies/test_entry_integrity.py`
- Modify: `tests/backtest/test_fast_parity.py`
- Modify: `xau_lab/strategies/base.py`
- Modify: `xau_lab/backtest/fast.py`

**Interfaces:**
- Consumes: arbitrary one-dimensional strategy signal output and `Sequence[int]`/NumPy signals passed to `run_fast_backtest`.
- Produces: contiguous `int8` signals containing only `-1,0,1`, with identical rejection behavior for invalid values and identical `entry_allowed` masking.

- [ ] **Step 1: Add invalid-float regression tests**

Add tests proving values such as `0.5`, `1.5`, NaN, and Inf are rejected before integer conversion. Add tests proving valid float-valued `-1.0,0.0,1.0` inputs are accepted if the current contract accepts them.

- [ ] **Step 2: Add no-copy fast-path test where observable**

For an already contiguous `int8` array with no `entry_allowed` mask, assert generation returns a valid contiguous array and does not require a Python-list conversion path. Keep the test behavioral; do not assert CPython object internals.

- [ ] **Step 3: Verify RED**

Run `python -m pytest tests/strategies/test_entry_integrity.py tests/backtest/test_fast_parity.py -v`.

- [ ] **Step 4: Implement single-normalization validation**

In `StrategyDefinition.generate()`, validate shape, finiteness/integer-equivalence, and `[-1,1]` bounds without `np.isin`; normalize once to contiguous `int8`; only copy when applying `entry_allowed`.

In `run_fast_backtest()`, remove `list(signals)`. Use `np.asarray(signals)` once, verify one dimension/length/domain, and make contiguous `int8` only when needed.

- [ ] **Step 5: Verify GREEN and parity**

Run `python -m pytest tests/strategies/test_entry_integrity.py tests/backtest/test_fast_parity.py -v`.

- [ ] **Step 6: Commit**

Commit message: `perf: eliminate full signal list conversions`

---

### Task 3: Bound Numba trade-output storage by possible trades, not bars

**Files:**
- Modify: `tests/backtest/test_fast_parity.py`
- Modify: `xau_lab/backtest/fast.py`

**Interfaces:**
- Consumes: same market bars, signals, symbol, cost, risk, and exit spec.
- Produces: identical `BacktestResult`; internal packed arrays contain only completed-trade capacity rather than twelve arrays of length `len(bars)`.

- [ ] **Step 1: Extend trade-for-trade parity coverage**

Add deterministic same-bar ambiguity, time-exit, trailing-stop, sparse-signal, dense-signal, long-only, short-only, and no-trade fixtures comparing every meaningful `Trade` field and `risk_skip_count` against `run_reference_backtest`.

- [ ] **Step 2: Add storage-capacity regression seam**

Refactor the internal kernel return into a private packed-result helper whose array lengths can be inspected in tests. Assert a long sparse market with one completed trade does not return output arrays sized to the market length.

- [ ] **Step 3: Verify RED**

Run `python -m pytest tests/backtest/test_fast_parity.py -v` and confirm the new capacity assertion fails under the current full-bar allocation design.

- [ ] **Step 4: Implement bounded storage**

Prefer a Numba-compatible geometric-growth buffer starting at a small capacity and doubling when a completed trade is appended. If Numba cannot safely reallocate all aligned fields, implement a first pass that counts completed trades/risk skips and a second pass that writes into exactly-sized arrays. Preserve control flow and exit precedence exactly.

- [ ] **Step 5: Verify parity**

Run `python -m pytest tests/backtest/test_fast_parity.py tests/backtest/test_reference_engine.py -v`.

- [ ] **Step 6: Commit**

Commit message: `perf: bound fast backtest trade buffers`

---

### Task 4: Avoid discovery-only annotation duplication

**Files:**
- Modify: `tests/runner/test_campaign.py`
- Modify: `tests/metrics/test_performance.py` if a metrics adapter is introduced
- Modify: `xau_lab/runner/single.py`
- Optionally modify: `xau_lab/metrics/performance.py`

**Interfaces:**
- Consumes: base `Trade` objects from fast backtest plus market broker-date lookup.
- Produces: identical master metrics; full annotated `Trade` tuples only when `include_trades=True`.

- [ ] **Step 1: Add discovery-output equivalence test**

Construct one deterministic `CompleteExperiment`/`MarketBundle` and call `run_experiment(..., include_trades=False)` and `run_experiment(..., include_trades=True)`. Assert `master_result` is identical and the false path returns `trades == ()`.

- [ ] **Step 2: Add annotation-path test**

Assert `include_trades=True` still fills experiment ID, fingerprint, broker date, sessions, timezone, and preserves all P&L fields.

- [ ] **Step 3: Verify RED**

Introduce a test seam around `_annotate_trade` (monkeypatch/counter) showing the false path should not annotate every trade. Current code should fail because it always annotates first.

- [ ] **Step 4: Implement discovery metrics without full annotation copy**

When `include_trades=False`, compute metrics from base trades while substituting broker date from `market_bundle.broker_date[entry_index]` where required. Keep full `_annotate_trade` behavior for `include_trades=True`.

- [ ] **Step 5: Verify GREEN**

Run `python -m pytest tests/runner/test_campaign.py tests/metrics/test_performance.py -v`.

- [ ] **Step 6: Commit**

Commit message: `perf: avoid discovery trade annotation copies`

---

### Task 5: Full regression, PR CI, and runtime gate preparation

**Files:**
- No production changes unless regression failures expose a semantic bug.
- Use: `tests/integration/test_smoke_cli.py`, `tests/runner/test_manifest.py`, `tests/runner/test_smoke_provenance.py`.

**Interfaces:**
- Produces: branch ready for authoritative PR CI and local canonical runtime validation.

- [ ] **Step 1: Run the complete local suite**

Run `python -m pytest -v`.

Expected: all tests pass with no warnings/errors that indicate behavior degradation.

- [ ] **Step 2: Confirm canonical catalog is untouched**

Run `Get-FileHash .\results\EXPERIMENT_CATALOG.csv -Algorithm SHA256` locally and require the existing SHA `AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9`.

- [ ] **Step 3: Open PR to `main`**

Opening the PR triggers `.github/workflows/tests.yml`; do not merge.

- [ ] **Step 4: Require green PR CI**

Record exact head SHA, workflow run ID, test count, and duration.

- [ ] **Step 5: Local canonical single-experiment runtime validation**

After pulling the branch locally, run the exact pending `EXP37889369D2F1` against the canonical feature file with one worker in an isolated temporary result root. Record elapsed time, peak RAM/private bytes, and errors.

- [ ] **Step 6: Fresh 100-smoke provenance namespace**

Run from a clean namespace such as `perf_smoke/results` so its checkpoint resolves under `perf_smoke/state`, while continuing to use the unchanged canonical catalog/features:

`python -m scripts.run_smoke --catalog results/EXPERIMENT_CATALOG.csv --features data/features/XAUUSD_M1_FEATURES.parquet --result-root perf_smoke/results --count 100 --workers 1`

Require exactly 100 selected experiments, zero errors, stable memory, and a throughput benchmark.

- [ ] **Step 7: Gate full campaign**

Do not run 50k if projected runtime is >24 hours, any selected experiment fails, any `MemoryError` occurs, or catalog/feature provenance differs.
