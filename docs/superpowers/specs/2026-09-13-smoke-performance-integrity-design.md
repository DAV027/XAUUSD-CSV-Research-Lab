# Smoke Performance Integrity Design

## Goal

Make the canonical 100-experiment smoke campaign complete in a practical amount of time and without memory failures while preserving the exact research semantics of the existing 50,000-experiment catalog.

## Context

The canonical feature dataset contains 3,403,685 XAUUSD M1 rows from 2017 onward. The current smoke campaign has 17 successful checkpointed experiments, but a resumed one-worker run made no new progress for roughly two hours. The next pending experiment is `EXP37889369D2F1`, bucket `breakout`, strategy `compression_breakout`, parameters `{"breakout_lookback":4,"compression_lookback":36,"compression_percentile":0.236588625}`.

Three earlier attempts also recorded `MemoryError` failures in three separate stages: strategy signal validation, the Numba backtest kernel, and trade annotation. Those failures later succeeded and therefore demonstrate resource pressure rather than deterministic strategy-invalidity.

## Root Causes

### 1. Quadratic expanding quantile in `compression_breakout`

`compression_breakout()` currently recalculates `np.quantile(ranges[:i], compression_percentile)` for every bar. On a 3.4 million-row dataset, each iteration revisits an ever-growing prefix, so work grows approximately quadratically. This explains why the first remaining breakout experiment can consume hours while using one CPU core continuously.

### 2. Per-experiment signal amplification

`run_fast_backtest()` converts the full signal vector to a Python list, scans the list in Python, then converts it back to a contiguous NumPy array. With 3.4 million elements, this creates unnecessary objects, memory pressure, and runtime overhead.

`StrategyDefinition.generate()` also validates values with `np.isin(signal, (-1, 0, 1))`, creating a full-length boolean temporary. It then copies the signal again before applying `entry_allowed`.

### 3. Full-length trade output buffers

The Numba kernel allocates multiple output arrays with length equal to all market bars, even though only the first `trade_count` slots are consumed. This creates hundreds of megabytes of transient allocation pressure per experiment and caused `MemoryError` in the kernel under multi-worker execution.

### 4. Discovery-only trade duplication

`run_experiment()` annotates every returned `Trade` into a second tuple even when `include_trades=False`. Discovery metrics need trade P&L, dates, direction, holding time and costs, but do not require experiment/fingerprint/session annotations on every trade object. This creates avoidable object allocation and copying.

## Requirements

1. Preserve the canonical catalog file and every experiment ID/fingerprint.
2. Preserve strategy semantics. Optimized `compression_breakout` must produce exactly identical signals to the current implementation for the same inputs, including NaNs and edge conditions.
3. Preserve backtest semantics. Optimized execution must produce the same trade sequence, prices, reasons, costs, P&L, and summary metrics as the current/reference implementation on deterministic fixtures.
4. Do not weaken integrity checks. Signal-domain validation, `entry_allowed`, cost/risk handling, and fail-closed behavior remain enforced.
5. Do not reuse old smoke results as if they were produced by the optimized commit. The manifest already binds campaign results to `software_git_commit`; optimized smoke validation must use a fresh result/checkpoint namespace or otherwise preserve provenance isolation.
6. Do not launch the 50,000-experiment campaign until the 100-smoke gate passes under measured memory and throughput constraints.
7. Do not change research start date, feature definitions, cost model, risk model, catalog budget, seed, allocation, or holdout methodology.

## Design

### A. Exact incremental expanding quantile

Replace repeated prefix-wide `np.quantile` calls in `compression_breakout` with an exact incremental order-statistics implementation that supports the NumPy linear quantile convention used by the current code.

For finite range values, maintain a sorted multiset of all prior values. For each bar `i`, query the percentile of values from `ranges[:i]` using the same interpolation formula as NumPy's default `method="linear"`: `h=(n-1)*q`, interpolate between floor/ceil ranks. If any prior value in `ranges[:i]` is non-finite, preserve current fail-closed behavior for that bar because `_finite(history)` would have been false.

Implementation may use a coordinate-compressed Fenwick tree over the immutable finite range vector. This provides exact rank selection in `O(log U)` per update/query after `O(n log n)` coordinate preparation and avoids approximate percentiles. A simpler structure is acceptable only if equivalence and runtime tests prove it practical at 3.4 million rows.

The existing `compression_lookback`, `breakout_lookback`, comparison order, and signal conditions remain unchanged.

### B. Zero-copy signal validation path

Change `run_fast_backtest()` to accept the existing NumPy signal vector directly when possible. Normalize once with `np.asarray(..., dtype=np.int8)` / `np.ascontiguousarray` only when necessary. Validate domain without materializing a Python list. Use scalar min/max or boolean expressions that minimize temporaries and retain exact rejection of values outside `{-1,0,1}`.

In `StrategyDefinition.generate()`, avoid `np.isin` for this three-value domain. Normalize to contiguous `int8` once after validating shape and integer-equivalent values, then apply `entry_allowed` in place on a private writable copy only when masking is required. Validation must not silently coerce values such as `0.5` into valid integers.

### C. Bounded backtest output storage

Replace twelve bar-count-sized output arrays with a bounded trade-record storage design.

The preferred implementation is a Numba-compatible growable/batched record buffer: start with a modest capacity, grow geometrically only when a completed trade is appended, and return arrays sliced to `trade_count`. If Numba constraints make geometric growth brittle, use a two-pass kernel only if benchmarks show the second scan is still substantially faster and lower-memory than the current architecture.

The storage change must not alter signal timing, same-bar stop/target precedence, trailing logic, risk sizing, or exit reason semantics.

### D. Avoid discovery annotation duplication

When `include_trades=False`, compute summary metrics using the backtest trades without creating a second annotated tuple. Supply broker dates required by metrics without mutating the canonical trade semantics. Full experiment/fingerprint/session annotation remains available on paths that explicitly request trades, including finalist export/validation workflows.

If a cleaner interface is required, introduce a metrics adapter that accepts base trades plus market date lookup instead of changing the public meaning of `Trade`.

## Provenance and Existing Results

The 17 existing successful smoke results remain historical evidence from commit `1b4db32803ae35aef5945ebcbcbc05a47c8fd087`. They must not be mixed into a completed smoke result set produced by the optimized commit.

After the optimized branch is validated, run smoke in a fresh result root/state namespace so `RUN_MANIFEST.json`, `MASTER_RESULTS.csv`, `ERRORS.csv`, checkpoint state, and throughput benchmark are all attributable to one software commit. The canonical catalog and feature Parquet remain unchanged and should retain their existing hashes.

## Test Strategy

### RED gates

1. Add a reference-only implementation of the current `compression_breakout` behavior in tests and assert exact equality against the optimized implementation over deterministic small arrays, randomized arrays, repeated values, percentile edge cases, and NaN/Inf cases.
2. Add a regression case matching `EXP37889369D2F1` parameters on a sufficiently large synthetic series that exposes the expanding-quantile performance defect without relying on wall-clock-fragile microbenchmarks.
3. Add tests proving signal validation rejects non-domain floating values without `np.isin`-style large temporaries.
4. Add backtest equivalence tests comparing optimized fast output to the trusted reference backtester across exits, long/short directions, costs, same-bar ambiguity, trailing stops, and time exits.
5. Add a capacity test proving output storage scales with completed trades rather than number of bars.
6. Add a discovery-path test proving `include_trades=False` does not perform annotation copies while summary metrics remain identical.

### GREEN gates

- Targeted new tests pass.
- Entire existing test suite passes.
- `compression_breakout` reference/optimized signals are exactly equal on all fixtures.
- Fast/reference backtest trades and metrics are exactly equal on all fixtures covered by current reference tests plus new edge cases.

## Runtime Validation Gates

Run in this order only after tests are green:

1. Benchmark the exact pending experiment `EXP37889369D2F1` against the canonical feature file with one worker. It must complete successfully and in a practical duration; record elapsed time and peak memory.
2. Run a small deterministic cross-family benchmark, one or more experiments per strategy family, to detect other pathological generators.
3. Run a fresh canonical 100-experiment stratified smoke with one worker first. Require zero experiment errors and complete selected-set count 100.
4. Measure projected 50k runtime and memory.
5. Increase to two workers only if one-worker memory headroom supports it. Do not infer safe worker count from CPU count alone.
6. The full 50,000 campaign remains blocked if projected runtime is greater than 24 hours, any `MemoryError` occurs, any selected experiment fails, or provenance/integrity checks fail.

## Non-Goals

- No strategy redesign or signal-quality tuning.
- No removal of `compression_breakout`.
- No data downsampling or shortening of the 2017-2026 research window.
- No approximate quantile algorithm.
- No catalog regeneration unless equivalence cannot be preserved.
- No live trading or MT5 validation changes.
- No automatic PR merge.

## Acceptance Criteria

The change is acceptable only when all of the following are true:

- Existing canonical catalog SHA remains unchanged.
- Feature Parquet hash remains unchanged.
- Optimized `compression_breakout` matches old/reference signals exactly in tests.
- Backtest output matches reference semantics exactly in tests.
- Full automated test suite is green.
- `EXP37889369D2F1` completes without `MemoryError` and without multi-hour execution.
- Fresh 100-smoke completes with exactly 100 selected experiments and zero errors.
- Memory remains stable at the chosen worker count.
- Throughput projection is recorded before any full 50k launch.
