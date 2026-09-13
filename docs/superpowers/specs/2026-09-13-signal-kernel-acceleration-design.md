# Signal Kernel Acceleration Design

**Date:** 2026-09-13

## Goal

Reduce canonical discovery runtime enough that the 50,000-experiment XAUUSD campaign projects to **24 hours or less**, while preserving the exact semantics, IDs, parameters, risk model, cost model, feature data, and result schema already validated by PR #16.

The current fresh 100-smoke completes correctly but measures **32.3029 seconds per experiment** at two workers, projecting the 50,000 campaign to **448.65 hours**. Profiling on eight representative experiments attributes **95.3% of total runtime to strategy signal generation**, versus 3.6% to backtesting and 1.1% to metrics. The required end-to-end target is approximately **1.728 seconds per experiment**, an overall improvement of roughly **18.7x**.

## Constraints

- Preserve `results/EXPERIMENT_CATALOG.csv` exactly, including all 50,000 experiment IDs and fingerprints.
- Preserve canonical feature artifact semantics and the current 72-column dataset, including the 62 pre-session columns and the frozen session features.
- Preserve strategy output exactly for identical inputs and parameters.
- Preserve warmups, window endpoints, comparison operators, threshold conventions, NaN/Inf handling, zero-division behavior, and fail-closed behavior.
- Preserve direction filtering, `entry_allowed`, costs, risk sizing, exit semantics, backtest ordering, and summary metrics.
- Keep PR #16 draft and unmerged until the new runtime gate is re-verified.
- Do not launch the full 50,000 campaign while projected runtime exceeds 24 hours or if any correctness/provenance gate fails.
- Do not introduce approximate rolling statistics or approximate quantiles.
- Do not add shared worker caches in the first implementation stage.

## Current Bottleneck

Representative canonical profiling on 3,403,685 bars produced:

| Bucket | Strategy | Signal | Backtest | Metrics | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| volatility | `volatility_contraction_reversion` | 189.002s | 2.177s | 0.586s | 191.771s |
| statistical | `standardized_return_signal` | 66.547s | 3.776s | 1.356s | 71.681s |
| exit_execution | `zscore_reversion` | 64.508s | 0.655s | 0.154s | 65.319s |
| mean_reversion | `bollinger_reversion` | 63.789s | 0.856s | 0.250s | 64.897s |
| trend_momentum | `regression_slope` | 36.541s | 0.763s | 0.224s | 37.532s |
| breakout | `compression_breakout` | 35.094s | 0.664s | 0.205s | 35.964s |
| price_action | `rejection_candle` | 14.466s | 3.394s | 1.042s | 18.907s |
| session | `opening_range_breakout` | 3.936s | 5.631s | 1.680s | 11.249s |

Across the profile: **signal generation 95.3%, backtest 3.6%, metrics 1.1%**.

The dominant anti-pattern is a Python loop across all bars that repeatedly calls NumPy reductions on small rolling slices. Examples include rolling `quantile`, `median`, `mean`, `std`, `max`, `min`, `corrcoef`, and `dot` inside a 3.4-million-iteration Python loop.

## Chosen Architecture

### Stage 1: Exact compiled signal kernels

Keep the existing public strategy API:

```python
strategy(ctx: StrategyContext, params: dict) -> np.ndarray
```

Each hot strategy becomes a thin Python wrapper that:

1. extracts/scalars parameters,
2. normalizes required arrays to contiguous NumPy arrays only when necessary,
3. invokes a private `@numba.njit(cache=True)` kernel,
4. returns the same contiguous `np.int8` signal array as today.

This limits change to implementation internals; registry contracts, experiment generation, runner code, catalog semantics, and backtest interfaces remain unchanged.

### Reusable compiled primitives

Introduce private Numba-compatible helpers for recurring exact operations:

- finite-window validation,
- rolling sum / mean,
- rolling sum-of-squares / population standard deviation,
- rolling minimum / maximum,
- regression slope,
- direction-from-body / threshold comparisons,
- exact rolling quantile with NumPy default linear interpolation,
- rolling autocorrelation where required.

Helpers must preserve current window inclusion/exclusion rules exactly. If a strategy currently uses `values[i-lookback:i]`, the compiled version must use that strict prior window. If it uses `i-lookback+1:i+1`, the compiled version must retain the inclusive-current window.

### Quantile requirements

Rolling quantile strategies are the highest-risk optimization because semantic drift is easy. The implementation must reproduce:

```python
np.quantile(window, q)
```

with NumPy's default linear interpolation for every tested finite window and must continue to skip the bar when any member of the required window is non-finite. No percentile approximation, histogram binning, sampling, or interpolation substitution is allowed.

For short rolling windows (current strategy domains are generally <=200), an exact Numba implementation may copy the small window into a fixed/temporary buffer, sort it, and perform the same linear rank interpolation. That is O(N * L log L), but with all work compiled and `L <= 200`, it avoids the dominant Python/NumPy call overhead while remaining straightforward to prove equivalent. A more complex rolling order-statistics structure is not required unless benchmarks show this exact compiled sort remains too slow.

### Strategy conversion order

Implement in benchmark-driven order:

1. **Volatility**
   - `volatility_contraction_reversion`
   - `atr_percentile_regime`
   - `volatility_expansion_direction`
2. **Mean reversion / exit execution**
   - `bollinger_reversion`
   - `zscore_reversion`
   - RSI, stochastic, CCI, Williams %R
3. **Statistical**
   - `standardized_return_signal`
   - rolling autocorrelation
   - range-position reversal
   - return continuation/reversal
4. **Trend / momentum**
   - `regression_slope`
   - SMA/EMA slope
   - efficiency trend
   - ROC momentum
5. **Breakout / price action / session**
   - compile remaining Python bar loops including the post-threshold loop in `compression_breakout`
   - compile simple candle/session condition loops where useful

The conversion sequence is ordered by measured runtime impact, not by module name.

## Semantic Oracle Strategy

Every optimized strategy must be guarded by a literal reference implementation representing the pre-optimization behavior.

Tests will compare reference and optimized outputs with `np.array_equal` over:

- deterministic fixtures,
- randomized finite data,
- NaN/Inf placements,
- repeated/equal values,
- flat windows / zero standard deviation,
- zero denominators,
- minimum and maximum parameter-domain values,
- threshold boundary equality cases,
- warmup boundaries,
- multiple random parameter samples per strategy.

The test oracle should live in tests, not production, so the slow Python implementation is not retained in the runtime path.

For strategies whose current implementation derives temporary arrays (returns, true range, typical price, bandwidth), oracle tests must compare final signal arrays, not merely intermediate statistics.

## TDD and Verification Flow

For each family:

1. Add semantic-oracle tests that pass against the current Python implementation.
2. Add a structural/performance regression seam where necessary to prove the optimized function actually calls a compiled kernel rather than another Python-per-bar loop.
3. Implement the Numba kernel.
4. Run the family-specific tests.
5. Run cross-family strategy tests and fast/reference backtest parity tests.
6. Commit only after GREEN evidence.

After all hot families are converted:

1. Run the complete pytest suite.
2. Re-run the exact eight-experiment profiler with the same feature artifact and experiments.
3. Compare every optimized experiment's full stored result against the current canonical result where an old result exists.
4. Run a fresh 100-smoke in a new result/checkpoint namespace because `software_git_commit` changes.
5. Require 100/100 unique experiments, exact bucket quotas, zero errors, correct hashes, and stable memory/pagefile.
6. Use the new throughput benchmark to gate the 50,000 campaign.

## Runtime Gates

### Intermediate profiler gate

The eight representative experiments must show a substantial reduction in signal-generation time with no signal/result changes. The target is not merely “faster”; aggregate profile time should indicate a plausible route to the 24-hour full-campaign requirement.

### Canonical 100-smoke gate

The fresh smoke must satisfy all existing correctness/provenance conditions and additionally report:

```text
projected_50000_hours <= 24
```

Only after that condition is met may the full campaign be considered.

### If Stage 1 is insufficient

If exact compiled kernels preserve correctness but the fresh smoke still projects above 24 hours, do **not** weaken the runtime gate. Move to Stage 2 only then.

## Stage 2 Contingency: Shared lookback-statistic caching

Stage 2 is intentionally out of scope for the first implementation. If required, design a separate cache layer that exploits repeated integer lookbacks across experiments within a worker.

Potential cacheable artifacts include:

- rolling mean/std for close by lookback,
- rolling high/low by lookback,
- range quantiles by lookback/percentile only when reuse justifies memory,
- regression coefficients by lookback,
- derived returns/true-range arrays independent of experiment parameters.

Any cache must be bounded, worker-local, provenance-neutral, and must not materially reintroduce the memory failure that motivated PR #16. It requires its own design review before implementation.

## Memory Design

Stage 1 should be approximately memory-neutral. Kernels return a single `int8` signal array and may allocate only small rolling scratch buffers or one strategy-specific derived array where already required by semantics.

Do not precompute matrices with shape `(number_of_lookbacks, bars)` during Stage 1. On 3.4M bars, such matrices can quickly consume gigabytes per worker and undermine the proven two-worker stability.

## Error Handling

- Invalid/non-finite strategy inputs continue to fail closed at the bar level exactly as before.
- Invalid parameter-domain values remain rejected by existing strategy-definition validation.
- Numba compilation/runtime exceptions remain isolated per experiment by the existing campaign worker boundary.
- No optimized kernel may silently substitute a fallback approximation after an error.

## Files Expected to Change

Likely production files:

- `xau_lab/strategies/volatility.py`
- `xau_lab/strategies/mean_reversion.py`
- `xau_lab/strategies/statistical.py`
- `xau_lab/strategies/trend.py`
- `xau_lab/strategies/breakout.py`
- `xau_lab/strategies/price_action.py`
- `xau_lab/strategies/session.py`
- optionally a focused private helper module such as `xau_lab/strategies/_kernels.py` if reuse materially improves clarity.

Likely tests:

- existing family strategy test files,
- new targeted semantic-equivalence tests if the existing files would become unwieldy.

No changes are planned to catalog generation, risk/cost models, backtest semantics, feature-generation semantics, or promotion criteria.

## Non-Goals

- Do not redesign trading strategies.
- Do not remove slow strategy families.
- Do not reduce the 50,000 budget to make runtime appear acceptable.
- Do not downsample the canonical 3.4M-row dataset.
- Do not approximate quantiles/correlations.
- Do not alter session definitions.
- Do not increase workers as a substitute for fixing the hot path.
- Do not optimize backtest or metrics before signal generation unless new profiling shows the bottleneck has moved there.
- Do not merge PR #16 without explicit user instruction.

## Success Criteria

The signal-kernel acceleration work is considered successful only when all of the following are freshly verified:

1. full test suite passes,
2. optimized strategy outputs match semantic oracles exactly,
3. canonical experiment result parity is preserved,
4. fresh 100-smoke completes 100/100 with zero errors,
5. canonical catalog and feature hashes match the approved artifacts,
6. resource usage remains stable at the chosen worker count,
7. `projected_50000_hours <= 24`.

Until all seven criteria pass, the full 50,000 campaign remains blocked.
