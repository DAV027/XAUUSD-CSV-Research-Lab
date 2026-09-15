# Stage 2 Packed Discovery Fast Path Design

**Date:** 2026-09-14

## Goal

Reduce canonical XAUUSD discovery runtime enough that the frozen 50,000-experiment campaign projects to **24 hours or less**, while preserving exact experiment semantics and every stored discovery result field.

Stage 1 successfully removed the original strategy-signal bottleneck, but the authoritative fresh 100-experiment smoke at benchmark commit `8e9a92602a185c57400f531059675c63b47e2d04` still required **373.777545 seconds** at two workers, or **3.737775 seconds per experiment**, projecting the 50,000 campaign to **186,888.773 seconds / 51.9135 hours**. The hard runtime gate therefore remains failed.

A follow-up exact-parity profile of the same frozen 100 smoke experiments identified the new bottleneck:

| Stage | Seconds | Share of accounted time | Average / experiment |
| --- | ---: | ---: | ---: |
| Strategy context | 0.7586 | 0.12% | 0.0076s |
| Signal generation | 130.2055 | 20.90% | 1.3021s |
| Direction filtering | 0.3276 | 0.05% | 0.0033s |
| Per-experiment setup | 0.0051 | ~0.00% | 0.0001s |
| Backtest | 369.0460 | 59.24% | 3.6905s |
| Metrics | 122.6151 | 19.68% | 1.2262s |

The profiling pipeline reproduced the authoritative smoke exactly on the checked result fields for all 100 experiments: `PARITY_MISMATCHES=0`, `PROFILE_PIPELINE_PARITY=PASS`.

The dominant slow experiments are also the highest-trade-count experiments. Representative examples include:

- `sma_slope`: ~740,974 trades average, ~13.94s backtest + ~4.56s metrics.
- `williams_r_extreme`: ~800,624 trades average, ~14.86s backtest + ~4.69s metrics.
- `stochastic_extreme`: ~654,727 trades average, ~12.38s backtest + ~3.97s metrics.
- Individual smoke experiments reached more than **1.17 million completed trades**.

The code path explains this scaling: the Numba backtest kernel already emits packed primitive arrays, but `run_fast_backtest()` then reconstructs a Python `_Position` and immutable `Trade` object for every completed trade, calls Python `_finalize_trade()` for every trade, and returns a tuple of those objects. `summarize_trades()` then traverses the Python objects again for P/L, drawdown, daily/monthly/yearly grouping, long/short metrics, concentration metrics, and medians.

Stage 2 will remove that object-materialization boundary from the normal discovery path without changing the detailed trade path used for inspection/export.

## Provenance and Frozen Inputs

The Stage 2 design is based on the following verified canonical inputs:

- Canonical feature artifact: `data/features/XAUUSD_M1_FEATURES.parquet`
- Canonical feature SHA256: `5D3680D043E1F8D4C759124AD177BC73E47C055101DD3BE7558564187B0C10EF`
- Canonical rows: `3,403,685`
- Canonical catalog: `results/EXPERIMENT_CATALOG.csv`
- Canonical catalog SHA256: `AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9`
- Catalog budget: `50,000`
- Campaign seed: `9,215,000`
- Authoritative smoke workers: `2`
- Authoritative smoke experiment count: `100`
- Benchmark software commit: `8e9a92602a185c57400f531059675c63b47e2d04`

The design document itself is based on current `main` at `f7460a49536d3f1b966dc6bdc2c3053844c0186c`. The runtime evidence remains explicitly tied to the benchmark commit above; implementation verification must create a new manifest bound to the implementation commit.

## Hard Constraints

- Preserve the canonical feature artifact and catalog exactly; no data regeneration, resampling, catalog edits, or experiment-ID changes.
- Preserve strategy signal arrays exactly.
- Preserve direction filtering, `entry_allowed`, cost model, risk sizing, entry/exit ordering, stop/target ambiguity rules, time exits, trailing stops, and end-of-data behavior exactly.
- Preserve the existing detailed `Trade` object API and detailed backtest behavior for callers that require trades.
- Preserve every discovery/master-result field exactly for identical inputs. Do not relax parity to `allclose` or tolerance-based comparison.
- Preserve trade ordering.
- Preserve arithmetic ordering wherever changing it would alter stored floating-point results.
- No approximate medians, quantiles, P/L aggregation, drawdown, or calendar metrics.
- No 50,000 campaign launch until a fresh 100-smoke projects `<= 24` hours with zero errors and correct provenance.
- Do not use more workers as a substitute for fixing the measured per-experiment bottleneck.
- Do not introduce giant worker-global trade matrices or shared lookback matrices.
- If packed discovery remains above 24 hours, stop and profile again before choosing the next optimization.

## Why the Previous Stage-2 Contingency Changes

The Stage-1 signal-kernel design listed shared lookback-statistic caching as a possible Stage-2 contingency. Fresh profiling after the signal-kernel work changes the evidence: signal generation is now only **20.90%** of accounted runtime, while backtest plus metrics are **78.92%**.

Therefore shared signal/lookback caching is not the first Stage-2 target. It remains a possible later optimization only if a new profile after packed discovery shows signal generation has become dominant again.

## Chosen Architecture

### Two execution modes, one semantic contract

Keep the existing detailed execution path for trade-level consumers and add a separate internal packed discovery path for the normal campaign case.

Conceptually:

```text
signal generation
      |
      v
compiled backtest kernel
      |
      +------------------------------+
      |                              |
      v                              v
packed discovery path          detailed trade path
(no Python Trade objects)      (existing Trade objects)
      |                              |
      v                              v
exact packed metrics           trade export / inspection
      |
      v
MASTER_RESULTS row
```

The normal campaign runner calls `run_experiment(..., include_trades=False)`. That path will use packed discovery. `include_trades=True` will continue to use the existing detailed `run_fast_backtest()` + `Trade` materialization path.

The detailed path remains the semantic oracle and compatibility path.

### Public API compatibility

Do **not** redefine `BacktestResult` or change the meaning of `run_fast_backtest()` for existing callers.

Preferred shape:

- Keep `run_fast_backtest(...) -> BacktestResult` unchanged.
- Add a private/internal packed backtest function, for example `run_packed_backtest(...)` or `_run_packed_backtest(...)`.
- Add a packed summary function, for example `summarize_packed_backtest(...)`.
- Change `run_experiment()` so `include_trades=False` uses the packed path and `include_trades=True` uses the detailed path.

Exact naming may follow existing module conventions during implementation, but the boundary above is part of the design.

## Packed Result Representation

The compiled kernel already produces primitive arrays for completed trades before Python object reconstruction. The packed path should expose only the primitive information needed to reproduce discovery metrics and `risk_skip_count`.

At minimum, packed discovery needs enough information to reproduce:

- trade direction,
- signal / entry / exit indices as needed,
- entry and exit raw prices or equivalent finalized economics,
- lot size,
- planned risk,
- exit reason only if required to preserve downstream semantics,
- completed trade count,
- risk skip count.

Implementation should prefer avoiding duplicate large arrays. If a metric can be computed from arrays already emitted by the kernel, do not allocate a second per-trade copy solely for convenience.

Two acceptable internal implementations are:

1. **Packed arrays + compiled finalization/summary:** expose the existing packed arrays up to `trade_count`, calculate per-trade economics and metrics without Python `Trade` objects.
2. **Kernel-emitted finalized economics:** extend the kernel to emit the numeric values needed by metrics directly, again without Python `Trade` objects.

The implementation plan should start with the least invasive option that proves exact parity. Do not fuse unrelated policy logic into the kernel merely for theoretical maximum speed.

## Exact Metrics Contract

The packed path must reproduce all metrics currently emitted by `summarize_trades()` and therefore all fields merged into the master result.

This includes at least:

- `completed_trades`
- `wins`
- `losses`
- `win_rate`
- `gross_profit`
- `gross_loss`
- `profit_factor`
- `after_cost_profit`
- `expectancy_usd`
- `max_drawdown_usd`
- `max_drawdown_pct`
- `max_loss_streak`
- `expectancy_R`
- `profit_per_active_day`
- `median_profit_per_active_day`
- `positive_year_fraction`
- `positive_month_fraction`
- `active_months`
- `worst_month`
- `long_PF`
- `short_PF`
- `long_trades`
- `short_trades`
- `trades_per_active_day`
- `median_hold_minutes`
- `top_5_trade_profit_fraction`
- `best_month_profit_fraction`
- `net_profit`
- `commission_cost`
- `spread_cost`
- `slippage_cost`
- `risk_skip_count`

The complete master-result schema, not this abbreviated list alone, is the final oracle.

### Arithmetic-order requirement

Current metrics frequently use Python left-to-right `sum()` over trades and ordered daily/monthly/yearly accumulation. A vectorized or parallel reduction can change the final IEEE-754 result even when mathematically equivalent.

Therefore:

- Do not use `fastmath`.
- Do not use unordered parallel reductions for parity-sensitive sums.
- Preserve trade order.
- Prefer explicit sequential compiled loops where necessary to reproduce Python accumulation order.
- If exact parity fails, fix the packed arithmetic order; do not weaken the test.

## Calendar Aggregation Design

Calendar metrics must use the canonical broker calendar, not recompute dates per trade through timezone conversion.

The current canonical feature artifact already contains:

- `broker_date`
- `year`
- `month`
- broker-local session/calendar columns

`MarketBundle` already carries `broker_date`, and the feature map includes numeric `year` and `month` arrays. Packed discovery should use the canonical per-bar calendar values at each trade's `entry_index`.

For efficient daily grouping, derive one compact numeric day key **once per loaded worker market**, not once per experiment or once per trade. An acceptable key is a lossless integer encoding of canonical `broker_date` (for example days-since-epoch or `YYYYMMDD`). This derived key is runtime-only and must not modify the canonical Parquet artifact.

Requirements:

- Day/month/year keys must come from the frozen broker-local calendar columns.
- No per-trade `ZoneInfo` or `datetime.fromtimestamp(...).astimezone(...)` conversion in discovery mode.
- Month/year grouping must preserve the same grouping semantics as current `summarize_trades()`.
- Daily/monthly/yearly P/L accumulation must follow trade order and exact floating arithmetic semantics.

The extra worker-local day-key array should be a compact integer array; approximately one `int32` per bar is acceptable (~13.6 MB for 3.4M rows) if profiling confirms it materially reduces trade-level calendar overhead.

## Median and Concentration Metrics

Several metrics require more than simple streaming sums:

- median active-day P/L,
- median hold minutes,
- top-five profitable trade contribution,
- best-month profit fraction.

The packed path may use bounded temporary arrays sized by the actual completed trade/day/month count where required, but must avoid Python `Trade` objects.

Design rules:

- `median_hold_minutes` may sort/copy a compact numeric hold-time array or use an exact selection algorithm. Approximation is forbidden.
- `median_profit_per_active_day` operates on active-day P/L totals, whose cardinality is tiny relative to trade count.
- Top-five profitable trades should be computed exactly. A five-element streaming top-k is preferred over sorting every positive trade, but only if it matches current ordering/value semantics exactly.
- Best-month profit uses exact accumulated month totals.

If a simpler exact implementation is already fast enough, prefer simplicity over a more complex algorithm.

## Detailed Trade Path

`include_trades=True` remains detailed and compatibility-oriented.

It must continue to return annotated `Trade` objects with:

- experiment/fingerprint metadata,
- entry/exit times and prices,
- stop/target,
- exit reason,
- cost breakdown,
- `pnl_R`,
- hold time,
- broker date/timezone,
- session flags,
- entry/exit indices.

No export, finalist, inspection, or debugging workflow may silently receive a reduced packed object in place of `Trade`.

Packed discovery is an optimization of the `include_trades=False` path only.

## Error Handling

- Existing per-experiment exception isolation remains unchanged.
- Invalid signals and parameter errors continue to fail through existing validation.
- Packed summary must fail loudly if packed array lengths/counts are inconsistent.
- Calendar lookup must fail loudly if an entry index exceeds the canonical calendar arrays.
- No fallback approximation is allowed after a packed-metrics error.
- During rollout, parity tests are the fallback mechanism: the detailed path is the oracle, not an automatic production fallback that could hide defects.

## Memory Design

The optimization goal is lower Python-object overhead and lower CPU time without reintroducing memory instability.

Requirements:

- Do not materialize Python `Trade` objects in discovery mode.
- Reuse existing packed arrays where possible.
- Slice/view packed arrays by `trade_count` rather than copying full capacity when practical.
- Avoid object-dtype per-trade arrays.
- Keep any derived calendar key worker-local and compact.
- Do not retain per-experiment packed arrays after the master result has been summarized and written.
- Preserve the proven parent-only result writer architecture.

A high-trade experiment with ~1M trades must remain bounded and should use substantially less Python heap than the current detailed path.

## TDD and Semantic Oracle

Implementation follows RED -> GREEN. The current detailed path is the oracle.

### Layer 1: Packed backtest economics

For deterministic and adversarial small fixtures, compare packed economics against `run_fast_backtest()` trade-by-trade for all values needed by discovery metrics.

Include cases covering:

- long and short entries,
- spread/slippage/commission,
- risk minimum and risk skips,
- stop exits,
- same-bar stop/target ambiguity,
- target exits,
- time exits,
- ATR trailing stops,
- end-of-data exit,
- NaN/invalid ATR behavior,
- direction filtering handled before the backtest boundary.

### Layer 2: Packed metric parity

For the same fixtures, compare the complete packed summary dict with `summarize_trades()` using exact equality.

Include:

- no-trade case,
- only wins,
- only losses,
- mixed long/short,
- multiple trades on one day,
- day/month/year transitions,
- odd/even median counts,
- equal P/L values,
- large trade counts,
- top-five ties,
- drawdown and loss-streak boundaries.

### Layer 3: `run_experiment()` parity

For representative experiments, require the complete `master_result` from:

```text
include_trades=False packed path
```

to equal the detailed/oracle result produced from the existing trade path for every stored field.

### Layer 4: Frozen canonical evidence

Before any runtime claim:

1. Run the complete pytest suite.
2. Run the frozen eight canonical representative experiments.
3. Compare the complete master-result dictionaries/CSV fields between packed and detailed paths.
4. Run the same frozen 100 smoke experiments through a parity harness and require **100/100 complete-result parity**.
5. Run a fresh authoritative 100-smoke at two workers in a new namespace with the canonical hashes.

No tolerance-based exception list is permitted.

## Performance Gates

### Stage-2 micro/profile gate

After correctness is GREEN, profile the same frozen 100 experiments using the same decomposition used to identify Stage 2.

Expected qualitative result:

- Backtest + metrics must fall substantially from the current 78.92% share.
- High-trade strategies must show the largest absolute improvement.
- Python object materialization must no longer scale linearly as a dominant cost in discovery mode.

This profile is diagnostic; the authoritative decision remains the fresh smoke.

### Authoritative 100-smoke gate

Use:

- canonical feature SHA `5D3680...C10EF`,
- canonical catalog SHA `AB2AD9...706A9`,
- seed `9215000`,
- 100 experiments,
- exact frozen bucket quotas,
- two workers,
- a fresh result/checkpoint namespace,
- zero errors,
- 100 completed unique IDs.

The hard performance condition remains:

```text
projected_50000_seconds <= 86400
projected_50000_hours <= 24
requires_profiling_before_full_run == false
```

Only then may the 50,000 campaign be launched.

### Required speedup

The current authoritative projection is 51.9135 hours, so Stage 2 needs at least **2.163x end-to-end throughput improvement** to pass with no margin.

The design should target materially more than 2.163x because machine noise and experiment mix can move the smoke result. A practical target is **<= 20-22 projected hours** before launching the full campaign.

## If Stage 2 Is Still Above 24 Hours

Do not broaden this implementation while debugging it.

If packed discovery is exact but the new smoke still projects above 24 hours:

1. stop,
2. re-profile the exact fresh 100 smoke,
3. identify the new dominant stage,
4. design the next optimization separately.

Likely later candidates, only if profiling justifies them, are the remaining signal hotspots:

- `atr_percentile_regime`,
- `volatility_contraction_reversion`,
- `range_expansion`,
- `bollinger_expansion`,
- `volatility_expansion_direction`.

Worker-count scaling may be benchmarked only after per-experiment packed performance is understood and memory remains stable. It does not replace the two-worker canonical gate unless the research protocol itself is separately changed and approved.

## Expected Files to Change

Likely production files:

- `xau_lab/backtest/fast.py` — expose/consume packed kernel output without mandatory `Trade` reconstruction.
- `xau_lab/metrics/performance.py` or a focused private packed-metrics module — exact packed summary implementation.
- `xau_lab/runner/single.py` — select packed path for `include_trades=False`, preserve detailed path for `include_trades=True`.
- `xau_lab/runner/campaign.py` — only if `MarketBundle` needs a compact precomputed broker-day key.
- `xau_lab/backtest/models.py` — only if a small private/public-neutral packed result dataclass materially improves clarity; do not alter `BacktestResult` semantics.

Likely tests:

- fast/reference backtest parity tests,
- new packed-backtest economics tests,
- new packed-metrics exact-equivalence tests,
- `run_experiment` packed-vs-detailed complete result parity,
- smoke/runtime verification tooling only if existing scripts cannot express the required evidence without production behavior changes.

Do not modify strategy implementations in this Stage-2 change unless a failing exactness test proves a directly related defect.

## Non-Goals

- No strategy redesign.
- No parameter-domain changes.
- No profitability filtering or optimization.
- No change to risk or cost assumptions.
- No approximate statistics.
- No replacement of the detailed trade/export path.
- No catalog or feature regeneration.
- No giant shared caches.
- No automatic 50,000 launch.
- No worker-count tuning as the primary fix.

## Acceptance Criteria

Stage 2 is complete only when all of the following are true:

1. Full test suite passes.
2. Detailed `run_fast_backtest()` compatibility remains intact.
3. `include_trades=True` returns the same detailed trade semantics as before.
4. Packed discovery emits no Python `Trade` objects in the normal campaign path.
5. Packed-vs-detailed complete result parity passes on deterministic/adversarial tests.
6. Frozen eight canonical experiments match exactly.
7. Frozen canonical 100-experiment parity passes 100/100 with zero mismatched stored fields.
8. Fresh authoritative 100-smoke completes 100 unique experiments with zero errors and correct hashes/manifest.
9. Fresh authoritative projection is `<= 24` hours.
10. Memory/pagefile behavior remains stable enough for the two-worker campaign configuration.

Until all ten are satisfied, the full 50,000 campaign remains blocked.
