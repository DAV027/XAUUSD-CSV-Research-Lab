# Exact compression and discovery memory optimization

## Scope and invariants

Base: 1b4db32803ae35aef5945ebcbcbc05a47c8fd087 (PR #12).
Branch: perf/exact-compression-memory. No merge authorized.
Preserve the canonical 3,403,685-row feature dataset and 50,000-row catalog.
Catalog SHA256: AB2AD9935FB4E9A229CE1D84E2E9B80C4D670A8CEB7E577BF3CAE0B70F0706A9.
No rolling-percentile replacement, strategy redesign, lookahead, or full 50k campaign.

## Implementation and verification sequence

1. Freeze the legacy compression implementation in regression tests; demonstrate repeated expanding scans (1,998,370 inspected values for 2,000 bars).
2. Replace quantile rescans with sorted coordinate compression and a Fenwick frequency tree. Insert only prior observations; select adjacent ranks and use NumPy's exact linear interpolation branch. Keep NumPy window means and breakout comparisons unchanged. Stop after the first nonfinite historical range, while preserving quantile overflow behavior for finite inputs.
3. Demonstrate RED failures for signal scalar materialization, full-market output buffers, full-signal validation temporaries, and discovery annotation duplication.
4. Validate signals in bounded 4096-element chunks before int8 conversion. Size output arrays to nonzero signals before the final bar, a proven upper bound on trades. Preserve all kernel trade logic.
5. Summarize original trades with entry broker-date overrides during discovery. Retain full annotated trade export when requested.
6. Run complete tests and exact signal/quantile/trade-field comparisons, then the blocked experiment, then eight cross-bucket experiments. Independently review changes before commit.
7. Commit software and start a fresh deterministic 100-smoke under a unique parent directory, with its own results/ and state/. Do not reuse the historical results or checkpoint. Never merge automatically.

## Test environment

Python 3.12; isolated pytest 8.4.2 (system pytest 9 is outside the declared project range).
Tests require repository write access for Numba cache files on this Windows environment; restricted cache access stalled collection.
Run: `python -m pytest -p no:cacheprovider` with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
First full RED run: 4 expected memory regressions failed, 352 passed.
First full GREEN run: 356 passed. Additional quantile and overflow regressions followed.
The original local pyproject.toml package-discovery edit belongs to the user and is excluded from this change.

## Benchmark reproduction

Run from the repository root:

```powershell
python -m scripts.benchmark_performance --catalog results/EXPERIMENT_CATALOG.csv --features data/features/XAUUSD_M1_FEATURES.parquet --experiment-id EXP37889369D2F1 --output results/<unique-benchmark>/single.json
```

The benchmark uses the real discovery runner and records signal generation, backtest, metrics, total time, and process-lifetime peak resident working set. The latter includes dataset loading and is cumulative across experiments in one process; it is not isolated per-experiment allocation. Compilation/cache warmup is included when it occurs. Benchmark JSON is separate from campaign results and is not eligible for resume.

Initial full-data single experiment: signals 32.895 s, backtest 0.637 s, metrics 0.207 s, total 33.810 s; 34,028 trades, 39,675 unfiltered signals. Load 4.010 s; peak working set 3,921,600,512 bytes. These are precommit measurements, not a controlled old/new runtime ratio. The old full-data experiment had been reported stalled after approximately two hours; it was not rerun to completion.

The fresh smoke command must use a new parent namespace because run_campaign places its checkpoint at result_root.parent/state/CHECKPOINT.json:

```powershell
python -m scripts.run_smoke --catalog results/EXPERIMENT_CATALOG.csv --features data/features/XAUUSD_M1_FEATURES.parquet --result-root results/perf-<commit>/results --count 100 --workers 1
```

A 100-smoke pass and acceptable measured throughput remain prerequisites for any 50,000-experiment run; this task does not authorize that run.

Final precommit suite: 363 passed in 27.94 seconds. Exact quantiles also verified at 129 expanding-history points spanning all 3,403,685 rows; exact legacy signals verified on three independent 8,192-row real-data fixtures. Independent read-only code review approved with no actionable findings.
