# Stage A verification

- Analysis commit: `d6c7e2211ed1ba9bbc7608188dbbe551009f56df`.
- Full suite: **619 passed, 0 failed, 0 errors, 0 skipped**, 40.75 seconds.
- 22 new synthetic tests; unchanged baseline suite contained 597 tests.
- Two analysis/diagnostic runs produced byte-identical copies of all six result files listed below.
- All pre-existing tracked repository files remain unchanged against frozen base `3d63a68218a921b50c5e1406eab355c58c026a88`.
- No candidate strategy, parameters, parity tolerances, or raw tick datasets were added or modified.
- Frozen replay reproduced 364 trades and its existing one-cent residual; the strict parity flag remains false.
- Original EA source was neither located nor modified. No new MT5 compilation or terminal test is claimed.
- Initial test development caught a test invocation error (replicate count passed as seed); corrected before the full passing run.
- Stage B and subsequent work remain deferred for owner review. No merge to main.

Python 3.12.0; package versions:

```json
{
  "numpy": "2.5.3",
  "polars": "1.44.2",
  "pyarrow": "21.0.0",
  "scipy": "1.18.1",
  "numba": "0.67.0",
  "pytest": "8.4.2",
  "tzdata": "2026.4"
}
```

Repeatability hashes:

```json
{
  "ENTRY_FEATURES.csv": "4afdbe9b76cb61d77b51f19ae977be7e1406079e77f6e127508a26714bbac6cd",
  "FEATURE_DECOMPOSITION.csv": "05c2e68e51701422a23c1381bd4fe5f323de50ad39d2f15748834d6fb04d5b29",
  "STAGE_A_SUMMARY.json": "af0e8038cedfb17d86f6fe5779d19b93ccf4cce00fcd3b4cb088bb03a1603ebc",
  "FIXED_LEDGER_COST_SENSITIVITY.csv": "c70f7653f8da8799d4ba750325f188214b04db3e3d3cc8e8dbea67b70543dc7a",
  "ROBUSTNESS_DIAGNOSTICS.json": "db5352ccd3bfd4f6d5b8c8fc8041e1542a27939add8b8c169798b015c7b11b42",
  "input_query.txt": "2eccafe10510f1ba9816422860d4799df07ec549775235b253046fb770825671"
}
```

Changed files are additions only: two Stage A scripts, `tests/test_v22_stage_a.py`, and the Stage A report/aggregate evidence under `docs/research/sma_rsi_htf_v22/`. The original 364-trade detail and raw input datasets are not copied into Git.

The user-facing output directory also contains the attributed trade CSV, original replay rerun output, diagnostics, and full test log. Paths in commands can be changed; permitted input dates and baseline identity checks deliberately cannot be widened by a CLI flag.
