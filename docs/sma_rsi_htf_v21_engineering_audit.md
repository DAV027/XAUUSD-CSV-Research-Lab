# Frozen V2.1 engineering audit

Audit baseline: `3d63a68218a921b50c5e1406eab355c58c026a88` (main).
Audit date: 2026-09-23. Scope: the registered V2.1 strategy, its shared indicator/data/runner dependencies, all V2.1 export/comparison/diagnostic scripts, tests, configuration and CI. Other strategy families are covered by the complete regression suite, not individually certified by this audit.

No real market dataset, MT5 report, or tester log is tracked in this checkout. No returns from October 2026 onward were opened or calculated. All new execution fixtures are synthetic and dated June 2026. The user's 450 signals / 364 trades / skip counts 1, 21, 64 / balances $5,080.25 and $5,080.26 are supplied validation evidence, not independently rerun results. The isolated one-cent residual remains unresolved; no historical-trade exception is proposed.

## Prioritized findings, before implementation

### CRITICAL

**C1 — Reserved data and replay horizon are not enforced.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:main`, `TickStore.__init__`, `last_available_tick`; `scripts/export_sma_rsi_htf_parity.py:main`.
- Problem: replay indexes every tick partition and uses the last available tick as its test end. Signal export loads and calculates over the complete feature file before applying optional output date filters. Neither path enforces the V2.1 reservation.
- Why it matters: appending later ticks can extend an existing development trade and change end-of-test price, swap and equity. An output filter does not prevent reserved data from being read. This is a research-isolation defect, not evidence of future prices in the closed-bar signal formulas.
- Historical results: **yes**, especially end-of-test positions. **STOP: no execution-boundary change made.**
- Recommended fix: approve an explicit, immutable development input manifest and replay interval, validate permitted partitions before reading them, and use a dedicated development-only feature artifact including agreed warmup. Do not silently truncate or extend the existing run.
- Required test: use inaccessible sentinel partitions beyond the allowed interval; assert they are never opened and appending them cannot change the replay. Verify the exact approved development ledger before adopting a new horizon.

### HIGH

**H1 — MT5 implementation is absent, blocking live lifecycle certification.**
- Location: `config/sma_rsi_htf_v21_frozen.json:mt5_source_sha256/mt5_ex5_sha256`; repository tree has no `.mq5`, `.mqh` or `.ex5`.
- Problem: hashes are recorded, but EA code cannot be inspected for OnInit/OnTick/OnTradeTransaction, physical protection, failed/partial orders, reinitialization, or ownership of a netting position. Python `main` starts balance and position state afresh and is a batch simulator, not a restartable EA.
- Why it matters: there is no evidence here that restart avoids duplicate entries or that foreign/manual symbol positions cannot be modified. Netting has a single position per symbol, potentially composed of multiple deals ([MetaQuotes](https://www.mql5.com/en/docs/trading/positionselect)).
- Historical results: **possibly**; not a confirmed EA bug. **STOP: no inferred EA fix.**
- Recommended fix: supply the source matching the recorded hash, tester build and settings; review symbol/account isolation and persisted/reconstructed last-bar and position state.
- Required test: restart while flat/open, restart within the same M5 bar, failed/partial fills, disconnection, foreign same-symbol deals, and physical SL/TP surviving terminal shutdown, in an isolated demo/test harness.

**H2 — Generic research execution is not frozen V2.1 execution.**
- Location: `xau_lab/runner/single.py:run_experiment`, `xau_lab/backtest/models.py:RiskModel`, `xau_lab/backtest/reference.py:_open_position`, `xau_lab/strategies/base.py:StrategyDefinition.generate`.
- Problem: generic execution uses fixed-dollar risk, base-bar ATR and configurable exit/cost geometry. Registration can also apply the shared `entry_allowed` mask, whereas the V2.1 exporter calls `build_sma_rsi_htf_state` directly. An empty parameter domain alone does not freeze the surrounding execution pipeline.
- Why it matters: running the named strategy through the generic campaign can silently produce a different trade sequence, sizing and P/L.
- Historical results: **yes**. **STOP: runner, masks and risk model unchanged.**
- Recommended fix: explicitly separate signal-only registration from the frozen replay, or reject V2.1 in generic execution after approval. Do not retrofit generic risk and call it parity.
- Required test: demonstrate the two routes with a masked signal and distinct M1/M5 ATR; an approved route guard must fail before simulation.

**H3 — Tick and signal inputs are trusted without a data contract.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:TickStore.__init__/_frame/window/main/_expected_volume`.
- Problem: empty partitions cause `int(None)`; missing ticks are treated as NO_TICK without coverage proof; no explicit finite/positive bid/ask, ask >= bid, valid BUY/SELL, finite positive ATR, unique signal bucket, partition disjointness or symbol validation. Any non-BUY direction takes SELL branches. Path order is assumed chronological for final-tick and first-exit selection.
- Why it matters: corrupt or overlapping input can create fills, hide exits, alter ordering or fail opaquely. Silent sorting is not proof that duplicate tick events are legitimate.
- Historical results: **yes** on malformed/incomplete data. **STOP: no filtering, deduplication or defaulting added.**
- Recommended fix: fail closed with row/partition diagnostics; distinguish a verified market closure from unavailable data. Preserve legitimate same-millisecond events and provider order.
- Required test: empty/missing schema, null/NaN/infinity/zero/crossed quotes, unknown direction, duplicate signals, overlapping partitions, and missing-day manifests; verify valid-input results remain exact.

**H4 — Delay and lifecycle ordering have unresolved boundary cases.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:main/_first_exit_tick`.
- Problem: fill quote is the last tick at or before request+250ms, but exits start after that quote's timestamp, not a separately represented activation event. All later events at the same millisecond are excluded by `after_ms + 1`. `request_ms < open_until_ms` allows a new request at the exact exit millisecond. Stable sorting retains ties but no event cursor survives selection.
- Why it matters: sparse ticks, protection touched within the delay, or tied entry/exit events can be processed in the wrong order. Existing development parity does not establish all these cases.
- Historical results: **yes** if semantics change. **STOP: preserve the validated delay and exit rules.**
- Recommended fix: obtain synthetic MT5 tester traces for boundary cases; model quote timestamp, execution activation and event sequence explicitly only after agreement.
- Required test: quote at +249/+250/+251ms, no tick during delay, several ticks at one millisecond, exit/re-entry tie, and SL/TP crossed before/at activation. No price fitting or per-trade exceptions.

**H5 — Broker specification is implicit and stop-distance checks are absent.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:constants/_round_price/_floor_volume/_expected_volume/main`; `xau_lab/backtest/models.py:SymbolSpec`; `scripts/export_mt5_ticks.py:main` metadata.
- Problem: replay hardcodes digits, point, contract, lot limits and costs without verifying dataset metadata. Decimal digits are not generally tick-size normalization. Replay has no stops-level, freeze-level, order rejection or margin model. Tick metadata does not include all these properties.
- Why it matters: the math is specific to the frozen USD XAUUSD.r contract. Balance-based 0.20% sizing represents equity only when flat with no other account activity; delayed fill can change realized risk. Do not substitute a universal broker model.
- Historical results: **yes** if quantization, order eligibility or sizing changes. **STOP: constants and calculations unchanged.**
- Recommended fix: record and validate the exact historical symbol/account specification; investigate stop rejection with the matching EA. Reject incompatible specifications rather than adapting silently.
- Required test: min/step/max lot boundaries, cent rounding, tick_size != point, stops-distance boundary, insufficient margin, flat equity and foreign account activity; establish MT5 agreement before execution changes.

**H6 — Timestamp semantics differ across tools.**
- Location: `scripts/compare_sma_rsi_parity_mt5.py:_parse_python_time`, `scripts/compare_sma_rsi_trade_construction_mt5.py:_parse_python_time`, `scripts/compare_sma_rsi_volume_mt5.py:_parse_python_time`; compare with `scripts/replay_sma_rsi_v21_standalone.py:_parse_python_time`; `xau_lab/runner/campaign.py:_time_epoch`.
- Problem: comparators remove an explicit offset without converting to UTC; standalone replay converts it. Existing `time_epoch` bypasses the loader's requested semantics. `_utc_ms` helpers replace tzinfo rather than normalize aware inputs. Tester wall time is assumed to match raw UTC for this particular run.
- Why it matters: equivalent instants expressed with +03:00 can disagree; a semantic flag does not establish the origin of an epoch column. MT5 Python tick APIs require UTC datetimes ([MetaQuotes](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)).
- Historical results: **possibly**, or comparison classifications. **STOP: no time conversion changes.**
- Recommended fix: attach explicit clock provenance to artifacts and consolidate conversions after preserving the known tester convention.
- Required test: equivalent offset/Z times, naive policy, DST gaps/folds, broker-date midnight, and epoch/provenance conflicts.

**H7 — Parsers can silently lose or mispair evidence.**
- Location: `scripts/compare_sma_rsi_lifecycle_mt5.py:_read_log_lines/_extract_v21_run/_extract_report_lifecycle/_to_float`; duplicated helpers in parity, construction and volume comparators.
- Problem: UTF-16 is attempted without BOM detection and some even-length UTF-8 input decodes as unrelated characters. XLSX assumes sheet1; invalid deal timestamps are skipped; absent money fields become zero; nonmatching symbols are ignored; orphan exits and partial exit volume are not validated by the lifecycle extractor; `if balance` ignores a valid zero. Latest matching run is selected without a frozen input identity.
- Why it matters: malformed/truncated reports can look like a shorter ledger, and selecting the latest run can select the wrong evidence.
- Historical results: replay decisions **no**, reported validation/accounting **yes**. **STOP: parser interpretation unchanged pending fixtures.**
- Recommended fix: BOM-aware decoding, explicit run and worksheet identity, strict required fields, complete lifecycle/volume reconciliation, clear unsupported-format errors.
- Required test: real-format synthetic XLSX with shared/inline strings, UTF-8/UTF-16 logs, zero balance, missing numeric fields, orphan/partial closes, multiple runs, malformed timestamps and truncated logs.

**H8 — Parity reporting has false-confidence paths.**
- Location: `scripts/compare_sma_rsi_parity_mt5.py:main/_extract_mt5`; `scripts/replay_sma_rsi_v21_standalone.py:main` summary/output.
- Problem: signal parity uses sets, so duplicate keys lose multiplicity, and empty inputs can report parity success. `_extract_mt5` invents an implicit exit at the next entry. Replay compares exit times only to seconds. CLI success exit codes do not depend on parity booleans. A one-cent exit-price discrepancy must still fail the existing strict full-parity boolean.
- Why it matters: successful process completion is not proof of a complete, ordered ledger match.
- Historical results: trade computation **no**, validation outcome **yes**. Existing assertions/tolerances must not be weakened.
- Recommended fix: add an explicit strict validation command with nonempty expected counts, ordered identity, complete unmatched tails and a failing exit status. Keep structural parity distinct from exact full parity.
- Required test: empty/duplicate/missing/extra trades, mismatches in each field, exit-time precision, and one-cent residual failing exact parity.

### MEDIUM

**M1 — Aggregation assumes globally valid M1 input.**
- Location: `xau_lab/strategies/sma_rsi_htf.py:_aggregate_complete_minutes/build_sma_rsi_htf_state`; `xau_lab/strategies/base.py:StrategyContext.__post_init__`.
- Problem: bucket-local checks do not enforce global increasing unique timestamps or minute alignment; malformed buckets may be silently omitted. A partial initial bucket is accepted without warmup provenance. Missing provider data and legitimate no-tick minutes are indistinguishable.
- Why it matters: `searchsorted` requires sorted HTF times, and warmup/omitted bars change indicators. For valid sorted inputs, HTF selection explicitly uses close <= signal close; no future-close lookup was found. Final signal placement requires the next bucket's existence, so prefix tests must supply that next row.
- Historical results: **yes** if bar acceptance changes. **STOP: preserve sparse-bar behavior.**
- Recommended fix: input validation and completeness provenance, not forced 5/15-row candles.
- Required test: future-price perturbations, prefix invariance with next-row timestamp present, M15 boundary, sparse minutes, out-of-order buckets and duplicate timestamps.

**M2 — Indicator formulas lack a versioned MT5 buffer reference.**
- Location: `xau_lab/strategies/sma_rsi_htf.py:_mt5_iatr`; `xau_lab/data/features.py:_rolling_sum/_rolling_mean/_rsi`; `tests/strategies/test_sma_rsi_htf.py`.
- Problem: ATR/RSI warmup is implemented explicitly, but MT5 build, warmup bars and exported buffers are unavailable. Cumulative-sum subtraction can differ in floating-point arithmetic from recursive MT5 buffers. Existing ATR tests check mathematical values, not bitwise MT5 equivalence.
- Why it matters: a tiny boundary difference can affect crossover comparisons, rounded protection or lot floors. This is not a finding that the existing development parity is wrong.
- Historical results: **possibly**. **STOP: no indicator replacement or epsilon adjustment.**
- Recommended fix: freeze warmup and buffer provenance; test against independently exported synthetic MT5 buffers.
- Required test: flat/rising/falling RSI, ATR seed and gaps, long-history numerical behavior, exact crossover/equality thresholds and future perturbation.

**M3 — Large tick processing repeatedly scans and materializes data.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:TickStore.window/_frame/_first_exit_tick`; `scripts/compare_sma_rsi_tick_exits_mt5.py:_candidate_partitions/_load_window`; entry/exit delay diagnostic `_load_window` functions.
- Problem: replay bounds scans read each partition, every small window filters full cached days and concatenates/sorts; exit search computes all hits in a window. Replay cache is bounded by six partitions, not bytes. Diagnostic caches are unbounded and exit diagnostics repeatedly scan partition bounds.
- Why it matters: 20M rows need at least ~480MB for only three 64-bit columns, before sort/filter copies and other fields. No 20M-row performance claim is verified here.
- Historical results: **should not** change for an equivalent refactor, but tick tie ordering is sensitive.
- Recommended fix: reusable validated index, bounded cache by bytes, binary-search slices/row groups, early exit iteration. Preserve event order and compare complete synthetic outputs before replacing code.
- Required test: stable same-millisecond ordering, eviction/reload equivalence, partition edges, and a synthetic 20M-row benchmark recording peak RSS and runtime.

**M4 — No executable standalone replay regression coverage.**
- Location: `tests/strategies/test_sma_rsi_htf.py:test_standalone_v21_replay_script_is_syntax_valid` and related syntax-only tests.
- Problem: compile checks cannot detect wrong delay selection, money/volume boundaries, lifecycle skips, swap rollover, stale outputs or comparator mistakes.
- Why it matters: changes can pass CI without running the V2.1 engine.
- Historical results: tests themselves **no**.
- Recommended fix: synthetic behavioral tests and an end-to-end replay fixture; retain all existing tests and tolerances.
- Required test: frozen constants, BUY/SELL fills and exits, spread/position/volume skips, compounding, Wednesday swap, midnight/weekend boundaries, report independence, repeatability and exact-parity failure on one cent.

**M5 — Output files can be stale or partially written.**
- Location: `scripts/replay_sma_rsi_v21_standalone.py:main` final CSV/JSON writes; analogous comparator output blocks.
- Problem: a successful zero-trade replay never replaces an older CSV. Direct writes can leave partial output when interrupted. CSV and summary are not a transactional pair.
- Why it matters: a new summary can be reviewed beside an old ledger.
- Historical results: **no**; this affects artifact integrity only.
- Recommended fix: always write a schema-bearing CSV, even when empty; stage files before replacement. State explicitly that two file replacements are not a transaction.
- Required test: stale CSV on empty run, byte-identical populated output, interrupted serialization retaining old destinations, and repeated deterministic writes.

**M6 — Raw datasets are not excluded from Git.**
- Location: `.gitignore`; `scripts/export_mt5_ticks.py:main` default `datasets/.../data/ticks`.
- Problem: only caches/worktrees are ignored. A broad add can include real ticks or feature/market artifacts.
- Why it matters: repository bloat and accidental publication of broker data.
- Historical results: **no**.
- Recommended fix: ignore canonical data/dataset/results roots and tick partition paths, including staged temporary Parquet files. Keep synthetic fixtures generated in temporary directories.
- Required test: `git check-ignore` for default/custom tick paths and verify source/config/test files remain trackable; inspect tracked files before every commit.

**M7 — Reproducibility metadata and restart export metadata are incomplete.**
- Location: `pyproject.toml`; `scripts/replay_sma_rsi_v21_standalone.py:main` summary; `scripts/export_mt5_ticks.py:main` existing-partition branch.
- Problem: dependencies are ranged, no exact environment or input hashes accompany replay; reused tick partitions update row totals but not first/last times, and no existing partition spec is checked. Export metadata describes an invocation, not necessarily all files under its root. Windows timezone support is not declared.
- Why it matters: identical-looking runs can use different input histories, library builds, symbol metadata or timezone databases.
- Historical results: **possibly** through input/environment drift; metadata-only correction **no**.
- Recommended fix: archive tested environment and an approved content-hashed input manifest; define resume metadata semantics and verify existing partitions. Do not lock an unverified replacement numerical environment as the historical reference.
- Required test: fresh vs resumed export metadata, mixed symbol/timezone refusal, input mutation detection, and two independent identical synthetic replays.

### LOW

**L1 — Documentation contradicts the merged sparse-bar implementation and freeze.**
- Location: `docs/sma_rsi_htf_v21_runbook.md:Signal timing convention/Stage 3/Current implementation files`; `config/sma_rsi_htf_v21_frozen.json:research_rule/development_period`.
- Problem: runbook requires complete contiguous candles although implementation accepts nonempty sparse buckets; stage 3 suggests optimization; replay is absent from file list. The exact checked-in end is August 31 exclusive, which is not the same as including all of August.
- Why it matters: an operator can reconstruct the wrong sample or think parity authorizes tuning.
- Historical results: documentation-only **no**; changing the date or config **yes**.
- Recommended fix: correct runbook and state the engineering freeze and validated evidence; preserve frozen config and explicitly flag the period wording discrepancy for confirmation.
- Required test: documentation contract references sparse buckets, replay, unchanged exclusive end, reserved periods and no optimization.

**L2 — Duplicated helpers and environment-sensitive test imports.**
- Location: `_xlsx_rows/_read_log_lines/_parse_python_time/_load_window` across V2.1 scripts; missing `tests/__init__.py`; `.github/workflows/tests.yml`.
- Problem: copies have already diverged; bare sibling script imports depend on invocation mode. Test imports `tests.strategies.*` collide with an installed `tests` package (eight collection errors observed in host Python). CI only runs pushes for one legacy branch, or PRs targeting main.
- Why it matters: tests can fail to collect or script behavior can differ across environments; changes on other branches lack push CI.
- Historical results: import/package/CI hygiene **no**; shared parser refactoring can alter diagnostics.
- Recommended fix: make test package ownership explicit; test direct script invocation; consolidate helpers later behind established fixtures; broaden CI in a separate reviewed change.
- Required test: competing `tests` package on PYTHONPATH, full suite in a clean environment, and CLI regression fixtures.

## Coverage of requested areas

| Requested areas | Findings / evidence |
|---|---|
| 1 correctness; 2 lookahead/repainting; 3 timezones; 4 aggregation | C1, H6, M1; closed-bar and sparse-bucket paths inspected |
| 5 indicators; 6 risk; 7 volume; 8 ticks/prices; 9 stops; 10 spread | H2, H3, H5, M2; frozen spread comparison and lot flooring retained |
| 11 delay; 12 real-tick protection; 13 commission/swap; 14 lifecycle | H4, H5, H7, M4; bid long / ask short exit sides inspected |
| 15 restart; 16 netting isolation | H1: unavailable EA source, not certified |
| 17 malformed/missing ticks; 18 large datasets | H3, M3; no real tick benchmark performed |
| 19 duplication; 20 weak tests; 21 parsers; 22 errors | H7, H8, M4, M5, L2 |
| 23 reproducibility; 24 determinism; 25 documentation | C1, M2, M7, L1, L2 |

## Safe implementation boundary

Only test/package and Git hygiene, behavioral regression tests, replay artifact writing, and documentation corrections are authorized in this pass. No signals, indicator math, risk/volume/price calculations, costs, delay selection, event order, period definitions, parser interpretation or parity tolerances will change. Any fix marked STOP above requires an explicit follow-up decision before altering behavior. No merge to main.
