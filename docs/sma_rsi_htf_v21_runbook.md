# SMA/RSI HTF V2.1 — reproduction runbook

## Purpose

This track reproduces the frozen MT5 V2.1 SMA/RSI + M15 trend-filter strategy inside the CSV research lab.

It is a **frozen reproduction and engineering-maintenance track**.
No strategy optimization is authorized. Do not change trading parameters or
signal rules, fit returns, or special-case individual historical trades.
The former `research/sma-rsi-htf-v1` work is merged.

Frozen strategy registration:

`sma_rsi_htf_v21`

## Frozen MT5 logic

M5 indicators:

- SMA 9
- SMA 21
- RSI 14
- ATR 14

BUY:

- SMA9 crosses above SMA21 on the completed M5 candle.
- RSI14 < 70 on that completed M5 candle.
- Latest completed M15 close > latest completed M15 SMA200.

SELL:

- SMA9 crosses below SMA21 on the completed M5 candle.
- RSI14 > 30 on that completed M5 candle.
- Latest completed M15 close < latest completed M15 SMA200.

Exit geometry:

- SL = 1.5 × M5 ATR14.
- TP = 1.5 × M5 ATR14 (1R).

MT5 risk:

- 0.20% of current equity per trade.
- One position at a time.
- Symbol: XAUUSD.r; maximum spread: 80 points.
- MT5 tester entry delay: 250 ms, using the validated last quote at or before
  request + 250 ms in the standalone replay.
- Physical SL/TP; long protective exits use bid, short exits use ask.
- No grid, martingale, recovery or averaging.

No parameter in the Python reproduction is tunable.

## Signal timing convention

The CSV lab consumes M1 bars and the generic backtester enters one base row after a signal.

To reproduce the MT5 EA's closed-bar timing:

1. Aggregate nonempty M5/M15 time buckets from sorted M1 bars. Individual missing
   no-tick minutes do not discard the bucket; do not fill them with synthetic prices.
2. Confirm the SMA crossover/RSI on the just-completed M5 bar.
3. Use only an M15 bar whose **close timestamp is <= the new M5 open timestamp**.
4. Write the signal onto the final M1 row of the completed M5 bar.
5. The next M1 row must belong to the immediately following M5 bucket. It can be
   later than the nominal bucket open if early minutes had no ticks.
6. Never carry a signal across an entirely empty M5 bucket or market-closure gap.

The HTF selection excludes candles closing after the signal. The next-row
timestamp establishes whether the next bucket exists; its OHLC is not used to
confirm the preceding signal. Input ordering, coverage and warmup still require
validation; sparse bars cannot distinguish missing provider data from no ticks.

## Frozen development reference

MT5 real-tick development test:

- Symbol: XAUUSD.r
- Timeframe: M5
- Period: 2026-06-01 to 2026-08-31 exclusive
- Initial balance: USD 5,000
- Leverage: 1:100
- Risk: 0.20% current equity per trade
- Execution delay: 250 ms
- Model: Every tick based on real ticks
- Trades: 364
- Net: +USD 80.25
- PF: 1.06
- Expected payoff: +USD 0.22
- Win rate: 50.82%
- Max equity DD: 2.99%
- Classification: INVESTIGATE FURTHER

The checked-in boundary is **August 31 exclusive**. Do not silently extend it
to September 1 to interpret “June–August” as three full calendar months.

Supplied validated development evidence:

- 450 raw signals; 364 executed trades.
- 1 spread skip; 21 position-open skips; 64 below-minimum-volume skips.
- 0 sequence, volume, entry-fill, exit-reason or exit-time mismatches.
- Commission and swap accounting validated.
- MT5 final balance: $5,080.25; standalone: $5,080.26.
- One isolated $0.01 exit-price residual remains unresolved.

This evidence is not a claim that a source-only checkout has rerun the real-tick
test. Exact full parity must still fail when an exit-price mismatch remains;
do not weaken assertions to relabel structural parity as exact parity.

The Python reproduction must not be tuned to make these P/L metrics match.

## Parity stages

### Stage 1 — signal parity

Compare Python and MT5 entry **timestamps and directions** over the known development period.

Required outputs:

- Python signal timestamp
- direction
- M5 SMA9
- M5 SMA21
- M5 RSI14
- M5 ATR14
- selected M15 close timestamp
- selected M15 close
- selected M15 SMA200

Parity interpretation:

- Every executed MT5 entry must have a matching Python signal at the corresponding new-M5 entry time and direction.
- Extra Python raw signals are not automatically mismatches because the frozen MT5 EA returns early while one of its positions is already open.
- Classify an extra Python signal as a true mismatch only when MT5 was flat and otherwise eligible to evaluate/enter at that time.
- Also account for spread rejection or other documented execution guards before calling a signal mismatch.

Target:

- Explain every genuine mismatch.
- Do not change strategy parameters to reduce mismatches.

### Stage 2 — execution-geometry parity

The generic lab backtester cannot yet be treated as exact V2.1 P/L parity for two reasons:

1. The generic engine reads `MarketBars.atr` for stop distance, while V2.1 requires the aligned **M5 ATR14** at the signal.
2. The generic `RiskModel` uses fixed USD risk, while MT5 V2.1 risks **0.20% of current equity**, so lot size compounds with realized equity.

Do not publish CSV-lab net profit as a reproduction of V2.1 until both differences are handled in a parity-safe adapter.

The strategy module already exposes `execution_atr` aligned to the M1 signal row for the first issue.

### Engineering maintenance

Use `scripts/replay_sma_rsi_v21_standalone.py` for the frozen real-tick execution
model. Its sizing uses flat-state balance * 0.20%, floors to 0.01-lot steps,
rejects below 0.01 lot, and caps at 100 lots. Frozen broker assumptions include
point/tick size 0.01, two digits, contract size 100, entry commission -$6/lot,
and points-mode swap -57.849 long / +36.963 short with Wednesday triple rollover.
Do not change these assumptions without reviewing historical behavior impact.

The generic campaign engine is not a V2.1 execution adapter. Its fixed-dollar
risk, base-bar ATR, configurable exits and integrity mask can differ from the
standalone route. Neither this runbook nor the legacy config `research_rule`
authorizes tuning after parity: the strategy remains frozen.

Before any real-data replay, use a separately identified development-only input
directory and approved feature history. The current tools do not enforce the
reserved intervals, and replay ends at the last tick in its directory. Do not
point them at a mixed development/prospective/OOS dataset. No later returns may
inform engineering choices. Do not run the price-rounding or delay diagnostics
to select changes from profitability or improve the isolated historical residual.

The replay CSV is replaced atomically and contains a header even for zero trades.
The CSV and JSON summary are not a transaction; a failure between writes can
still leave mismatched artifacts. Use separate output locations for separate runs.

Run the complete synthetic test suite with `python -m pytest` after installing
the declared development dependencies. All V2.1 replay tick/report fixtures are
generated in temporary directories. Keep real tick data out of Git, including
when using a custom output path. Never force-add broker exports.

See [the engineering audit](sma_rsi_htf_v21_engineering_audit.md) for prioritized
risks and changes that require review before modifying historical behavior.

## Reserved periods

Prospective validation:

- 2026-10-01 to 2027-01-01 exclusive

Final OOS:

- 2027-01-01 to 2027-04-01 exclusive
- Do not inspect returns or use this period for development.

These reservations are specific to this SMA/RSI track and do not replace other holdouts already present in the repository.

## Frozen hashes

MT5 source:

`7650ecabe60c1a5fbe9735cf9fb9977877865ae2694eb9f2c329c1a2ae6a4313`

MT5 EX5:

`cf57eec574a1658dba36e3dd01a9fe6e188f53c504d11f7723aabfeb879e7548`

## Current implementation files

- `xau_lab/strategies/sma_rsi_htf.py`
- `tests/strategies/test_sma_rsi_htf.py`
- `config/sma_rsi_htf_v21_frozen.json`
- `scripts/export_sma_rsi_htf_parity.py`
- `scripts/replay_sma_rsi_v21_standalone.py`
- `scripts/compare_sma_rsi_lifecycle_mt5.py`
- `tests/strategies/test_v21_replay.py`
- `tests/strategies/test_v21_causality.py`

The EA source and EX5 are not tracked here. Their hashes alone cannot establish
restart safety, netting isolation or physical-order behavior.

## Promotion rule

Do not call a CSV-lab variant a replacement for V2.1 until it:

1. passes internal tests,
2. has signal parity audited against MT5,
3. survives rolling/expanding research folds,
4. survives realistic cost stress,
5. is exported back to MT5,
6. survives MT5 real-tick confirmation,
7. then faces the preregistered validation protocol.
