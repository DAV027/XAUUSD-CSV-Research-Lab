# SMA/RSI HTF V2.1 — reproduction runbook

## Purpose

This track reproduces the frozen MT5 V2.1 SMA/RSI + M15 trend-filter strategy inside the CSV research lab.

It is a **reproduction and parity exercise first**, not an optimization campaign.

Branch:

`research/sma-rsi-htf-v1`

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

No parameter in the Python reproduction is tunable.

## Signal timing convention

The CSV lab consumes M1 bars and the generic backtester enters one base row after a signal.

To reproduce the MT5 EA's closed-bar timing:

1. Aggregate only complete contiguous M1 bars into M5 and M15 bars.
2. Confirm the SMA crossover/RSI on the just-completed M5 bar.
3. Use only an M15 bar whose **close timestamp is <= the new M5 open timestamp**.
4. Write the signal onto the final M1 row of the completed M5 bar.
5. The generic engine's next-row entry then occurs at the next M1 open, which is the new M5 open.
6. Never carry a signal across a data gap.

This avoids look-ahead and avoids firing Friday's final signal on Monday after a market closure.

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

### Stage 3 — research

Only after signal and execution parity are understood may this track run controlled research experiments.

Change one major concept at a time.

Examples:

- RSI momentum confirmation
- SMA slope
- SMA separation
- volatility regime
- ADX/trend-strength filter
- session filter
- asymmetric BUY/SELL treatment
- exit geometry

Do not conduct an unrestricted parameter sweep.

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

## Promotion rule

Do not call a CSV-lab variant a replacement for V2.1 until it:

1. passes internal tests,
2. has signal parity audited against MT5,
3. survives rolling/expanding research folds,
4. survives realistic cost stress,
5. is exported back to MT5,
6. survives MT5 real-tick confirmation,
7. then faces the preregistered validation protocol.
