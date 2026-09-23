# V2.2 Stage A — frozen V2.1 trade attribution

Date: 2026-09-23. Status: **Stage A complete for review; no candidate implemented or selected.**
Branch: `research/sma-rsi-htf-v22`. Frozen base: `3d63a68218a921b50c5e1406eab355c58c026a88`.

## Decision for review

V2.1's modest aggregate profit is concentrated and varies substantially by month and direction.
This analysis identifies associations worth discussing, but cannot establish why trades won or
whether changing an entry condition would improve a newly replayed strategy. No filters, exits,
risk settings, stop levels, signal rules, or historical trade exceptions have been introduced.
Stages B–H remain deferred at the owner's request: **complete Stage A for review first**.

## Data boundary and reconciliation

The permitted development interval is **2026-06-01 00:00 through 2026-08-31 00:00 exclusive**.
Earlier M1 history is used only for indicator warmup. The source M1 clock is EET; signal/report
comparison clocks are UTC. The lazy M1 scan applies the cutoff to the source time string before
collecting OHLC values, then converts permitted timestamps through EET to UTC. It collects
1,158,680 rows, from 2017-01-03 02:00 to 2026-08-29 02:49 broker time. No market observations at
or after the cutoff were collected or analyzed. Prospective and final OOS windows remain untouched.
Cutoff tests use synthetic sentinel values, not actual future observations.

All 364 report trades reconcile to unique exported signal bucket/direction pairs and saved replay
rows. Recomputed frozen ATR and SMA21 match the signal export within 1e-7; M15 close time is no
later than M5 close time, which is no later than entry. The report supplies actual net outcomes;
the saved replay supplies request-time spread. Hashes are in `STAGE_A_SUMMARY.json`.

The original, unchanged real-tick replay was also rerun independently over the permitted inputs:

- 450 raw signals; 364 executions; 1 spread skip; 21 position skips; 64 volume skips.
- Zero sequence, volume, entry-fill, exit-reason, or exit-second mismatches.
- Commission and swap checks pass under the existing assertions.
- MT5 final balance $5,080.25; replay $5,080.26.
- The existing one-cent exit-price residual remains. The original strict replay parity flag
  remains **false**; this report does not relabel it a full parity pass or relax any tolerance.

Tick metadata accounts for 21,607,135 rows across 76 permitted partitions. Every partition's
row-group maximum timestamp is before the cutoff. The diagnostics script reads metadata only;
the separate frozen replay reads the permitted real ticks. No raw ticks are committed.

## Baseline outcomes

All dollar figures below are net of reported commission and swap. Subset figures are attribution,
not counterfactual strategy results. Removing trades changes later position availability and
equity-based sizing, so these rows cannot be used as a candidate backtest.

| Scope | Trades | Winners | Net $ | Expectancy $ | PF | Win rate | Average win $ | Average loss $ | Longest loss run | Net without top 5 winners $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 364 | 185 | +80.25 | 0.220 | 1.059 | 50.82% | 7.76 | -7.57 | 8 | -15.88 |
| BUY | 182 | 91 | +19.81 | 0.109 | 1.028 | 50.00% | 8.11 | -7.89 | 10 | -75.94 |
| SELL | 182 | 94 | +60.44 | 0.332 | 1.095 | 51.65% | 7.42 | -7.24 | 5 | +10.65 |
| 2026-06 | 116 | 57 | +49.32 | 0.425 | 1.114 | 49.14% | 8.48 | -7.36 | 6 | -45.36 |
| 2026-07 | 134 | 64 | -80.28 | -0.599 | 0.860 | 47.76% | 7.68 | -8.16 | 8 | -131.09 |
| 2026-08 | 114 | 64 | +111.21 | 0.976 | 1.318 | 56.14% | 7.21 | -7.00 | 5 | +62.78 |

Average frequency is 121.3 trades/month, with 57, 64, and 64 profitable trades in June, July,
and August. August excludes August 31. Longest runs in side-only subsets count consecutive
trades of that side, not consecutive account trades. Closed-trade balance drawdown is 2.854%;
the supplied MT5 maximum equity drawdown is approximately 2.99%. Intratrade equity drawdown
was not reconstructed by Stage A. Subset drawdown fields use a fresh hypothetical $5,000 balance.

Price profit totals $120.66, commission −$28.56 and swap −$11.85, leaving $80.25.
Commission and swap consume $40.41, or 33.5% of the price-profit surplus. Price profit already
reflects actual bid/ask execution; spread must not be deducted again as if it were absent.

The five largest net winners contribute $96.13. Removing them leaves **−$15.88**. The single
largest winner contributes $54.24; removing only it leaves $26.01. These are concentration
diagnostics, not proposed trade exclusions. BUY is especially fragile after top-winner removal.

## Feature definitions and timing

Features use the last completed signal M5 bar and the frozen latest-completed M15 mapping.
No outcome, future high/low, or holding duration enters feature calculations. Holding duration
is reported separately as an outcome diagnostic. Winner/loser labels are retrospective outcomes.

| Feature | Definition |
|---|---|
| Trend efficiency | Absolute 20-bar close displacement divided by the sum of 20 absolute close changes |
| SMA separation | Direction-signed (SMA9−SMA21) divided by M5 ATR14 |
| SMA slope | Direction-signed three-bar SMA21 change divided by ATR14 |
| HTF distance | Direction-signed latest closed M15 close minus SMA200, divided by M5 ATR14 |
| RSI / RSI momentum | Frozen RSI14 level / direction-signed three-bar change |
| ATR percentile | Current completed-bar ATR rank against the previous 288 completed M5 ATR values; reference excludes current bar |
| Spread | Saved replay request-time spread in points, also divided by ATR after point-to-price conversion |
| Clock / weekday | UTC entry hour, EET broker entry hour, UTC weekday |
| Session | London 08:00–17:00 local and New York 08:00–17:00 local; exclusive overlap/London/New York/other categories |
| Recent extremes | Close distance from preceding 20-bar high/low, excluding signal bar, divided by ATR |
| Candle structure | Absolute and direction-signed body/range, upper and lower wick/range |
| Breakout | Signal close beyond preceding 20-bar extreme in trade direction |
| Pullback tag | Signal candle touches SMA21 and closes on the trade-direction side |
| Volatility expansion | Mean range of last 5 completed bars divided by mean range of the preceding 20 |
| M5/M15 alignment | Direction agrees with SMA9−SMA21 and M15 close−SMA200; true for all baseline entries |
| Slope/HTF alignment | Separate tag: direction-signed SMA21 slope is positive; not the frozen alignment condition |
| Previous-bar structure | Previous inside bar, signal inside bar, and direction-signed previous body/ATR |

Numeric low/middle/high groups use broad June-only tertiles, fixed when describing July and
August. Thresholds are published in the summary JSON. June thresholds, all feature inspection,
and July/August observations are **development research**, not independent validation. These
are descriptive partitions, not optimized thresholds or pre-registered confirmatory tests.
Ties can make groups unequal or empty. RSI levels are not direction-normalized; inspect side
decomposition before interpreting pooled values. Session definitions are analytical conventions.

## Winner/loser distributions

| Entry feature | Winner median | Loser median | Winner mean | Loser mean |
|---|---:|---:|---:|---:|
| trend_efficiency | 0.10778 | 0.09656 | 0.12131 | 0.10991 |
| sma_separation_atr | 0.08052 | 0.08511 | 0.10302 | 0.10432 |
| sma_slope_aligned_atr | -0.00920 | 0.00541 | -0.01783 | -0.00114 |
| htf_distance_atr | 7.53576 | 7.03573 | 9.92032 | 8.69005 |
| rsi | 49.50456 | 49.70155 | 49.29432 | 49.89119 |
| rsi_momentum_aligned | 3.94713 | 2.97016 | 4.22000 | 3.60400 |
| atr | 4.16143 | 3.97500 | 4.09016 | 4.08485 |
| atr_percentile | 0.38194 | 0.38542 | 0.39298 | 0.39364 |
| spread_points | 29.00000 | 29.00000 | 33.81622 | 32.89944 |
| spread_atr | 0.07596 | 0.07519 | 0.09539 | 0.08989 |
| distance_high_atr | 2.12487 | 1.91620 | 2.10707 | 1.95619 |
| distance_low_atr | 1.95939 | 2.07749 | 2.04391 | 2.08655 |
| candle_body_range | 0.50159 | 0.48265 | 0.49282 | 0.47022 |
| candle_body_aligned | 0.30709 | 0.24324 | 0.17592 | 0.18509 |
| upper_wick_fraction | 0.23438 | 0.19672 | 0.27537 | 0.22990 |
| lower_wick_fraction | 0.18400 | 0.25466 | 0.23180 | 0.29988 |
| volatility_expansion | 0.89461 | 0.94940 | 0.96660 | 1.02580 |
| previous_body_aligned | 0.20000 | 0.11405 | 0.23629 | 0.16973 |

Each distribution has 185 winners and 179 losers. Differences are associations with overlapping
features and confounding by month, side, volatility, costs, and realized sizing. They do not
identify a causal mechanism or justify changing frozen V2.1.

## Broad descriptive comparisons

Every numeric bin and categorical group is included below; no unfavorable groups are suppressed.
Each cell contains **trade count / net dollars**. The accompanying `FEATURE_DECOMPOSITION.csv`
also includes expectancy, PF, win rate, average win/loss, streak, and top-five removal for every
group and each side/month. Hour and weekday tables remain in that CSV to keep this report readable.

| Feature | Group | ALL n / $ | BUY n / $ | SELL n / $ | June n / $ | July n / $ | August n / $ |
|---|---|---:|---:|---:|---:|---:|---:|
| session | london | 97 / +90.55 | 45 / -30.85 | 52 / +121.40 | 30 / +73.41 | 34 / -8.52 | 33 / +25.66 |
| session | new_york | 53 / +2.31 | 30 / +16.26 | 23 / -13.95 | 19 / -15.83 | 21 / +11.80 | 13 / +6.34 |
| session | other | 147 / +53.17 | 76 / +36.44 | 71 / +16.73 | 45 / +13.14 | 54 / -66.36 | 48 / +106.39 |
| session | overlap | 67 / -65.78 | 31 / -2.04 | 36 / -63.74 | 22 / -21.40 | 25 / -17.20 | 20 / -27.18 |
| breakout | False | 310 / +70.82 | 154 / +8.56 | 156 / +62.26 | 102 / +57.18 | 119 / -73.31 | 89 / +86.95 |
| breakout | True | 54 / +9.43 | 28 / +11.25 | 26 / -1.82 | 14 / -7.86 | 15 / -6.97 | 25 / +24.26 |
| pullback | False | 293 / +94.76 | 151 / +64.93 | 142 / +29.83 | 92 / +28.63 | 106 / -32.97 | 95 / +99.10 |
| pullback | True | 71 / -14.51 | 31 / -45.12 | 40 / +30.61 | 24 / +20.69 | 28 / -47.31 | 19 / +12.11 |
| slope_htf_alignment | False | 182 / +86.24 | 103 / +41.09 | 79 / +45.15 | 58 / +18.28 | 69 / +17.87 | 55 / +50.09 |
| slope_htf_alignment | True | 182 / -5.99 | 79 / -21.28 | 103 / +15.29 | 58 / +31.04 | 65 / -98.15 | 59 / +61.12 |
| m5_m15_alignment | True | 364 / +80.25 | 182 / +19.81 | 182 / +60.44 | 116 / +49.32 | 134 / -80.28 | 114 / +111.21 |
| previous_inside | False | 318 / +121.07 | 160 / +54.28 | 158 / +66.79 | 103 / +56.76 | 115 / -44.65 | 100 / +108.96 |
| previous_inside | True | 46 / -40.82 | 22 / -34.47 | 24 / -6.35 | 13 / -7.44 | 19 / -35.63 | 14 / +2.25 |
| signal_inside | False | 310 / +129.91 | 153 / +60.04 | 157 / +69.87 | 96 / +72.21 | 114 / -58.55 | 100 / +116.25 |
| signal_inside | True | 54 / -49.66 | 29 / -40.23 | 25 / -9.43 | 20 / -22.89 | 20 / -21.73 | 14 / -5.04 |
| trend_efficiency | low | 122 / +9.93 | 67 / -17.91 | 55 / +27.84 | 39 / +35.87 | 47 / -42.62 | 36 / +16.68 |
| trend_efficiency | middle | 114 / -71.59 | 51 / +22.21 | 63 / -93.80 | 38 / -60.43 | 42 / -51.16 | 34 / +40.00 |
| trend_efficiency | high | 128 / +141.91 | 64 / +15.51 | 64 / +126.40 | 39 / +73.88 | 45 / +13.50 | 44 / +54.53 |
| sma_separation_atr | low | 113 / +175.02 | 51 / +172.41 | 62 / +2.61 | 39 / +110.65 | 41 / +8.69 | 33 / +55.68 |
| sma_separation_atr | middle | 133 / -129.11 | 71 / -152.96 | 62 / +23.85 | 38 / -39.45 | 54 / -94.08 | 41 / +4.42 |
| sma_separation_atr | high | 118 / +34.34 | 60 / +0.36 | 58 / +33.98 | 39 / -21.88 | 39 / +5.11 | 40 / +51.11 |
| sma_slope_aligned_atr | low | 120 / +161.93 | 65 / +113.18 | 55 / +48.75 | 39 / +52.89 | 45 / +43.93 | 36 / +65.11 |
| sma_slope_aligned_atr | middle | 110 / -144.62 | 62 / -83.53 | 48 / -61.09 | 38 / -48.40 | 37 / -74.51 | 35 / -21.71 |
| sma_slope_aligned_atr | high | 134 / +62.94 | 55 / -9.84 | 79 / +72.78 | 39 / +44.83 | 52 / -49.70 | 43 / +67.81 |
| htf_distance_atr | low | 96 / +18.90 | 49 / +1.23 | 47 / +17.67 | 39 / -45.37 | 25 / -0.60 | 32 / +64.87 |
| htf_distance_atr | middle | 156 / -67.39 | 69 / -21.44 | 87 / -45.95 | 38 / -46.09 | 74 / +5.02 | 44 / -26.32 |
| htf_distance_atr | high | 112 / +128.74 | 64 / +40.02 | 48 / +88.72 | 39 / +140.78 | 35 / -84.70 | 38 / +72.66 |
| rsi | low | 101 / +105.29 | 10 / -2.81 | 91 / +108.10 | 39 / +104.85 | 37 / -45.29 | 25 / +45.73 |
| rsi | middle | 106 / -46.77 | 41 / +38.56 | 65 / -85.33 | 38 / -19.56 | 35 / -34.89 | 33 / +7.68 |
| rsi | high | 157 / +21.73 | 131 / -15.94 | 26 / +37.67 | 39 / -35.97 | 62 / -0.10 | 56 / +57.80 |
| rsi_momentum_aligned | low | 130 / +2.42 | 67 / +44.03 | 63 / -41.61 | 39 / +34.38 | 53 / -34.17 | 38 / +2.21 |
| rsi_momentum_aligned | middle | 127 / +17.45 | 67 / -28.20 | 60 / +45.65 | 38 / -50.84 | 46 / +11.89 | 43 / +56.40 |
| rsi_momentum_aligned | high | 107 / +60.38 | 48 / +3.98 | 59 / +56.40 | 39 / +65.78 | 35 / -58.00 | 33 / +52.60 |
| atr | low | 159 / +12.58 | 79 / +20.92 | 80 / -8.34 | 39 / +27.56 | 73 / -47.84 | 47 / +32.86 |
| atr | middle | 113 / +32.89 | 53 / +1.91 | 60 / +30.98 | 38 / -25.70 | 28 / +10.69 | 47 / +47.90 |
| atr | high | 92 / +34.78 | 50 / -3.02 | 42 / +37.80 | 39 / +47.46 | 33 / -43.13 | 20 / +30.45 |
| atr_percentile | low | 116 / +58.99 | 53 / +74.66 | 63 / -15.67 | 39 / +13.53 | 48 / +12.29 | 29 / +33.17 |
| atr_percentile | middle | 93 / -35.46 | 43 / -57.62 | 50 / +22.16 | 37 / -0.54 | 24 / -58.99 | 32 / +24.07 |
| atr_percentile | high | 155 / +56.72 | 86 / +2.77 | 69 / +53.95 | 40 / +36.33 | 62 / -33.58 | 53 / +53.97 |
| spread_points | low | 37 / +53.13 | 14 / -6.94 | 23 / +60.07 | 37 / +53.13 | 0 / +0.00 | 0 / +0.00 |
| spread_points | middle | 158 / +30.95 | 83 / +58.29 | 75 / -27.34 | 35 / +75.69 | 73 / -54.86 | 50 / +10.12 |
| spread_points | high | 169 / -3.83 | 85 / -31.54 | 84 / +27.71 | 44 / -79.50 | 61 / -25.42 | 64 / +101.09 |
| spread_atr | low | 65 / +67.38 | 29 / -31.38 | 36 / +98.76 | 39 / +123.26 | 18 / -36.84 | 8 / -19.04 |
| spread_atr | middle | 101 / -76.10 | 59 / -11.32 | 42 / -64.78 | 38 / -97.68 | 29 / -36.14 | 34 / +57.72 |
| spread_atr | high | 198 / +88.97 | 94 / +62.51 | 104 / +26.46 | 39 / +23.74 | 87 / -7.30 | 72 / +72.53 |
| distance_high_atr | low | 154 / -81.02 | 131 / -69.65 | 23 / -11.37 | 39 / -106.74 | 62 / -50.03 | 53 / +75.75 |
| distance_high_atr | middle | 102 / +124.41 | 36 / +83.21 | 66 / +41.20 | 38 / +81.30 | 37 / -15.05 | 27 / +58.16 |
| distance_high_atr | high | 108 / +36.86 | 15 / +6.25 | 93 / +30.61 | 39 / +74.76 | 35 / -15.20 | 34 / -22.70 |
| distance_low_atr | low | 85 / +48.05 | 4 / -37.98 | 81 / +86.03 | 39 / +95.69 | 25 / -76.30 | 21 / +28.66 |
| distance_low_atr | middle | 120 / -11.09 | 49 / +58.63 | 71 / -69.72 | 38 / -17.69 | 43 / -1.16 | 39 / +7.76 |
| distance_low_atr | high | 159 / +43.29 | 129 / -0.84 | 30 / +44.13 | 39 / -28.68 | 66 / -2.82 | 54 / +74.79 |
| candle_body_range | low | 115 / +2.74 | 66 / +15.34 | 49 / -12.60 | 39 / +45.50 | 48 / -56.46 | 28 / +13.70 |
| candle_body_range | middle | 146 / +81.56 | 76 / +6.52 | 70 / +75.04 | 38 / -4.56 | 55 / +70.92 | 53 / +15.20 |
| candle_body_range | high | 103 / -4.05 | 40 / -2.05 | 63 / -2.00 | 39 / +8.38 | 31 / -94.74 | 33 / +82.31 |
| candle_body_aligned | low | 131 / +14.66 | 74 / +34.36 | 57 / -19.70 | 39 / +23.10 | 51 / -36.26 | 41 / +27.82 |
| candle_body_aligned | middle | 127 / +14.13 | 66 / -33.80 | 61 / +47.93 | 38 / +12.04 | 47 / -8.42 | 42 / +10.51 |
| candle_body_aligned | high | 106 / +51.46 | 42 / +19.25 | 64 / +32.21 | 39 / +14.18 | 36 / -35.60 | 31 / +72.88 |
| upper_wick_fraction | low | 106 / -63.89 | 46 / -50.60 | 60 / -13.29 | 39 / +40.70 | 36 / -134.74 | 31 / +30.15 |
| upper_wick_fraction | middle | 131 / +47.02 | 60 / -8.22 | 71 / +55.24 | 38 / +3.86 | 43 / +32.10 | 50 / +11.06 |
| upper_wick_fraction | high | 127 / +97.12 | 76 / +78.63 | 51 / +18.49 | 39 / +4.76 | 55 / +22.36 | 33 / +70.00 |
| lower_wick_fraction | low | 118 / +107.76 | 60 / +36.38 | 58 / +71.38 | 39 / +21.06 | 45 / +48.66 | 34 / +38.04 |
| lower_wick_fraction | middle | 132 / +101.70 | 59 / +84.52 | 73 / +17.18 | 38 / +59.67 | 45 / -72.79 | 49 / +114.82 |
| lower_wick_fraction | high | 114 / -129.21 | 63 / -101.09 | 51 / -28.12 | 39 / -31.41 | 44 / -56.15 | 31 / -41.65 |
| volatility_expansion | low | 151 / +125.78 | 80 / +65.23 | 71 / +60.55 | 39 / +7.62 | 64 / +72.21 | 48 / +45.95 |
| volatility_expansion | middle | 93 / -8.16 | 38 / -18.76 | 55 / +10.60 | 38 / +76.87 | 31 / -122.17 | 24 / +37.14 |
| volatility_expansion | high | 120 / -37.37 | 64 / -26.66 | 56 / -10.71 | 39 / -35.17 | 39 / -30.32 | 42 / +28.12 |
| previous_body_aligned | low | 122 / -56.97 | 62 / -13.33 | 60 / -43.64 | 39 / -60.82 | 47 / -26.95 | 36 / +30.80 |
| previous_body_aligned | middle | 114 / +82.92 | 54 / +52.36 | 60 / +30.56 | 38 / +98.54 | 48 / -34.88 | 28 / +19.26 |
| previous_body_aligned | high | 128 / +54.30 | 66 / -19.22 | 62 / +73.52 | 39 / +11.60 | 39 / -18.45 | 50 / +61.15 |

### Interpretation for review, not selection

- **Stronger slope is not established as an improvement.** Slope-aligned trades total −$5.99
  and include a −$98.15 July. Numeric slope groups are non-monotonic. Frozen M5/M15 alignment
  already holds for every trade and cannot discriminate baseline winners from losers.
- **Trend efficiency merits discussion, with a concentration caveat.** The high group has
  128 trades and +$141.91 across three positive months, but BUY's +$15.51 becomes −$78.92
  without its five largest winners. Pooled strength is not symmetric evidence.
- **Contraction is descriptive, not a validated regime gate.** Low recent range expansion
  has 151 trades, +$125.78 and positive monthly totals. BUY still becomes negative after its
  top five winners are removed. ATR-percentile and spread/ATR bins are not monotonic.
- **Session overlap is consistently weak in this sample.** Its 67 trades lose $65.78, with
  losses in each month. SELL contributes most of that loss. Excluding the overlap still needs
  a fresh sequential replay; it cannot be justified by subtracting this subset alone.
- **Inside-bar structure shows a repeated negative association.** Signal-inside trades total
  54 and −$49.66, negative by both side and month. Multiple inspected features and small
  monthly counts prevent treating this as independent evidence of an exploitable rule.
- **Existing pullback/breakout tags do not validate new entry methods.** Pullback-tagged
  baseline trades lose $14.51 over 71 trades; breakout-tagged trades gain $9.43 over 54.
  These tags describe crossover entries, not a separately timed pullback or breakout strategy.
- Median holding time is 23.37 minutes for winners and 19.43 for losers. Duration is observed
  after entry, so this does not justify a time exit or establish its effect on unrealized paths.

## Fixed-ledger cost sensitivity

This is arithmetic on the same executed trades and original volumes. It is **not** a wider-spread,
slippage, execution-delay, or re-compounded equity replay. One additional point per fill costs
$9.52 in total (4.76 lots × two fills × $1 per lot-point). Scenarios do not change spread skips,
position availability, lot rounding, stops, or fill timing. Side/month results are in the CSV.

| Scenario | Net $ | PF | Net without top 5 $ |
|---|---:|---:|---:|
| reference | +80.25 | 1.059 | -15.88 |
| commission_plus_25pct | +73.11 | 1.054 | -22.90 |
| commission_plus_50pct | +65.97 | 1.048 | -29.92 |
| plus_1_point_each_fill | +70.73 | 1.052 | -25.24 |
| plus_2_points_each_fill | +61.21 | 1.045 | -34.60 |
| commission_25pct_plus_1_point_each_fill | +63.59 | 1.047 | -32.26 |

## Sampling uncertainty

A moving-block bootstrap resamples 65 UTC weekdays, including zero-trade days. Whole trade net
is assigned to entry date; these are daily attributed dollars, not mark-to-market daily returns.
Seed 222026, 5,000 replicates, non-circular overlapping blocks, sampled paths truncated to 65 days.

| Block length | 2.5% total $ | Median total $ | 97.5% total $ |
|---|---:|---:|---:|
| 3 days | -194.43 | +106.72 | +395.28 |
| 5 days | -199.24 | +106.40 | +388.20 |
| 7 days | -188.23 | +109.65 | +353.49 |

Every interval includes material losses. This resampling does not repair the short development
sample, selection bias, nonstationarity, or multiple comparisons. Positive-resample fractions are
not probabilities that the EA will profit live. No statistical significance claim is made.

## Reproduce and verify

Use Python 3.12 and repository dependencies plus pytest and Windows `tzdata`. From repository root:

```powershell
python scripts/analyze_sma_rsi_v22_stage_a.py --source-root E:/XAUUSD-CSV-Research-Lab --output <separate-output-directory>
python scripts/summarize_sma_rsi_v22_stage_a.py --dataset-root E:/XAUUSD-CSV-Research-Lab/datasets/fxify_xauusdr --output <same-output-directory>
python -m pytest
```

Source evidence is read-only. The analysis rejects output inside the frozen source result or
dataset directories. Input reconciliation and partition checks raise errors even with `python -O`.
The synthetic tests cover metrics, top-five concentration, invalid inputs, cutoff exclusion,
trailing-feature prefix invariance, source-output isolation, and bootstrap repeatability.
No pre-existing parity assertion has been changed. See `VERIFICATION.md` for the final suite,
repeatability, file-freeze checks, environment versions, and commit references.

## Unresolved limits and next step

The isolated one-cent V2.1 residual remains unresolved. The original EA has not been located or
modified; no claim is made about freshly compiling it or testing restart/netting behavior here.
Stage A does not rerun the entire earlier engineering audit. Existing parser/execution risks
remain those documented in the separate engineering audit branch. Broker EET/DST assumptions
remain inherited from the validated pipeline; the original terminal's timezone configuration
has not been independently recovered. This stage reconciles all executed entries, not a new
independent proof of every possible raw signal or missing-data behavior.

No candidate parameter-neighborhood tests, walk-forward candidate trials, real execution stress,
or exit-path experiments have occurred. Descriptive groups can be correlated and confounded;
outlier dependence, BUY/SELL imbalance and monthly variation remain substantial.

**Recommended next step:** review these Stage A findings and their limitations. Only after that
review should Stage B define a small, explicit hypothesis list and rejection criteria. No future
stage, candidate implementation, prospective test, OOS test, or merge is authorized by this report.
