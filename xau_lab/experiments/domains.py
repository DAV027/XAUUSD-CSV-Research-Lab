from __future__ import annotations

FAMILY_BUDGET_WEIGHTS: dict[str, int] = {
    "trend_momentum": 8000,
    "breakout": 8000,
    "mean_reversion": 8000,
    "price_action": 7000,
    "session": 5000,
    "volatility": 5000,
    "statistical": 5000,
    "exit_execution": 4000,
}

DIRECTION_MODES = ("long", "short", "combined")
STOP_ATR_VALUES = (0.5, 0.75, 1.0, 1.5, 2.0)
EXIT_TYPES = ("target_r", "time", "atr_trail", "target_time")
TARGET_R_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0)
TIME_EXIT_MINUTES = (15, 30, 60, 120, 240)
ATR_TRAIL_VALUES = (0.5, 1.0, 1.5, 2.0)

DISCOVERY_COMMISSION_ROUND_TRIP_PER_LOT = 6.0
DISCOVERY_SLIPPAGE_POINTS_PER_FILL = 5.0
SAMPLER_VERSION = "v1"
