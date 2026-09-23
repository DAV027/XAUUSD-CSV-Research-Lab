from __future__ import annotations

import json

import MetaTrader5 as mt5


def main() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        symbol = "XAUUSD.r"
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info returned no data for {symbol}")

        data = info._asdict() if hasattr(info, "_asdict") else {}
        keys = [
            "name",
            "digits",
            "point",
            "trade_contract_size",
            "trade_tick_size",
            "trade_tick_value",
            "trade_tick_value_profit",
            "trade_tick_value_loss",
            "currency_profit",
            "volume_min",
            "volume_max",
            "volume_step",
            "swap_mode",
            "swap_long",
            "swap_short",
            "swap_rollover3days",
        ]
        payload = {key: data.get(key, getattr(info, key, None)) for key in keys}

        # Include MT5 enum constants when exposed by the Python package so the
        # numeric swap_mode / rollover weekday can be interpreted without guessing.
        constants = {}
        for name in dir(mt5):
            if name.startswith("SYMBOL_SWAP_MODE_") or name.startswith("DAY_OF_WEEK_"):
                value = getattr(mt5, name)
                if isinstance(value, int):
                    constants[name] = value
        payload["available_mt5_constants"] = constants

        print(json.dumps(payload, indent=2, default=str))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
