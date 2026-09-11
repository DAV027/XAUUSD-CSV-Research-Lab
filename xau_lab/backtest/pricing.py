from __future__ import annotations

from xau_lab.backtest.models import CostModel, SymbolSpec


def _validate_spread(spread_points: int | float) -> None:
    if spread_points < 0:
        raise ValueError("spread_points cannot be negative")


def _round_price(price: float, symbol: SymbolSpec) -> float:
    return round(float(price), symbol.digits)


def buy_entry_price(
    bid_open: float,
    spread_points: int | float,
    symbol: SymbolSpec,
    cost: CostModel,
) -> float:
    """Executable buy fill from a bid-side M1 open."""
    _validate_spread(spread_points)
    price = bid_open + spread_points * symbol.point + cost.slippage_points_per_fill * symbol.point
    return _round_price(price, symbol)


def sell_entry_price(
    bid_open: float,
    spread_points: int | float,
    symbol: SymbolSpec,
    cost: CostModel,
) -> float:
    """Executable sell fill from a bid-side M1 open."""
    _validate_spread(spread_points)
    price = bid_open - cost.slippage_points_per_fill * symbol.point
    return _round_price(price, symbol)


def fill_exit_price(
    bid_price: float,
    spread_points: int | float,
    direction: int,
    symbol: SymbolSpec,
    cost: CostModel,
) -> float:
    """Return an adverse executable exit fill for an existing position.

    Long positions exit by selling at bid minus slippage. Short positions
    exit by buying at ask plus slippage.
    """
    _validate_spread(spread_points)
    slippage = cost.slippage_points_per_fill * symbol.point
    if direction == 1:
        return _round_price(bid_price - slippage, symbol)
    if direction == -1:
        ask = bid_price + spread_points * symbol.point
        return _round_price(ask + slippage, symbol)
    raise ValueError("direction must be -1 or +1")
