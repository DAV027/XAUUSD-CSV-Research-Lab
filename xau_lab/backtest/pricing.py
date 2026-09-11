from .models import SymbolSpec, CostModel

def buy_entry_price(bid_open: float, spread_points: float, symbol: SymbolSpec, cost: CostModel) -> float:
    return bid_open + (spread_points + cost.slippage_points_per_fill) * symbol.point

def sell_entry_price(bid_open: float, spread_points: float, symbol: SymbolSpec, cost: CostModel) -> float:
    return bid_open - cost.slippage_points_per_fill * symbol.point

def fill_exit_price(raw_bid_price: float, direction: int, spread_points: float, symbol: SymbolSpec, cost: CostModel) -> float:
    slip=cost.slippage_points_per_fill*symbol.point
    spread=spread_points*symbol.point
    if direction==1:
        return raw_bid_price-slip
    if direction==-1:
        return raw_bid_price+spread+slip
    raise ValueError("direction must be +/-1")
