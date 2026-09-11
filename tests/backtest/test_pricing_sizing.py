import pytest

from xau_lab.backtest.models import CostModel, RiskModel, SymbolSpec
from xau_lab.backtest.pricing import buy_entry_price, fill_exit_price, sell_entry_price
from xau_lab.backtest.sizing import size_for_stop


SPEC = SymbolSpec(
    point=0.01,
    digits=2,
    contract_size=100.0,
    volume_min=0.01,
    volume_step=0.01,
)


def test_buy_entry_pays_spread_and_slippage_but_sell_uses_bid_minus_slippage():
    cost = CostModel(slippage_points_per_fill=5)
    assert buy_entry_price(2000.00, 35, SPEC, cost) == pytest.approx(2000.40)
    assert sell_entry_price(2000.00, 35, SPEC, cost) == pytest.approx(1999.95)


def test_long_exit_is_bid_minus_slippage_and_short_exit_pays_ask_plus_slippage():
    cost = CostModel(slippage_points_per_fill=5)
    assert fill_exit_price(2001.00, 35, 1, SPEC, cost) == pytest.approx(2000.95)
    assert fill_exit_price(1999.00, 35, -1, SPEC, cost) == pytest.approx(1999.40)


def test_minimum_lot_is_skipped_when_stop_risk_exceeds_hard_cap():
    risk = RiskModel(preferred_risk_usd=2.0, hard_risk_usd=5.0)
    sized = size_for_stop(entry_price=2000.0, stop_price=1994.0, symbol=SPEC, risk=risk)
    assert sized.feasible is False
    assert sized.lot == 0.0


def test_minimum_lot_is_allowed_only_as_hard_cap_fallback():
    risk = RiskModel(preferred_risk_usd=2.0, hard_risk_usd=5.0)
    sized = size_for_stop(entry_price=2000.0, stop_price=1996.0, symbol=SPEC, risk=risk)
    assert sized.feasible is True
    assert sized.lot == pytest.approx(0.01)
    assert sized.risk_usd == pytest.approx(4.0)


def test_preferred_size_is_floored_to_broker_step_and_never_rounded_up():
    risk = RiskModel(preferred_risk_usd=12.0, hard_risk_usd=20.0, max_lot=1.0)
    sized = size_for_stop(entry_price=2000.0, stop_price=1997.0, symbol=SPEC, risk=risk)
    assert sized.feasible is True
    assert sized.lot == pytest.approx(0.04)
    assert sized.risk_usd == pytest.approx(12.0)


def test_zero_stop_distance_is_rejected():
    with pytest.raises(ValueError):
        size_for_stop(2000.0, 2000.0, SPEC, RiskModel())
