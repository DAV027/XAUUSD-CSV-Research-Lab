import numpy as np
import pytest

from xau_lab.backtest.models import CostModel, ExitSpec, MarketBars, RiskModel, SymbolSpec


def test_symbol_rejects_nonpositive_point():
    with pytest.raises(ValueError):
        SymbolSpec(point=0.0, digits=2, contract_size=100.0, volume_min=0.01, volume_step=0.01)


def test_symbol_rejects_invalid_volume_ladder():
    with pytest.raises(ValueError):
        SymbolSpec(point=0.01, digits=2, contract_size=100.0, volume_min=0.02, volume_step=0.01, volume_max=0.01)


def test_exit_spec_rejects_unknown_directionless_configuration():
    with pytest.raises(ValueError):
        ExitSpec(stop_atr=0.0, target_r=1.0, time_exit_minutes=None, atr_trail=None)


def test_discovery_cost_defaults_match_spec():
    c = CostModel()
    assert c.commission_round_trip_per_lot == 6.0
    assert c.slippage_points_per_fill == 5.0


def test_risk_model_rejects_preferred_risk_above_hard_cap():
    with pytest.raises(ValueError):
        RiskModel(preferred_risk_usd=6.0, hard_risk_usd=5.0)


def test_market_bars_require_equal_length_arrays():
    with pytest.raises(ValueError):
        MarketBars(
            time_epoch=np.array([1, 2], dtype=np.int64),
            open=np.array([100.0]),
            high=np.array([101.0, 102.0]),
            low=np.array([99.0, 100.0]),
            close=np.array([100.0, 101.0]),
            spread=np.array([10, 10], dtype=np.int64),
            atr=np.array([1.0, 1.0]),
        )
