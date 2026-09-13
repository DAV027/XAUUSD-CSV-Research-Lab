import numpy as np
import pytest
from xau_lab.backtest.fast import run_fast_backtest
from xau_lab.backtest.reference import run_reference_backtest
from xau_lab.backtest.models import CostModel, ExitSpec, RiskModel, SymbolSpec
from .test_fast_parity import _random_market

SPEC = SymbolSpec(point=.01, digits=2, contract_size=100., volume_min=.01, volume_step=.01)

class NoIteration(np.ndarray):
    def __iter__(self):
        raise AssertionError("NumPy signals must not become a Python list")


def test_numpy_signals_do_not_materialize_python_scalars():
    bars, signals = _random_market(200)
    expected = run_reference_backtest(bars, signals, SPEC, CostModel(), RiskModel(), ExitSpec(stop_atr=1.))
    actual = run_fast_backtest(bars, signals.view(NoIteration), SPEC, CostModel(), RiskModel(), ExitSpec(stop_atr=1.))
    assert actual == expected


@pytest.mark.parametrize('bad', [2, 256, .5, np.nan, np.inf])
def test_invalid_signals_rejected_before_int8_conversion(bad):
    bars, signals = _random_market(20)
    signals = signals.astype(float)
    signals[3] = bad
    with pytest.raises(ValueError):
        run_fast_backtest(bars, signals, SPEC, CostModel(), RiskModel(), ExitSpec(stop_atr=1.))


@pytest.mark.parametrize('density', [0., .01, 1.])
@pytest.mark.parametrize('exit_spec', [ExitSpec(stop_atr=1., target_r=1.5),
    ExitSpec(stop_atr=1., time_exit_minutes=8), ExitSpec(stop_atr=1.5, target_r=2., atr_trail=.8)])
def test_exact_all_trade_fields(density, exit_spec):
    bars, _ = _random_market(1000)
    rng = np.random.default_rng(712)
    signals = rng.choice([-1, 0, 1], 1000, p=[density/2, 1-density, density/2])
    args = (bars, signals, SPEC, CostModel(), RiskModel(), exit_spec)
    assert run_fast_backtest(*args) == run_reference_backtest(*args)



def test_sparse_signals_bound_output_capacity(monkeypatch):
    import xau_lab.backtest.fast as fast
    bars, signals = _random_market(1000)
    signals[:] = 0
    signals[20] = 1
    original = fast._kernel
    def measured(*args):
        packed = original(*args)
        assert all(len(array) <= 1 for array in packed[2:]), "full-market output buffers"
        return packed
    monkeypatch.setattr(fast, '_kernel', measured)
    assert fast.run_fast_backtest(bars, signals, SPEC, CostModel(), RiskModel(), ExitSpec(stop_atr=1.)) == run_reference_backtest(bars, signals, SPEC, CostModel(), RiskModel(), ExitSpec(stop_atr=1.))
