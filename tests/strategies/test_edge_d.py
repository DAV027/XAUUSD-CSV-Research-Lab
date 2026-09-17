from __future__ import annotations

import numpy as np

from xau_lab.strategies.base import StrategyContext
from xau_lab.strategies.edge_d import prior_session_sweep_reclaim
from xau_lab.strategies.registry import get_strategy


def _ctx(
    *,
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    atr: list[float] | None = None,
    asia: list[bool] | None = None,
    london: list[bool] | None = None,
    new_york: list[bool] | None = None,
) -> StrategyContext:
    n = len(close)
    assert len(open_) == len(high) == len(low) == n
    atr_values = atr if atr is not None else [1.0] * n
    features = {
        "session_asia": np.asarray(asia if asia is not None else [False] * n, dtype=bool),
        "session_london": np.asarray(london if london is not None else [False] * n, dtype=bool),
        "session_new_york": np.asarray(
            new_york if new_york is not None else [False] * n, dtype=bool
        ),
    }
    return StrategyContext(
        open=np.asarray(open_, dtype=np.float64),
        high=np.asarray(high, dtype=np.float64),
        low=np.asarray(low, dtype=np.float64),
        close=np.asarray(close, dtype=np.float64),
        spread=np.zeros(n, dtype=np.float64),
        atr14=np.asarray(atr_values, dtype=np.float64),
        time_epoch=np.arange(n, dtype=np.int64),
        features=features,
    )


def _params(transition: str = "asia_to_london", **overrides) -> dict[str, object]:
    params: dict[str, object] = {
        "transition": transition,
        "entry_window_bars": 10,
        "sweep_atr": 0.25,
        "max_reclaim_bars": 2,
        "reclaim_depth_atr": 0.10,
    }
    params.update(overrides)
    return params


def test_asia_range_freezes_at_first_london_bar_and_upper_reclaim_short() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 101.0, 100.8, 100.2],
        high=[101, 102, 103, 102, 104.0, 103.5, 101.0],
        low=[99, 98, 99, 98.5, 100.5, 100.0, 99.5],
        close=[100, 101, 102, 100, 103.5, 100.7, 100.0],
        asia=[True, True, True, True, False, False, False],
        london=[False, False, False, False, True, True, True],
    )
    signal = prior_session_sweep_reclaim(ctx, _params())
    assert signal.tolist() == [0, 0, 0, 0, 0, -1, 0]


def test_lower_sweep_then_bullish_reclaim_emits_long() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 98.0, 98.4, 99.5],
        high=[101, 102, 103, 102, 98.5, 100.0, 100.5],
        low=[99, 98, 99, 98.5, 97.0, 97.5, 99.0],
        close=[100, 101, 102, 100, 97.4, 99.2, 100.0],
        asia=[True, True, True, True, False, False, False],
        london=[False, False, False, False, True, True, True],
    )
    signal = prior_session_sweep_reclaim(ctx, _params())
    assert signal.tolist() == [0, 0, 0, 0, 0, 1, 0]


def test_london_pre_ny_reference_excludes_first_new_york_bar() -> None:
    ctx = _ctx(
        open_=[100, 100, 101, 101, 102.0, 101.5, 100.5],
        high=[101, 102, 103, 104, 110.0, 105.0, 102.0],
        low=[99, 99, 100, 100, 101.0, 100.0, 99.5],
        close=[100, 101, 102, 103, 109.0, 101.0, 100.0],
        london=[True, True, True, True, True, True, True],
        new_york=[False, False, False, False, True, True, True],
    )
    signal = prior_session_sweep_reclaim(
        ctx,
        _params("london_pre_ny_to_new_york", sweep_atr=0.5, reclaim_depth_atr=0.0),
    )
    # The frozen London high is 104, not the first NY bar's high of 110.
    assert signal.tolist() == [0, 0, 0, 0, 0, -1, 0]


def test_atr_is_strictly_lagged_and_frozen_for_transition() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 101, 100.8],
        high=[101, 102, 103, 102, 103.4, 103.0],
        low=[99, 98, 99, 98.5, 100, 100],
        close=[100, 101, 102, 100, 103.2, 100.5],
        atr=[1.0, 1.0, np.nan, 2.0, 0.1, 50.0],
        asia=[True, True, True, True, False, False],
        london=[False, False, False, False, True, True],
    )
    signal = prior_session_sweep_reclaim(
        ctx,
        _params(sweep_atr=0.5, reclaim_depth_atr=0.0),
    )
    # Frozen ATR must be 2.0 from index 3; threshold high=103 + 1.0, so 103.4 is not a sweep.
    assert signal.tolist() == [0, 0, 0, 0, 0, 0]


def test_sweep_without_reclaim_emits_no_signal() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 103, 104, 104],
        high=[101, 102, 103, 102, 104, 105, 105],
        low=[99, 98, 99, 98.5, 102, 103, 103],
        close=[100, 101, 102, 100, 103.8, 104.2, 104.0],
        asia=[True, True, True, True, False, False, False],
        london=[False, False, False, False, True, True, True],
    )
    assert not np.any(prior_session_sweep_reclaim(ctx, _params()))


def test_reclaim_after_max_reclaim_window_is_rejected() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 103, 104, 104, 101],
        high=[101, 102, 103, 102, 104, 105, 105, 102],
        low=[99, 98, 99, 98.5, 102, 103, 103, 99],
        close=[100, 101, 102, 100, 103.8, 104.2, 104.0, 100],
        asia=[True, True, True, True, False, False, False, False],
        london=[False, False, False, False, True, True, True, True],
    )
    signal = prior_session_sweep_reclaim(ctx, _params(max_reclaim_bars=1))
    assert not np.any(signal)


def test_same_bar_reclaim_allowed_when_max_reclaim_zero() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 104.0],
        high=[101, 102, 103, 102, 104.0],
        low=[99, 98, 99, 98.5, 100.0],
        close=[100, 101, 102, 100, 101.0],
        asia=[True, True, True, True, False],
        london=[False, False, False, False, True],
    )
    signal = prior_session_sweep_reclaim(
        ctx,
        _params(max_reclaim_bars=0, reclaim_depth_atr=0.0),
    )
    assert signal.tolist() == [0, 0, 0, 0, -1]


def test_same_bar_double_sweep_invalidates_transition() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 104],
        high=[101, 102, 103, 102, 104],
        low=[99, 98, 99, 98.5, 97],
        close=[100, 101, 102, 100, 101],
        asia=[True, True, True, True, False],
        london=[False, False, False, False, True],
    )
    assert not np.any(prior_session_sweep_reclaim(ctx, _params(max_reclaim_bars=0)))


def test_sequential_opposite_sweep_before_reclaim_invalidates() -> None:
    ctx = _ctx(
        open_=[100, 100, 100, 100, 103.5, 99.0, 101.0],
        high=[101, 102, 103, 102, 104.0, 100.0, 102.0],
        low=[99, 98, 99, 98.5, 102.0, 97.0, 99.0],
        close=[100, 101, 102, 100, 103.8, 98.0, 100.0],
        asia=[True, True, True, True, False, False, False],
        london=[False, False, False, False, True, True, True],
    )
    assert not np.any(prior_session_sweep_reclaim(ctx, _params()))


def test_one_signal_per_transition_and_new_transition_can_signal_again() -> None:
    ctx = _ctx(
        open_=[100,100,100,100,104,101,104,101, 100,100,100,100,104,101],
        high=[101,102,103,102,104,102,105,102, 101,102,103,102,104,102],
        low=[99,98,99,98.5,100,99,100,99, 99,98,99,98.5,100,99],
        close=[100,101,102,100,101,100,101,100, 100,101,102,100,101,100],
        asia=[True,True,True,True,False,False,False,False, True,True,True,True,False,False],
        london=[False,False,False,False,True,True,True,True, False,False,False,False,True,True],
    )
    signal = prior_session_sweep_reclaim(
        ctx,
        _params(max_reclaim_bars=0, reclaim_depth_atr=0.0),
    )
    assert np.flatnonzero(signal == -1).tolist() == [4, 12]


def test_entry_window_expiry_blocks_late_sweep() -> None:
    ctx = _ctx(
        open_=[100,100,100,100,100,100,104],
        high=[101,102,103,102,102,102,104],
        low=[99,98,99,98.5,99,99,100],
        close=[100,101,102,100,100,100,101],
        asia=[True,True,True,True,False,False,False],
        london=[False,False,False,False,True,True,True],
    )
    assert not np.any(
        prior_session_sweep_reclaim(
            ctx,
            _params(entry_window_bars=2, max_reclaim_bars=0, reclaim_depth_atr=0.0),
        )
    )


def test_invalid_or_missing_lagged_atr_produces_no_signal() -> None:
    ctx = _ctx(
        open_=[100,100,100,100,104],
        high=[101,102,103,102,104],
        low=[99,98,99,98.5,100],
        close=[100,101,102,100,101],
        atr=[np.nan, 0.0, -1.0, np.nan, 1.0],
        asia=[True,True,True,True,False],
        london=[False,False,False,False,True],
    )
    assert not np.any(prior_session_sweep_reclaim(ctx, _params(max_reclaim_bars=0)))


def test_registry_identity_and_domain_are_frozen() -> None:
    definition = get_strategy("prior_session_sweep_reclaim")
    assert definition.family == "edge_d_session_sweep_reclaim"
    assert definition.name == "prior_session_sweep_reclaim"
    assert dict(definition.parameter_domain) == {
        "transition": ("asia_to_london", "london_pre_ny_to_new_york"),
        "entry_window_bars": (30, 180),
        "sweep_atr": (0.10, 1.00),
        "max_reclaim_bars": (0, 10),
        "reclaim_depth_atr": (0.00, 0.50),
    }
