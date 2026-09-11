import numpy as np
import pytest

from xau_lab.backtest.models import MarketBars, RiskModel, SymbolSpec
from xau_lab.experiments.spec import CompleteExperiment
from xau_lab.runner.single import MarketBundle
from xau_lab.strategies.base import StrategyContext


def _market(*, times=None, opens=None, spreads=None, features=None):
    n = 40
    t = np.arange(n, dtype=np.int64) * 60 if times is None else np.asarray(times, dtype=np.int64)
    o = np.full(n, 2000.0) if opens is None else np.asarray(opens, dtype=float)
    h = o + 1.0
    l = o - 1.0
    c = o + 0.2
    sp = np.full(n, 25.0) if spreads is None else np.asarray(spreads, dtype=float)
    atr = np.full(n, 2.0)
    bars = MarketBars(t, o, h, l, c, sp, atr)
    feats = features or {}
    ctx = StrategyContext(o, h, l, c, sp, atr, t, feats)
    return MarketBundle(bars, ctx, SymbolSpec(0.01, 2, 100.0, 0.01, 0.01), RiskModel())


def _experiment(name="sma_slope", family="trend", eid="EXP1", fp="a" * 64):
    return CompleteExperiment(
        experiment_id=eid,
        fingerprint=fp,
        strategy_family=family,
        strategy_name=name,
        parameters={"window": 9},
        direction_mode="combined",
        stop_atr=1.0,
        exit_mode="target",
        target_r=1.5,
        time_exit_minutes=0,
        atr_trail=0.0,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        sampler_version="v1",
        seed=9215000,
    )


def test_preflight_accepts_valid_inputs():
    from xau_lab.runner.preflight import validate_campaign_inputs

    report = validate_campaign_inputs(_market(), [_experiment()], expected_catalog_size=1)
    assert report.ok is True
    assert report.market_rows == 40
    assert report.catalog_rows == 1


def test_preflight_rejects_non_monotonic_time():
    from xau_lab.runner.preflight import CampaignPreflightError, validate_campaign_inputs

    t = np.arange(40, dtype=np.int64) * 60
    t[10] = t[9]
    with pytest.raises(CampaignPreflightError, match="strictly increasing"):
        validate_campaign_inputs(_market(times=t), [_experiment()])


def test_preflight_rejects_invalid_prices_and_spread():
    from xau_lab.runner.preflight import CampaignPreflightError, validate_campaign_inputs

    o = np.full(40, 2000.0)
    o[5] = 0.0
    with pytest.raises(CampaignPreflightError, match="positive finite OHLC"):
        validate_campaign_inputs(_market(opens=o), [_experiment()])

    sp = np.full(40, 25.0)
    sp[7] = -1.0
    with pytest.raises(CampaignPreflightError, match="non-negative finite spread"):
        validate_campaign_inputs(_market(spreads=sp), [_experiment()])


def test_preflight_rejects_duplicate_catalog_identity():
    from xau_lab.runner.preflight import CampaignPreflightError, validate_campaign_inputs

    e1 = _experiment()
    e2 = _experiment()
    with pytest.raises(CampaignPreflightError, match="duplicate experiment_id"):
        validate_campaign_inputs(_market(), [e1, e2])

    # Simulate a corrupted in-memory catalog where the row ID was changed but
    # the canonical strategy fingerprint was duplicated.
    e3 = _experiment()
    object.__setattr__(e3, "experiment_id", "EXPCORRUPTED")
    with pytest.raises(CampaignPreflightError, match="duplicate fingerprint"):
        validate_campaign_inputs(_market(), [e1, e3])


def test_preflight_requires_session_features_when_catalog_contains_session_strategy():
    from xau_lab.runner.preflight import CampaignPreflightError, validate_campaign_inputs

    session_exp = _experiment(name="opening_range_breakout", family="session")
    with pytest.raises(CampaignPreflightError, match="session features"):
        validate_campaign_inputs(_market(), [session_exp])

    features = {
        "session_asia": np.zeros(40, dtype=np.int8),
        "session_london": np.ones(40, dtype=np.int8),
        "session_new_york": np.zeros(40, dtype=np.int8),
        "session_overlap": np.zeros(40, dtype=np.int8),
        "broker_date": np.array(["2026-01-01"] * 40),
        "hour": np.arange(40) % 24,
        "minute": np.zeros(40, dtype=np.int16),
    }
    assert validate_campaign_inputs(_market(features=features), [session_exp]).ok


def test_preflight_enforces_expected_catalog_size():
    from xau_lab.runner.preflight import CampaignPreflightError, validate_campaign_inputs

    with pytest.raises(CampaignPreflightError, match="expected 50000"):
        validate_campaign_inputs(_market(), [_experiment()], expected_catalog_size=50_000)
