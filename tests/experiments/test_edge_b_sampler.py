from xau_lab.experiments.sampler import APPROVED_FAMILIES, generate_catalog
from xau_lab.experiments.edge_b_sampler import (
    EDGE_B_DEFAULT_BUDGET,
    EDGE_B_DEFAULT_SEED,
    EDGE_B_SAMPLER_VERSION,
    generate_edge_b_catalog,
)
from xau_lab.strategies.registry import get_strategy, list_strategies


def test_edge_b_catalog_is_deterministic_unique_and_isolated():
    first = generate_edge_b_catalog(total_budget=500, seed=EDGE_B_DEFAULT_SEED)
    second = generate_edge_b_catalog(total_budget=500, seed=EDGE_B_DEFAULT_SEED)

    assert first == second
    assert len(first) == 500
    assert len({row.fingerprint for row in first}) == 500
    assert len({row.experiment_id for row in first}) == 500
    assert all(row.family == "edge_b_trend_pullback" for row in first)
    assert all(row.allocation_bucket == "edge_b_trend_pullback" for row in first)
    assert all(row.strategy_name == "trend_pullback_recovery" for row in first)
    assert all(row.direction_mode == "combined" for row in first)
    assert all(row.sampler_version == EDGE_B_SAMPLER_VERSION for row in first)


def test_edge_b_parameter_and_execution_domains_are_frozen():
    catalog = generate_edge_b_catalog(total_budget=1000, seed=EDGE_B_DEFAULT_SEED)

    assert EDGE_B_DEFAULT_BUDGET == 10_000
    assert EDGE_B_DEFAULT_SEED == 9_216_000
    assert EDGE_B_SAMPLER_VERSION == "edge_b_v1"
    for row in catalog:
        params = row.parameters
        assert 60 <= params["trend_lookback"] <= 240
        assert 1.0 <= params["trend_threshold_atr"] <= 4.0
        assert 3 <= params["pullback_lookback"] <= 20
        assert 0.25 <= params["pullback_threshold_atr"] <= 1.5
        assert row.stop_atr in {0.5, 0.75, 1.0, 1.5, 2.0}
        assert row.exit_type in {"target_r", "time", "atr_trail", "target_time"}
        assert row.commission_round_trip_per_lot == 6.0
        assert row.slippage_points_per_fill == 5.0


def test_original_sampler_v1_excludes_registered_edge_b_family():
    assert get_strategy("trend_pullback_recovery").family == "edge_b_trend_pullback"
    assert any(row.family == "edge_b_trend_pullback" for row in list_strategies())
    assert "edge_b_trend_pullback" not in APPROVED_FAMILIES

    first = generate_catalog(total_budget=500, seed=9_215_000)
    second = generate_catalog(total_budget=500, seed=9_215_000)

    assert first == second
    assert all(row.family != "edge_b_trend_pullback" for row in first)
