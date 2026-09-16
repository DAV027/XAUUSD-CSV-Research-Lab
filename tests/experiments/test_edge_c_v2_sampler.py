from xau_lab.experiments.edge_c_sampler import (
    EDGE_C_DEFAULT_SEED,
    EDGE_C_FAMILY,
    EDGE_C_SAMPLER_VERSION,
    EDGE_C_STRATEGY,
    generate_edge_c_catalog,
)
from xau_lab.experiments.edge_c_v2_sampler import (
    EDGE_C_V2_DEFAULT_BUDGET,
    EDGE_C_V2_DEFAULT_SEED,
    EDGE_C_V2_FAMILY,
    EDGE_C_V2_SAMPLER_VERSION,
    EDGE_C_V2_STRATEGY,
    generate_edge_c_v2_catalog,
)


def test_edge_c_v2_identity_defaults_are_frozen():
    assert EDGE_C_V2_DEFAULT_BUDGET == 10_000
    assert EDGE_C_V2_DEFAULT_SEED == 9_216_300
    assert EDGE_C_V2_SAMPLER_VERSION == "edge_c_v2"
    assert EDGE_C_V2_FAMILY == "edge_c_breakout_retest_v2"
    assert EDGE_C_V2_STRATEGY == "compression_breakout_retest_v2"


def test_edge_c_v2_catalog_is_deterministic_and_10000_rows_are_unique():
    first = generate_edge_c_v2_catalog(total_budget=10_000, seed=EDGE_C_V2_DEFAULT_SEED)
    second = generate_edge_c_v2_catalog(total_budget=10_000, seed=EDGE_C_V2_DEFAULT_SEED)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert len(first) == 10_000
    assert len({row.experiment_id for row in first}) == 10_000
    assert len({row.fingerprint for row in first}) == 10_000


def test_edge_c_v2_catalog_uses_exact_domains_exits_and_costs():
    rows = generate_edge_c_v2_catalog(total_budget=1024, seed=EDGE_C_V2_DEFAULT_SEED)

    assert {row.family for row in rows} == {EDGE_C_V2_FAMILY}
    assert {row.allocation_bucket for row in rows} == {EDGE_C_V2_FAMILY}
    assert {row.strategy_name for row in rows} == {EDGE_C_V2_STRATEGY}
    assert {row.sampler_version for row in rows} == {EDGE_C_V2_SAMPLER_VERSION}
    assert {row.direction_mode for row in rows} == {"combined"}
    assert {row.stop_atr for row in rows}.issubset({1.0, 1.5, 2.0, 3.0})
    assert {row.exit_type for row in rows}.issubset({"target_r", "atr_trail"})
    assert {row.commission_round_trip_per_lot for row in rows} == {6.0}
    assert {row.slippage_points_per_fill for row in rows} == {5.0}

    for row in rows:
        p = row.parameters
        assert set(p) == {
            "compression_lookback",
            "compression_atr_ratio",
            "breakout_buffer_atr",
            "breakout_body_atr",
            "retest_window",
            "retest_tolerance_atr",
            "confirmation_atr",
        }
        assert 20 <= p["compression_lookback"] <= 120
        assert 0.50 <= p["compression_atr_ratio"] <= 1.10
        assert 0.25 <= p["breakout_buffer_atr"] <= 1.50
        assert 0.25 <= p["breakout_body_atr"] <= 1.50
        assert 2 <= p["retest_window"] <= 20
        assert 0.10 <= p["retest_tolerance_atr"] <= 0.75
        assert 0.10 <= p["confirmation_atr"] <= 1.00
        assert row.time_exit_minutes is None
        if row.exit_type == "target_r":
            assert row.target_r in {1.5, 2.0, 3.0, 4.0}
            assert row.atr_trail is None
        else:
            assert row.target_r is None
            assert row.atr_trail in {1.0, 1.5, 2.0}


def test_edge_c_v2_sampler_does_not_mutate_v1_identity_or_output():
    assert EDGE_C_DEFAULT_SEED == 9_216_200
    assert EDGE_C_SAMPLER_VERSION == "edge_c_v1"
    assert EDGE_C_FAMILY == "edge_c_breakout_retest"
    assert EDGE_C_STRATEGY == "compression_breakout_retest"

    before = [row.to_dict() for row in generate_edge_c_catalog(total_budget=256, seed=EDGE_C_DEFAULT_SEED)]
    _ = generate_edge_c_v2_catalog(total_budget=256, seed=EDGE_C_V2_DEFAULT_SEED)
    after = [row.to_dict() for row in generate_edge_c_catalog(total_budget=256, seed=EDGE_C_DEFAULT_SEED)]
    assert before == after
