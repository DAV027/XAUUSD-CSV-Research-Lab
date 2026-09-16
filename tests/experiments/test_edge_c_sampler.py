from xau_lab.experiments.edge_c_sampler import (
    EDGE_C_DEFAULT_SEED,
    EDGE_C_SAMPLER_VERSION,
    generate_edge_c_catalog,
)
from xau_lab.experiments.sampler import generate_catalog


def test_edge_c_catalog_is_deterministic_and_unique():
    first = generate_edge_c_catalog(total_budget=128, seed=EDGE_C_DEFAULT_SEED)
    second = generate_edge_c_catalog(total_budget=128, seed=EDGE_C_DEFAULT_SEED)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert len({row.experiment_id for row in first}) == 128
    assert len({row.fingerprint for row in first}) == 128


def test_edge_c_catalog_uses_frozen_family_domains_and_costs():
    rows = generate_edge_c_catalog(total_budget=512, seed=EDGE_C_DEFAULT_SEED)

    assert {row.family for row in rows} == {"edge_c_breakout_retest"}
    assert {row.allocation_bucket for row in rows} == {"edge_c_breakout_retest"}
    assert {row.strategy_name for row in rows} == {"compression_breakout_retest"}
    assert {row.sampler_version for row in rows} == {EDGE_C_SAMPLER_VERSION}
    assert {row.direction_mode for row in rows} == {"combined"}
    assert {row.stop_atr for row in rows}.issubset({1.0, 1.5, 2.0, 3.0})
    assert {row.exit_type for row in rows}.issubset({"target_r", "atr_trail"})
    assert {row.commission_round_trip_per_lot for row in rows} == {6.0}
    assert {row.slippage_points_per_fill for row in rows} == {5.0}

    for row in rows:
        p = row.parameters
        assert 20 <= p["compression_lookback"] <= 120
        assert 0.35 <= p["compression_atr_ratio"] <= 0.80
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


def test_importing_edge_c_does_not_change_or_allocate_original_sampler_v1():
    before = [row.to_dict() for row in generate_catalog(total_budget=500, seed=9_215_000)]
    _ = generate_edge_c_catalog(total_budget=64, seed=EDGE_C_DEFAULT_SEED)
    after = [row.to_dict() for row in generate_catalog(total_budget=500, seed=9_215_000)]

    assert before == after
    assert all(row["family"] != "edge_c_breakout_retest" for row in after)
    assert all(row["strategy_name"] != "compression_breakout_retest" for row in after)
