from xau_lab.experiments.edge_b_sampler import generate_edge_b_catalog
from xau_lab.experiments.edge_b_v2_sampler import (
    EDGE_B_V2_DEFAULT_SEED,
    EDGE_B_V2_SAMPLER_VERSION,
    generate_edge_b_v2_catalog,
)


def test_edge_b_v2_catalog_is_deterministic_and_unique():
    first = generate_edge_b_v2_catalog(total_budget=64, seed=EDGE_B_V2_DEFAULT_SEED)
    second = generate_edge_b_v2_catalog(total_budget=64, seed=EDGE_B_V2_DEFAULT_SEED)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert len({row.experiment_id for row in first}) == 64
    assert len({row.fingerprint for row in first}) == 64


def test_edge_b_v2_catalog_uses_separate_version_and_cost_aware_exit_space():
    rows = generate_edge_b_v2_catalog(total_budget=256, seed=EDGE_B_V2_DEFAULT_SEED)

    assert {row.family for row in rows} == {"edge_b_trend_pullback_v2"}
    assert {row.strategy_name for row in rows} == {"trend_pullback_recovery_v2"}
    assert {row.sampler_version for row in rows} == {EDGE_B_V2_SAMPLER_VERSION}
    assert {row.direction_mode for row in rows} == {"combined"}
    assert {row.stop_atr for row in rows}.issubset({1.5, 2.0, 3.0, 4.0})
    assert {row.exit_type for row in rows}.issubset({"target_r", "atr_trail"})


def test_importing_v2_does_not_change_edge_b_v1_catalog():
    before = [row.to_dict() for row in generate_edge_b_catalog(total_budget=32, seed=9_216_000)]
    _ = generate_edge_b_v2_catalog(total_budget=32, seed=EDGE_B_V2_DEFAULT_SEED)
    after = [row.to_dict() for row in generate_edge_b_catalog(total_budget=32, seed=9_216_000)]
    assert before == after
