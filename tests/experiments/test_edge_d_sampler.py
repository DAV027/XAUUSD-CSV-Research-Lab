from xau_lab.experiments.edge_c_v2_sampler import (
    EDGE_C_V2_DEFAULT_SEED,
    generate_edge_c_v2_catalog,
)
from xau_lab.experiments.edge_d_sampler import (
    EDGE_D_DEFAULT_BUDGET,
    EDGE_D_DEFAULT_SEED,
    EDGE_D_FAMILY,
    EDGE_D_SAMPLER_VERSION,
    EDGE_D_STRATEGY,
    generate_edge_d_catalog,
)
from xau_lab.experiments.spec import canonical_fingerprint, canonical_json


def test_edge_d_identity_defaults_are_frozen():
    assert EDGE_D_DEFAULT_BUDGET == 10_000
    assert EDGE_D_DEFAULT_SEED == 9_216_400
    assert EDGE_D_SAMPLER_VERSION == "edge_d_v1"
    assert EDGE_D_FAMILY == "edge_d_session_sweep_reclaim"
    assert EDGE_D_STRATEGY == "prior_session_sweep_reclaim"


def test_edge_d_catalog_is_deterministic_and_10000_rows_are_unique():
    first = generate_edge_d_catalog(total_budget=10_000, seed=EDGE_D_DEFAULT_SEED)
    second = generate_edge_d_catalog(total_budget=10_000, seed=EDGE_D_DEFAULT_SEED)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert len(first) == 10_000
    assert len({row.experiment_id for row in first}) == 10_000
    assert len({row.fingerprint for row in first}) == 10_000


def test_edge_d_catalog_uses_exact_frozen_domains_exits_and_costs():
    rows = generate_edge_d_catalog(total_budget=2048, seed=EDGE_D_DEFAULT_SEED)

    assert {row.family for row in rows} == {EDGE_D_FAMILY}
    assert {row.allocation_bucket for row in rows} == {EDGE_D_FAMILY}
    assert {row.strategy_name for row in rows} == {EDGE_D_STRATEGY}
    assert {row.sampler_version for row in rows} == {EDGE_D_SAMPLER_VERSION}
    assert {row.direction_mode for row in rows} == {"combined"}
    assert {row.stop_atr for row in rows} == {0.75, 1.0, 1.5, 2.0}
    assert {row.exit_type for row in rows} == {"target_r", "time"}
    assert {row.commission_round_trip_per_lot for row in rows} == {6.0}
    assert {row.slippage_points_per_fill for row in rows} == {5.0}
    assert {row.parameters["transition"] for row in rows} == {
        "asia_to_london",
        "london_pre_ny_to_new_york",
    }

    seen_target_r = set()
    seen_time_exit = set()
    for row in rows:
        p = row.parameters
        assert set(p) == {
            "transition",
            "entry_window_bars",
            "sweep_atr",
            "max_reclaim_bars",
            "reclaim_depth_atr",
        }
        assert 30 <= p["entry_window_bars"] <= 180
        assert 0.10 <= p["sweep_atr"] <= 1.00
        assert 0 <= p["max_reclaim_bars"] <= 10
        assert 0.00 <= p["reclaim_depth_atr"] <= 0.50
        assert row.atr_trail is None

        if row.exit_type == "target_r":
            assert row.target_r in {1.0, 1.5, 2.0, 3.0}
            assert row.time_exit_minutes is None
            seen_target_r.add(row.target_r)
        else:
            assert row.target_r is None
            assert row.time_exit_minutes in {30, 60, 120, 240}
            seen_time_exit.add(row.time_exit_minutes)

    assert seen_target_r == {1.0, 1.5, 2.0, 3.0}
    assert seen_time_exit == {30, 60, 120, 240}


def test_edge_d_rows_have_canonical_json_and_recomputable_fingerprint():
    rows = generate_edge_d_catalog(total_budget=256, seed=EDGE_D_DEFAULT_SEED)

    for row in rows:
        assert row.canonical_parameters_json == canonical_json(row.parameters)
        assert row.fingerprint == canonical_fingerprint(
            strategy_name=row.strategy_name,
            canonical_parameters=row.parameters,
            direction_mode=row.direction_mode,
            stop_atr=row.stop_atr,
            exit_type=row.exit_type,
            target_r=row.target_r,
            time_exit_minutes=row.time_exit_minutes,
            atr_trail=row.atr_trail,
            commission_round_trip_per_lot=row.commission_round_trip_per_lot,
            slippage_points_per_fill=row.slippage_points_per_fill,
            strategy_seed=row.strategy_seed,
            sampler_version=row.sampler_version,
        )
        assert row.experiment_id == "EXP" + row.fingerprint[:12].upper()


def test_edge_d_sampler_does_not_mutate_existing_edge_c_v2_output():
    before = [
        row.to_dict()
        for row in generate_edge_c_v2_catalog(total_budget=256, seed=EDGE_C_V2_DEFAULT_SEED)
    ]
    _ = generate_edge_d_catalog(total_budget=256, seed=EDGE_D_DEFAULT_SEED)
    after = [
        row.to_dict()
        for row in generate_edge_c_v2_catalog(total_budget=256, seed=EDGE_C_V2_DEFAULT_SEED)
    ]

    assert before == after
