import json

from xau_lab.experiments.sampler import FAMILY_BUDGET_WEIGHTS, generate_catalog
from xau_lab.experiments.spec import canonical_fingerprint


def test_catalog_is_deterministic_unique_and_budgeted():
    a = generate_catalog(total_budget=1000, seed=9215000)
    b = generate_catalog(total_budget=1000, seed=9215000)
    assert a == b
    assert len(a) == 1000
    assert len({x.fingerprint for x in a}) == 1000
    assert len({x.experiment_id for x in a}) == 1000


def test_budget_counts_are_complete_experiments_not_exit_expansion():
    catalog = generate_catalog(total_budget=500, seed=9215000)
    assert len(catalog) == 500
    assert all(x.strategy_name for x in catalog)
    assert all(x.direction_mode in {"long", "short", "combined"} for x in catalog)
    assert all(x.stop_atr in {0.5, 0.75, 1.0, 1.5, 2.0} for x in catalog)
    assert all(x.exit_type in {"target_r", "time", "atr_trail", "target_time"} for x in catalog)


def test_approved_budget_weights_sum_to_50000_guidance():
    assert sum(FAMILY_BUDGET_WEIGHTS.values()) == 50000
    assert FAMILY_BUDGET_WEIGHTS["trend_momentum"] == 8000
    assert FAMILY_BUDGET_WEIGHTS["exit_execution"] == 4000


def test_canonical_fingerprint_is_order_independent_and_uses_10_sig_digit_floats():
    left = canonical_fingerprint(
        strategy_name="demo",
        canonical_parameters={"b": 2, "a": 1.23456789004},
        direction_mode="long",
        stop_atr=1.0,
        exit_type="target_r",
        target_r=2.0,
        time_exit_minutes=None,
        atr_trail=None,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        strategy_seed=123,
        sampler_version="v1",
    )
    right = canonical_fingerprint(
        strategy_name="demo",
        canonical_parameters={"a": 1.23456789003, "b": 2},
        direction_mode="long",
        stop_atr=1.0,
        exit_type="target_r",
        target_r=2.0,
        time_exit_minutes=None,
        atr_trail=None,
        commission_round_trip_per_lot=6.0,
        slippage_points_per_fill=5.0,
        strategy_seed=123,
        sampler_version="v1",
    )
    assert left == right
    assert len(left) == 64


def test_catalog_stores_canonical_parameter_json():
    experiment = generate_catalog(total_budget=1, seed=9215000)[0]
    decoded = json.loads(experiment.canonical_parameters_json)
    assert decoded == experiment.parameters
    assert experiment.experiment_id == "EXP" + experiment.fingerprint[:12].upper()
