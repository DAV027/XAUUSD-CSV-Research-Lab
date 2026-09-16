from pathlib import Path

import scripts.create_edge_c_v2_catalog as create_cli
import scripts.run_edge_c_v2_activation as activation_cli


def test_edge_c_v2_catalog_cli_defaults_are_isolated():
    parser = create_cli.build_parser()
    args = parser.parse_args([])

    assert args.budget == 10_000
    assert args.seed == 9_216_300
    assert args.output == Path("edge_c_v2/results/EXPERIMENT_CATALOG.csv")


def test_write_edge_c_v2_catalog_uses_v2_generator(monkeypatch, tmp_path):
    calls = {}

    class FakeExperiment:
        def to_dict(self):
            return {"experiment_id": "EXP1", "family": "edge_c_breakout_retest_v2"}

    def fake_generate(*, total_budget, seed):
        calls["args"] = (total_budget, seed)
        return [FakeExperiment()]

    monkeypatch.setattr(create_cli, "generate_edge_c_v2_catalog", fake_generate)
    output = tmp_path / "edge_c_v2" / "results" / "EXPERIMENT_CATALOG.csv"
    count = create_cli.write_edge_c_v2_catalog(output, budget=7, seed=123)

    assert count == 1
    assert calls["args"] == (7, 123)
    assert output.exists()
    assert "edge_c_breakout_retest_v2" in output.read_text(encoding="utf-8")


def test_edge_c_v2_activation_cli_defaults_are_isolated():
    args = activation_cli.build_parser().parse_args([])

    assert args.catalog == Path("edge_c_v2/results/EXPERIMENT_CATALOG.csv")
    assert args.features == Path("data/features/XAUUSD_M1_FEATURES.parquet")
    assert args.output == Path("edge_c_v2/results/ACTIVATION_REPORT.json")
