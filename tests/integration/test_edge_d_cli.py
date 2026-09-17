from pathlib import Path

import scripts.create_edge_d_catalog as create_cli


def test_edge_d_catalog_cli_defaults_are_isolated():
    parser = create_cli.build_parser()
    args = parser.parse_args([])

    assert args.budget == 10_000
    assert args.seed == 9_216_400
    assert args.output == Path("edge_d/results/EXPERIMENT_CATALOG.csv")


def test_write_edge_d_catalog_uses_edge_d_generator(monkeypatch, tmp_path):
    calls = {}

    class FakeExperiment:
        def to_dict(self):
            return {
                "experiment_id": "EXP1",
                "family": "edge_d_session_sweep_reclaim",
            }

    def fake_generate(*, total_budget, seed):
        calls["args"] = (total_budget, seed)
        return [FakeExperiment()]

    monkeypatch.setattr(create_cli, "generate_edge_d_catalog", fake_generate)
    output = tmp_path / "edge_d" / "results" / "EXPERIMENT_CATALOG.csv"
    count = create_cli.write_edge_d_catalog(output, budget=7, seed=123)

    assert count == 1
    assert calls["args"] == (7, 123)
    assert output.exists()
    assert not output.with_name(output.name + ".tmp").exists()
    assert "edge_d_session_sweep_reclaim" in output.read_text(encoding="utf-8")
