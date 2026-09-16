from pathlib import Path

import scripts.run_edge_b_campaign as edge_b_cli


def test_edge_b_campaign_cli_defaults_are_isolated():
    args = edge_b_cli.build_parser().parse_args([])

    assert args.catalog == Path("edge_b/results/EXPERIMENT_CATALOG.csv")
    assert args.features == Path("data/features/XAUUSD_M1_FEATURES.parquet")
    assert args.result_root == Path("edge_b/results")
    assert args.seed == 9_216_000
    assert args.workers == 2
    assert args.limit is None
    assert args.allow_slow is False


def test_run_edge_b_campaign_wires_existing_engine(monkeypatch, tmp_path):
    calls = {}
    catalog = tmp_path / "edge_b" / "results" / "EXPERIMENT_CATALOG.csv"
    features = tmp_path / "features.parquet"
    result_root = tmp_path / "edge_b" / "results"

    def fake_guard(root, allow_slow):
        calls["guard"] = (root, allow_slow)

    def fake_manifest(**kwargs):
        calls["manifest_args"] = kwargs
        return {"manifest": True}

    def fake_write(root, manifest):
        calls["write"] = (root, manifest)

    def fake_run(catalog_path, feature_path, root, *, workers, limit):
        calls["run"] = (catalog_path, feature_path, root, workers, limit)

    monkeypatch.setattr(edge_b_cli, "_guard_slow_campaign", fake_guard)
    monkeypatch.setattr(edge_b_cli, "build_run_manifest", fake_manifest)
    monkeypatch.setattr(edge_b_cli, "write_or_validate_run_manifest", fake_write)
    monkeypatch.setattr(edge_b_cli, "run_campaign", fake_run)

    edge_b_cli.run_edge_b_campaign(
        catalog_path=catalog,
        feature_path=features,
        result_root=result_root,
        workers=2,
        seed=9_216_000,
        limit=25,
        allow_slow=True,
    )

    assert calls["guard"] == (result_root, True)
    assert calls["manifest_args"]["feature_path"] == features
    assert calls["manifest_args"]["catalog_path"] == catalog
    assert calls["manifest_args"]["workers"] == 2
    assert calls["manifest_args"]["campaign_seed"] == 9_216_000
    assert calls["write"] == (result_root, {"manifest": True})
    assert calls["run"] == (catalog, features, result_root, 2, 25)
