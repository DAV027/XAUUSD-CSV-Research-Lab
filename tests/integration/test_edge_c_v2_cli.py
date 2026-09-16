import hashlib
import json
from pathlib import Path

import pytest

import scripts.create_edge_c_v2_catalog as create_cli
import scripts.run_edge_c_v2_activation as activation_cli
import scripts.run_edge_c_v2_campaign as campaign_cli


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _activation_payload(catalog: Path, features: Path, *, passed=True):
    return {
        "schema_version": 1,
        "name": "edge_c_v2_activation",
        "passed": passed,
        "family": "edge_c_breakout_retest_v2",
        "strategy": "compression_breakout_retest_v2",
        "sampler_version": "edge_c_v2",
        "campaign_seed": 9_216_300,
        "catalog_budget": 10_000,
        "catalog_sha256": _sha(catalog),
        "source_data_sha256": _sha(features),
    }


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


def test_edge_c_v2_campaign_cli_defaults_are_isolated():
    args = campaign_cli.build_parser().parse_args([])

    assert args.catalog == Path("edge_c_v2/results/EXPERIMENT_CATALOG.csv")
    assert args.features == Path("data/features/XAUUSD_M1_FEATURES.parquet")
    assert args.result_root == Path("edge_c_v2/results")
    assert args.activation_report == Path("edge_c_v2/results/ACTIVATION_REPORT.json")
    assert args.seed == 9_216_300
    assert args.workers == 2
    assert args.limit is None
    assert args.allow_slow is False


def test_activation_report_guard_accepts_matching_pass_and_rejects_missing_malformed_or_fail(tmp_path):
    catalog = tmp_path / "catalog.csv"
    features = tmp_path / "features.parquet"
    report = tmp_path / "ACTIVATION_REPORT.json"
    catalog.write_text("catalog", encoding="utf-8")
    features.write_text("features", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        campaign_cli.validate_activation_report(report, catalog, features)

    report.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        campaign_cli.validate_activation_report(report, catalog, features)

    payload = _activation_payload(catalog, features, passed=False)
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not pass"):
        campaign_cli.validate_activation_report(report, catalog, features)

    payload["passed"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert campaign_cli.validate_activation_report(report, catalog, features) == payload


def test_activation_report_guard_rejects_wrong_identity_or_artifact_hash(tmp_path):
    catalog = tmp_path / "catalog.csv"
    features = tmp_path / "features.parquet"
    report = tmp_path / "ACTIVATION_REPORT.json"
    catalog.write_text("catalog", encoding="utf-8")
    features.write_text("features", encoding="utf-8")

    payload = _activation_payload(catalog, features)
    payload["sampler_version"] = "wrong"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        campaign_cli.validate_activation_report(report, catalog, features)

    payload = _activation_payload(catalog, features)
    payload["catalog_sha256"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog"):
        campaign_cli.validate_activation_report(report, catalog, features)


def test_run_edge_c_v2_campaign_validates_activation_then_wires_existing_engine(monkeypatch, tmp_path):
    calls = []
    catalog = tmp_path / "catalog.csv"
    features = tmp_path / "features.parquet"
    result_root = tmp_path / "edge_c_v2" / "results"
    activation_report = result_root / "ACTIVATION_REPORT.json"

    def fake_validate(report, catalog_path, feature_path):
        calls.append(("activation", report, catalog_path, feature_path))
        return {"passed": True}

    def fake_guard(root, allow_slow):
        calls.append(("guard", root, allow_slow))

    def fake_manifest(**kwargs):
        calls.append(("manifest", kwargs))
        return {"manifest": True}

    def fake_write(root, manifest):
        calls.append(("write", root, manifest))

    def fake_run(catalog_path, feature_path, root, *, workers, limit):
        calls.append(("run", catalog_path, feature_path, root, workers, limit))

    monkeypatch.setattr(campaign_cli, "validate_activation_report", fake_validate)
    monkeypatch.setattr(campaign_cli, "_guard_slow_campaign", fake_guard)
    monkeypatch.setattr(campaign_cli, "build_run_manifest", fake_manifest)
    monkeypatch.setattr(campaign_cli, "write_or_validate_run_manifest", fake_write)
    monkeypatch.setattr(campaign_cli, "run_campaign", fake_run)

    campaign_cli.run_edge_c_v2_campaign(
        catalog_path=catalog,
        feature_path=features,
        result_root=result_root,
        activation_report=activation_report,
        workers=2,
        seed=9_216_300,
        limit=25,
        allow_slow=True,
    )

    assert calls[0] == ("activation", activation_report, catalog, features)
    assert ("guard", result_root, True) in calls
    assert ("run", catalog, features, result_root, 2, 25) in calls
