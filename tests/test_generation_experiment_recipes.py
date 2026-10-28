"""Offline regressions for isolated, versioned generation recipes."""
import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from sentinel_crop_seg.core.pipeline.generation_config import load_generation_config, generation_config_report
from sentinel_crop_seg.cli import generate as generate_cli
from sentinel_crop_seg.cli.audit_generation_config import main as audit_main


def base_config(tmp_path):
    root = tmp_path / "config"
    root.mkdir()
    files = {
        "generation.yaml": {"generation": {"satellite": "sentinel2", "label_source": "eurocrops"}},
        "paths.yaml": {"paths": {"sentinel_data_root": "raw", "dataset_output_dir": "out"}},
        "eurocrops.yaml": {"eurocrops": {"shapefile_path": "a.shp"}},
        "catalog.yaml": {"catalog": {"stac": {"max_cloud_cover": 10, "limit": 50}}},
        "dataset.yaml": {"dataset": {"patch": {"size": 256, "stride": 128},
            "roi": {"strategy": "grid"}, "bands": {"B02": {"resolution": 10}},
            "target": {"class_id": None, "total_patches": 100},
            "max_cloud_fraction": 0.01}},
        "classes.yaml": {"classes": {
            "background": {"id": 0, "include": [], "exclude": []},
            "wheat": {"id": 1, "include": ["wheat"], "exclude": []}}},
        "sentinel.yaml": {"sentinel": {"cache": {"max_size_gb": 10}}},
    }
    for name, payload in files.items():
        (root / name).write_text(yaml.safe_dump(payload), encoding="utf-8")
    return root


def recipe(tmp_path, settings, *, version=1):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": version, "name": "demo", "settings": settings
    }), encoding="utf-8")
    return path


def test_recipe_precedence_isolation_and_source_integrity(tmp_path):
    root = base_config(tmp_path)
    (root / "zz_training.yaml").write_text("dataset:\n  patch:\n    size: 999\n")
    path = recipe(tmp_path, {"dataset": {"patch": {"size": 128}, "max_cloud_fraction": 0.005}})
    original = (root / "dataset.yaml").read_bytes()
    config = load_generation_config(root, mode="isolated", experiment_recipe=path,
                                    overrides=["dataset.max_cloud_fraction=0.002"])
    assert config["dataset"]["patch"] == {"size": 128, "stride": 128}
    assert config["dataset"]["max_cloud_fraction"] == 0.002
    assert config["experiment_recipe"]["name"] == "demo"
    assert (root / "dataset.yaml").read_bytes() == original
    assert load_generation_config(root, mode="legacy")["dataset"]["patch"]["size"] == 999


def test_preflight_reproducible_and_sensitive_to_overrides(tmp_path):
    root = base_config(tmp_path)
    path = recipe(tmp_path, {"dataset": {"max_cloud_fraction": 0.005}})
    one = generation_config_report(root, experiment_recipe=path)
    two = generation_config_report(root, experiment_recipe=path)
    three = generation_config_report(root, experiment_recipe=path,
                                     overrides=["dataset.max_cloud_fraction=0.004"])
    assert one["resolved_config_sha256"] == two["resolved_config_sha256"]
    assert one["resolved_config_sha256"] != three["resolved_config_sha256"]
    assert len(one["experiment_recipe"]["sha256"]) == 64


@pytest.mark.parametrize("settings, message", [
    ({"dataset": {"patch": {"sizze": 128}}}, "Unknown experiment setting"),
    ({"unexpected": {"key": 1}}, "Unknown recipe sections"),
    ({"dataset": {"patch": 32}}, "nonempty mapping"),
    ({"dataset": {}}, "nonempty mapping"),
    ({"classes": {"wheat": {"id": 1, "include": ["wheat"], "exclude": []}}}, "background=0"),
])
def test_rejects_invalid_recipe_settings(tmp_path, settings, message):
    root = base_config(tmp_path)
    with pytest.raises(ValueError, match=message):
        load_generation_config(root, mode="isolated", experiment_recipe=recipe(tmp_path, settings))


def test_taxonomy_replacement_without_mutation(tmp_path):
    root = base_config(tmp_path)
    custom = {"background": {"id": 0, "include": [], "exclude": []},
              "corn": {"id": 1, "include": ["corn"], "exclude": []}}
    config = load_generation_config(root, mode="isolated",
                                    experiment_recipe=recipe(tmp_path, {"classes": custom}))
    assert config["classes"] == custom
    assert "wheat" in load_generation_config(root, mode="isolated")["classes"]


def test_invalid_schema_and_legacy_mode(tmp_path):
    root = base_config(tmp_path)
    path = recipe(tmp_path, {"dataset": {"patch": {"size": 128}}}, version=2)
    with pytest.raises(ValueError, match="schema_version"):
        load_generation_config(root, mode="isolated", experiment_recipe=path)
    with pytest.raises(ValueError, match="isolated"):
        load_generation_config(root, mode="legacy", experiment_recipe=path)


def test_preflight_cli_uses_recipe_without_writing(tmp_path, monkeypatch, capsys):
    root = base_config(tmp_path)
    path = recipe(tmp_path, {"dataset": {"patch": {"size": 128}}})
    monkeypatch.setattr(sys, "argv", ["audit", "--config-dir", str(root),
                                      "--experiment", str(path)])
    audit_main()
    result = json.loads(capsys.readouterr().out)
    assert result["patch_size"] == 128
    assert result["experiment_recipe"]["name"] == "demo"


def test_generate_cli_defaults_for_new_and_legacy_runs(monkeypatch):
    calls = {}
    def stub(**kwargs):
        calls.update(kwargs)
        # search_failures is part of the result contract; zero here keeps this
        # an empty-but-healthy run, which is what this test is about.
        return SimpleNamespace(output_dir="out", total_samples=0, selected_rois=0,
            processed_products=0, strategy="grid", satellite="sentinel2",
            label_source="eurocrops", run_tag="x", processed_state_path="state",
            search_failures=0)
    monkeypatch.setattr(generate_cli, "generate_dataset", stub)
    monkeypatch.setattr(sys, "argv", ["generate", "--experiment", "recipe.yaml"])
    generate_cli.main()
    assert (calls["config_mode"], calls["resume_policy"]) == ("isolated", "strict")
    assert calls["experiment_recipe"] == "recipe.yaml"
    monkeypatch.setattr(sys, "argv", ["generate"])
    generate_cli.main()
    assert (calls["config_mode"], calls["resume_policy"]) == ("legacy", "legacy")
    assert calls["experiment_recipe"] is None
