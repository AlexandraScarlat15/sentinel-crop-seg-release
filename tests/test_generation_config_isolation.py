"""Offline configuration-isolation regressions for dataset generation.

No Sentinel-2 downloads, EuroCrops vectors, output files or training runs.
The default legacy config merge stays identical; isolated mode is opt-in.
"""
from __future__ import annotations

import json
import sys

import pytest
import yaml

from sentinel_crop_seg.core.config_loader import load_config
from sentinel_crop_seg.core.pipeline.generation_config import (
    GENERATION_YAML_FILES,
    generation_config_report,
    load_generation_config,
)
from sentinel_crop_seg.cli.audit_generation_config import main as preflight_main


def write_minimal_generator_config(root):
    root.mkdir()
    payloads = {
        "generation.yaml": {"generation": {"satellite": "sentinel2", "label_source": "eurocrops"}},
        "paths.yaml": {"paths": {"sentinel_data_root": "data/raw/sentinel",
                                 "dataset_output_dir": "data/processed/generated"}},
        "eurocrops.yaml": {"eurocrops": {"shapefile_path": "data/external/country.shp"}},
        "catalog.yaml": {"catalog": {"stac": {"max_cloud_cover": 10, "limit": 10}}},
        "dataset.yaml": {"dataset": {
            "patch": {"size": 256, "stride": 128},
            "roi": {"strategy": "grid"}, "bands": {"B02": {"resolution": 10}},
            "target": {"class_id": None, "total_patches": 100},
            "max_cloud_fraction": 0.01}},
        "classes.yaml": {"classes": {"wheat": {"id": 3, "include": ["wheat"]}}},
        "sentinel.yaml": {"sentinel": {"cache": {"max_size_gb": 10}}},
    }
    assert set(payloads) == set(GENERATION_YAML_FILES)
    for filename, payload in payloads.items():
        (root / filename).write_text(yaml.safe_dump(payload), encoding="utf-8")
    return root


def test_isolated_ignores_unrelated_root_experiment_yaml(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    (root / "zz_training_experiment.yaml").write_text(
        "dataset:\n  patch:\n    size: 999\n  max_cloud_fraction: 0.50\n",
        encoding="utf-8",
    )
    legacy = load_generation_config(root, mode="legacy")
    isolated = load_generation_config(root, mode="isolated")
    assert legacy["dataset"]["patch"]["size"] == 999
    assert legacy["dataset"]["max_cloud_fraction"] == 0.5
    assert isolated["dataset"]["patch"]["size"] == 256
    assert isolated["dataset"]["max_cloud_fraction"] == 0.01


def test_legacy_loader_default_remains_unchanged(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    (root / "zz_training_experiment.yaml").write_text(
        "dataset:\n  patch:\n    stride: 8\n", encoding="utf-8"
    )
    assert load_config(root) == load_generation_config(root, mode="legacy")
    assert load_config(root)["dataset"]["patch"]["stride"] == 8


def test_explicit_file_precedence_and_dotted_overrides(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    result = load_config(root, yaml_files=["dataset.yaml", "classes.yaml"])
    assert "paths" not in result
    assert result["dataset"]["patch"]["size"] == 256
    isolated = load_generation_config(
        root, mode="isolated",
        overrides=["dataset.patch.size=128", "dataset.max_cloud_fraction=0.005"],
    )
    assert isolated["dataset"]["patch"]["size"] == 128
    assert isolated["dataset"]["patch"]["stride"] == 128
    assert isolated["dataset"]["max_cloud_fraction"] == 0.005


@pytest.mark.parametrize("filename", ["../secret.yaml", "/tmp/test.yaml", "nested/extra.yaml",
                                      "wrong.yml", "", r"dir\other.yaml"])
def test_explicit_loader_rejects_unsafe_or_unknown_path(tmp_path, filename):
    root = write_minimal_generator_config(tmp_path / "config")
    with pytest.raises(ValueError, match="Invalid explicit YAML"):
        load_config(root, yaml_files=[filename])


def test_explicit_loader_rejects_duplicate_files_and_missing_required(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(root, yaml_files=["dataset.yaml", "dataset.yaml"])
    (root / "classes.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="classes.yaml"):
        load_generation_config(root, mode="isolated")


def test_explicit_loader_rejects_malformed_top_level_yaml(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    (root / "classes.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_generation_config(root, mode="isolated")


def test_isolated_recipe_digest_is_repeatable_and_sensitive(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    first = generation_config_report(root)
    second = generation_config_report(root)
    modified = generation_config_report(root, overrides=["dataset.max_cloud_fraction=0.015"])
    assert first["resolved_config_sha256"] == second["resolved_config_sha256"]
    assert first["resolved_config_sha256"] != modified["resolved_config_sha256"]
    assert first["selected_config_files"] == list(GENERATION_YAML_FILES)
    assert first["bands"] == ["B02"]
    assert first["patch_size"] == 256
    assert len(first["resolved_config_sha256"]) == 64


def test_isolated_recipe_does_not_include_training_experiment(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    before = generation_config_report(root)
    (root / "zz_training_experiment.yaml").write_text(
        "dataset:\n  max_cloud_fraction: 0.5\n", encoding="utf-8"
    )
    after = generation_config_report(root)
    assert before["resolved_config_sha256"] == after["resolved_config_sha256"]
    legacy = generation_config_report(root, mode="legacy")
    assert legacy["resolved_config_sha256"] != after["resolved_config_sha256"]


def test_preflight_command_is_offline_and_outputs_summary(tmp_path, monkeypatch, capsys):
    root = write_minimal_generator_config(tmp_path / "config")
    monkeypatch.setattr(sys, "argv", [
        "sentinel-crop-audit-generation-config", "--config-dir", str(root),
        "--config-mode", "isolated", "--override", "dataset.patch.stride=64",
    ])
    preflight_main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["config_mode"] == "isolated"
    assert printed["stride"] == 64
    assert printed["bands"] == ["B02"]
    assert "run_tag alone" in " ".join(printed["notes"])
    assert not (root / "processed.json").exists()


def test_rejects_unknown_mode_and_invalid_hotcell_contract(tmp_path):
    root = write_minimal_generator_config(tmp_path / "config")
    with pytest.raises(ValueError, match="Unknown generation config mode"):
        load_generation_config(root, mode="unknown")
    (root / "dataset.yaml").write_text(yaml.safe_dump({
        "dataset": {
            "patch": {"size": 256, "stride": 128},
            "roi": {"strategy": "hotcell"},
            "bands": {"B02": {"resolution": 10}},
            "target": {"class_id": None},
        }
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="requires dataset.target.class_id"):
        generation_config_report(root)
