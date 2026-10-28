import pytest
import yaml

from sentinel_crop_seg.core.config_loader import load_config


def write_yaml(path, data):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_load_config_merges_yaml_files_recursively(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    write_yaml(
        config_dir / "01_paths.yaml",
        {
            "paths": {
                "dataset_root": "data/datasets",
                "output_dir": "outputs",
            },
            "dataset": {
                "patch": {
                    "size": 256,
                    "stride": 128,
                }
            },
        },
    )

    write_yaml(
        config_dir / "02_override.yaml",
        {
            "paths": {
                "output_dir": "custom_outputs",
            },
            "dataset": {
                "patch": {
                    "stride": 64,
                },
                "bands": {
                    "B02": {"resolution": 10},
                },
            },
        },
    )

    config = load_config(str(config_dir))

    assert config["paths"]["dataset_root"] == "data/datasets"
    assert config["paths"]["output_dir"] == "custom_outputs"

    assert config["dataset"]["patch"]["size"] == 256
    assert config["dataset"]["patch"]["stride"] == 64
    assert config["dataset"]["bands"]["B02"]["resolution"] == 10


def test_load_config_rejects_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing_config"

    with pytest.raises(FileNotFoundError, match="Config directory not found"):
        load_config(str(missing_dir))
