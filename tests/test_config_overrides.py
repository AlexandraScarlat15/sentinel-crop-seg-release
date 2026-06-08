from __future__ import annotations

import pytest

from src.config_loader import apply_overrides, parse_override_value


def test_parse_override_value_uses_yaml_types() -> None:
    assert parse_override_value("8") == 8
    assert parse_override_value("0.5") == 0.5
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("null") is None
    assert parse_override_value("[0, 1, 2]") == [0, 1, 2]
    assert parse_override_value("data/processed/example") == "data/processed/example"


def test_apply_overrides_updates_nested_values() -> None:
    config = {
        "dataset": {
            "target": {
                "total_patches": 10000,
            }
        },
        "training": {
            "epochs": 260,
        },
        "loader": {
            "pin_memory": True,
        },
    }

    apply_overrides(
        config,
        [
            "dataset.target.total_patches=8",
            "training.epochs=1",
            "loader.pin_memory=false",
            "paths.manifest_csv=null",
        ],
    )

    assert config["dataset"]["target"]["total_patches"] == 8
    assert config["training"]["epochs"] == 1
    assert config["loader"]["pin_memory"] is False
    assert config["paths"]["manifest_csv"] is None


def test_apply_overrides_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected format"):
        apply_overrides({}, ["dataset.target.total_patches"])