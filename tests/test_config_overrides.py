from __future__ import annotations

from pathlib import Path

import pytest

from sentinel_crop_seg.core.config_loader import apply_overrides, parse_override_value


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
        ],
    )

    assert config["dataset"]["target"]["total_patches"] == 8
    assert config["training"]["epochs"] == 1
    assert config["loader"]["pin_memory"] is False


def test_apply_overrides_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected format"):
        apply_overrides({}, ["dataset.target.total_patches"])

def test_a_mistyped_override_path_is_rejected_instead_of_creating_a_subtree() -> None:
    """A dotted path used to create whatever it named.

    So `--override dataset.targe.total_patches=8` built a `targe` section
    nothing reads, the real `total_patches` kept its configured value, and the
    run proceeded and reported success. Same failure shape as the converter
    defects: plausible artifact, wrong setting, no error anywhere.
    """
    config = {"dataset": {"target": {"total_patches": 10000}}}

    with pytest.raises(ValueError, match="Unknown configuration path"):
        apply_overrides(config, ["dataset.targe.total_patches=8"])

    assert "targe" not in config["dataset"], "a rejected override must write nothing"
    assert config["dataset"]["target"]["total_patches"] == 10000


def test_a_mistyped_final_key_is_rejected_too() -> None:
    """Intermediate sections are not the only place a typo lands.

    Checking only the path prefix would accept `dataset.target.total_patchs`,
    which is the likelier mistake of the two.
    """
    config = {"dataset": {"target": {"total_patches": 10000}}}

    with pytest.raises(ValueError, match="Unknown configuration key"):
        apply_overrides(config, ["dataset.target.total_patchs=8"])

    assert set(config["dataset"]["target"]) == {"total_patches"}


def test_an_unknown_top_level_section_is_rejected() -> None:
    config = {"dataset": {"target": {"total_patches": 10000}}}

    with pytest.raises(ValueError, match="Unknown configuration path"):
        apply_overrides(config, ["trainng.epochs=1"])

    assert set(config) == {"dataset"}


def test_the_error_names_the_keys_that_do_exist() -> None:
    """A rejection is only useful if it says what the right spelling was."""
    config = {"dataset": {"target": {"total_patches": 10000, "stride": 128}}}

    with pytest.raises(ValueError) as caught:
        apply_overrides(config, ["dataset.target.total_patchs=8"])

    message = str(caught.value)
    assert "stride" in message and "total_patches" in message


def test_a_path_through_a_non_mapping_is_rejected_under_any_permission() -> None:
    """No value can live inside a scalar, so this is never a permissions
    question.

    It must fail identically whether or not new keys are authorized,
    otherwise `allow_new_keys` would look like a way to make it work.
    """
    for permitted in (False, True):
        config = {"dataset": {"target": 10000}}
        with pytest.raises(ValueError, match="not a section"):
            apply_overrides(
                config,
                ["dataset.target.total_patches=8"],
                allow_new_keys=permitted,
            )
        assert config["dataset"]["target"] == 10000


def test_an_empty_path_component_is_rejected() -> None:
    config = {"dataset": {"target": {"total_patches": 1}}}

    with pytest.raises(ValueError, match="Empty path component"):
        apply_overrides(config, ["dataset..total_patches=8"])


def test_a_declared_optional_path_is_accepted_though_the_file_omits_it() -> None:
    """Strictness must not reject a setting the consumer reads with a default.

    A configuration file is a subset of the setting space wherever the reader
    supplies a fallback, so validating against the file alone would refuse
    real settings. The caller declares those paths instead.
    """
    config = {"data": {"in_channels": 10}}

    apply_overrides(
        config,
        ["data.parcel_boundary_cache_manifest=some/cache.json"],
        known_paths=frozenset({"data.parcel_boundary_cache_manifest"}),
    )

    assert config["data"]["parcel_boundary_cache_manifest"] == "some/cache.json"

    # A typo near a declared path is still rejected.
    with pytest.raises(ValueError, match="Unknown configuration key"):
        apply_overrides(
            config,
            ["data.parcel_boundary_cache_manifes=some/cache.json"],
            known_paths=frozenset({"data.parcel_boundary_cache_manifest"}),
        )


def test_an_explicitly_authorized_new_key_still_works() -> None:
    """The permissive path remains available for a caller that means it."""
    config = {"dataset": {}}

    apply_overrides(config, ["dataset.extra.value=3"], allow_new_keys=True)

    assert config["dataset"]["extra"]["value"] == 3


def test_the_trainers_declared_optional_paths_match_what_it_actually_reads() -> None:
    """Derive the set from the source, so a hand-written list cannot drift.

    A declared path that is no longer read would silently keep accepting a
    dead override; a read path that is not declared would reject a legitimate
    one the moment a user tried it. Both directions are checked here rather
    than discovered in a run.
    """
    import re

    from sentinel_crop_seg.core.training.train_unet import (
        KNOWN_OPTIONAL_OVERRIDE_PATHS,
    )

    source = (
        Path(__file__).resolve().parents[1]
        / "sentinel_crop_seg" / "core" / "training" / "train_unet.py"
    ).read_text(encoding="utf-8")

    read_with_default = {
        f"{section}.{key}"
        for section, key in re.findall(
            r'raw\["(\w+)"\]\s*\.\s*get\(\s*\n?\s*"(\w+)"', source
        )
    }
    sections = {"loss": "loss", "loader": "loader", "labels": "labels"}
    read_with_default |= {
        f"{sections[var]}.{key}"
        for var, key in re.findall(
            r'(\w+)_cfg\s*\.\s*get\(\s*\n?\s*"(\w+)"', source
        )
        if var in sections
    }

    assert read_with_default, "the scan must find the reads it is checking"
    assert KNOWN_OPTIONAL_OVERRIDE_PATHS == read_with_default, (
        "declared but not read: "
        f"{sorted(KNOWN_OPTIONAL_OVERRIDE_PATHS - read_with_default)}; "
        "read but not declared: "
        f"{sorted(read_with_default - KNOWN_OPTIONAL_OVERRIDE_PATHS)}"
    )
