from __future__ import annotations

from pathlib import Path

import pytest

from sentinel_crop_seg.core.evaluation.metadata_filters import (
    month_is_allowed,
    parse_scene_month,
    read_allowed_manifest_sample_keys,
    read_allowed_sample_names_by_month,
    validate_month_filter_args,
    resolve_sample_name_column,
)


def test_month_is_allowed_for_normal_interval():
    assert month_is_allowed(4, 4, 9) is True
    assert month_is_allowed(9, 4, 9) is True
    assert month_is_allowed(3, 4, 9) is False
    assert month_is_allowed(10, 4, 9) is False


def test_month_is_allowed_for_wraparound_interval():
    assert month_is_allowed(11, 11, 3) is True
    assert month_is_allowed(1, 11, 3) is True
    assert month_is_allowed(3, 11, 3) is True
    assert month_is_allowed(6, 11, 3) is False


def test_parse_scene_month():
    assert parse_scene_month("2021-08-10") == 8
    assert parse_scene_month("2021-12-31T10:30:00") == 12

    with pytest.raises(ValueError):
        parse_scene_month("not-a-date")


def test_read_allowed_sample_names_by_month(tmp_path: Path):
    metadata_csv = tmp_path / "metadata.csv"

    metadata_csv.write_text(
        "\n".join(
            [
                "sample_name,scene_date",
                "sample_000001.npy,2021-04-15",
                "sample_000002.npy,2021-08-20",
                "sample_000003.npy,2021-11-05",
            ]
        ),
        encoding="utf-8",
    )

    allowed = read_allowed_sample_names_by_month(
        metadata_csv,
        month_start=4,
        month_end=9,
    )

    assert allowed == {
        "sample_000001.npy",
        "sample_000002.npy",
    }


def test_read_allowed_sample_names_by_month_requires_columns(tmp_path: Path):
    metadata_csv = tmp_path / "metadata.csv"
    metadata_csv.write_text("sample,date\nsample_000001.npy,2021-04-15\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_allowed_sample_names_by_month(
            metadata_csv,
            month_start=4,
            month_end=9,
        )


def test_validate_month_filter_args():
    assert validate_month_filter_args(
        metadata_csv=None,
        month_start=None,
        month_end=None,
    ) is False

    assert validate_month_filter_args(
        metadata_csv="metadata.csv",
        month_start=4,
        month_end=9,
    ) is True

    with pytest.raises(ValueError):
        validate_month_filter_args(
            metadata_csv=None,
            month_start=4,
            month_end=9,
        )

    with pytest.raises(ValueError):
        validate_month_filter_args(
            metadata_csv="metadata.csv",
            month_start=4,
            month_end=None,
        )

    with pytest.raises(ValueError):
        validate_month_filter_args(
            metadata_csv="metadata.csv",
            month_start=13,
            month_end=9,
        )
        
def test_resolve_sample_name_column_prefers_sample_name():
    assert resolve_sample_name_column(["sample_name", "sample_id", "scene_date"]) == "sample_name"


def test_resolve_sample_name_column_accepts_sample_id():
    assert resolve_sample_name_column(["sample_id", "scene_date"]) == "sample_id"


def test_read_allowed_sample_names_by_month_accepts_sample_id_column(tmp_path: Path):
    metadata_csv = tmp_path / "patches_log.csv"

    metadata_csv.write_text(
        "\n".join(
            [
                "sample_id,scene_date",
                "sample_000001,2021-04-15",
                "sample_000002,2021-08-20",
                "sample_000003,2021-11-05",
            ]
        ),
        encoding="utf-8",
    )

    allowed = read_allowed_sample_names_by_month(
        metadata_csv,
        month_start=4,
        month_end=9,
    )

    assert allowed == {
        "sample_000001",
        "sample_000002",
    }


def test_read_allowed_manifest_sample_keys_normalizes_patch_suffix(tmp_path: Path):
    sidecar = tmp_path / "phenology.csv"
    sidecar.write_text(
        "\n".join(
            [
                "dataset,sample_id,phenology_group",
                "AT_2021_beet_v2,AT_2021_beet_v2/sample_000001.npy,WEIGHTED90",
                "AT_2021_beet_v2,sample_000002,BELOW90",
            ]
        ),
        encoding="utf-8",
    )

    assert read_allowed_manifest_sample_keys(sidecar, group="WEIGHTED90") == {
        ("AT_2021_beet_v2", "sample_000001")
    }
