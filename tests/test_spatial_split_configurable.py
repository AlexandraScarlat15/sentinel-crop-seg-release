"""Data-free tests for the reusable spatial splitting API and CLI contract."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel_crop_seg.core.splitting.spatial_split import (
    SpatialSplitConfig,
    allocation_units,
    build_windows,
    conflict_pairs,
    exclude_conflicts,
    minimum_footprint_gap,
    prepare_manifest,
    run_from_files,
    split_dataframe,
)


def fixture_manifest(n: int = 12) -> pd.DataFrame:
    """Well-separated 30 km blocks in EPSG:3035; two dates per footprint."""
    rows = []
    for i in range(n):
        x = 4_000_000.0 + i * 80_000.0
        for month in (4, 5):
            rows.append({
                "dataset": "AT_2021", "sample_id": f"s{i}_{month}",
                "epsg": 3035, "bbox_minx": x, "bbox_miny": 3_000_000.0,
                "bbox_maxx": x + 2560.0, "bbox_maxy": 3_002_560.0,
                "country": "AT", "cohort": "AT_2021",
                "acquisition_date": f"2021-{month:02d}-01",
                "parcel_uid": f"parcel-{i}",
                "p0": 100, "p1": i + 1,
            })
    return pd.DataFrame(rows)


def test_config_defaults_and_invalid_guard(tmp_path):
    config = tmp_path / "spatial.yaml"
    config.write_text("splitting:\n  strategy: spatial_block\n", encoding="utf-8")
    assert SpatialSplitConfig.from_yaml(config).guard_distance_m == 1280
    config.write_text("splitting:\n  spatial:\n    guard_distance_m: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="guard_distance"):
        SpatialSplitConfig.from_yaml(config)
    config.write_text("splitting:\n  spatial:\n    missing_knob: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="splitting.spatial"):
        SpatialSplitConfig.from_yaml(config)


def test_config_rejects_disabling_leakage_or_window_atomicity(tmp_path):
    cfg = tmp_path / "split.yaml"
    cfg.write_text("splitting:\n  quality:\n    fail_on_leakage: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Leakage"):
        SpatialSplitConfig.from_yaml(cfg)
    cfg.write_text("splitting:\n  grouping:\n    same_window_same_partition: false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="atomicity"):
        SpatialSplitConfig.from_yaml(cfg)


def test_config_rejects_non_metric_target_crs():
    with pytest.raises(ValueError, match="metric|metre|projected"):
        SpatialSplitConfig(crs="EPSG:4326").validate()


def test_guard_checks_footprints_not_only_centres():
    windows = pd.DataFrame({
        "x": [0.0, 3000.0], "y": [0.0, 0.0],
        "half_x": [1000.0, 1000.0], "half_y": [1000.0, 1000.0],
    })
    assert len(conflict_pairs(windows, 999.0)) == 0
    assert len(conflict_pairs(windows, 1000.0)) == 0
    assert len(conflict_pairs(windows, 1280.0)) == 1
    assert len(conflict_pairs(windows, 2560.0)) == 1


def test_touching_footprints_conflict_even_if_guard_is_zero():
    windows = pd.DataFrame({
        "x": [0.0, 2000.0], "y": [0.0, 0.0],
        "half_x": [1000.0, 1000.0], "half_y": [1000.0, 1000.0],
    })
    assert len(conflict_pairs(windows, 0.0)) == 1


def test_minimum_footprint_gap_not_nearest_centroid():
    windows = pd.DataFrame({
        "x": [0.0, 300.0, 500.0], "y": [0.0] * 3,
        "half_x": [100.0, 10.0, 400.0], "half_y": [100.0, 10.0, 400.0],
    })
    assert minimum_footprint_gap(windows, np.array([0]), np.array([1, 2])) == 0.0


def test_priority_protects_test_and_removes_validation_before_training():
    dropped = exclude_conflicts(
        np.array(["train", "val", "test"], dtype=object),
        np.array([[0, 1], [1, 2]], dtype=int),
    )
    assert dropped.tolist() == [False, True, False]


def test_same_geographic_footprint_across_dates_is_atomic():
    data = prepare_manifest(fixture_manifest(1), SpatialSplitConfig())
    windows = build_windows(data, SpatialSplitConfig())
    assert len(windows) == 1
    assert windows.iloc[0]["sample_count"] == 2


def test_parcel_identity_can_group_different_geographic_cells():
    cfg = SpatialSplitConfig(parcel_id_column="parcel_uid")
    data = fixture_manifest(4)
    data.loc[data["sample_id"].str.startswith("s1_"), "parcel_uid"] = "parcel-0"
    data = prepare_manifest(data, cfg)
    windows = allocation_units(data, build_windows(data, cfg), cfg)
    assert windows["unit_id"].nunique() == 3
    shared_cells = windows[windows["_window"].isin(data.loc[
        data["parcel_uid"] == "parcel-0", "_window"
    ])]
    assert shared_cells["unit_id"].nunique() == 1


def test_missing_parcel_identity_fails_closed():
    data = fixture_manifest().drop(columns="parcel_uid")
    with pytest.raises(ValueError, match="parcel_id_column"):
        split_dataframe(data, SpatialSplitConfig(parcel_id_column="parcel_uid"))


def test_integration_split_preserves_dates_and_has_zero_leakage():
    data = fixture_manifest()
    result, summary, audit, classes = split_dataframe(data, SpatialSplitConfig(candidate_seeds=30))
    assert audit["clean"]
    assert set(result["partition"]) == {"train", "val", "test"}
    assert len(result) == len(data)
    assert result.groupby("bbox_minx")["partition"].nunique().max() == 1
    assert sum(summary["partitions"][p]["samples"] for p in ("train", "val", "test")) == len(data)
    assert not classes.empty
    assert set(classes["class_id"]) == {0, 1}
    assert all(row["minimum_footprint_gap_m"] >= 1280 for row in audit["pairs"].values())


def test_integration_is_reproducible_across_input_row_order():
    cfg = SpatialSplitConfig(candidate_seeds=30)
    first, _, _, _ = split_dataframe(fixture_manifest(), cfg)
    shuffled, _, _, _ = split_dataframe(fixture_manifest().sample(frac=1, random_state=9), cfg)
    a = first.set_index("sample_id")["partition"].sort_index()
    b = shuffled.set_index("sample_id")["partition"].sort_index()
    pd.testing.assert_series_equal(a, b)


def test_rejects_ambiguous_ids_and_nonpositive_bounding_box():
    data = fixture_manifest(1)
    data.loc[1, "sample_id"] = data.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="unique"):
        prepare_manifest(data, SpatialSplitConfig())
    data = fixture_manifest(1)
    data.loc[0, "bbox_maxx"] = data.loc[0, "bbox_minx"]
    with pytest.raises(ValueError, match="positive"):
        prepare_manifest(data, SpatialSplitConfig())


def test_refuses_infeasible_three_way_split():
    with pytest.raises(ValueError, match="three geographical cells"):
        split_dataframe(fixture_manifest(2), SpatialSplitConfig())


def test_file_workflow_writes_audit_and_never_overwrites(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("splitting:\n  strategy: spatial_block\n  allocation:\n    candidate_seeds: 30\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    source = fixture_manifest()
    source["split"] = "historical"
    source.to_csv(manifest, index=False)
    out = tmp_path / "result"
    preview = run_from_files(config, manifest, out, validate_only=True)
    assert preview["samples_input"] == len(source)
    assert not out.exists()
    result = run_from_files(config, manifest, out)
    assert result["leakage_audit_clean"]
    assert json.loads((out / "leakage_audit.json").read_text())["clean"]
    csv = pd.read_csv(out / "split_manifest.csv")
    assert set(csv["source_split"]) == {"historical"}
    assert set(csv["split"]) == {"train", "val", "test"}
    assert (out / "class_distribution.csv").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        run_from_files(config, manifest, out)


def test_optional_map_output_is_created(tmp_path):
    config = tmp_path / "with_map.yaml"
    config.write_text(
        "splitting:\n  allocation:\n    candidate_seeds: 12\n"
        "  outputs:\n    save_split_map: true\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "samples.csv"
    fixture_manifest().to_csv(manifest, index=False)
    out = tmp_path / "mapped"
    run_from_files(config, manifest, out)
    assert (out / "spatial_split_map.png").stat().st_size > 0
