"""Synthetic, purely local tests for the reusable geographical OOF planner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sentinel_crop_seg.core.splitting.spatial_oof_folds import (
    SpatialOOFConfig, plan_spatial_oof, run_spatial_oof, _guard_drop, _check_pairs,
)


def population(n: int = 9):
    rows = []
    for i in range(n):
        x = 10000 + 60000 * i
        part = "train" if i % 3 else "val"
        rows.append({
            "dataset": "AT_2021_oats_v2",
            "sample_id": f"s{i}", "cohort": "AT_2021_oats",
            "split": part, "partition": part, "window_id": f"W{i}",
            "final_spatial_tier": "EXACT", "epsg": 3035,
            "bbox_minx": x, "bbox_miny": 10000,
            "bbox_maxx": x + 2560, "bbox_maxy": 12560,
            "img_path": f"images/s{i}.npy", "mask_path": f"masks/s{i}.npy",
            "acquisition_date": "2021-07-01",
        })
    # Two acquisition dates of the SAME geographic window must never split.
    twin = rows[0].copy()
    twin["sample_id"] = "s0_second_date"
    twin["acquisition_date"] = "2021-08-01"
    rows.append(twin)
    test = rows[1].copy()
    test.update(sample_id="frozen_test", split="test", partition="test",
                window_id="TEST_WINDOW")
    rows.append(test)
    excluded = rows[2].copy()
    excluded.update(sample_id="weak_spatial", window_id="",
                    final_spatial_tier="INFERRED_TRAIN_ONLY")
    rows.append(excluded)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("folds", [2, 3, 4, 5, 6])
def test_configurable_k_keeps_test_out_and_every_eligible_item_held_once(folds):
    data = population()
    source = data.copy(deep=True)
    tables, summary = plan_spatial_oof(data, SpatialOOFConfig(folds=folds))
    assert data.equals(source)
    assert summary["status"] == "validated_fold_manifests_only_NO_OOF_predictions"
    assert summary["test_rows_reserved_untouched"] == 1
    assert summary["excluded_development_rows"] == 1
    assert summary["eligible_development_rows"] == 10
    assert summary["all_eligible_samples_held_out_exactly_once"]
    assert "frozen_test" not in str(tables)
    excluded = tables["oof_ineligible_development.csv"]
    assert excluded["sample_id"].tolist() == ["weak_spatial"]
    held = pd.concat([tables[f"fold_{fold:02d}_held_out_manifest.csv"] for fold in range(folds)])
    assert len(held) == 10
    assert held.duplicated(["dataset", "sample_id"]).sum() == 0
    assert held.loc[held["sample_id"].isin(["s0", "s0_second_date"]), "outer_fold_id"].nunique() == 1
    for fold in range(folds):
        train = tables[f"fold_{fold:02d}_training_manifest.csv"]
        oof = tables[f"fold_{fold:02d}_held_out_manifest.csv"]
        assert set(train["split"]) == {"train", "val"}
        assert set(oof["split"]) == {"held_out_fold"}
        assert not (set(train["unit_id"]) & set(oof["unit_id"]))
        roles = tables[f"fold_{fold:02d}_roles.csv"]
        assert len(roles) == 10 and not roles.duplicated(["dataset", "sample_id"]).any()
        assert "test" not in roles["split"].unique()


def test_optional_multiple_parcels_can_connect_cells_across_geography():
    data = population()
    links = pd.DataFrame([
        {"dataset": "AT_2021_oats_v2", "sample_id": "s1",
         "cohort": "AT_2021_oats", "parcel_key": "fid_2021_a", "partition": "train"},
        {"dataset": "AT_2021_oats_v2", "sample_id": "s5",
         "cohort": "AT_2021_oats", "parcel_key": "fid_2021_a", "partition": "train"},
    ])
    tables, report = plan_spatial_oof(data, SpatialOOFConfig(folds=3), parcel_links=links)
    assign = tables["fold_assignments.csv"].set_index("sample_id")
    assert assign.loc["s1", "outer_fold_id"] == assign.loc["s5", "outer_fold_id"]
    assert assign.loc["s1", "unit_id"] == assign.loc["s5", "unit_id"]
    assert report["parcel_linkage"]["eligible_samples_with_parcel_link"] == 2
    assert not report["parcel_linkage"]["parcel_level_coverage_complete"]


def test_reject_test_parcel_links_so_test_never_guides_grouping():
    links = pd.DataFrame([{
        "cohort": "AT_2021_oats", "parcel_key": "p", "sample_id": "frozen_test",
        "dataset": "AT_2021_oats_v2", "partition": "test"
    }])
    with pytest.raises(ValueError, match="TEST/nondevelopment"):
        plan_spatial_oof(population(), SpatialOOFConfig(folds=3), parcel_links=links)


def test_uncertified_geometry_missing_window_and_missing_geo_rejected_or_excluded():
    data = population()
    _, report = plan_spatial_oof(data, SpatialOOFConfig(folds=3))
    assert report["excluded_development_rows"] == 1
    data = data.drop(columns=["final_spatial_tier"])
    with pytest.raises(ValueError, match="final_spatial_tier"):
        plan_spatial_oof(data, SpatialOOFConfig(folds=3))
    data = population().drop(columns=["window_id"])
    with pytest.raises(ValueError, match="window_id"):
        plan_spatial_oof(data, SpatialOOFConfig(folds=3))


@pytest.mark.parametrize("folds", [1, 0, 33, 3.2, "6", True, None])
def test_invalid_fold_counts_fail_closed(folds):
    with pytest.raises(ValueError, match="folds must be"):
        plan_spatial_oof(population(), SpatialOOFConfig(folds=folds))


def test_more_folds_than_independent_geography_is_refused():
    with pytest.raises(ValueError, match="independent spatial units"):
        plan_spatial_oof(population(4), SpatialOOFConfig(folds=4))


def test_guard_can_only_drop_training_and_guard_violations_are_detected():
    roles = np.array(["held_out_fold", "outer_train", "outer_train"], dtype=object)
    pairs = np.array([[0, 1]], dtype=int)
    dropped = _guard_drop(roles, pairs, held="held_out_fold", training="outer_train")
    assert dropped.tolist() == [False, True, False]
    roles[dropped] = "excluded_outer_guard"
    _check_pairs(roles, pairs, "held_out_fold", "outer_train")
    with pytest.raises(ValueError, match="Geographic guard violation"):
        _check_pairs(np.array(["held_out_fold", "outer_train"], dtype=object),
                     pairs, "held_out_fold", "outer_train")


def test_source_partition_drift_rejected():
    data = population()
    data.loc[0, "partition"] = "test"
    with pytest.raises(ValueError, match="split/partition disagree"):
        plan_spatial_oof(data, SpatialOOFConfig(folds=3))


def test_no_sources_changed_and_validate_only_does_not_create_output(tmp_path):
    data = population()
    original = tmp_path / "frozen.csv"
    data.to_csv(original, index=False)
    cfg_file = tmp_path / "folds.yaml"
    cfg_file.write_text(yaml.safe_dump({"oof_folds": {
        "folds": 3, "seed": 31, "inner_val_fraction": 0.12,
    }}), encoding="utf-8")
    unchanged = original.read_bytes()
    out = tmp_path / "oof_new"
    preview = run_spatial_oof(cfg_file, original, out, validate_only=True)
    assert preview["folds"] == 3 and not out.exists()
    produced = run_spatial_oof(cfg_file, original, out)
    assert produced["input_sha256"]["manifest"] == preview["input_sha256"]["manifest"]
    assert (out / "oof_fold_audit.json").is_file()
    assert (out / "fold_02_held_out_manifest.csv").is_file()
    assert original.read_bytes() == unchanged
    archive = pd.read_csv(out / "fold_00_training_manifest.csv")
    assert archive["img_path"].iloc[0]  # rebased asset path
    assert set(archive["split"]) == {"train", "val"}
    assert not list(tmp_path.glob("oof_new.tmp"))
    with pytest.raises(FileExistsError, match="overwrite"):
        run_spatial_oof(cfg_file, original, out)


def test_config_yaml_unknown_controls_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("oof_folds:\n  folds: 6\n  leak_test: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        SpatialOOFConfig.from_yaml(path)


def test_no_inner_early_stopping_on_heldout_fold():
    tables, _ = plan_spatial_oof(population(), SpatialOOFConfig(folds=3))
    for f in range(3):
        train = tables[f"fold_{f:02d}_training_manifest.csv"]
        held = tables[f"fold_{f:02d}_held_out_manifest.csv"]
        assert not set(train["sample_id"]) & set(held["sample_id"])
        assert set(train["oof_role"]) == {"inner_val", "inner_train"}


def test_inadequate_geography_fails_before_any_output(tmp_path):
    data = population(2)
    source = tmp_path / "source.csv"
    data.to_csv(source, index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("oof_folds:\n  folds: 3\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="independent spatial units"):
        run_spatial_oof(cfg, source, out)
    assert not out.exists()



def test_linked_reserved_test_identity_is_blocked_despite_forged_train_label():
    links = pd.DataFrame([{
        "cohort": "AT_2021_oats", "parcel_key": "p",
        "dataset": "AT_2021_oats_v2", "sample_id": "frozen_test",
        "partition": "train",
    }])
    with pytest.raises(ValueError, match="reserved TEST sample"):
        plan_spatial_oof(population(), SpatialOOFConfig(folds=3), parcel_links=links)


def test_explicit_empty_or_unmatched_parcel_links_are_not_silently_ignored():
    with pytest.raises(ValueError, match="empty"):
        plan_spatial_oof(
            population(), SpatialOOFConfig(folds=3),
            parcel_links=pd.DataFrame(columns=[
                "cohort", "parcel_key", "dataset", "sample_id"
            ])
        )
    missing = pd.DataFrame([{
        "cohort": "AT_2021_oats", "parcel_key": "p",
        "dataset": "AT_2021_oats_v2", "sample_id": "nonexistent",
    }])
    with pytest.raises(ValueError, match="no eligible matching"):
        plan_spatial_oof(population(), SpatialOOFConfig(folds=3), parcel_links=missing)


def test_source_partition_is_retained_separately_from_inner_train_val():
    tables, _ = plan_spatial_oof(population(), SpatialOOFConfig(folds=3))
    for i in range(3):
        training = tables[f"fold_{i:02d}_training_manifest.csv"]
        assert (training["partition"] == training["split"]).all()
        assert set(training["source_partition"]) <= {"train", "val"}
        heldout = tables[f"fold_{i:02d}_held_out_manifest.csv"]
        assert set(heldout["source_partition"]) <= {"train", "val"}
        assert (heldout["partition"] == "held_out_fold").all()
