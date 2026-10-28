"""Synthetic checks for safe cross-cohort aggregation before spatial splitting."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from sentinel_crop_seg.cli import merge_cohorts
from sentinel_crop_seg.core.data.cohort_manifest_merge import merge_cohort_manifests
from sentinel_crop_seg.core.splitting.spatial_split import SpatialSplitConfig, split_dataframe


def cohort(tmp_path, dataset="AT_2021", country="AT", cohort_id="AT_2021", offset=0, count=6):
    source = tmp_path / dataset
    source.mkdir()
    rows = []
    for i in range(count):
        img = source / f"image_{i}.npy"
        mask = source / f"mask_{i}.npy"
        img.write_bytes(b"unmodified synthetic fixture")
        mask.write_bytes(b"unchanged mask fixture")
        x = 4000000 + offset + 85000 * i
        rows.append({
            "dataset": dataset, "sample_id": f"sample_{i:02d}",
            "country": country, "cohort": cohort_id, "epsg": "3035",
            "bbox_minx": str(x), "bbox_miny": "3000000",
            "bbox_maxx": str(x + 2560), "bbox_maxy": "3002560",
            "img_path": img.name, "mask_path": mask.name,
            "band_order": "B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12",
            "label_source": "eurocrops", "label_source_ref": f"{country}_SOURCE_2021",
            "p0": "300", "p3": "12" if country == "AT" else "0",
            "acquisition_date": "2021-06-01",
            "product_id": f"scene_{dataset}",
        })
        if country == "FR":
            rows[-1].pop("p3")
            rows[-1]["p6"] = "18"
    path = source / "adapted.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_merged_manifest_rebases_both_inputs_and_is_spatial_split_ready(tmp_path):
    first = cohort(tmp_path, count=6)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", offset=1500000, count=6)
    inputs = {p: p.read_bytes() for p in (first, second)}
    dest = tmp_path / "new" / "aggregate"
    preview = merge_cohort_manifests(
        [first, second], dest, verify_assets=True, validate_only=True
    )
    assert preview["status"] == "validated_only"
    assert preview["samples"] == 12
    assert preview["class_ids_in_histograms"] == [0, 3, 6]
    assert not dest.exists()
    result = merge_cohort_manifests(
        [first, second], dest, verify_assets=True, validate_only=False
    )
    assert result["status"] == "written"
    merged = pd.read_csv(dest / "combined_manifest.csv")
    assert set(merged["cohort"]) == {"AT_2021", "FR_2018"}
    assert int(merged.loc[merged["country"] == "FR", "p3"].sum()) == 0
    assert int(merged.loc[merged["country"] == "AT", "p6"].sum()) == 0
    for col in ("img_path", "mask_path"):
        assert all((dest / value).resolve().is_file() for value in merged[col])
    assert inputs == {p: p.read_bytes() for p in (first, second)}
    audit = json.loads((dest / "merge_audit.json").read_text(encoding="utf-8"))
    assert audit["split_assigned"] is False
    assert len(audit["source_manifests"]) == 2
    with pytest.raises(FileExistsError, match="Refusing"):
        merge_cohort_manifests([first, second], dest, validate_only=True)

    # One shared spatial partition is assigned AFTER aggregation.
    split, summary, leakage, _ = split_dataframe(
        merged, SpatialSplitConfig(candidate_seeds=100)
    )
    assert summary["leakage_audit_clean"]
    assert leakage["clean"]
    assert {"train", "val", "test"}.issubset(set(split["split"]))


def test_duplicate_dataset_identity_or_input_path_fails(tmp_path):
    first = cohort(tmp_path, count=2)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", 1000000, 2)
    frame = pd.read_csv(second, dtype=str)
    frame["dataset"] = "AT_2021"
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="distinct dataset and cohort"):
        merge_cohort_manifests([first, second], tmp_path / "output")
    with pytest.raises(ValueError, match="distinct"):
        merge_cohort_manifests([first, first], tmp_path / "output")


def test_mixed_band_order_and_split_partition_fail_without_outputs(tmp_path):
    first = cohort(tmp_path, count=2)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", 1000000, 2)
    frame = pd.read_csv(second, dtype=str)
    frame["band_order"] = "B03,B02"
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="band_order"):
        merge_cohort_manifests([first, second], tmp_path / "output")
    frame["band_order"] = "B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12"
    frame["split"] = "test"
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match="already split"):
        merge_cohort_manifests([first, second], tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(("field", "bad", "match"), [
    ("country", "fr", "country"), ("p0", "-4", "histogram"),
    ("p3", "4.1", "histogram"), ("bbox_maxx", "0", "bounds"),
    ("img_path", "", "empty img_path"),
    ("dataset", "", "empty dataset"),
])
def test_bad_provenance_and_histograms_are_rejected(tmp_path, field, bad, match):
    first = cohort(tmp_path, count=2)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", 1000000, 2)
    frame = pd.read_csv(second, dtype=str, keep_default_na=False)
    if field == "p3":
        frame["p6"] = bad
    else:
        frame[field] = bad
    frame.to_csv(second, index=False)
    with pytest.raises(ValueError, match=match):
        merge_cohort_manifests([first, second], tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_preview_and_apply_without_source_asset_modification(tmp_path, monkeypatch, capsys):
    first = cohort(tmp_path, count=2)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", 1000000, 2)
    dest = tmp_path / "merged"
    monkeypatch.setattr(sys, "argv", [
        "sentinel-crop-merge-cohorts", "--manifest", str(first),
        "--manifest", str(second), "--output-dir", str(dest), "--verify-assets",
    ])
    merge_cohorts.main()
    assert json.loads(capsys.readouterr().out)["status"] == "validated_only"
    assert not dest.exists()
    monkeypatch.setattr(sys, "argv", [
        "sentinel-crop-merge-cohorts", "--manifest", str(first),
        "--manifest", str(second), "--output-dir", str(dest), "--apply",
    ])
    merge_cohorts.main()
    assert json.loads(capsys.readouterr().out)["status"] == "written"
    assert (dest / "combined_manifest.csv").exists()


def test_missing_asset_is_rejected_before_any_output(tmp_path):
    first = cohort(tmp_path, count=1)
    second = cohort(tmp_path, "FR_2018", "FR", "FR_2018", 1000000, 1)
    (second.parent / "image_0.npy").unlink()
    with pytest.raises(FileNotFoundError, match="unresolved img_path"):
        merge_cohort_manifests([first, second], tmp_path / "merged", verify_assets=True)
    assert not (tmp_path / "merged").exists()
