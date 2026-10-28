"""Integration regressions for spatial-split output and the existing NPY loader.

The test files are synthetic 10-band images and integer masks; no source
datasets, model artifacts, or historical split files are accessed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel_crop_seg.core.splitting.spatial_split import (
    SpatialSplitConfig,
    rebase_asset_paths,
    run_from_files,
    split_dataframe,
)
from sentinel_crop_seg.core.training.dataset_manifest import ManifestSegmentationDataset


def make_source_manifest(tmp_path: Path, *, absolute_paths: bool = False) -> tuple[Path, pd.DataFrame]:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    rows = []
    for index in range(12):
        sample_id = f"patch_{index:03d}"
        image = root / "images" / f"{sample_id}.npy"
        mask = root / "masks" / f"{sample_id}.npy"
        np.save(image, np.full((10, 8, 8), index + 1, dtype=np.float32))
        np.save(mask, np.full((8, 8), index % 11, dtype=np.uint8))
        x = 4_000_000.0 + index * 80_000.0
        rows.append({
            "dataset": "synthetic", "sample_id": sample_id,
            "img_path": str(image.resolve()) if absolute_paths else f"images/{sample_id}.npy",
            "mask_path": str(mask.resolve()) if absolute_paths else f"masks/{sample_id}.npy",
            "epsg": 3035, "bbox_minx": x, "bbox_maxx": x + 2560,
            "bbox_miny": 3_000_000.0, "bbox_maxy": 3_002_560.0,
            "country": "AT", "cohort": "AT_2021",
            "partition": ("train", "val", "test")[index % 3],
            "split": ("val", "test", "train")[index % 3],
        })
    manifest = root / "source_manifest.csv"
    original = pd.DataFrame(rows)
    original.to_csv(manifest, index=False)
    return manifest, original


def write_settings(tmp_path: Path) -> Path:
    settings = tmp_path / "split.yaml"
    settings.write_text(
        "splitting:\n"
        "  allocation:\n"
        "    random_seed: 42\n"
        "    candidate_seeds: 30\n",
        encoding="utf-8",
    )
    return settings


def test_in_memory_split_retains_both_historical_partition_columns(tmp_path):
    manifest, original = make_source_manifest(tmp_path)
    result, summary, audit, _ = split_dataframe(
        original, SpatialSplitConfig(candidate_seeds=30)
    )
    keyed = result.set_index("sample_id")
    old = original.set_index("sample_id")
    assert keyed["source_partition"].to_dict() == old["partition"].to_dict()
    assert keyed["source_split"].to_dict() == old["split"].to_dict()
    assert keyed["partition"].isin(["train", "val", "test", "excluded"]).all()
    assert keyed["split"].equals(keyed["partition"])
    assert audit["clean"] and summary["leakage_audit_clean"]


def test_output_manifest_rebases_relative_paths_and_loads_every_partition(tmp_path):
    manifest, original = make_source_manifest(tmp_path)
    config = write_settings(tmp_path)
    destination = tmp_path / "results" / "nested" / "spatial_split"

    source_bytes = manifest.read_bytes()
    image_bytes = (manifest.parent / original.iloc[0]["img_path"]).read_bytes()
    mask_bytes = (manifest.parent / original.iloc[0]["mask_path"]).read_bytes()

    summary = run_from_files(config, manifest, destination)
    produced = pd.read_csv(destination / "split_manifest.csv")
    assert summary["leakage_audit_clean"]
    assert len(produced) == len(original)
    assert "source_partition" in produced and "source_split" in produced

    keyed = produced.set_index("sample_id")
    for record in original.itertuples(index=False):
        row = keyed.loc[record.sample_id]
        assert row["source_partition"] == record.partition
        assert row["source_split"] == record.split
        for column in ("img_path", "mask_path"):
            written = Path(str(row[column]))
            assert not written.is_absolute()
            resolved = (destination / written).resolve()
            expected = (manifest.parent / getattr(record, column)).resolve()
            assert resolved == expected
            assert resolved.is_file()

    for partition in ("train", "val", "test"):
        loader = ManifestSegmentationDataset(
            manifest_csv=destination / "split_manifest.csv",
            split=partition, shuffle=False, limit=1,
        )
        assert len(loader) == 1
        image, mask = loader[0]
        assert image.shape == (10, 8, 8)
        assert mask.shape == (8, 8)

    assert manifest.read_bytes() == source_bytes
    assert (manifest.parent / original.iloc[0]["img_path"]).read_bytes() == image_bytes
    assert (manifest.parent / original.iloc[0]["mask_path"]).read_bytes() == mask_bytes


def test_absolute_paths_remain_unchanged(tmp_path):
    manifest, original = make_source_manifest(tmp_path, absolute_paths=True)
    output = tmp_path / "other" / "output"
    rebased = rebase_asset_paths(original, manifest, output)
    assert rebased["img_path"].tolist() == original["img_path"].tolist()
    assert rebased["mask_path"].tolist() == original["mask_path"].tolist()


@pytest.mark.parametrize(
    ("original", "preserved"),
    [
        ("partition", "source_partition"),
        ("split", "source_split"),
    ],
)
def test_conflicting_historical_columns_fail_before_output(tmp_path, original, preserved):
    manifest, data = make_source_manifest(tmp_path)
    data[preserved] = "already_present"
    data.to_csv(manifest, index=False)
    output = tmp_path / "new_split"
    with pytest.raises(ValueError, match=f"Cannot preserve existing {original}"):
        run_from_files(write_settings(tmp_path), manifest, output)
    assert not output.exists()


def test_blank_relative_asset_path_refuses_output_without_changes(tmp_path):
    manifest, data = make_source_manifest(tmp_path)
    data.loc[0, "mask_path"] = ""
    data.to_csv(manifest, index=False)
    output = tmp_path / "new_split"
    settings = write_settings(tmp_path)
    with pytest.raises(ValueError, match="mask_path contains an empty asset path"):
        run_from_files(settings, manifest, output, validate_only=True)
    assert not output.exists()
    with pytest.raises(ValueError, match="mask_path contains an empty asset path"):
        run_from_files(settings, manifest, output)
    assert not output.exists()


def test_foreign_windows_absolute_paths_are_not_misinterpreted_as_relative(tmp_path):
    source = tmp_path / "source.csv"
    paths = pd.DataFrame({
        "img_path": [r"D:\\remote\\images\\sample.npy"],
        "mask_path": [r"\\\\server\\dataset\\sample_mask.npy"],
    })
    result = rebase_asset_paths(paths, source, tmp_path / "output")
    assert result["img_path"].tolist() == paths["img_path"].tolist()
    assert result["mask_path"].tolist() == paths["mask_path"].tolist()


def test_windows_style_relative_asset_paths_work_across_platforms(tmp_path):
    source = tmp_path / "inputs" / "manifest.csv"
    output = tmp_path / "outputs" / "new_split"
    manifest = pd.DataFrame({
        "img_path": [r"images\\sample.npy"],
        "mask_path": [r"masks\\sample.npy"],
    })
    result = rebase_asset_paths(manifest, source, output)
    for kind in ("img_path", "mask_path"):
        directory = "images" if kind == "img_path" else "masks"
        expected = (source.parent / directory / "sample.npy").resolve()
        assert (output / result.loc[0, kind]).resolve() == expected
