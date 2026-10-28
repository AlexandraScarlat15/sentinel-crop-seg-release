import json
from pathlib import Path

import numpy as np
import pytest

from sentinel_crop_seg.core.training.dataset_manifest import ManifestSegmentationDataset
from sentinel_crop_seg.core.training.mask_resolver import (
    CORRECTED_V1_VERSION,
    HISTORICAL_VERSION,
    MaskResolverSettings,
    VersionedMaskResolver,
    build_mask_resolver,
    require_manifest_data_source,
)


def write_correction_manifest(tmp_path, patches):
    manifest = tmp_path / "combined_manifest.json"
    manifest.write_text(json.dumps({"revision": CORRECTED_V1_VERSION, "patches": patches}), encoding="utf-8")
    return manifest


def test_resolver_uses_dataset_mask_identity_not_sample_id_alone(tmp_path):
    corrected_root = tmp_path / "corrected"
    manifest = write_correction_manifest(tmp_path, [{
        "subset": "alpha_subset", "sample_id": "sample_000001",
        "corrected_path": r"C:\old\corrected_masks_v1\masks\alpha_subset\sample_000001.npy",
    }])
    resolver = VersionedMaskResolver(
        version=CORRECTED_V1_VERSION, corrected_manifest=manifest,
        corrected_mask_root=corrected_root, subset_to_dataset={"alpha_subset": "alpha_v2"},
        expected_corrected_records=1,
    )
    alpha = resolver.resolve(dataset="alpha_v2", sample_id="alpha_v2/sample_000001.npy",
                             original_mask_path="/dataset/alpha_v2/masks/sample_000001.npy")
    beta = resolver.resolve(dataset="beta_v2", sample_id="beta_v2/sample_000001.npy",
                            original_mask_path="/dataset/beta_v2/masks/sample_000001.npy")
    assert alpha.corrected and alpha.path == corrected_root / "masks/alpha_subset/sample_000001.npy"
    assert not beta.corrected
    assert beta.path == Path("/dataset/beta_v2/masks/sample_000001.npy")


def test_historical_resolver_always_returns_original_path():
    resolver = VersionedMaskResolver(version=HISTORICAL_VERSION)
    resolution = resolver.resolve(dataset="alpha_v2", sample_id="sample_000001",
                                  original_mask_path="/dataset/alpha_v2/masks/sample_000001.npy")
    assert not resolution.corrected
    assert resolution.path.as_posix().endswith("alpha_v2/masks/sample_000001.npy")


def test_manifest_dataset_replaces_only_in_memory_mask_path(tmp_path):
    dataset_root = tmp_path / "dataset_v2"
    images = dataset_root / "images"; masks = dataset_root / "masks"
    images.mkdir(parents=True); masks.mkdir()
    image_path = images / "sample_000001.npy"; original_mask = masks / "sample_000001.npy"
    np.save(image_path, np.ones((2, 4, 4), dtype=np.float32)); np.save(original_mask, np.zeros((4, 4), dtype=np.uint8))
    corrected_root = tmp_path / "corrected"; corrected_mask = corrected_root / "masks/subset/sample_000001.npy"
    corrected_mask.parent.mkdir(parents=True); np.save(corrected_mask, np.full((4, 4), 10, dtype=np.uint8))
    manifest_csv = tmp_path / "training.csv"
    source = "dataset,sample_id,img_path,mask_path,split,extra\n" + \
        f"dataset_v2,dataset_v2/sample_000001.npy,{image_path},{original_mask},train,preserved\n"
    manifest_csv.write_text(source, encoding="utf-8")
    corrected_manifest = write_correction_manifest(tmp_path, [{
        "subset": "subset", "sample_id": "sample_000001", "corrected_path": "/old/root/masks/subset/sample_000001.npy",
    }])
    resolver = VersionedMaskResolver(version=CORRECTED_V1_VERSION, corrected_manifest=corrected_manifest,
        corrected_mask_root=corrected_root, subset_to_dataset={"subset": "dataset_v2"}, expected_corrected_records=1)
    dataset = ManifestSegmentationDataset(manifest_csv, "train", shuffle=False, mask_resolver=resolver)
    assert dataset.samples[0].image_path == image_path
    assert dataset.samples[0].sample_id == "dataset_v2/sample_000001.npy"
    assert dataset.samples[0].split == "train"
    assert dataset.samples[0].mask_path == corrected_mask
    assert manifest_csv.read_text(encoding="utf-8") == source
    _image, label = dataset[0]
    assert np.all(label == 10)


def test_resolver_rejects_duplicate_corrected_identity(tmp_path):
    patches = [{"subset": "subset", "sample_id": "sample_000001", "corrected_path": "/old/masks/subset/sample_000001.npy"}] * 2
    with pytest.raises(ValueError, match="Ambiguous corrected-mask key"):
        VersionedMaskResolver(version=CORRECTED_V1_VERSION, corrected_manifest=write_correction_manifest(tmp_path, patches),
            corrected_mask_root=tmp_path / "root", subset_to_dataset={"subset": "dataset_v2"})


def test_shared_label_config_factory_supports_historical_and_corrected(tmp_path):
    historical_settings = MaskResolverSettings.from_mapping({"version": "historical"})
    assert historical_settings.version == HISTORICAL_VERSION
    assert build_mask_resolver(historical_settings).corrected_record_count == 0

    corrected_manifest = write_correction_manifest(tmp_path, [{
        "subset": "subset", "sample_id": "sample_000001", "corrected_path": "/old/masks/subset/sample_000001.npy",
    }])
    settings = MaskResolverSettings.from_mapping({
        "version": CORRECTED_V1_VERSION,
        "corrected_manifest": corrected_manifest,
        "corrected_mask_root": tmp_path / "root",
        "expected_corrected_records": 1,
        "subset_to_dataset": {"subset": "dataset_v2"},
    })
    assert build_mask_resolver(settings).corrected_record_count == 1
    require_manifest_data_source(settings, "manifest")
    with pytest.raises(ValueError, match="requires manifest-based"):
        require_manifest_data_source(settings, "root")
