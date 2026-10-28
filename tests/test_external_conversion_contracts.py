"""The external converters must not produce silently wrong inputs.

Three defects found by review of the first archive that shipped them. Each one
would have produced a plausible-looking tensor and a wrong number, with no
error anywhere: the channel count, the dtype and the shape were all correct.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel_crop_seg.core.external.dacia5_conversion import (
    DACIA5_12BAND_ORDER,
    PROJECT_BANDS,
    compute_project_band_indices,
    ensure_output_dirs,
    load_project_image,
)
from sentinel_crop_seg.core.external.pastis_conversion import (
    DEFAULT_PASTIS_BAND_ORDER,
    PROJECT_BAND_ORDER,
    get_output_band_order,
    parse_band_reorder,
    project_band_reorder,
)


def test_pastis_defaults_to_the_project_band_order():
    """The old default left channels 3-6 permuted, and nothing could tell.

    PASTIS orders its bands B02,B03,B04,B05,B06,B07,B08,... while this project
    puts B08 fourth. A model trained on the project order would read B05 as
    near-infrared. The tensor still has ten channels, so inference succeeds.
    """
    assert DEFAULT_PASTIS_BAND_ORDER != PROJECT_BAND_ORDER, "fixture assumption"
    assert get_output_band_order(parse_band_reorder(None)) == PROJECT_BAND_ORDER
    assert get_output_band_order(parse_band_reorder("")) == PROJECT_BAND_ORDER


def test_the_pastis_permutation_is_derived_not_hard_coded():
    """A written-out permutation would drift if either band list changed."""
    indices = project_band_reorder()
    assert [DEFAULT_PASTIS_BAND_ORDER[i] for i in indices] == PROJECT_BAND_ORDER
    assert sorted(indices) == list(range(10))


def test_pastis_can_still_opt_out_explicitly():
    """A copy already in project order must be able to skip the reorder."""
    assert parse_band_reorder("none") is None
    assert get_output_band_order(None) == DEFAULT_PASTIS_BAND_ORDER


def test_pastis_rejects_a_reorder_that_is_not_a_permutation():
    for bad in ("0,1,2", "0,1,2,3,4,5,6,7,8,8"):
        with pytest.raises(ValueError):
            parse_band_reorder(bad)


def test_dacia5_band_order_is_also_derived_from_the_names():
    indices = compute_project_band_indices()
    assert [DACIA5_12BAND_ORDER[i] for i in indices] == list(PROJECT_BANDS)


def test_dacia5_keeps_the_source_reflectance_scale(tmp_path):
    """Pre-dividing by 10,000 made a scale_10000 checkpoint scale twice.

    The project's own generator writes raw reflectance and evaluation applies
    the checkpoint's normalize_mode, so the converter must not normalize too.
    A doubly scaled patch is 10,000x too small and raises nothing.
    """
    rasterio = pytest.importorskip("rasterio")

    path = tmp_path / "scene.tif"
    value = 5000
    with rasterio.open(
        path, "w", driver="GTiff", width=4, height=4,
        count=len(DACIA5_12BAND_ORDER), dtype="uint16",
    ) as dst:
        for band in range(1, len(DACIA5_12BAND_ORDER) + 1):
            dst.write(np.full((4, 4), value, dtype=np.uint16), band)

    image = load_project_image(path)
    assert image.shape[0] == len(PROJECT_BANDS)
    # The source value must survive untouched. 0.5 here would mean the
    # converter had already applied scale_10000.
    assert np.allclose(image, float(value)), image.ravel()[:4]


def test_dacia5_refuses_to_overwrite_existing_results(tmp_path):
    """Repeating a conversion must not destroy what it is compared against."""
    out = tmp_path / "converted"
    images, _ = ensure_output_dirs(out)
    np.save(images / "sample_000000.npy", np.zeros((10, 4, 4), dtype=np.float32))

    with pytest.raises(FileExistsError, match="already contains results"):
        ensure_output_dirs(out)

    # Explicit authorization still works, and an empty directory is not a
    # result, so a fresh re-run is not blocked.
    ensure_output_dirs(out, overwrite=True)
    ensure_output_dirs(tmp_path / "fresh")


def test_the_shipped_document_states_the_ignore_semantics_correctly():
    """Unmapped labels become ignore, not background.

    The distinction decides how a cross-dataset metric reads: ignored pixels
    are excluded from the score, whereas background pixels would be counted
    as errors. The document previously said background, which inverted it.
    """
    root = Path(__file__).resolve().parents[1]
    text = " ".join(
        (root / "docs" / "external_dataset_conversion.md")
        .read_text(encoding="utf-8").split()
    )
    assert "255" in text
    assert "excluded from" in text
    assert "not background" in text
    assert "ignore value" in text
    # And it must not keep asserting the inverted version, in which the model
    # would be penalized for crops the taxonomy does not model.
    assert "become background" not in text
    assert "sends to background" not in text


def test_the_mappings_really_use_ignore_for_unmapped_crops():
    """Guards the claim above against the configuration drifting away."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    pastis = yaml.safe_load(
        (root / "config" / "pastis" / "label_mapping.yaml").read_text(encoding="utf-8")
    )["pastis"]
    assert pastis["ignore_value"] == 255
    remap = pastis["remap_to_project_class_id"]
    assert remap[0] == 0, "background must stay class 0"
    ignored = [k for k, v in remap.items() if v == 255]
    assert ignored, "the mapping must send unmodelled crops to ignore"


def test_an_authorized_overwrite_leaves_no_patches_from_the_previous_run(tmp_path):
    """A reconversion must not mix two runs' patches in one directory.

    Found by review of the second archive. `--overwrite` was authorized but
    only permitted the write; it did not clear `images/` and `masks/`, so a
    run emitting fewer patches than its predecessor left the surplus behind.
    The metadata CSV then described the new patches while the directory held
    the old ones too, and an evaluator reading the directory would score
    samples no manifest mentions.
    """
    out = tmp_path / "converted"
    images, masks = ensure_output_dirs(out)
    for index in range(5):
        (images / f"patch_{index:04d}.npy").write_bytes(b"old")
        (masks / f"patch_{index:04d}.npy").write_bytes(b"old")
    (out / "metadata.csv").write_text("stale\n", encoding="utf-8")

    images, masks = ensure_output_dirs(out, overwrite=True)

    assert list(images.iterdir()) == []
    assert list(masks.iterdir()) == []
    assert not (out / "metadata.csv").exists()


def test_a_directory_holding_only_a_stale_report_still_counts_as_used(tmp_path):
    """Emptiness is judged over the whole directory, not the subdirectories.

    The earlier DACIA5 check looked only at `images/` and `masks/`, so a
    directory left holding a previous `metadata.csv` was treated as unused and
    the new conversion wrote beside the stale report.
    """
    out = tmp_path / "converted"
    out.mkdir()
    (out / "metadata.csv").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains results"):
        ensure_output_dirs(out)

    ensure_output_dirs(out, overwrite=True)
    assert not (out / "metadata.csv").exists()


def test_an_overwrite_refuses_a_directory_holding_files_no_conversion_wrote(tmp_path):
    """Authorizing an overwrite does not authorize deleting unrelated files.

    Clearing a fixed filename list would silently keep them; clearing
    everything would destroy the user's data. Refusing is the only branch that
    neither loses files nor leaves a directory whose contents cannot be
    attributed to one run.
    """
    out = tmp_path / "converted"
    out.mkdir()
    (out / "notes_i_care_about.txt").write_text("mine\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="no conversion wrote"):
        ensure_output_dirs(out, overwrite=True)

    assert (out / "notes_i_care_about.txt").read_text(encoding="utf-8") == "mine\n"


def test_both_converters_share_one_output_contract():
    """The two must not drift apart again.

    Three different versions of this rule existed across two files, each
    wrong in its own way. They now resolve to the same function, and the
    artifact list covers exactly what the two converters write.
    """
    from sentinel_crop_seg.core.external import dacia5_conversion, pastis_conversion
    from sentinel_crop_seg.core.external.output_contract import (
        CONVERSION_ARTIFACTS,
        prepare_conversion_output,
    )

    assert dacia5_conversion.prepare_conversion_output is prepare_conversion_output
    assert pastis_conversion.prepare_conversion_output is prepare_conversion_output

    root = Path(__file__).resolve().parents[1] / "sentinel_crop_seg" / "core" / "external"
    written = set()
    for name in ("dacia5_conversion.py", "pastis_conversion.py"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if 'output_dir / "' in line:
                written.add(line.split('output_dir / "')[1].split('"')[0])
    assert written, "the scan must find the top-level writes it is checking"
    assert written <= set(CONVERSION_ARTIFACTS), (
        f"these artifacts survive an overwrite: {sorted(written - set(CONVERSION_ARTIFACTS))}"
    )
