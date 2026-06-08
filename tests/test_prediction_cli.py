from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.prediction.predict_cli import (
    build_color_table,
    colorize_mask,
    hex_to_rgb,
    json_safe,
    predict_batch,
    safe_prediction_filename,
    sample_to_record,
)


@dataclass(frozen=True)
class DummyManifestSample:
    dataset: str
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


@dataclass(frozen=True)
class DummyRootSample:
    image_path: Path
    mask_path: Path


class DummyUNet(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        logits = torch.zeros((batch_size, 2, height, width), dtype=torch.float32)
        logits[:, 0, :, :] = 0.25
        logits[:, 1, :, :] = 0.75

        return logits


def test_safe_prediction_filename_sanitizes_manifest_sample_id():
    filename = safe_prediction_filename(
        output_index=3,
        sample_id="AT_2021/sample_000001.npy",
    )

    assert filename == "000003_AT_2021__sample_000001.npy"


def test_sample_to_record_for_manifest_sample():
    sample = DummyManifestSample(
        dataset="AT_2021",
        sample_id="AT_2021/sample_000001.npy",
        image_path=Path("images/sample_000001.npy"),
        mask_path=Path("masks/sample_000001.npy"),
        split="test",
    )

    record = sample_to_record(sample)

    assert record["dataset"] == "AT_2021"
    assert record["sample_id"] == "AT_2021/sample_000001.npy"
    assert record["split"] == "test"
    assert record["image_path"] == "images/sample_000001.npy"
    assert record["mask_path"] == "masks/sample_000001.npy"


def test_sample_to_record_for_root_sample():
    sample = DummyRootSample(
        image_path=Path("dataset/images/sample_000001.npy"),
        mask_path=Path("dataset/masks/sample_000001.npy"),
    )

    record = sample_to_record(sample)

    assert record["dataset"] == "dataset"
    assert record["sample_id"] == "dataset/sample_000001.npy"
    assert record["split"] == ""
    assert record["image_path"] == "dataset/images/sample_000001.npy"
    assert record["mask_path"] == "dataset/masks/sample_000001.npy"


def test_json_safe_converts_paths_numpy_and_tuples():
    payload = {
        "path": Path("outputs/test"),
        "array": np.array([1, 2, 3], dtype=np.int64),
        "value": np.float32(0.5),
        "tuple": (1, 2),
    }

    safe = json_safe(payload)

    assert safe == {
        "path": "outputs/test",
        "array": [1, 2, 3],
        "value": 0.5,
        "tuple": [1, 2],
    }


def test_predict_batch_unet_returns_argmax_masks():
    model = DummyUNet()
    images = torch.zeros((2, 10, 4, 4), dtype=torch.float32)

    predictions = predict_batch(
        model_type="unet",
        predictor=model,
        images=images,
        device=torch.device("cpu"),
    )

    assert predictions.shape == (2, 4, 4)
    assert predictions.dtype == np.int64
    assert np.all(predictions == 1)

def test_hex_to_rgb():
    assert hex_to_rgb("#ffd300") == (255, 211, 0)
    assert hex_to_rgb("000000") == (0, 0, 0)


def test_build_color_table_contains_project_classes_and_ignore_color():
    color_table = build_color_table()

    assert color_table.shape == (12, 3)
    assert color_table.dtype == np.uint8

    # class 0 background is black
    assert color_table[0].tolist() == [0, 0, 0]

    # last color is ignore label color #d3d3d3
    assert color_table[-1].tolist() == [211, 211, 211]


def test_colorize_mask_maps_classes_and_ignore_label():
    mask = np.array(
        [
            [0, 1],
            [255, 10],
        ],
        dtype=np.int64,
    )

    color = colorize_mask(mask, ignore_index=255)

    assert color.shape == (2, 2, 3)
    assert color.dtype == np.uint8

    assert color[0, 0].tolist() == [0, 0, 0]
    assert color[1, 0].tolist() == [211, 211, 211]