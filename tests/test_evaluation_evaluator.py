from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluator import evaluate_segmentation_predictor


class TinySegmentationDataset(Dataset):
    def __init__(self) -> None:
        self.images = [
            torch.tensor(
                [
                    [
                        [0, 1],
                        [1, 1],
                    ]
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [
                        [1, 1],
                        [0, 0],
                    ]
                ],
                dtype=torch.float32,
            ),
        ]

        self.masks = [
            torch.tensor(
                [
                    [0, 1],
                    [1, 0],
                ],
                dtype=torch.int64,
            ),
            torch.tensor(
                [
                    [1, 1],
                    [0, 0],
                ],
                dtype=torch.int64,
            ),
        ]

        self.sample_ids = ["sample_000001", "sample_000002"]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index], self.sample_ids[index]


class ImageAsPredictionPredictor:
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, 0, :, :].to(torch.int64)


def test_evaluate_segmentation_predictor_computes_metrics_and_saves_report(tmp_path: Path):
    dataset = TinySegmentationDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    result = evaluate_segmentation_predictor(
        predictor=ImageAsPredictionPredictor(),
        dataloader=dataloader,
        num_classes=2,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        ignore_index=None,
        ignore_background=False,
        class_names=["background", "crop"],
        summary_metadata={
            "model_type": "dummy",
            "data_source": "tiny",
        },
        progress_every=1,
    )

    assert result.confusion_matrix.tolist() == [
        [3, 1],
        [0, 4],
    ]

    assert result.summary["model_type"] == "dummy"
    assert result.summary["data_source"] == "tiny"
    assert result.summary["num_samples_evaluated"] == 2
    assert result.summary["pixel_accuracy"] == pytest.approx(7.0 / 8.0)
    assert result.summary["mIoU_all_classes"] == pytest.approx((0.75 + 0.8) / 2.0)
    assert result.summary["mIoU_present_classes"] == pytest.approx((0.75 + 0.8) / 2.0)

    assert result.evaluated_sample_ids == ["sample_000001", "sample_000002"]

    expected_files = {
        "confusion_matrix_counts",
        "confusion_matrix_row_norm",
        "metrics_per_class",
        "summary_json",
        "summary_txt",
        "evaluated_samples",
    }

    assert set(result.saved_paths.keys()) == expected_files

    for path in result.saved_paths.values():
        assert path.exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_type"] == "dummy"
    assert summary["pixel_accuracy"] == pytest.approx(7.0 / 8.0)


def test_evaluate_segmentation_predictor_supports_torch_module_logits():
    class TinyLogitModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Return logits with shape (N, C, H, W).
            pred_class = images[:, 0, :, :].to(torch.int64)
            logits = torch.zeros(
                (images.shape[0], 2, images.shape[2], images.shape[3]),
                dtype=torch.float32,
                device=images.device,
            )

            logits[:, 0, :, :] = (pred_class == 0).to(torch.float32) * 10.0
            logits[:, 1, :, :] = (pred_class == 1).to(torch.float32) * 10.0

            return logits

    dataset = TinySegmentationDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    result = evaluate_segmentation_predictor(
        predictor=TinyLogitModel(),
        dataloader=dataloader,
        num_classes=2,
        device=torch.device("cpu"),
        output_dir=None,
        progress_every=1,
    )

    assert result.confusion_matrix.tolist() == [
        [3, 1],
        [0, 4],
    ]

    assert result.saved_paths == {}


def test_evaluate_segmentation_predictor_can_ignore_background():
    dataset = TinySegmentationDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    result = evaluate_segmentation_predictor(
        predictor=ImageAsPredictionPredictor(),
        dataloader=dataloader,
        num_classes=2,
        device=torch.device("cpu"),
        output_dir=None,
        ignore_background=True,
        progress_every=1,
    )

    assert result.confusion_matrix.tolist() == [
        [0, 0],
        [0, 4],
    ]

    assert result.summary["pixel_accuracy"] == pytest.approx(1.0)


def test_evaluate_segmentation_predictor_rejects_invalid_num_classes():
    dataset = TinySegmentationDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with pytest.raises(ValueError):
        evaluate_segmentation_predictor(
            predictor=ImageAsPredictionPredictor(),
            dataloader=dataloader,
            num_classes=0,
            device=torch.device("cpu"),
        )