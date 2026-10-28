from __future__ import annotations

import pytest
import torch

from sentinel_crop_seg.core.evaluation.metrics import (
    compute_iou_from_confusion_matrix,
    compute_mean_class_accuracy_from_confusion_matrix,
    compute_mean_iou_from_confusion_matrix,
    compute_per_class_accuracy_from_confusion_matrix,
    compute_pixel_accuracy_from_confusion_matrix,
    update_confusion_matrix_from_predictions,
)


def test_update_confusion_matrix_from_predictions_counts_pixels_correctly():
    cm = torch.zeros((3, 3), dtype=torch.int64)

    target = torch.tensor(
        [
            [0, 1, 1],
            [2, 2, 2],
        ],
        dtype=torch.int64,
    )

    pred = torch.tensor(
        [
            [0, 1, 2],
            [2, 0, 2],
        ],
        dtype=torch.int64,
    )

    update_confusion_matrix_from_predictions(
        confusion_matrix=cm,
        predictions=pred,
        targets=target,
        num_classes=3,
    )

    expected = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 2],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(cm, expected)


def test_update_confusion_matrix_supports_ignore_index_and_background_ignore():
    cm = torch.zeros((3, 3), dtype=torch.int64)

    target = torch.tensor([0, 1, 1, 2, 255], dtype=torch.int64)
    pred = torch.tensor([0, 1, 2, 2, 1], dtype=torch.int64)

    update_confusion_matrix_from_predictions(
        confusion_matrix=cm,
        predictions=pred,
        targets=target,
        num_classes=3,
        ignore_index=255,
        ignore_background=True,
    )

    expected = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(cm, expected)


def test_metric_values_from_confusion_matrix():
    cm = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 2],
        ],
        dtype=torch.int64,
    )

    iou = compute_iou_from_confusion_matrix(cm)
    acc = compute_per_class_accuracy_from_confusion_matrix(cm)
    pixel_acc = compute_pixel_accuracy_from_confusion_matrix(cm)

    assert iou.tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert acc.tolist() == pytest.approx([1.0, 0.5, 2.0 / 3.0])
    assert pixel_acc == pytest.approx(4.0 / 6.0)

    assert compute_mean_iou_from_confusion_matrix(cm) == pytest.approx(0.5)
    assert compute_mean_iou_from_confusion_matrix(
        cm,
        present_classes_only=True,
    ) == pytest.approx(0.5)

    assert compute_mean_class_accuracy_from_confusion_matrix(cm) == pytest.approx(
        (1.0 + 0.5 + 2.0 / 3.0) / 3.0
    )


def test_metric_values_with_absent_class_present_only_average():
    cm = torch.tensor(
        [
            [2, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
        ],
        dtype=torch.int64,
    )

    assert compute_mean_iou_from_confusion_matrix(cm) == pytest.approx(
        (1.0 + 0.0 + 0.5) / 3.0
    )

    assert compute_mean_iou_from_confusion_matrix(
        cm,
        present_classes_only=True,
    ) == pytest.approx((1.0 + 0.5) / 2.0)

    assert compute_mean_class_accuracy_from_confusion_matrix(
        cm,
        present_classes_only=True,
    ) == pytest.approx((1.0 + 0.5) / 2.0)


def test_update_confusion_matrix_rejects_size_mismatch():
    cm = torch.zeros((2, 2), dtype=torch.int64)

    with pytest.raises(ValueError):
        update_confusion_matrix_from_predictions(
            confusion_matrix=cm,
            predictions=torch.tensor([0, 1, 1]),
            targets=torch.tensor([0, 1]),
            num_classes=2,
        )