from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch.utils.data import DataLoader

from sentinel_crop_seg.core.evaluation.metrics import (
    compute_iou_from_confusion_matrix,
    compute_mean_class_accuracy_from_confusion_matrix,
    compute_mean_iou_from_confusion_matrix,
    compute_per_class_accuracy_from_confusion_matrix,
    compute_pixel_accuracy_from_confusion_matrix,
    update_confusion_matrix_from_predictions,
)
from sentinel_crop_seg.core.evaluation.reporting import save_evaluation_report


class SegmentationPredictor(Protocol):
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return either logits with shape (N, C, H, W) or class masks with shape (N, H, W).
        """
        ...


@dataclass(frozen=True)
class EvaluationResult:
    confusion_matrix: torch.Tensor
    summary: dict[str, Any]
    evaluated_sample_ids: list[str]
    saved_paths: dict[str, Path]


def _predict(
    predictor: SegmentationPredictor | torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    if hasattr(predictor, "predict"):
        return predictor.predict(images)

    if callable(predictor):
        return predictor(images)

    raise TypeError(
        "predictor must either implement predict(images) or be callable."
    )


def _prediction_to_class_ids(prediction: torch.Tensor) -> torch.Tensor:
    """
    Convert model output to class-id masks.

    Supported forms:
    - logits/probabilities: (N, C, H, W)
    - class masks:          (N, H, W)
    - single class mask:    (H, W)
    """
    if prediction.ndim == 4:
        return torch.argmax(prediction, dim=1)

    if prediction.ndim in {2, 3}:
        return prediction.to(torch.int64)

    raise ValueError(
        "Expected prediction with shape (N, C, H, W), (N, H, W), or (H, W), "
        f"got shape={tuple(prediction.shape)}."
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Sequence[Any] | None]:
    """
    Supported batch layouts:
    - (images, masks)
    - (images, masks, sample_ids)
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(
            "Expected batch to be a tuple/list: (images, masks) or "
            "(images, masks, sample_ids)."
        )

    if len(batch) == 2:
        images, masks = batch
        sample_ids = None
    elif len(batch) >= 3:
        images, masks, sample_ids = batch[0], batch[1], batch[2]
    else:
        raise ValueError("Batch must contain at least images and masks.")

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected images to be a torch.Tensor, got {type(images).__name__}")

    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"Expected masks to be a torch.Tensor, got {type(masks).__name__}")

    return images, masks, sample_ids


def _sample_ids_to_list(sample_ids: Sequence[Any] | None, batch_size: int) -> list[str]:
    if sample_ids is None:
        return []

    if isinstance(sample_ids, torch.Tensor):
        return [str(item) for item in sample_ids.detach().cpu().tolist()]

    if isinstance(sample_ids, (list, tuple)):
        return [str(item) for item in sample_ids]

    return [str(sample_ids)] * batch_size


@torch.no_grad()
def evaluate_segmentation_predictor(
    *,
    predictor: SegmentationPredictor | torch.nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    output_dir: str | Path | None = None,
    ignore_index: int | None = None,
    ignore_background: bool = False,
    class_names: Sequence[str] | None = None,
    summary_metadata: Mapping[str, Any] | None = None,
    progress_every: int = 50,
    log_prefix: str = "[EVAL]",
) -> EvaluationResult:
    """
    Evaluate a segmentation predictor using a shared confusion-matrix based pipeline.

    The predictor may be:
    - a PyTorch model returning logits (N, C, H, W);
    - an object implementing predict(images);
    - a callable returning class masks (N, H, W).

    The dataloader must yield either:
    - (images, masks)
    - (images, masks, sample_ids)
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if progress_every <= 0:
        raise ValueError(f"progress_every must be positive, got {progress_every}")

    if hasattr(predictor, "eval"):
        predictor.eval()

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
        device=device,
    )

    evaluated_sample_ids: list[str] = []
    num_samples_evaluated = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        images, masks, sample_ids = _unpack_batch(batch)

        images = images.to(device, non_blocking=(device.type == "cuda"))
        masks = masks.to(device, non_blocking=(device.type == "cuda"))

        raw_prediction = _predict(predictor, images)
        prediction = _prediction_to_class_ids(raw_prediction)

        update_confusion_matrix_from_predictions(
            confusion_matrix=confusion_matrix,
            predictions=prediction,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
            ignore_background=ignore_background,
        )

        batch_size = int(images.shape[0]) if images.ndim > 0 else 1
        num_samples_evaluated += batch_size
        evaluated_sample_ids.extend(_sample_ids_to_list(sample_ids, batch_size))

        if batch_idx % progress_every == 0 or batch_idx == len(dataloader):
            print(f"{log_prefix} processed batches: {batch_idx}/{len(dataloader)}")

    iou = compute_iou_from_confusion_matrix(confusion_matrix).detach().cpu()
    per_class_accuracy = compute_per_class_accuracy_from_confusion_matrix(
        confusion_matrix
    ).detach().cpu()

    pixel_accuracy = compute_pixel_accuracy_from_confusion_matrix(confusion_matrix)
    mean_iou_all = compute_mean_iou_from_confusion_matrix(
        confusion_matrix,
        present_classes_only=False,
    )
    mean_iou_present = compute_mean_iou_from_confusion_matrix(
        confusion_matrix,
        present_classes_only=True,
    )
    mean_class_accuracy_all = compute_mean_class_accuracy_from_confusion_matrix(
        confusion_matrix,
        present_classes_only=False,
    )
    mean_class_accuracy_present = compute_mean_class_accuracy_from_confusion_matrix(
        confusion_matrix,
        present_classes_only=True,
    )

    summary: dict[str, Any] = dict(summary_metadata or {})
    summary.update(
        {
            "num_samples_evaluated": num_samples_evaluated,
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "ignore_background": ignore_background,
            "mIoU_all_classes": mean_iou_all,
            "mIoU_present_classes": mean_iou_present,
            "pixel_accuracy": pixel_accuracy,
            "mean_class_accuracy_all_classes": mean_class_accuracy_all,
            "mean_class_accuracy_present_classes": mean_class_accuracy_present,
            "iou_per_class": [float(value) for value in iou.tolist()],
            "accuracy_per_class": [float(value) for value in per_class_accuracy.tolist()],
        }
    )

    saved_paths: dict[str, Path] = {}

    if output_dir is not None:
        saved_paths = save_evaluation_report(
            output_dir=output_dir,
            confusion_matrix=confusion_matrix,
            summary=summary,
            class_names=class_names,
            evaluated_sample_ids=evaluated_sample_ids,
        )

    return EvaluationResult(
        confusion_matrix=confusion_matrix.detach().cpu(),
        summary=summary,
        evaluated_sample_ids=evaluated_sample_ids,
        saved_paths=saved_paths,
    )