from __future__ import annotations

import torch


def update_confusion_matrix_from_predictions(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int | None = None,
    ignore_background: bool = False,
) -> None:
    """
    Update a confusion matrix from predicted and target segmentation masks.

    Parameters
    ----------
    confusion_matrix:
        Tensor with shape (num_classes, num_classes). Rows are target classes,
        columns are predicted classes.
    predictions:
        Predicted class ids. Shape may be (N, H, W), (H, W), or flat.
    targets:
        Ground-truth class ids. Shape must be compatible with predictions.
    num_classes:
        Number of valid project classes.
    ignore_index:
        Optional target value ignored during metric computation.
    ignore_background:
        If True, pixels with target class 0 are excluded from metric computation.
    """
    pred = predictions.reshape(-1).to(torch.int64)
    tgt = targets.reshape(-1).to(torch.int64)

    if pred.numel() != tgt.numel():
        raise ValueError(
            "Predictions and targets must contain the same number of pixels "
            f"(got predictions={pred.numel()}, targets={tgt.numel()})."
        )

    valid = (tgt >= 0) & (tgt < num_classes)

    if ignore_index is not None:
        valid = valid & (tgt != int(ignore_index))

    if ignore_background:
        valid = valid & (tgt != 0)

    pred = pred[valid]
    tgt = tgt[valid]

    if tgt.numel() == 0:
        return

    pred = torch.clamp(pred, min=0, max=num_classes - 1)

    encoded = tgt * num_classes + pred

    counts = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confusion_matrix += counts.to(
        device=confusion_matrix.device,
        dtype=confusion_matrix.dtype,
    )


def compute_iou_from_confusion_matrix(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class Intersection over Union from a confusion matrix.
    """
    cm = confusion_matrix.to(torch.float32)

    true_positive = torch.diag(cm)
    false_positive = cm.sum(dim=0) - true_positive
    false_negative = cm.sum(dim=1) - true_positive

    denominator = true_positive + false_positive + false_negative

    return torch.where(
        denominator > 0,
        true_positive / denominator,
        torch.zeros_like(denominator),
    )


def compute_pixel_accuracy_from_confusion_matrix(confusion_matrix: torch.Tensor) -> float:
    """
    Compute global pixel accuracy from a confusion matrix.
    """
    cm = confusion_matrix.to(torch.float32)
    total = cm.sum()

    if float(total.item()) <= 0.0:
        return 0.0

    correct = torch.diag(cm).sum()

    return float((correct / total).item())


def compute_per_class_accuracy_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-class accuracy/recall from a confusion matrix.
    """
    cm = confusion_matrix.to(torch.float32)

    true_positive = torch.diag(cm)
    support = cm.sum(dim=1)

    return torch.where(
        support > 0,
        true_positive / support,
        torch.zeros_like(true_positive),
    )


def compute_mean_class_accuracy_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    *,
    present_classes_only: bool = False,
) -> float:
    """
    Compute mean class accuracy.

    If present_classes_only=True, only classes with target support > 0 are averaged.
    """
    per_class_accuracy = compute_per_class_accuracy_from_confusion_matrix(
        confusion_matrix
    )

    if present_classes_only:
        support = confusion_matrix.sum(dim=1) > 0

        if not bool(support.any()):
            return 0.0

        return float(per_class_accuracy[support].mean().item())

    return float(per_class_accuracy.mean().item())


def compute_mean_iou_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    *,
    present_classes_only: bool = False,
) -> float:
    """
    Compute mean IoU.

    If present_classes_only=True, only classes with target support > 0 are averaged.
    """
    iou = compute_iou_from_confusion_matrix(confusion_matrix)

    if present_classes_only:
        support = confusion_matrix.sum(dim=1) > 0

        if not bool(support.any()):
            return 0.0

        return float(iou[support].mean().item())

    return float(iou.mean().item())