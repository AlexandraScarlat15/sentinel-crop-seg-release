from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sentinel_crop_seg.core.evaluation.metrics import (
    compute_iou_from_confusion_matrix,
    compute_per_class_accuracy_from_confusion_matrix,
)


DEFAULT_CLASS_NAMES = [
    "background",
    "maize",
    "rapeseed",
    "wheat",
    "barley",
    "oats",
    "sunflower",
    "sugar_beet",
    "soybean",
    "potato",
    "pasture",
]


def _to_numpy_confusion_matrix(confusion_matrix: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(confusion_matrix, torch.Tensor):
        matrix = confusion_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(confusion_matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Expected a square confusion matrix with shape (num_classes, num_classes), "
            f"got shape={matrix.shape}."
        )

    return matrix


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def save_summary_json(path: str | Path, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(summary)), f, indent=2)


def save_summary_txt(path: str | Path, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {_json_safe(value)}\n")


def save_confusion_matrices(
    output_dir: str | Path,
    confusion_matrix: torch.Tensor | np.ndarray,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = _to_numpy_confusion_matrix(confusion_matrix)

    counts_path = output_dir / "confusion_matrix_counts.npy"
    row_norm_path = output_dir / "confusion_matrix_row_norm.npy"

    np.save(counts_path, matrix)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=row_sums > 0,
    )

    np.save(row_norm_path, row_norm)

    return {
        "confusion_matrix_counts": counts_path,
        "confusion_matrix_row_norm": row_norm_path,
    }


def save_per_class_metrics_csv(
    path: str | Path,
    confusion_matrix: torch.Tensor | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix_np = _to_numpy_confusion_matrix(confusion_matrix)
    matrix_torch = torch.as_tensor(matrix_np, dtype=torch.int64)

    iou = compute_iou_from_confusion_matrix(matrix_torch)
    per_class_accuracy = compute_per_class_accuracy_from_confusion_matrix(matrix_torch)

    num_classes = int(matrix_np.shape[0])
    names = list(class_names) if class_names is not None else DEFAULT_CLASS_NAMES

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "class_id",
                "class_name",
                "iou",
                "accuracy",
                "support_pixels",
            ]
        )

        for class_id in range(num_classes):
            class_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
            support_pixels = int(matrix_np[class_id].sum())

            writer.writerow(
                [
                    class_id,
                    class_name,
                    float(iou[class_id].item()),
                    float(per_class_accuracy[class_id].item()),
                    support_pixels,
                ]
            )


def save_evaluated_samples_txt(
    path: str | Path,
    sample_ids: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")


def save_evaluation_report(
    output_dir: str | Path,
    *,
    confusion_matrix: torch.Tensor | np.ndarray,
    summary: Mapping[str, Any],
    class_names: Sequence[str] | None = None,
    evaluated_sample_ids: Sequence[str] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    saved_paths.update(
        save_confusion_matrices(
            output_dir=output_dir,
            confusion_matrix=confusion_matrix,
        )
    )

    per_class_path = output_dir / "metrics_per_class.csv"
    save_per_class_metrics_csv(
        path=per_class_path,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
    )
    saved_paths["metrics_per_class"] = per_class_path

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    save_summary_json(summary_json_path, summary)
    save_summary_txt(summary_txt_path, summary)

    saved_paths["summary_json"] = summary_json_path
    saved_paths["summary_txt"] = summary_txt_path

    if evaluated_sample_ids is not None:
        evaluated_samples_path = output_dir / "evaluated_samples.txt"
        save_evaluated_samples_txt(evaluated_samples_path, evaluated_sample_ids)
        saved_paths["evaluated_samples"] = evaluated_samples_path

    return saved_paths