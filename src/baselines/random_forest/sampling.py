from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np


def _build_target_array(
    target_pixels_per_class: int | Sequence[int],
    num_classes: int,
) -> np.ndarray:
    if isinstance(target_pixels_per_class, int):
        if target_pixels_per_class < 0:
            raise ValueError("target_pixels_per_class must be non-negative.")
        return np.full(num_classes, target_pixels_per_class, dtype=np.int64)

    target = np.asarray(target_pixels_per_class, dtype=np.int64)

    if target.shape != (num_classes,):
        raise ValueError(
            f"Expected target_pixels_per_class with shape ({num_classes},), "
            f"got {target.shape}."
        )

    if np.any(target < 0):
        raise ValueError("target_pixels_per_class values must be non-negative.")

    return target


def flatten_patch_features(x: np.ndarray) -> np.ndarray:
    """
    Convert a segmentation patch from (C, H, W) to a tabular format (H * W, C).

    Random Forest does not receive an image patch directly. It receives one row
    per pixel, where each row contains the spectral/spatial features of that pixel.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected image patch with shape (C, H, W), got {x.shape}.")

    x_hw_c = np.moveaxis(x, 0, -1)
    return x_hw_c.reshape(-1, x_hw_c.shape[-1])


def extract_pixel_samples(
    x: np.ndarray,
    y: np.ndarray,
    flat_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract selected pixel rows from an image patch and its mask.

    Parameters
    ----------
    x:
        Image patch with shape (C, H, W).
    y:
        Mask patch with shape (H, W).
    flat_indices:
        Flat pixel indices selected from the mask.

    Returns
    -------
    X_selected:
        Feature matrix with shape (N, C).
    y_selected:
        Label vector with shape (N,).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (C, H, W), got {x.shape}.")

    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (H, W), got {y.shape}.")

    if x.shape[1:] != y.shape:
        raise ValueError(f"Image/mask shape mismatch: x={x.shape}, y={y.shape}.")

    if flat_indices.ndim != 1:
        raise ValueError(f"Expected flat_indices as 1D array, got {flat_indices.shape}.")

    X_all = flatten_patch_features(x)
    y_all = y.reshape(-1)

    if len(flat_indices) == 0:
        return (
            np.empty((0, X_all.shape[1]), dtype=X_all.dtype),
            np.empty((0,), dtype=y_all.dtype),
        )

    return X_all[flat_indices], y_all[flat_indices]


@dataclass
class StratifiedPixelSampler:
    """
    Select pixels for Random Forest training in a class-aware way.

    This sampler prevents the training set from being dominated by background or
    frequent crop classes. It tries to collect up to a target number of pixels
    for each class, while also limiting how many pixels can be taken from the
    same class inside a single patch.
    """

    num_classes: int
    target_pixels_per_class: int | Sequence[int] = 100_000
    max_pixels_per_class_per_patch: int = 64
    seed: int = 42
    ignore_index: int | None = None
    include_background: bool = True

    rng: np.random.Generator = field(init=False, repr=False)
    target_per_class: np.ndarray = field(init=False)
    selected_per_class: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        if self.max_pixels_per_class_per_patch <= 0:
            raise ValueError("max_pixels_per_class_per_patch must be positive.")

        self.rng = np.random.default_rng(self.seed)

        self.target_per_class = _build_target_array(
            self.target_pixels_per_class,
            self.num_classes,
        )

        if not self.include_background and self.num_classes > 0:
            self.target_per_class[0] = 0

        self.selected_per_class = np.zeros(self.num_classes, dtype=np.int64)

    def remaining_per_class(self) -> Dict[int, int]:
        remaining = np.maximum(self.target_per_class - self.selected_per_class, 0)
        return {int(class_id): int(value) for class_id, value in enumerate(remaining)}

    def selected_summary(self) -> Dict[int, int]:
        return {
            int(class_id): int(value)
            for class_id, value in enumerate(self.selected_per_class)
        }

    def target_summary(self) -> Dict[int, int]:
        return {
            int(class_id): int(value)
            for class_id, value in enumerate(self.target_per_class)
        }

    def is_complete(self) -> bool:
        return bool(np.all(self.selected_per_class >= self.target_per_class))

    def sample_patch_indices(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Select flat pixel indices from a mask, stratified by class.

        The method updates the internal selected_per_class counter.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected mask with shape (H, W), got {mask.shape}.")

        flat_mask = mask.reshape(-1)

        selected_indices = []
        selected_counts: Dict[int, int] = {}

        for class_id in range(self.num_classes):
            if class_id == 0 and not self.include_background:
                continue

            if self.ignore_index is not None and class_id == self.ignore_index:
                continue

            remaining = int(self.target_per_class[class_id] - self.selected_per_class[class_id])
            if remaining <= 0:
                continue

            candidate_indices = np.flatnonzero(flat_mask == class_id)
            if len(candidate_indices) == 0:
                continue

            take = min(
                len(candidate_indices),
                self.max_pixels_per_class_per_patch,
                remaining,
            )

            chosen = self.rng.choice(candidate_indices, size=take, replace=False)

            selected_indices.append(chosen)
            selected_counts[class_id] = int(take)
            self.selected_per_class[class_id] += int(take)

        if not selected_indices:
            return np.empty((0,), dtype=np.int64), selected_counts

        indices = np.concatenate(selected_indices).astype(np.int64, copy=False)
        self.rng.shuffle(indices)

        return indices, selected_counts