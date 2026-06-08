from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch

from src.baselines.random_forest.features import FeatureMode, build_feature_patch


def predict_in_batches(
    model: Any,
    features: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    predictions = []

    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        predictions.append(model.predict(features[start:end]))

    if not predictions:
        return np.empty((0,), dtype=np.int64)

    return np.concatenate(predictions, axis=0)


@dataclass
class RandomForestSegmentationPredictor:
    model: Any
    feature_mode: FeatureMode
    predict_batch_size: int = 32768
    spatial_window_sizes: Sequence[int] | None = None
    spatial_band_indices: Sequence[int] | None = None
    invalid_prediction_value: int = 0

    @classmethod
    def from_joblib(
        cls,
        model_path: str | Path,
        *,
        feature_mode: FeatureMode,
        predict_batch_size: int = 32768,
        spatial_window_sizes: Sequence[int] | None = None,
        spatial_band_indices: Sequence[int] | None = None,
    ) -> "RandomForestSegmentationPredictor":
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Random Forest model not found: {model_path}")

        model = joblib.load(model_path)

        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1, verbose=0)

        return cls(
            model=model,
            feature_mode=feature_mode,
            predict_batch_size=predict_batch_size,
            spatial_window_sizes=spatial_window_sizes,
            spatial_band_indices=spatial_band_indices,
        )

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation masks for a batch of image patches.

        Parameters
        ----------
        images:
            Tensor with shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Predicted class masks with shape (N, H, W).
        """
        if images.ndim != 4:
            raise ValueError(
                "RandomForestSegmentationPredictor expects images with shape "
                f"(N, C, H, W), got {tuple(images.shape)}."
            )

        device = images.device
        images_np = images.detach().cpu().numpy().astype(np.float32, copy=False)

        predicted_masks = []

        for image in images_np:
            predicted_masks.append(self._predict_single_patch(image))

        predictions_np = np.stack(predicted_masks, axis=0).astype(np.int64, copy=False)

        return torch.from_numpy(predictions_np).to(device=device, dtype=torch.int64)

    def _predict_single_patch(self, image_chw: np.ndarray) -> np.ndarray:
        if image_chw.ndim != 3:
            raise ValueError(
                f"Expected single image patch with shape (C, H, W), got {image_chw.shape}."
            )

        feature_patch = build_feature_patch(
            image_chw,
            feature_mode=self.feature_mode,
            spatial_window_sizes=self.spatial_window_sizes,
            spatial_band_indices=self.spatial_band_indices,
        )

        if feature_patch.ndim != 3:
            raise ValueError(
                "build_feature_patch must return an array with shape (F, H, W), "
                f"got {feature_patch.shape}."
            )

        _, height, width = feature_patch.shape

        features = np.moveaxis(feature_patch, 0, -1).reshape(-1, feature_patch.shape[0])
        valid = np.isfinite(features).all(axis=1)

        predicted_flat = np.full(
            shape=(features.shape[0],),
            fill_value=self.invalid_prediction_value,
            dtype=np.int64,
        )

        if np.any(valid):
            valid_features = features[valid].astype(np.float32, copy=False)

            predicted_flat[valid] = predict_in_batches(
                self.model,
                valid_features,
                batch_size=self.predict_batch_size,
            ).astype(np.int64, copy=False)

        return predicted_flat.reshape(height, width)