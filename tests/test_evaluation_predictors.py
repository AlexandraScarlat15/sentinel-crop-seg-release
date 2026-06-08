from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.predictors import (
    RandomForestSegmentationPredictor,
    predict_in_batches,
)


class ThresholdModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] > 0.5).astype(np.int64)


def test_predict_in_batches_returns_expected_predictions():
    model = ThresholdModel()

    X = np.array(
        [
            [0.1, 1.0],
            [0.7, 2.0],
            [0.2, 3.0],
            [0.9, 4.0],
        ],
        dtype=np.float32,
    )

    y = predict_in_batches(
        model,
        X,
        batch_size=2,
    )

    assert y.tolist() == [0, 1, 0, 1]


def test_predict_in_batches_rejects_invalid_batch_size():
    model = ThresholdModel()
    X = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        predict_in_batches(
            model,
            X,
            batch_size=0,
        )


def test_random_forest_segmentation_predictor_predicts_spectral_masks():
    predictor = RandomForestSegmentationPredictor(
        model=ThresholdModel(),
        feature_mode="spectral",
        predict_batch_size=2,
    )

    images = torch.tensor(
        [
            [
                [
                    [0.1, 0.7],
                    [0.2, 0.9],
                ],
                [
                    [10.0, 10.0],
                    [10.0, 10.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )

    pred = predictor.predict(images)

    assert pred.shape == (1, 2, 2)
    assert pred.dtype == torch.int64
    assert pred.tolist() == [
        [
            [0, 1],
            [0, 1],
        ]
    ]


def test_random_forest_segmentation_predictor_rejects_non_batched_input():
    predictor = RandomForestSegmentationPredictor(
        model=ThresholdModel(),
        feature_mode="spectral",
    )

    image = torch.zeros((2, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError):
        predictor.predict(image)