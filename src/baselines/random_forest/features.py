from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.ndimage import uniform_filter


FeatureMode = Literal["spectral", "spatial"]


def validate_image_patch(x: np.ndarray) -> None:
    """
    Validate that an image patch follows the expected segmentation format.

    Expected shape:
        (C, H, W)

    where:
        C = number of channels / bands
        H = patch height
        W = patch width
    """
    if x.ndim != 3:
        raise ValueError(f"Expected image patch with shape (C, H, W), got {x.shape}.")

    if x.shape[0] <= 0:
        raise ValueError(f"Expected at least one channel, got shape {x.shape}.")

    if x.shape[1] <= 0 or x.shape[2] <= 0:
        raise ValueError(f"Invalid spatial dimensions for patch: {x.shape}.")


def _validate_band_indices(
    num_channels: int,
    band_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if band_indices is None:
        return tuple(range(num_channels))

    if len(band_indices) == 0:
        raise ValueError("band_indices must be None or a non-empty sequence.")

    out = tuple(int(idx) for idx in band_indices)

    for idx in out:
        if idx < 0 or idx >= num_channels:
            raise ValueError(
                f"Band index {idx} is out of range for image with "
                f"{num_channels} channels."
            )

    return out


def local_mean_std(
    band: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local mean and standard deviation for a single 2D band.

    The standard deviation is computed using:
        Var(X) = E[X^2] - E[X]^2
    """
    if band.ndim != 2:
        raise ValueError(f"Expected 2D band, got shape {band.shape}.")

    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")

    mean = uniform_filter(band, size=window_size, mode="reflect")
    mean_sq = uniform_filter(band * band, size=window_size, mode="reflect")

    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)

    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def add_spatial_context(
    x: np.ndarray,
    *,
    window_sizes: Sequence[int] = (3, 5),
    band_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Add local spatial-context features to a Sentinel-2 patch.

    Input:
        x shape = (C, H, W)

    Output:
        features shape = (C + 2 * len(window_sizes) * selected_channels, H, W)

    For each selected band and each window size, the function adds:
        local mean
        local standard deviation

    Example:
        If x has 10 channels and window_sizes=(3, 5), using all bands:

        original channels: 10
        mean/std 3x3:     20
        mean/std 5x5:     20

        total channels:   50
    """
    validate_image_patch(x)

    x = x.astype(np.float32, copy=False)
    selected_band_indices = _validate_band_indices(x.shape[0], band_indices)

    features: list[np.ndarray] = [x]

    for window_size in window_sizes:
        means = []
        stds = []

        for band_idx in selected_band_indices:
            mean, std = local_mean_std(x[band_idx], int(window_size))
            means.append(mean)
            stds.append(std)

        features.append(np.stack(means, axis=0))
        features.append(np.stack(stds, axis=0))

    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def build_feature_patch(
    x: np.ndarray,
    *,
    feature_mode: FeatureMode,
    spatial_window_sizes: Sequence[int] = (3, 5),
    spatial_band_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Build the feature patch used by the Random Forest baseline.

    feature_mode="spectral":
        Returns the original spectral channels.

    feature_mode="spatial":
        Returns the original spectral channels plus local spatial-context
        statistics.
    """
    validate_image_patch(x)

    x = x.astype(np.float32, copy=False)

    if feature_mode == "spectral":
        return x

    if feature_mode == "spatial":
        return add_spatial_context(
            x,
            window_sizes=spatial_window_sizes,
            band_indices=spatial_band_indices,
        )

    raise ValueError(
        f"Unknown feature_mode={feature_mode!r}. "
        "Expected one of: 'spectral', 'spatial'."
    )


def count_output_channels(
    *,
    input_channels: int,
    feature_mode: FeatureMode,
    spatial_window_sizes: Sequence[int] = (3, 5),
    spatial_band_indices: Sequence[int] | None = None,
) -> int:
    """
    Compute how many channels the feature patch will have.

    This is useful for logging and debugging.
    """
    if input_channels <= 0:
        raise ValueError("input_channels must be positive.")

    if feature_mode == "spectral":
        return input_channels

    if feature_mode == "spatial":
        selected_channels = (
            input_channels
            if spatial_band_indices is None
            else len(spatial_band_indices)
        )

        return input_channels + 2 * len(spatial_window_sizes) * selected_channels

    raise ValueError(
        f"Unknown feature_mode={feature_mode!r}. "
        "Expected one of: 'spectral', 'spatial'."
    )