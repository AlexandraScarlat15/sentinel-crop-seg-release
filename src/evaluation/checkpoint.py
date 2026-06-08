from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.training.model_unet_4levels import UNet4LevelsGN


@dataclass(frozen=True)
class UNetCheckpointConfig:
    in_channels: int = 10
    num_classes: int = 11
    base: int = 48
    dropout_bottleneck: float = 0.0
    dropout_decoder: float = 0.0
    up_mode: str = "bilinear"

    band_indices: tuple[int, ...] | None = None
    normalize_mode: str = "none"
    scale_divisor: float = 10000.0


@dataclass(frozen=True)
class LoadedUNetCheckpoint:
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    config: UNetCheckpointConfig
    model: UNet4LevelsGN


def safe_torch_load(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint while remaining compatible with multiple PyTorch versions.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint to be a dictionary, got {type(checkpoint).__name__}"
        )

    return checkpoint


def _parse_band_indices(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None

    if isinstance(value, tuple):
        return tuple(int(item) for item in value)

    if isinstance(value, list):
        return tuple(int(item) for item in value)

    raise ValueError(
        "band_indices must be None, list[int], or tuple[int, ...], "
        f"got {type(value).__name__}"
    )


def extract_unet_checkpoint_config(
    checkpoint: dict[str, Any],
) -> UNetCheckpointConfig:
    """
    Extract the U-Net architecture and preprocessing configuration from a checkpoint.

    Older checkpoints may contain only part of the config, so safe defaults are used.
    """
    raw_config = checkpoint.get("config", {})

    if raw_config is None:
        raw_config = {}

    if not isinstance(raw_config, dict):
        raise ValueError(
            f"Expected checkpoint['config'] to be a dictionary, got {type(raw_config).__name__}"
        )

    return UNetCheckpointConfig(
        in_channels=int(raw_config.get("in_channels", 10)),
        num_classes=int(raw_config.get("num_classes", 11)),
        base=int(raw_config.get("base", 48)),
        dropout_bottleneck=float(raw_config.get("dropout_bottleneck", 0.0)),
        dropout_decoder=float(raw_config.get("dropout_decoder", 0.0)),
        up_mode=str(raw_config.get("up_mode", "bilinear")),
        band_indices=_parse_band_indices(raw_config.get("band_indices", None)),
        normalize_mode=str(raw_config.get("normalize_mode", "none")),
        scale_divisor=float(raw_config.get("scale_divisor", 10000.0)),
    )


def build_unet_from_config(config: UNetCheckpointConfig) -> UNet4LevelsGN:
    """
    Build a U-Net model from checkpoint configuration.
    """
    return UNet4LevelsGN(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        base=config.base,
        dropout_bottleneck=config.dropout_bottleneck,
        dropout_decoder=config.dropout_decoder,
        up_mode=config.up_mode,
    )


def load_unet_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedUNetCheckpoint:
    """
    Load a U-Net checkpoint, rebuild the model, load weights, and switch to eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path, device)

    if "model_state" not in checkpoint:
        raise KeyError(
            "Checkpoint is missing 'model_state'. "
            "Expected a checkpoint saved by the U-Net training script."
        )

    config = extract_unet_checkpoint_config(checkpoint)
    model = build_unet_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return LoadedUNetCheckpoint(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        model=model,
    )