from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.evaluation.checkpoint import (
    UNetCheckpointConfig,
    build_unet_from_config,
    extract_unet_checkpoint_config,
    load_unet_checkpoint,
    safe_torch_load,
)


def test_extract_unet_checkpoint_config_reads_values():
    checkpoint = {
        "config": {
            "in_channels": 10,
            "num_classes": 11,
            "base": 48,
            "dropout_bottleneck": 0.1,
            "dropout_decoder": 0.2,
            "up_mode": "bilinear",
            "band_indices": [0, 1, 2, 3],
            "normalize_mode": "scale_10000",
            "scale_divisor": 10000.0,
        }
    }

    cfg = extract_unet_checkpoint_config(checkpoint)

    assert cfg.in_channels == 10
    assert cfg.num_classes == 11
    assert cfg.base == 48
    assert cfg.dropout_bottleneck == pytest.approx(0.1)
    assert cfg.dropout_decoder == pytest.approx(0.2)
    assert cfg.up_mode == "bilinear"
    assert cfg.band_indices == (0, 1, 2, 3)
    assert cfg.normalize_mode == "scale_10000"
    assert cfg.scale_divisor == pytest.approx(10000.0)


def test_extract_unet_checkpoint_config_uses_defaults_for_old_checkpoints():
    cfg = extract_unet_checkpoint_config({})

    assert cfg == UNetCheckpointConfig()


def test_extract_unet_checkpoint_config_rejects_invalid_band_indices():
    checkpoint = {
        "config": {
            "band_indices": "0,1,2",
        }
    }

    with pytest.raises(ValueError):
        extract_unet_checkpoint_config(checkpoint)


def test_safe_torch_load_rejects_non_dict_checkpoint(tmp_path: Path):
    path = tmp_path / "bad_checkpoint.pt"
    torch.save([1, 2, 3], path)

    with pytest.raises(ValueError):
        safe_torch_load(path, torch.device("cpu"))


def test_load_unet_checkpoint_rebuilds_model_and_loads_state(tmp_path: Path):
    device = torch.device("cpu")

    cfg = UNetCheckpointConfig(
        in_channels=10,
        num_classes=11,
        base=8,
        dropout_bottleneck=0.0,
        dropout_decoder=0.0,
        up_mode="bilinear",
        band_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        normalize_mode="scale_10000",
        scale_divisor=10000.0,
    )

    model = build_unet_from_config(cfg)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "in_channels": cfg.in_channels,
                "num_classes": cfg.num_classes,
                "base": cfg.base,
                "dropout_bottleneck": cfg.dropout_bottleneck,
                "dropout_decoder": cfg.dropout_decoder,
                "up_mode": cfg.up_mode,
                "band_indices": list(cfg.band_indices),
                "normalize_mode": cfg.normalize_mode,
                "scale_divisor": cfg.scale_divisor,
            },
        },
        checkpoint_path,
    )

    loaded = load_unet_checkpoint(checkpoint_path, device)

    assert loaded.checkpoint_path == checkpoint_path
    assert loaded.config == cfg
    assert loaded.model.training is False

    x = torch.randn(1, 10, 32, 32)
    with torch.no_grad():
        y = loaded.model(x)

    assert y.shape == (1, 11, 32, 32)


def test_load_unet_checkpoint_requires_model_state(tmp_path: Path):
    path = tmp_path / "missing_model_state.pt"
    torch.save({"config": {}}, path)

    with pytest.raises(KeyError):
        load_unet_checkpoint(path, torch.device("cpu"))