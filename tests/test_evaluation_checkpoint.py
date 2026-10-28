from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentinel_crop_seg.core.evaluation import checkpoint as checkpoint_module
from sentinel_crop_seg.core.evaluation.checkpoint import (
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

class _ObjectOnlyAPickleCanHold:
    """Stands in for an optimizer wrapper or config class in a checkpoint."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_a_plain_tensor_checkpoint_loads_without_authorizing_pickle(tmp_path: Path):
    """The restricted reader must handle what this project's trainer writes.

    Hardening the loader is only acceptable if it does not break the normal
    case, so this pins it: tensors, nested containers and primitives load with
    no flag.
    """
    path = tmp_path / "plain.pt"
    torch.save(
        {
            "model_state": {"conv.weight": torch.zeros(2, 2)},
            "band_indices": [0, 1, 2],
            "normalize_mode": "scale_10000",
            "scale_divisor": 10000.0,
            "epoch": 7,
        },
        path,
    )

    checkpoint = safe_torch_load(path, torch.device("cpu"))

    assert checkpoint["normalize_mode"] == "scale_10000"
    assert checkpoint["band_indices"] == [0, 1, 2]


def test_a_checkpoint_carrying_pickled_objects_is_refused_by_default(tmp_path: Path):
    """`safe_torch_load` was named `safe_` while passing weights_only=False.

    Loading such a file executes code from whoever wrote it, so the default
    must refuse, and the message must say how to proceed deliberately rather
    than leaving the caller to guess.
    """
    path = tmp_path / "pickled.pt"
    torch.save(
        {
            "model_state": {"conv.weight": torch.zeros(2, 2)},
            "trainer": _ObjectOnlyAPickleCanHold(3),
        },
        path,
    )

    with pytest.raises(ValueError, match="arbitrary pickle deserialization"):
        safe_torch_load(path, torch.device("cpu"))

    # And the escape hatch works, so a checkpoint whose origin the user can
    # vouch for stays loadable.
    checkpoint = safe_torch_load(
        path, torch.device("cpu"), allow_unsafe_pickle=True
    )
    assert checkpoint["trainer"].value == 3


def test_an_old_pytorch_without_the_restricted_reader_fails_instead_of_pickling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fallback must not re-enable pickle on the installations that cannot
    restrict it.

    The first hardening attempt caught `TypeError` and retried without
    `weights_only`, so a PyTorch too old to know the argument silently got the
    unsafe loader - no authorization, no warning, on exactly the versions with
    the weakest reader.
    """
    path = tmp_path / "plain.pt"
    torch.save({"model_state": {"conv.weight": torch.zeros(2, 2)}}, path)

    def _old_torch_load(f, map_location=None):
        """Stands in for a `torch.load` predating `weights_only`."""
        return {"model_state": {"conv.weight": torch.zeros(2, 2)}}

    monkeypatch.setattr(checkpoint_module.torch, "load", _old_torch_load)

    with pytest.raises(RuntimeError, match="no restricted checkpoint reader"):
        safe_torch_load(path, torch.device("cpu"))

    # Authorized explicitly, the old loader is still usable - refusing
    # outright would strand anyone pinned to that PyTorch.
    loaded = safe_torch_load(path, torch.device("cpu"), allow_unsafe_pickle=True)
    assert "model_state" in loaded


def test_a_corrupt_checkpoint_is_not_blamed_on_the_restricted_reader(tmp_path: Path):
    """A damaged file must not be reported as an authorization problem.

    The blanket `except Exception` told the owner of any unreadable
    checkpoint to enable pickle, which cannot help and invites them to widen
    the loader for nothing. Only the restricted reader's own refusal is
    reported that way now.
    """
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"this is not a torch file at all")

    with pytest.raises(Exception) as caught:
        safe_torch_load(path, torch.device("cpu"))

    assert "allow_unsafe_pickle" not in str(caught.value)
    assert "arbitrary pickle deserialization" not in str(caught.value)


def test_a_truncated_checkpoint_is_not_blamed_on_the_restricted_reader(tmp_path: Path):
    """Same distinction for a file that starts valid and stops early."""
    source = tmp_path / "whole.pt"
    torch.save({"model_state": {"conv.weight": torch.zeros(8, 8)}}, source)
    raw = source.read_bytes()
    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(Exception) as caught:
        safe_torch_load(truncated, torch.device("cpu"))

    assert "allow_unsafe_pickle" not in str(caught.value)


def test_the_reader_capability_is_probed_by_signature_not_by_catching_typeerror():
    """A `TypeError` from inside a load must not be read as "argument
    unsupported".

    An unpickled object rejecting its own constructor arguments raises
    `TypeError` too, and the two are indistinguishable at the call site.
    """
    assert checkpoint_module._restricted_reader_available() is True
