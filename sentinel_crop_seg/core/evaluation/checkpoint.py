from __future__ import annotations

import inspect
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sentinel_crop_seg.core.training.model_unet_4levels import UNet4LevelsGN

#: Substring PyTorch's restricted reader puts in its own refusal. Used to tell
#: that refusal apart from a corrupt pickle stream, which raises the same
#: exception type but is not fixed by authorizing a wider loader.
_RESTRICTED_REFUSAL_MARKER = "Weights only load failed"


def _restricted_reader_available() -> bool:
    """Whether this PyTorch accepts ``weights_only``.

    Checked by signature rather than by catching ``TypeError`` from the call:
    a ``TypeError`` raised from inside a load - an unpickled object rejecting
    its arguments - is indistinguishable from the argument being unsupported,
    and treating it as the latter is what let an unauthorized pickle load
    through.
    """
    return "weights_only" in inspect.signature(torch.load).parameters


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
    calendar_features: str = "none"


@dataclass(frozen=True)
class LoadedUNetCheckpoint:
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    config: UNetCheckpointConfig
    model: UNet4LevelsGN


def safe_torch_load(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    allow_unsafe_pickle: bool = False,
) -> dict[str, Any]:
    """Load a PyTorch checkpoint, restricted to tensor data by default.

    This function was named `safe_` while passing `weights_only=False`, which
    permits arbitrary pickle execution: loading a checkpoint was equivalent to
    running its author's code. The name now matches the behaviour.

    `weights_only=True` reads tensors, plain containers and primitives. A
    checkpoint that also pickled a custom object - an optimizer wrapper, an
    argparse namespace, a config class - cannot be read this way, and that is
    a property of the file, not of the caller. Such a file needs
    `allow_unsafe_pickle=True`, which is a statement about the file's
    provenance, so the error says so rather than silently widening the load.

    Args:
        checkpoint_path: File to load.
        device: Map location for the stored tensors.
        allow_unsafe_pickle: Permit arbitrary pickle deserialization. Only for
            a checkpoint whose origin you can vouch for.

    Returns:
        The checkpoint dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If this PyTorch has no restricted reader and unsafe
            loading was not authorized.
        ValueError: If the checkpoint is not a dictionary, or if it needs
            pickle deserialization that was not authorized.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not _restricted_reader_available():
        # Falling back to a plain `torch.load` here would re-enable pickle
        # execution on exactly the installations that cannot restrict it, and
        # it would do so without the caller asking. Refusing is the only
        # branch that keeps the authorization meaningful.
        if not allow_unsafe_pickle:
            raise RuntimeError(
                f"This PyTorch ({torch.__version__}) has no restricted "
                f"checkpoint reader, so {checkpoint_path} can only be read by "
                f"arbitrary pickle deserialization. Upgrade PyTorch, or pass "
                f"allow_unsafe_pickle=True (CLI: --allow-unsafe-checkpoint) "
                f"if you can vouch for the file's origin."
            )
        return _as_checkpoint_dict(
            torch.load(checkpoint_path, map_location=device), checkpoint_path
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=not allow_unsafe_pickle,
        )
    except pickle.UnpicklingError as exc:
        # Only the restricted reader's own refusal may be reported as
        # something authorization would fix. A corrupt or truncated file also
        # fails here, and telling its owner to enable pickle would send them
        # to re-run with a wider loader that cannot help either - the earlier
        # blanket handler did exactly that.
        if allow_unsafe_pickle or _RESTRICTED_REFUSAL_MARKER not in str(exc):
            raise
        raise ValueError(
            f"Refusing to load {checkpoint_path} with arbitrary pickle "
            f"deserialization. It stores objects the restricted reader cannot "
            f"reconstruct, so reading it would execute code from whoever "
            f"wrote the file. If you can vouch for its origin, pass "
            f"allow_unsafe_pickle=True (CLI: --allow-unsafe-checkpoint). "
            f"Underlying error: {exc}"
        ) from exc

    return _as_checkpoint_dict(checkpoint, checkpoint_path)


def _as_checkpoint_dict(checkpoint: Any, checkpoint_path: Path) -> dict[str, Any]:
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
        calendar_features=str(raw_config.get("calendar_features", "none")),
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
    *,
    allow_unsafe_pickle: bool = False,
) -> LoadedUNetCheckpoint:
    """
    Load a U-Net checkpoint, rebuild the model, load weights, and switch to eval mode.

    Args:
        checkpoint_path: File to load.
        device: Device to place the rebuilt model on.
        allow_unsafe_pickle: Permit arbitrary pickle deserialization. See
            :func:`safe_torch_load`; only for a checkpoint whose origin you
            can vouch for.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = safe_torch_load(
        checkpoint_path, device, allow_unsafe_pickle=allow_unsafe_pickle
    )

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
