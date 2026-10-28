"""Render training curves from persisted U-Net metrics without training or inference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc", "lr"
)
SUPPORTED_FORMATS = ("png", "pdf", "svg")


def load_training_history(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Validate the recorded epoch metrics and their checkpoint selection.

    Args:
        run_dir: Existing training output directory.

    Returns:
        Sorted metric history and the original training summary.

    Raises:
        FileNotFoundError: If either required training report is missing.
        ValueError: If epoch identities, metric values or best-epoch evidence disagree.
    """
    csv_path = run_dir / "training_metrics.csv"
    summary_path = run_dir / "training_summary.json"
    if not csv_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Expected training_metrics.csv and training_summary.json")
    frame = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(frame)
    if missing:
        raise ValueError("Missing training history columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Training history must not be empty")
    for column in REQUIRED_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Training history has nonnumeric values in {column}") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Training history contains nonfinite values in {column}")
    epoch = frame["epoch"].to_numpy(dtype=float)
    if np.any(epoch <= 0) or np.any(epoch != np.floor(epoch)) or frame["epoch"].duplicated().any():
        raise ValueError("Epochs must be unique positive integers")
    if (frame["lr"] <= 0).any():
        raise ValueError("Learning rates must be positive")
    for column in ("val_mIoU", "val_pixel_acc"):
        if ((frame[column] < 0) | (frame[column] > 1)).any():
            raise ValueError(f"{column} must be in [0, 1]")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or "best_epoch" not in summary or "best_val_mIoU" not in summary:
        raise ValueError("Training summary lacks best checkpoint evidence")
    best_epoch = summary["best_epoch"]
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool):
        raise ValueError("best_epoch must be an integer")
    frame = frame.sort_values("epoch")
    selected = frame.loc[frame["epoch"] == best_epoch]
    if len(selected) != 1:
        raise ValueError("Best epoch is absent from training history")
    best_iou = float(summary["best_val_mIoU"])
    if not np.isfinite(best_iou) or abs(float(selected.iloc[0]["val_mIoU"]) - best_iou) > 1e-5:
        raise ValueError("Best validation mIoU differs from training history")
    return frame, summary


def render_training_curves(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "U-Net",
    formats: tuple[str, ...] = SUPPORTED_FORMATS,
    dpi: int = 220,
) -> dict[str, Path]:
    """Plot loss, validation scores and learning rate from an existing run.

    Args:
        run_dir: Folder with training_metrics.csv and training_summary.json.
        output_dir: New folder for plots; existing destinations are rejected.
        model_name: Figure title prefix.
        formats: File extensions selected from png, pdf and svg.
        dpi: Raster figure resolution.

    Returns:
        Mapping from generated figure filenames to their new paths.

    Raises:
        FileExistsError: If the output directory already exists.
        ValueError: If the input metrics, formats or plot settings are invalid.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite plot directory: {output_dir}")
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(f not in SUPPORTED_FORMATS for f in formats):
        raise ValueError("Formats must be unique selections of png, pdf, svg")
    if dpi < 72 or not model_name.strip():
        raise ValueError("DPI must be >= 72 and model name must be nonempty")
    frame, summary = load_training_history(run_dir)
    epoch = frame["epoch"].to_numpy(dtype=int)
    best_epoch = int(summary["best_epoch"])
    output_dir.mkdir(parents=True, exist_ok=False)
    written: dict[str, Path] = {}

    def save(figure: plt.Figure, stem: str) -> None:
        try:
            for fmt in formats:
                target = output_dir / f"{stem}.{fmt}"
                figure.savefig(target, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
                written[target.name] = target
        finally:
            plt.close(figure)

    def axes(title: str, y_label: str) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.set_title(f"{model_name} — {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.axvline(best_epoch, linestyle="--", linewidth=1.2, color="#72449E",
                   label=f"Best epoch ({best_epoch})")
        ax.grid(alpha=0.25)
        return fig, ax

    fig, ax = axes("training and validation loss", "Loss")
    ax.plot(epoch, frame["train_loss"], label="Training loss", color="#355C7D")
    ax.plot(epoch, frame["val_loss"], label="Validation loss", color="#D46A26")
    ax.legend()
    save(fig, "training_validation_loss")

    fig, ax = axes("validation performance", "Score (%)")
    ax.plot(epoch, 100 * frame["val_mIoU"], label="Validation mIoU", color="#168B7A")
    ax.plot(epoch, 100 * frame["val_pixel_acc"], label="Validation pixel accuracy", color="#C68618")
    ax.set_ylim(0, 100)
    ax.legend()
    save(fig, "validation_performance")

    fig, ax = axes("learning-rate schedule", "Learning rate")
    ax.step(epoch, frame["lr"], where="post", label="Learning rate", color="#5941A1")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend()
    save(fig, "learning_rate_schedule")
    return written
