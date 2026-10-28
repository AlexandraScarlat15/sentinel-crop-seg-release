"""Create confusion-matrix figures and class metrics from saved evaluation files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from sentinel_crop_seg.core.evaluation.reporting import DEFAULT_CLASS_NAMES

SUPPORTED_FORMATS = ("png", "pdf", "svg")


def read_evaluation(
    evaluation_dir: str | Path, *, class_names_json: str | Path | None = None
) -> tuple[np.ndarray, dict, list[str]]:
    """Validate a saved evaluation report and resolve the displayed taxonomy.

    Args:
        evaluation_dir: Folder containing confusion_matrix_counts.npy and summary.json.
        class_names_json: Optional JSON array of class names in numerical class-ID order.

    Returns:
        Nonnegative integer count matrix, saved summary and class names.

    Raises:
        FileNotFoundError: If required inputs are missing.
        ValueError: If class identities or count matrix are inconsistent.
    """
    evaluation_dir = Path(evaluation_dir)
    counts = evaluation_dir / "confusion_matrix_counts.npy"
    summary_path = evaluation_dir / "summary.json"
    if not counts.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Expected confusion_matrix_counts.npy and summary.json")
    matrix = np.load(counts, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("Confusion matrix must be a nonempty square array")
    if matrix.dtype.kind not in ("i", "u") or np.any(matrix < 0):
        raise ValueError("Confusion matrix counts must be nonnegative integers")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Evaluation summary must be a JSON object")
    size = int(matrix.shape[0])
    if "num_classes" in summary and (
        isinstance(summary["num_classes"], bool) or
        not isinstance(summary["num_classes"], int) or
        summary["num_classes"] != size
    ):
        raise ValueError("Summary class count does not match confusion matrix")

    if class_names_json is not None:
        names = json.loads(Path(class_names_json).read_text(encoding="utf-8"))
    elif (evaluation_dir / "metrics_per_class.csv").is_file():
        with (evaluation_dir / "metrics_per_class.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        try:
            ids = [int(row["class_id"]) for row in rows]
            names = [row["class_name"] for row in rows]
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid class identity columns in metrics_per_class.csv") from exc
        if ids != list(range(size)):
            raise ValueError("metrics_per_class.csv class IDs do not match the confusion matrix")
    elif size == len(DEFAULT_CLASS_NAMES):
        names = list(DEFAULT_CLASS_NAMES)
    else:
        names = [f"class_{i}" for i in range(size)]
    if (
        not isinstance(names, list) or len(names) != size or
        any(not isinstance(name, str) or not name.strip() for name in names) or
        len(set(names)) != size
    ):
        raise ValueError("Class names must be unique, nonempty strings in class-ID order")
    return matrix, summary, names


def metrics_from_counts(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate per-class scores from rows=truth, columns=prediction counts.

    Args:
        matrix: Valid nonnegative integer count matrix.

    Returns:
        Precision, recall, IoU, Dice/F1, support and row-normalized fractions.
        Undefined zero-denominator rates are represented as 0, not NaN.
    """
    matrix = matrix.astype(np.float64)
    tp = np.diag(matrix)
    true = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    union = true + predicted - tp
    return {
        "precision": divide(tp, predicted),
        "recall": divide(tp, true),
        "iou": divide(tp, union),
        "dice_f1": divide(2 * tp, true + predicted),
        "support": true.astype(np.int64),
        "predicted": predicted.astype(np.int64),
        "normalized": divide(matrix, true[:, None]),
    }


def render_evaluation_plots(
    evaluation_dir: str | Path,
    output_dir: str | Path,
    *,
    class_names_json: str | Path | None = None,
    model_name: str = "U-Net",
    formats: tuple[str, ...] = SUPPORTED_FORMATS,
    dpi: int = 220,
) -> dict[str, Path]:
    """Create confusion matrices, IoU bars and tabular scores without inference.

    Args:
        evaluation_dir: Precomputed numeric evaluation report directory.
        output_dir: Fresh output directory; existing paths are not overwritten.
        class_names_json: Optional JSON list of class labels in class-ID order.
        model_name: Figure title prefix.
        formats: Figure formats selected from png, pdf and svg.
        dpi: Resolution of PNG images.

    Returns:
        Mapping of produced figure and CSV filenames to output paths.

    Raises:
        FileExistsError: If the output destination already exists.
        ValueError: If counts, labels, formats or plot settings are invalid.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite plot directory: {output_dir}")
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(f not in SUPPORTED_FORMATS for f in formats):
        raise ValueError("Formats must be unique selections of png, pdf, svg")
    if dpi < 72 or not model_name.strip():
        raise ValueError("DPI must be >= 72 and model name must be nonempty")
    cm, summary, names = read_evaluation(evaluation_dir, class_names_json=class_names_json)
    measures = metrics_from_counts(cm)
    n = len(names)
    output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Path] = {}

    def save(fig: plt.Figure, stem: str) -> None:
        try:
            for fmt in formats:
                path = output_dir / f"{stem}.{fmt}"
                fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
                result[path.name] = path
        finally:
            plt.close(fig)

    def matrix_plot(data: np.ndarray, title: str, stem: str, normalized: bool) -> None:
        size = max(7.5, min(17, 0.82 * n + 4))
        fig, ax = plt.subplots(figsize=(size + 1.5, size))
        shown = data * 100 if normalized else data
        im = ax.imshow(shown, cmap="viridis",
                       norm=Normalize(vmin=0, vmax=100) if normalized else None,
                       interpolation="nearest")
        bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label("Share of true class (%)" if normalized else "Pixel count")
        ax.set_xticks(range(n), names, rotation=45, ha="right")
        ax.set_yticks(range(n), names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground-truth class")
        ax.set_title(f"{model_name} — {title}")
        if n <= 20:
            mid = (float(np.max(shown)) + float(np.min(shown))) / 2
            for i in range(n):
                for j in range(n):
                    text = f"{shown[i, j]:.1f}" if normalized else f"{int(shown[i, j]):,}"
                    ax.text(j, i, text, ha="center", va="center",
                            color="white" if shown[i, j] < mid else "black",
                            fontsize=max(6.5, 10 - n / 5))
        save(fig, stem)

    matrix_plot(cm, "confusion counts", "confusion_matrix_counts", False)
    matrix_plot(measures["normalized"], "row-normalized confusion (%)",
                "confusion_matrix_row_norm", True)

    fig, ax = plt.subplots(figsize=(9, max(4.5, n * 0.46)))
    y = np.arange(n)
    ax.barh(y, measures["iou"] * 100, color="#456B9B")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Intersection over Union (%)")
    ax.set_title(f"{model_name} — per-class IoU")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "iou_per_class")

    csv_path = output_dir / "metrics_per_class_pretty.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "iou_percent", "dice_f1_percent",
                         "precision_percent", "recall_percent", "support_pixels", "predicted_pixels"])
        for index, name in enumerate(names):
            writer.writerow([
                index, name,
                *(round(100 * float(measures[key][index]), 4)
                  for key in ("iou", "dice_f1", "precision", "recall")),
                int(measures["support"][index]), int(measures["predicted"][index]),
            ])
    result[csv_path.name] = csv_path

    counts_csv = output_dir / "confusion_matrix_counts.csv"
    norm_csv = output_dir / "confusion_matrix_row_norm_percent.csv"
    np.savetxt(counts_csv, cm, fmt="%d", delimiter=",")
    np.savetxt(norm_csv, 100 * measures["normalized"], fmt="%.6f", delimiter=",")
    result[counts_csv.name] = counts_csv
    result[norm_csv.name] = norm_csv

    summary_path = output_dir / "plots_summary.json"
    summary_path.write_text(json.dumps({
        "source_evaluation_dir": str(evaluation_dir),
        "num_classes": n,
        "class_names": names,
        "num_samples_evaluated": summary.get("num_samples_evaluated"),
        "generated_from_saved_counts": True,
        "no_inference_or_training_performed": True,
        "notes": "Undefined per-class fractions with zero denominators are plotted as zero.",
    }, indent=2) + "\n", encoding="utf-8")
    result[summary_path.name] = summary_path
    return result
