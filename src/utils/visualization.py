from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.utils.class_palette import (
    CLASS_INFO,
    IGNORE_COLOR,
    IGNORE_LABEL,
    IGNORE_NAME,
)


def make_cmap_and_norm(
    include_ignore: bool = True,
) -> tuple[ListedColormap, BoundaryNorm]:
    colors = [color for _, _, color in CLASS_INFO]

    if include_ignore:
        colors.append(IGNORE_COLOR)

    cmap = ListedColormap(colors)
    boundaries = [index - 0.5 for index in range(len(colors) + 1)]
    norm = BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


def mask_for_display(mask: np.ndarray, ignore_index: int = IGNORE_LABEL) -> np.ndarray:
    display = np.full(mask.shape, len(CLASS_INFO), dtype=np.int32)

    for display_index, (class_id, _, _) in enumerate(CLASS_INFO):
        display[mask == class_id] = display_index

    display[mask == ignore_index] = len(CLASS_INFO)

    return display


def make_legend_handles(include_ignore: bool = True) -> list[mpatches.Patch]:
    handles = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            label=f"{class_id}: {name}",
        )
        for class_id, name, color in CLASS_INFO
    ]

    if include_ignore:
        handles.append(
            mpatches.Patch(
                facecolor=IGNORE_COLOR,
                edgecolor="black",
                label=f"{IGNORE_LABEL}: {IGNORE_NAME}",
            )
        )

    return handles


def save_legend_png(output_path: Path) -> None:
    handles = make_legend_handles(include_ignore=True)

    fig_height = 0.35 * len(handles) + 0.5
    fig, ax = plt.subplots(figsize=(3.2, fig_height), dpi=220)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        handleheight=1.0,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def stretch_rgb(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] < 3:
        raise ValueError(f"Expected at least 3 channels for RGB, got {image_chw.shape}")

    # Project order:
    # 0 B02, 1 B03, 2 B04, 3 B08, 4 B05, 5 B06, 6 B07, 7 B8A, 8 B11, 9 B12.
    rgb = np.stack(
        [
            image_chw[2],
            image_chw[1],
            image_chw[0],
        ],
        axis=-1,
    ).astype(np.float32)

    out = np.zeros_like(rgb, dtype=np.float32)

    for channel_index in range(3):
        channel = rgb[..., channel_index]
        valid = np.isfinite(channel)

        if not np.any(valid):
            continue

        low, high = np.percentile(channel[valid], [2, 98])

        if high <= low:
            high = low + 1.0

        out[..., channel_index] = np.clip((channel - low) / (high - low), 0.0, 1.0)

    return out


def save_visualization(
    output_path: Path,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    ignore_index: int,
    title: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    include_legend: bool,
) -> None:
    gt_display = mask_for_display(gt_mask, ignore_index=ignore_index)
    pred_display = mask_for_display(pred_mask, ignore_index=ignore_index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(gt_display, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(pred_display, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11)

    if include_legend:
        handles = make_legend_handles(include_ignore=True)
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            fontsize=7,
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)