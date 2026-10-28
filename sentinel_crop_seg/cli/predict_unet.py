"""U-Net-only mask prediction CLI for manifest or root-folder samples.

Creates predicted NPY class masks and optionally colored PNG previews;
does not modify the input images, labels, manifest or model checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import re

import numpy as np
import torch

from sentinel_crop_seg.cli._unet_runtime import add_dataset_arguments, prepare_runtime
from sentinel_crop_seg.core.utils.class_palette import CLASS_INFO


def safe_name(index: int, sample_id: str) -> str:
    stem = str(sample_id).replace("\\", "/").strip("/").replace("/", "__")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:120] or "sample"
    return f"{index:06d}_{stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--save-color", action="store_true",
                        help="Also write a colorized PNG for each predicted mask.")
    args = parser.parse_args()
    runtime = prepare_runtime(args)
    if args.save_color and runtime.checkpoint.config.num_classes > len(CLASS_INFO):
        raise ValueError("Colored previews require the standard 11-class taxonomy")

    target = args.output_dir
    target.mkdir(parents=True, exist_ok=False)
    masks_dir = target / "pred_masks_npy"
    masks_dir.mkdir()
    if args.save_color:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = np.array([
            [int(color[pos:pos+2], 16) for pos in (1, 3, 5)]
            for _, _, color in CLASS_INFO
        ], dtype=np.uint8)
        color_dir = target / "pred_color"
        color_dir.mkdir()

    rows: list[dict[str, str | int]] = []
    count = 0
    runtime.model.eval()
    with torch.inference_mode():
        for images, _ in runtime.dataloader:
            logits = runtime.model(images.to(runtime.device))
            if logits.ndim != 4 or logits.shape[1] != runtime.checkpoint.config.num_classes:
                raise ValueError("Unexpected model output shape for U-Net prediction")
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            for prediction in predictions:
                sample = runtime.dataset.samples[count]
                dataset_name = getattr(sample, "dataset", sample.image_path.parent.parent.name)
                sample_id = getattr(sample, "sample_id", sample.image_path.stem)
                stem = safe_name(count, str(sample_id))
                npy_file = masks_dir / (stem + ".npy")
                np.save(npy_file, prediction.astype(np.uint8 if len(CLASS_INFO) >= runtime.checkpoint.config.num_classes else np.int16))
                if args.save_color:
                    plt.imsave(color_dir / (stem + ".png"), colors[prediction])
                rows.append({
                    "output_index": count,
                    "dataset": str(dataset_name),
                    "sample_id": str(sample_id),
                    "split": str(getattr(sample, "split", "")),
                    "image_path": str(sample.image_path),
                    "prediction_path": npy_file.relative_to(target).as_posix(),
                })
                count += 1

    with (target / "prediction_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "output_index", "dataset", "sample_id", "split",
            "image_path", "prediction_path",
        ])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **runtime.provenance, "num_predictions": count,
        "prediction_samples_csv": "prediction_samples.csv",
        "pred_masks_npy_dir": "pred_masks_npy",
        "pred_color_dir": "pred_color" if args.save_color else None,
    }
    (target / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[U-NET] predicted {count} samples; outputs: {target}")


if __name__ == "__main__":
    main()
