"""U-Net-only segmentation evaluation without Random Forest dependencies.

Example:
  python -m sentinel_crop_seg.cli.evaluate_unet --checkpoint model.pt \
    --manifest-csv split_manifest.csv --split test --output-dir outputs/eval_unet
"""
from __future__ import annotations

import argparse

from sentinel_crop_seg.cli._unet_runtime import add_dataset_arguments, prepare_runtime
from sentinel_crop_seg.core.evaluation.evaluator import evaluate_segmentation_predictor
from sentinel_crop_seg.core.evaluation.reporting import DEFAULT_CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--ignore-background", action="store_true")
    args = parser.parse_args()

    runtime = prepare_runtime(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = evaluate_segmentation_predictor(
        predictor=runtime.model,
        dataloader=runtime.dataloader,
        num_classes=runtime.checkpoint.config.num_classes,
        device=runtime.device,
        output_dir=args.output_dir,
        ignore_index=args.ignore_index,
        ignore_background=args.ignore_background,
        class_names=(
            DEFAULT_CLASS_NAMES
            if runtime.checkpoint.config.num_classes == len(DEFAULT_CLASS_NAMES)
            else None
        ),
        summary_metadata=runtime.provenance,
    )
    print(f"[U-NET] evaluated {result.summary['num_samples_evaluated']} samples")
    print(f"[U-NET] mIoU all classes: {result.summary['mIoU_all_classes']:.6f}")
    print(f"[U-NET] summary: {result.saved_paths['summary_json']}")


if __name__ == "__main__":
    main()
