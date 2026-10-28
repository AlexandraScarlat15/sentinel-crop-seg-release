"""Render existing U-Net training history as publication figures without training."""
from __future__ import annotations

import argparse
from pathlib import Path

from sentinel_crop_seg.core.training.plots import SUPPORTED_FORMATS, render_training_curves


def main() -> None:
    """Plot recorded training curves without loading checkpoint or input data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing training run with training_metrics.csv and training_summary.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh destination for figures.")
    parser.add_argument("--model-name", default="U-Net", help="Title prefix for figures.")
    parser.add_argument("--formats", nargs="+", choices=SUPPORTED_FORMATS,
                        default=list(SUPPORTED_FORMATS), help="Figure output formats.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution; minimum 72.")
    args = parser.parse_args()
    paths = render_training_curves(
        args.run_dir, args.out_dir, model_name=args.model_name,
        formats=tuple(args.formats), dpi=args.dpi,
    )
    print(f"[PLOT] Training: wrote {len(paths)} files into {args.out_dir}")


if __name__ == "__main__":
    main()
