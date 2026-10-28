"""Plot saved U-Net confusion matrices and IoU without inference or training."""
from __future__ import annotations

import argparse
from pathlib import Path

from sentinel_crop_seg.core.evaluation.plots import SUPPORTED_FORMATS, render_evaluation_plots


def main() -> None:
    """Plot precomputed evaluation reports without loading a model checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Existing report with confusion_matrix_counts.npy and summary.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh destination for figures.")
    parser.add_argument("--class-names-json", type=Path,
                        help="Optional JSON list of class labels in numeric class-ID order.")
    parser.add_argument("--model-name", default="U-Net", help="Title prefix for figures.")
    parser.add_argument("--formats", nargs="+", choices=SUPPORTED_FORMATS,
                        default=list(SUPPORTED_FORMATS), help="Figure output formats.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution; minimum 72.")
    args = parser.parse_args()
    paths = render_evaluation_plots(
        args.eval_dir, args.out_dir, class_names_json=args.class_names_json,
        model_name=args.model_name, formats=tuple(args.formats), dpi=args.dpi,
    )
    print(f"[PLOT] Evaluation: wrote {len(paths)} files into {args.out_dir}")


if __name__ == "__main__":
    main()
