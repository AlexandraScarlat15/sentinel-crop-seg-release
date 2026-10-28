"""Train-only pixel-weighted crop-calendar sensitivity audit and safe selection."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.quality.weighted_calendar_selection import build_weighted_calendar_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest with split, country, date and p0..pN")
    parser.add_argument("--calendar", default="config/crop_calendar_recalibrated_v0_3.csv")
    parser.add_argument("--taxonomy", default="config/classes.yaml")
    parser.add_argument("--binding", default="config/calendars/class_calendar_binding_v1.yaml")
    parser.add_argument("--threshold", default="0.90", help="Fraction of compatible agricultural pixels.")
    parser.add_argument("--sweep", nargs="+", default=["0.75", "0.90", "1.00"],
                        help="Comparison thresholds; TRAIN only, never VAL or TEST.")
    parser.add_argument("--output-dir", required=True,
                        help="New directory for optional derived manifest and audit files.")
    parser.add_argument("--apply", action="store_true",
                        help="Write a NEW training-filtered manifest; default only validates.")
    args = parser.parse_args()
    report = build_weighted_calendar_selection(
        args.manifest, args.calendar, args.taxonomy, args.binding,
        args.output_dir, threshold=args.threshold, sweep=tuple(args.sweep),
        validate_only=not args.apply,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
