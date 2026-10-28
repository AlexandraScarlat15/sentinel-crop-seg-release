"""Apply explicit human-approved TRAIN-only exclusions from OOF label review."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.quality.post_training_filter import review_and_filter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="Original immutable manifest, including split and cohort.")
    parser.add_argument("--parcel-review", required=True,
                        help="OOF parcel_review_levels.csv from the six-fold protocol.")
    parser.add_argument("--parcel-links", required=True,
                        help="Parcel admissibility join: cohort/parcel_key/dataset/sample_id.")
    parser.add_argument("--decisions",
                        help="Optional manually reviewed cohort,parcel_key,decision CSV.")
    parser.add_argument("--output-dir", required=True,
                        help="Fresh output directory; never overwrites source inputs.")
    parser.add_argument("--apply-approved-exclusions", action="store_true",
                        help="Explicitly exclude *whole train patches* for approved decisions.")
    args = parser.parse_args()
    result = review_and_filter(
        args.manifest, args.parcel_review, args.parcel_links, args.output_dir,
        decisions_file=args.decisions,
        apply_approved_exclusions=args.apply_approved_exclusions,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
