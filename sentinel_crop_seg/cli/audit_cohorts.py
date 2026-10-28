"""Audit the country/year cohorts and fixed crop-calendar coverage of a manifest."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.quality.calendar_cohort_audit import audit_calendar_cohorts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Frozen or draft manifest CSV.")
    parser.add_argument("--calendar", default="config/crop_calendar_recalibrated_v0_3.csv")
    parser.add_argument("--output-dir", help="New directory for summary and coverage CSVs.")
    parser.add_argument("--require-full-calendar", action="store_true",
                        help="Fail if any country/crop calendar pair is missing.")
    args = parser.parse_args()
    print(json.dumps(audit_calendar_cohorts(
        args.manifest, args.calendar, output_dir=args.output_dir,
        require_full_calendar=args.require_full_calendar,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
