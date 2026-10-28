"""Convert DataEngine patches_log.csv v3 into a georeferenced training manifest."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.data.patch_log_manifest import adapt_patch_log_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a generator patch log v3 into a manifest for Spatial Splitting."
    )
    parser.add_argument("--patch-log", required=True, help="Original patches_log.csv (read-only).")
    parser.add_argument("--output", required=True, help="New CSV path; refuses to overwrite.")
    parser.add_argument("--dataset-id", required=True, help="Explicit unique dataset/cohort namespace.")
    parser.add_argument("--format", choices=["npy", "geotiff"], default="npy",
                        dest="asset_format", help="Use NPY or GeoTIFF image-mask pairs.")
    parser.add_argument("--country", help="Explicit country code; never inferred.")
    parser.add_argument("--cohort", help="Explicit country/year cohort; never inferred.")
    parser.add_argument("--label-source", help="Optional declared label provider, e.g. eurocrops.")
    parser.add_argument("--label-source-ref",
                        help="Optional exact annotation source/version identifier.")
    parser.add_argument("--verify-assets", action="store_true",
                        help="Check the chosen image/mask files exist (does not read imagery).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate and summarize without writing any output.")
    args = parser.parse_args()

    report = adapt_patch_log_file(
        patch_log=args.patch_log,
        output_manifest=args.output,
        dataset_id=args.dataset_id,
        asset_format=args.asset_format,
        country=args.country,
        cohort=args.cohort,
        label_source=args.label_source,
        label_source_ref=args.label_source_ref,
        verify_assets=args.verify_assets,
        validate_only=args.validate_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
