"""Combine independently adapted country/year manifests without generating data."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.data.cohort_manifest_merge import merge_cohort_manifests


def main() -> None:
    """Preview or write a new merged manifest for subsequent spatial splitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True,
                        help="Input adapted, unsplit CSV; repeat for each cohort (at least two).")
    parser.add_argument("--output-dir", required=True,
                        help="Fresh directory for combined_manifest.csv and merge_audit.json.")
    parser.add_argument("--verify-assets", action="store_true",
                        help="Verify existing image/mask file paths; never modify assets.")
    parser.add_argument("--apply", action="store_true",
                        help="Write outputs; default is a read-only JSON preview.")
    args = parser.parse_args()
    print(json.dumps(merge_cohort_manifests(
        args.manifest, args.output_dir,
        verify_assets=args.verify_assets,
        validate_only=not args.apply,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
