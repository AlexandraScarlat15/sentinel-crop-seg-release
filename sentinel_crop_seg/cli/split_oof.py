"""Plan geographic out-of-fold training, inner-val and held-out manifests.

No model is trained here. The historical experimental six-fold artifacts are
never modified; publication is a separate process after complete validation.
"""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.splitting.spatial_oof_folds import run_spatial_oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML with oof_folds settings.")
    parser.add_argument("--manifest", required=True,
                        help="Frozen source split with spatial/temporal provenance.")
    parser.add_argument("--out-dir", required=True,
                        help="Fresh destination; no files are overwritten.")
    parser.add_argument("--parcel-links",
                        help="Optional cohort/parcel_key/dataset/sample_id link table; "
                             "improves physical parcel leakage protection when complete.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Audit every fold without creating any output.")
    args = parser.parse_args()
    report = run_spatial_oof(
        args.config, args.manifest, args.out_dir,
        parcel_links_path=args.parcel_links, validate_only=args.validate_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
