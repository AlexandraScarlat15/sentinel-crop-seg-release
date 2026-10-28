"""Command-line entry point for geographically separated dataset splits."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.splitting.spatial_split import run_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create audited spatial train/val/test splits.")
    parser.add_argument("--config", required=True, help="Standalone YAML splitting configuration.")
    parser.add_argument("--manifest", required=True, help="Georeferenced input CSV.")
    parser.add_argument("--out-dir", required=True, help="New output directory (must not exist).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Compute and validate all split constraints without writing files.")
    args = parser.parse_args()
    result = run_from_files(args.config, args.manifest, args.out_dir,
                            validate_only=args.validate_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
