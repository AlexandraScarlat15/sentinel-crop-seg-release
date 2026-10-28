"""Offline audit for the resolved Sentinel-2/EuroCrops generator recipe."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.pipeline.generation_config import generation_config_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect generator YAML selection, patch sampling and recipe fingerprint."
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--config-mode", choices=("isolated", "legacy"), default="isolated")
    parser.add_argument("--experiment", help="Versioned YAML generation experiment recipe.")
    parser.add_argument("--override", action="append", default=[],
                        help="Repeatable dotted-key override (same syntax as generate CLI).")
    args = parser.parse_args()
    report = generation_config_report(
        args.config_dir, mode=args.config_mode, overrides=args.override,
        experiment_recipe=args.experiment
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
