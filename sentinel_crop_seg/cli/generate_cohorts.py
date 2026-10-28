"""Preview or execute independently configured Sentinel-2/EuroCrops cohorts."""
from __future__ import annotations

import argparse
import json

from sentinel_crop_seg.core.pipeline.cohort_plan import run_cohorts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Versioned multi-cohort YAML plan.")
    parser.add_argument("--config-dir", default="config",
                        help="Directory containing the seven baseline generator YAMLs.")
    parser.add_argument("--execute", action="store_true",
                        help="Download/process each cohort; default is read-only validation.")
    args = parser.parse_args()
    print(json.dumps(run_cohorts(
        args.plan, config_dir=args.config_dir, execute=args.execute
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
