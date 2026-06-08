from __future__ import annotations

import argparse

from sentinel_crop_seg import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Sentinel-2 / EuroCrops segmentation dataset."
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory to load.",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="v2",
        help=(
            "Suffix used for the processed state file, "
            "for example v1, v2, hotcell_small, or e2e_smoke."
        ),
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between processed products, useful for polite API/cache usage.",
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override configuration values using dotted keys. "
            "Example: --override dataset.target.total_patches=8. "
            "Can be used multiple times."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = generate_dataset(
        config_dir=args.config_dir,
        run_tag=args.run_tag,
        sleep_seconds=args.sleep_seconds,
        overrides=args.override,
    )

    print("\n[CLI] Dataset generation result")
    print(f"[CLI] output_dir={result.output_dir}")
    print(f"[CLI] total_samples={result.total_samples}")
    print(f"[CLI] selected_rois={result.selected_rois}")
    print(f"[CLI] processed_products={result.processed_products}")
    print(f"[CLI] strategy={result.strategy}")
    print(f"[CLI] satellite={result.satellite}")
    print(f"[CLI] label_source={result.label_source}")
    print(f"[CLI] run_tag={result.run_tag}")
    print(f"[CLI] processed_state_path={result.processed_state_path}")


if __name__ == "__main__":
    main()