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
        "--config-mode",
        choices=("legacy", "isolated"),
        default=None,
        help=(
            "legacy merges every config/*.yaml (historical behavior); "
            "isolated reads only generator YAMLs in declared order."
        ),
    )

    parser.add_argument(
        "--resume-policy",
        choices=("legacy", "strict"),
        default=None,
        help=(
            "legacy keeps historical processed-state behavior; "
            "strict binds new isolated runs to a fingerprinted, fail-closed ledger."
        ),
    )

    parser.add_argument("--experiment", help="Versioned YAML experiment recipe.")

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

    config_mode = args.config_mode or ("isolated" if args.experiment else "legacy")
    resume_policy = args.resume_policy or ("strict" if args.experiment else "legacy")
    if args.experiment and config_mode != "isolated":
        raise ValueError("--experiment requires --config-mode isolated")

    result = generate_dataset(
        config_dir=args.config_dir,
        run_tag=args.run_tag,
        sleep_seconds=args.sleep_seconds,
        overrides=args.override,
        config_mode=config_mode,
        resume_policy=resume_policy,
        experiment_recipe=args.experiment,
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
    print(f"[CLI] search_failures={result.search_failures}")
    print(f"[CLI] processed_state_path={result.processed_state_path}")

    # A run that saved nothing *and* faulted is a failure, not an empty
    # region. Reporting success here let a misconfigured run -- a rejected
    # search parameter, wrong credentials, an unreachable catalogue -- exit
    # zero with an empty output directory, which is worse than a crash
    # because a script downstream cannot tell the two apart.
    if result.total_samples == 0 and result.search_failures:
        raise SystemExit(
            f"[CLI] FAILED: no samples were generated and the catalog search "
            f"faulted for {result.search_failures} of {result.selected_rois} "
            f"ROIs. The output directory is empty. Check credentials, "
            f"network access and the catalog.stac settings; a search "
            f"parameter rejected by the API faults every ROI identically."
        )


if __name__ == "__main__":
    main()