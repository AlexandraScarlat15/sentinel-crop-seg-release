from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_command_for_log(command: list[str]) -> str:
    redacted: list[str] = []

    for item in command:
        if item.startswith("eurocrops.shapefile_path="):
            redacted.append("eurocrops.shapefile_path=<EUROCROPS_SHAPEFILE_PATH>")
        else:
            redacted.append(item)

    return " ".join(redacted)


def run_command(command: list[str], cwd: Path) -> None:
    print()
    print("[E2E] Running command:")
    print("[E2E] " + format_command_for_log(command))

    result = subprocess.run(command, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}: {format_command_for_log(command)}"
        )


def require_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")


def require_dir(path: Path, message: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{message}: {path}")


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0

    return sum(1 for _ in path.glob(pattern))


def remove_path(path: Path) -> None:
    if not path.exists():
        return

    print(f"[E2E] Removing: {path}")

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def require_cli(command_name: str) -> None:
    if shutil.which(command_name) is None:
        raise RuntimeError(
            f"CLI command not found: {command_name}. "
            "Run `python -m pip install -e .` from the project root first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal end-to-end smoke test for the Sentinel crop "
            "segmentation pipeline. The test runs dataset generation, U-Net "
            "training, evaluation, and prediction using the public CLI commands."
        )
    )

    parser.add_argument(
        "--eurocrops-shapefile",
        default=os.environ.get("EUROCROPS_SHAPEFILE_PATH"),
        help=(
            "Path to the EuroCrops shapefile. If omitted, the script uses the "
            "EUROCROPS_SHAPEFILE_PATH environment variable."
        ),
    )

    parser.add_argument(
        "--config-dir",
        default="config",
        help="Configuration directory used by sentinel-crop-generate.",
    )

    parser.add_argument(
        "--train-config",
        default="config/train_unet.yaml",
        help="Training configuration used by sentinel-crop-train-unet.",
    )

    parser.add_argument(
        "--run-tag",
        default="e2e_smoke",
        help="Run tag used by dataset generation.",
    )

    parser.add_argument(
        "--dataset-root",
        default="data/processed/e2e_smoke_dataset/generated",
        help="Generated dataset root used for train/evaluate/predict.",
    )

    parser.add_argument(
        "--train-output-dir",
        default="outputs/e2e_smoke/unet_generated",
        help="Training output directory.",
    )

    parser.add_argument(
        "--eval-output-dir",
        default="outputs/e2e_smoke/eval_generated",
        help="Evaluation output directory.",
    )

    parser.add_argument(
        "--predict-output-dir",
        default="outputs/e2e_smoke/predict_generated",
        help="Prediction output directory.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device used for evaluation and prediction.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Sample limit used for training, evaluation, and prediction.",
    )

    parser.add_argument(
        "--target-patches",
        type=int,
        default=8,
        help="Number of dataset patches requested from the generator.",
    )

    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation and reuse an existing generated dataset.",
    )

    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip the prediction step.",
    )

    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove previous E2E outputs before running.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT

    print("[E2E] Sentinel crop segmentation smoke test")
    print(f"[E2E] Project root: {project_root}")

    config_dir = Path(args.config_dir)
    train_config = Path(args.train_config)
    dataset_root = Path(args.dataset_root)
    train_output_dir = Path(args.train_output_dir)
    eval_output_dir = Path(args.eval_output_dir)
    predict_output_dir = Path(args.predict_output_dir)

    if not (project_root / config_dir).is_dir():
        raise FileNotFoundError(f"Config directory not found: {project_root / config_dir}")

    if not (project_root / train_config).is_file():
        raise FileNotFoundError(f"Training config not found: {project_root / train_config}")

    require_cli("sentinel-crop-generate")
    require_cli("sentinel-crop-train-unet")
    require_cli("sentinel-crop-evaluate")

    if not args.skip_predict:
        require_cli("sentinel-crop-predict")

    if not args.skip_generate:
        if not args.eurocrops_shapefile:
            raise ValueError(
                "EuroCrops shapefile path is required. Provide it with "
                "`--eurocrops-shapefile` or set EUROCROPS_SHAPEFILE_PATH."
            )

        eurocrops_shapefile = Path(args.eurocrops_shapefile).expanduser()

        if not eurocrops_shapefile.is_file():
            raise FileNotFoundError(f"EuroCrops shapefile not found: {eurocrops_shapefile}")

        eurocrops_shapefile_value = str(eurocrops_shapefile.resolve()).replace("\\", "/")
    else:
        eurocrops_shapefile_value = ""

    if not args.no_clean:
        remove_path(project_root / "data/processed/e2e_smoke_dataset")
        remove_path(project_root / "outputs/e2e_smoke")

        remove_path(
            project_root
            / "data/raw/sentinel"
            / f"processed_hotcell_{args.run_tag}.json"
        )
        remove_path(
            project_root
            / "data/raw/sentinel"
            / f"processed_hotcell_{args.run_tag}.json"
        )

    if not args.skip_generate:
        print()
        print("[E2E] Step 1/4: dataset generation")

        generate_command = [
            "sentinel-crop-generate",
            "--config-dir",
            args.config_dir,
            "--run-tag",
            args.run_tag,
            "--sleep-seconds",
            "0.5",
            "--override",
            f"eurocrops.shapefile_path={eurocrops_shapefile_value}",
            "--override",
            "paths.dataset_output_dir=data/processed/e2e_smoke_dataset/generated",
            "--override",
            f"dataset.target.total_patches={args.target_patches}",
            "--override",
            "dataset.max_patches_per_roi=2",
            "--override",
            "dataset.roi.top_n_rois=5",
            "--override",
            "dataset.roi.hotcell.top_k_cells=20",
            "--override",
            "dataset.roi.hotcell.rois_from_cells=5",
            "--override",
            "catalog.stac.limit=5",
        ]

        run_command(generate_command, cwd=project_root)
    else:
        print()
        print("[E2E] Step 1/4: dataset generation skipped")

    full_dataset_root = project_root / dataset_root
    images_dir = full_dataset_root / "images"
    masks_dir = full_dataset_root / "masks"
    images_tif_dir = full_dataset_root / "images_tif"
    masks_tif_dir = full_dataset_root / "masks_tif"
    patches_log_path = full_dataset_root / "patches_log.csv"

    require_dir(images_dir, "Generated images directory not found")
    require_dir(masks_dir, "Generated masks directory not found")
    require_dir(images_tif_dir, "Generated GeoTIFF images directory not found")
    require_dir(masks_tif_dir, "Generated GeoTIFF masks directory not found")
    require_file(patches_log_path, "Generated patches log not found")

    image_count = count_files(images_dir, "*.npy")
    mask_count = count_files(masks_dir, "*.npy")
    image_tif_count = count_files(images_tif_dir, "*.tif")
    mask_tif_count = count_files(masks_tif_dir, "*.tif")

    print(f"[E2E] Generated NPY image patches:     {image_count}")
    print(f"[E2E] Generated NPY mask patches:      {mask_count}")
    print(f"[E2E] Generated GeoTIFF image patches: {image_tif_count}")
    print(f"[E2E] Generated GeoTIFF mask patches:  {mask_tif_count}")
    print(f"[E2E] Generated patch log:             {patches_log_path.relative_to(project_root)}")

    if image_count == 0 or mask_count == 0:
        raise RuntimeError("No generated .npy image/mask patches found.")

    if image_tif_count == 0 or mask_tif_count == 0:
        raise RuntimeError("No generated GeoTIFF image/mask patches found.")

    if image_count != mask_count:
        raise RuntimeError(
            f"NPY image/mask count mismatch: images={image_count}, masks={mask_count}"
        )

    if image_tif_count != mask_tif_count:
        raise RuntimeError(
            "GeoTIFF image/mask count mismatch: "
            f"images_tif={image_tif_count}, masks_tif={mask_tif_count}"
        )

    if image_count != image_tif_count:
        raise RuntimeError(
            "NPY/GeoTIFF image count mismatch: "
            f"npy_images={image_count}, tif_images={image_tif_count}"
        )

    if mask_count != mask_tif_count:
        raise RuntimeError(
            "NPY/GeoTIFF mask count mismatch: "
            f"npy_masks={mask_count}, tif_masks={mask_tif_count}"
        )

    print()
    print("[E2E] Step 2/4: U-Net training")

    train_command = [
        "sentinel-crop-train-unet",
        "--config",
        args.train_config,
        "--override",
        "paths.manifest_csv=null",
        "--override",
        "paths.train_root=data/processed/e2e_smoke_dataset/generated",
        "--override",
        "paths.val_root=data/processed/e2e_smoke_dataset/generated",
        "--override",
        "paths.test_root=data/processed/e2e_smoke_dataset/generated",
        "--override",
        "paths.out_dir=outputs/e2e_smoke/unet_generated",
        "--override",
        "paths.resume_ckpt=null",
        "--override",
        f"data.train_limit={args.limit}",
        "--override",
        f"data.val_limit={args.limit}",
        "--override",
        "model.base=16",
        "--override",
        "training.epochs=1",
        "--override",
        "training.batch_size=1",
        "--override",
        "loader.num_workers=0",
        "--override",
        "loader.pin_memory=false",
        "--override",
        "loader.persistent_workers=false",
        "--override",
        "loader.prefetch_factor=null",
    ]

    run_command(train_command, cwd=project_root)

    checkpoint_path = project_root / train_output_dir / "last.pt"

    require_file(checkpoint_path, "Training checkpoint not found")
    require_file(
        project_root / train_output_dir / "training_summary.json",
        "Training summary not found",
    )
    require_file(
        project_root / train_output_dir / "resolved_config.json",
        "Resolved training config not found",
    )

    print()
    print("[E2E] Step 3/4: evaluation")

    eval_command = [
        "sentinel-crop-evaluate",
        "--model-type",
        "unet",
        "--data-source",
        "root",
        "--checkpoint",
        str(checkpoint_path.relative_to(project_root)),
        "--dataset-root",
        args.dataset_root,
        "--sample-format",
        "npy",
        "--output-dir",
        args.eval_output_dir,
        "--batch-size",
        "1",
        "--limit",
        str(args.limit),
        "--num-workers",
        "0",
        "--device",
        args.device,
        "--ignore-background",
    ]

    run_command(eval_command, cwd=project_root)

    require_file(
        project_root / eval_output_dir / "summary.json",
        "Evaluation summary not found",
    )
    require_file(
        project_root / eval_output_dir / "metrics_per_class.csv",
        "Evaluation per-class metrics not found",
    )
    require_file(
        project_root / eval_output_dir / "confusion_matrix_counts.npy",
        "Evaluation confusion matrix not found",
    )

    if not args.skip_predict:
        print()
        print("[E2E] Step 4/4: prediction")

        predict_command = [
            "sentinel-crop-predict",
            "--model-type",
            "unet",
            "--data-source",
            "root",
            "--checkpoint",
            str(checkpoint_path.relative_to(project_root)),
            "--dataset-root",
            args.dataset_root,
            "--sample-format",
            "npy",
            "--output-dir",
            args.predict_output_dir,
            "--batch-size",
            "1",
            "--limit",
            str(args.limit),
            "--num-workers",
            "0",
            "--device",
            args.device,
            "--save-color",
            "--save-panels",
        ]

        run_command(predict_command, cwd=project_root)

        require_file(
            project_root / predict_output_dir / "prediction_summary.json",
            "Prediction summary not found",
        )
        require_file(
            project_root / predict_output_dir / "prediction_samples.csv",
            "Prediction samples CSV not found",
        )
        require_dir(
            project_root / predict_output_dir / "panels",
            "Prediction panels directory not found",
        )
    else:
        print()
        print("[E2E] Step 4/4: prediction skipped")

    print()
    print("[E2E] Smoke test passed.")
    print(f"[E2E] Dataset root:       {args.dataset_root}")
    print(f"[E2E] Training outputs:   {args.train_output_dir}")
    print(f"[E2E] Evaluation outputs: {args.eval_output_dir}")

    if not args.skip_predict:
        print(f"[E2E] Prediction outputs: {args.predict_output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"[E2E] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)