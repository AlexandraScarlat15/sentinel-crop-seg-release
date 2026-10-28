"""End-to-end local smoke tests of the RF-free public U-Net commands."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_crop_seg.cli import evaluate_unet, predict_unet
from sentinel_crop_seg.cli._unet_runtime import prepare_runtime
from sentinel_crop_seg.core.training.model_unet_4levels import UNet4LevelsGN


def fixture(tmp_path: Path, *, calendar: bool = False) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for index in range(2):
        image = np.full((3, 32, 32), 1000 + index, dtype=np.float32)
        mask = np.full((32, 32), index % 3, dtype=np.uint8)
        ipath = dataset / f"image_{index}.npy"
        mpath = dataset / f"mask_{index}.npy"
        np.save(ipath, image)
        np.save(mpath, mask)
        rows.append({
            "dataset": "AT_2021", "sample_id": f"sample_{index:06d}",
            "img_path": ipath.name, "mask_path": mpath.name,
            "split": "test", "acquisition_date": "2021-05-01",
        })
    manifest = dataset / "test_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    model_config = {
        "in_channels": 5 if calendar else 3,
        "num_classes": 3, "base": 4,
        "dropout_bottleneck": 0.0,
        "normalize_mode": "scale_10000",
        "scale_divisor": 10000.0,
        "band_indices": [0, 1, 2],
        "calendar_features": "doy_sin_cos" if calendar else "none",
    }
    model = UNet4LevelsGN(
        in_channels=model_config["in_channels"], num_classes=3, base=4,
        dropout_bottleneck=0.0,
    )
    checkpoint = tmp_path / "unet.pt"
    torch.save({"config": model_config, "model_state": model.state_dict()}, checkpoint)
    return checkpoint, manifest


def run_cli(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", ["command", *map(str, args)])
    module.main()


def test_evaluate_unet_emits_existing_scientific_reports(tmp_path, monkeypatch):
    checkpoint, manifest = fixture(tmp_path)
    output = tmp_path / "eval"
    run_cli(monkeypatch, evaluate_unet, "--checkpoint", checkpoint,
            "--manifest-csv", manifest, "--output-dir", output,
            "--device", "cpu", "--batch-size", "1")
    result = json.loads((output / "summary.json").read_text())
    assert result["model_type"] == "unet"
    assert result["num_samples_evaluated"] == 2
    assert result["num_classes"] == 3
    assert result["band_indices"] == [0, 1, 2]
    assert len(result["checkpoint_sha256"]) == 64
    assert (output / "metrics_per_class.csv").is_file()
    assert np.load(output / "confusion_matrix_counts.npy").shape == (3, 3)
    assert len((output / "evaluated_samples.txt").read_text().splitlines()) == 0
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_cli(monkeypatch, evaluate_unet, "--checkpoint", checkpoint,
                "--manifest-csv", manifest, "--output-dir", output,
                "--device", "cpu")


def test_predict_unet_writes_npy_png_and_auditable_csv(tmp_path, monkeypatch):
    checkpoint, manifest = fixture(tmp_path)
    target = tmp_path / "pred"
    run_cli(monkeypatch, predict_unet, "--checkpoint", checkpoint,
            "--manifest-csv", manifest, "--output-dir", target,
            "--device", "cpu", "--save-color", "--batch-size", "2")
    masks = sorted((target / "pred_masks_npy").glob("*.npy"))
    previews = sorted((target / "pred_color").glob("*.png"))
    assert len(masks) == len(previews) == 2
    assert np.load(masks[0]).shape == (32, 32)
    assert np.load(masks[1]).max() < 3
    with (target / "prediction_samples.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    assert len(entries) == 2
    assert entries[0]["sample_id"] == "sample_000000"
    assert (target / entries[0]["prediction_path"]).exists()
    summary = json.loads((target / "prediction_summary.json").read_text())
    assert summary["num_predictions"] == 2
    assert summary["pred_color_dir"] == "pred_color"


def test_unet_calendar_channels_follow_checkpoint_config(tmp_path, monkeypatch):
    checkpoint, manifest = fixture(tmp_path, calendar=True)
    result = tmp_path / "calendar_eval"
    run_cli(monkeypatch, evaluate_unet, "--checkpoint", checkpoint,
            "--manifest-csv", manifest, "--output-dir", result, "--device", "cpu")
    summary = json.loads((result / "summary.json").read_text())
    assert summary["calendar_features"] == "doy_sin_cos"
    assert summary["in_channels"] == 5


def test_unet_refuses_missing_manifest_and_channel_mismatch(tmp_path, monkeypatch):
    checkpoint, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest-csv is required"):
        run_cli(monkeypatch, evaluate_unet, "--checkpoint", checkpoint,
                "--output-dir", tmp_path / "missing", "--device", "cpu")
    with pytest.raises(ValueError, match="Input channel mismatch"):
        run_cli(monkeypatch, evaluate_unet, "--checkpoint", checkpoint,
                "--manifest-csv", manifest, "--output-dir", tmp_path / "bad",
                "--band-indices", "0,1", "--device", "cpu")
    assert not (tmp_path / "bad").exists()


def test_evaluation_namespace_no_longer_eagerly_imports_random_forest():
    code = (
        "import sentinel_crop_seg.core.evaluation; import sentinel_crop_seg.cli.evaluate_unet; "
        "import sentinel_crop_seg.cli.predict_unet; import sys; "
        "assert 'sentinel_crop_seg.core.evaluation.predictors' not in sys.modules; "
        "assert 'sentinel_crop_seg.core.baselines.random_forest.features' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_safe_name_does_not_embed_relative_path():
    output = predict_unet.safe_name(12, "../../BE/test\\sample:1")
    assert "/" not in output and "\\" not in output
    assert output.startswith("000012_")
