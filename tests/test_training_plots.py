"""Synthetic regression tests for read-only, checkpoint-free training plots."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from sentinel_crop_seg.cli import plot_training
from sentinel_crop_seg.core.training.plots import load_training_history, render_training_curves


def make_history(tmp_path: Path) -> Path:
    source = tmp_path / "training"
    source.mkdir()
    pd.DataFrame([
        {"epoch": 1, "train_loss": 1.2, "val_loss": 1.1, "val_mIoU": 0.4,
         "val_pixel_acc": 0.6, "lr": 0.001},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_mIoU": 0.6,
         "val_pixel_acc": 0.8, "lr": 0.0005},
    ]).to_csv(source / "training_metrics.csv", index=False)
    (source / "training_summary.json").write_text(
        json.dumps({"best_epoch": 2, "best_val_mIoU": 0.6}), encoding="utf-8"
    )
    return source


def test_training_plots_render_png_pdf_and_svg_without_modifying_inputs(tmp_path):
    source = make_history(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / "plots"
    files = render_training_curves(source, target, formats=("png", "pdf", "svg"), dpi=72)
    assert len(files) == 9
    for stem in ("training_validation_loss", "validation_performance", "learning_rate_schedule"):
        for extension in ("png", "pdf", "svg"):
            assert (target / f"{stem}.{extension}").stat().st_size > 300
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert not (source / "best.pt").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render_training_curves(source, target)


def test_training_plots_reject_summary_mismatch_without_creating_output(tmp_path):
    source = make_history(tmp_path)
    (source / "training_summary.json").write_text(
        json.dumps({"best_epoch": 2, "best_val_mIoU": 0.1}), encoding="utf-8"
    )
    target = tmp_path / "plots"
    with pytest.raises(ValueError, match="differs"):
        render_training_curves(source, target, formats=("png",))
    assert not target.exists()


def test_training_plots_reject_duplicate_epochs_and_nonfinite_values(tmp_path):
    source = make_history(tmp_path)
    table = pd.read_csv(source / "training_metrics.csv")
    table.loc[1, "epoch"] = 1
    table.to_csv(source / "training_metrics.csv", index=False)
    with pytest.raises(ValueError, match="unique positive"):
        load_training_history(source)
    table.loc[1, "epoch"] = 2
    table.loc[1, "val_mIoU"] = float("nan")
    table.to_csv(source / "training_metrics.csv", index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        load_training_history(source)


def test_training_plot_cli_is_independent_of_generator_and_checkpoint(tmp_path, monkeypatch):
    source = make_history(tmp_path)
    target = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [
        "sentinel-crop-plot-training", "--run-dir", str(source),
        "--out-dir", str(target), "--formats", "png", "--dpi", "72",
    ])
    plot_training.main()
    assert sorted(p.suffix for p in target.iterdir()) == [".png"] * 3


def test_training_plot_invalid_formats_do_not_create_directory(tmp_path):
    source = make_history(tmp_path)
    target = tmp_path / "plots"
    with pytest.raises(ValueError, match="Formats"):
        render_training_curves(source, target, formats=("png", "png"))
    assert not target.exists()
