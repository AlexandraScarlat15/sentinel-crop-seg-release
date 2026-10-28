"""Synthetic regression tests for read-only evaluation plots with flexible classes."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sentinel_crop_seg.cli import plot_evaluation
from sentinel_crop_seg.core.evaluation.plots import metrics_from_counts, read_evaluation, render_evaluation_plots


def make_evaluation(tmp_path: Path, matrix: np.ndarray | None = None) -> Path:
    source = tmp_path / "evaluation"
    source.mkdir()
    if matrix is None:
        matrix = np.array([[2, 0, 0], [1, 3, 0], [0, 1, 4]], dtype=np.int64)
    np.save(source / "confusion_matrix_counts.npy", matrix)
    (source / "summary.json").write_text(json.dumps({
        "num_classes": int(matrix.shape[0]), "num_samples_evaluated": 5
    }), encoding="utf-8")
    with (source / "metrics_per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name"])
        for i in range(matrix.shape[0]):
            writer.writerow([i, f"crop_{i}"])
    return source


def test_eval_plots_render_all_formats_and_correct_percentages(tmp_path):
    source = make_evaluation(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / "plots"
    files = render_evaluation_plots(source, target, formats=("png", "pdf", "svg"), dpi=72)
    assert len(files) == 13
    for stem in ("confusion_matrix_counts", "confusion_matrix_row_norm", "iou_per_class"):
        for ext in ("png", "pdf", "svg"):
            assert (target / f"{stem}.{ext}").stat().st_size > 300
    row_norm = np.loadtxt(target / "confusion_matrix_row_norm_percent.csv", delimiter=",")
    assert row_norm[0].tolist() == [100., 0., 0.]
    assert row_norm[1, 0] == pytest.approx(25.0)
    assert row_norm[1, 1] == pytest.approx(75.0)
    with (target / "metrics_per_class_pretty.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["class_name"] for row in rows] == ["crop_0", "crop_1", "crop_2"]
    assert float(rows[1]["dice_f1_percent"]) == pytest.approx(75.0)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render_evaluation_plots(source, target)


def test_eval_custom_class_names_json_in_numeric_order(tmp_path):
    source = make_evaluation(tmp_path)
    custom = tmp_path / "class_names.json"
    custom.write_text(json.dumps(["background", "wheat", "barley"]), encoding="utf-8")
    target = tmp_path / "plots"
    render_evaluation_plots(source, target, class_names_json=custom, formats=("png",), dpi=72)
    meta = json.loads((target / "plots_summary.json").read_text(encoding="utf-8"))
    assert meta["class_names"] == ["background", "wheat", "barley"]


@pytest.mark.parametrize("names", [
    ["background", "wheat"], ["same", "same", "other"], ["a", "", "c"],
])
def test_eval_rejects_invalid_names_before_output_creation(tmp_path, names):
    source = make_evaluation(tmp_path)
    file = tmp_path / "names.json"
    file.write_text(json.dumps(names), encoding="utf-8")
    target = tmp_path / "plots"
    with pytest.raises(ValueError, match="Class names"):
        render_evaluation_plots(source, target, class_names_json=file, formats=("png",))
    assert not target.exists()


def test_eval_rejects_non_integer_or_negative_counts(tmp_path):
    source = make_evaluation(tmp_path)
    np.save(source / "confusion_matrix_counts.npy", np.eye(3, dtype=np.float64))
    with pytest.raises(ValueError, match="integer"):
        read_evaluation(source)
    np.save(source / "confusion_matrix_counts.npy", -np.eye(3, dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative"):
        read_evaluation(source)


def test_eval_undefined_rates_are_zero_and_do_not_produce_nan():
    matrix = np.array([[2, 1, 0], [0, 0, 0], [1, 0, 1]], dtype=np.int64)
    stats = metrics_from_counts(matrix)
    assert stats["normalized"][1].tolist() == [0, 0, 0]
    assert np.isfinite(stats["normalized"]).all()
    assert stats["iou"][1] == 0


def test_eval_plot_cli_does_not_require_checkpoint_or_training(tmp_path, monkeypatch):
    source = make_evaluation(tmp_path)
    target = tmp_path / "plots"
    monkeypatch.setattr(sys, "argv", [
        "sentinel-crop-plot-evaluation", "--eval-dir", str(source),
        "--out-dir", str(target), "--formats", "png", "--dpi", "72",
    ])
    plot_evaluation.main()
    assert (target / "confusion_matrix_row_norm.png").is_file()
    assert (target / "metrics_per_class_pretty.csv").is_file()
