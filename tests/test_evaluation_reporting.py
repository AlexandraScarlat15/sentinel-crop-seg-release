from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.reporting import (
    save_confusion_matrices,
    save_evaluated_samples_txt,
    save_evaluation_report,
    save_per_class_metrics_csv,
    save_summary_json,
    save_summary_txt,
)


def test_save_confusion_matrices_writes_counts_and_row_normalized(tmp_path: Path):
    cm = torch.tensor(
        [
            [2, 0],
            [1, 1],
        ],
        dtype=torch.int64,
    )

    saved = save_confusion_matrices(tmp_path, cm)

    counts = np.load(saved["confusion_matrix_counts"])
    row_norm = np.load(saved["confusion_matrix_row_norm"])

    assert counts.tolist() == [[2, 0], [1, 1]]
    assert row_norm.tolist() == [[1.0, 0.0], [0.5, 0.5]]


def test_save_per_class_metrics_csv(tmp_path: Path):
    cm = torch.tensor(
        [
            [2, 0],
            [1, 1],
        ],
        dtype=torch.int64,
    )

    path = tmp_path / "metrics_per_class.csv"

    save_per_class_metrics_csv(
        path=path,
        confusion_matrix=cm,
        class_names=["background", "crop"],
    )

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2

    assert rows[0]["class_id"] == "0"
    assert rows[0]["class_name"] == "background"
    assert float(rows[0]["iou"]) == pytest.approx(2.0 / 3.0)
    assert float(rows[0]["accuracy"]) == pytest.approx(1.0)
    assert int(rows[0]["support_pixels"]) == 2

    assert rows[1]["class_id"] == "1"
    assert rows[1]["class_name"] == "crop"
    assert float(rows[1]["iou"]) == pytest.approx(0.5)
    assert float(rows[1]["accuracy"]) == pytest.approx(0.5)
    assert int(rows[1]["support_pixels"]) == 2


def test_save_summary_json_and_txt_are_json_safe(tmp_path: Path):
    summary = {
        "checkpoint": tmp_path / "best.pt",
        "score": np.float32(0.75),
        "values": np.array([1, 2, 3], dtype=np.int64),
        "tensor": torch.tensor([4, 5]),
    }

    json_path = tmp_path / "summary.json"
    txt_path = tmp_path / "summary.txt"

    save_summary_json(json_path, summary)
    save_summary_txt(txt_path, summary)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    text = txt_path.read_text(encoding="utf-8")

    assert loaded["checkpoint"].endswith("best.pt")
    assert loaded["score"] == pytest.approx(0.75)
    assert loaded["values"] == [1, 2, 3]
    assert loaded["tensor"] == [4, 5]

    assert "checkpoint:" in text
    assert "score:" in text


def test_save_evaluated_samples_txt(tmp_path: Path):
    path = tmp_path / "evaluated_samples.txt"

    save_evaluated_samples_txt(path, ["sample_000001", "sample_000002"])

    assert path.read_text(encoding="utf-8").splitlines() == [
        "sample_000001",
        "sample_000002",
    ]


def test_save_evaluation_report_writes_all_expected_files(tmp_path: Path):
    cm = torch.tensor(
        [
            [2, 0],
            [1, 1],
        ],
        dtype=torch.int64,
    )

    summary = {
        "model_type": "unet",
        "mIoU": 0.5,
    }

    saved = save_evaluation_report(
        output_dir=tmp_path,
        confusion_matrix=cm,
        summary=summary,
        class_names=["background", "crop"],
        evaluated_sample_ids=["sample_000001"],
    )

    expected_keys = {
        "confusion_matrix_counts",
        "confusion_matrix_row_norm",
        "metrics_per_class",
        "summary_json",
        "summary_txt",
        "evaluated_samples",
    }

    assert set(saved.keys()) == expected_keys

    for path in saved.values():
        assert path.exists()

    loaded_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["model_type"] == "unet"
    assert loaded_summary["mIoU"] == pytest.approx(0.5)