from pathlib import Path

from sentinel_crop_seg import DatasetGenerationResult, generate_dataset
from sentinel_crop_seg.dataset_generation import generate_dataset as generate_dataset_from_module


def test_public_api_exports_dataset_generation_function():
    assert generate_dataset is generate_dataset_from_module
    assert DatasetGenerationResult is not None


def test_public_dataset_generation_result_can_be_created():
    result = DatasetGenerationResult(
        output_dir=Path("dataset"),
        total_samples=10,
        selected_rois=3,
        processed_products=2,
        strategy="grid",
        run_tag="demo",
        processed_state_path=Path("processed_grid_demo.json"),
    )

    assert result.total_samples == 10
    assert result.strategy == "grid"
    assert result.run_tag == "demo"
