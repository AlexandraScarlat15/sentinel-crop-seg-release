from pathlib import Path

from sentinel_crop_seg.core.pipeline import DatasetGenerationResult, generate_dataset
from sentinel_crop_seg.core.pipeline.dataset_generation import make_roi_key, _read_generation_config, _read_target_config


def test_pipeline_api_is_importable():
    assert generate_dataset is not None
    assert DatasetGenerationResult is not None


def test_dataset_generation_result_dataclass():
    result = DatasetGenerationResult(
        output_dir=Path("output_dataset"),
        total_samples=5,
        selected_rois=2,
        processed_products=1,
        strategy="grid",
        run_tag="smoke",
        processed_state_path=Path("processed_grid_smoke.json"),
    )

    assert result.output_dir == Path("output_dataset")
    assert result.total_samples == 5
    assert result.selected_rois == 2
    assert result.processed_products == 1
    assert result.strategy == "grid"
    assert result.run_tag == "smoke"


def test_make_roi_key_rounds_coordinates():
    roi = (1.234, 2.345, 3.456, 4.567)

    assert make_roi_key(roi, decimals=1) == "1.2_2.3_3.5_4.6"


def test_read_target_config_handles_null_target_class():
    config = {
        "dataset": {
            "target": {
                "class_id": None,
                "min_target_ratio": 0.05,
                "min_target_pixels": 800,
                "total_patches": 100,
            }
        }
    }

    class_id, min_ratio, min_pixels, total_patches = _read_target_config(config)

    assert class_id is None
    assert min_ratio == 0.05
    assert min_pixels == 800
    assert total_patches == 100


def test_read_target_config_handles_integer_target_class():
    config = {
        "dataset": {
            "target": {
                "class_id": 6,
                "min_target_ratio": 0.1,
                "min_target_pixels": 500,
                "total_patches": 20,
            }
        }
    }

    class_id, min_ratio, min_pixels, total_patches = _read_target_config(config)

    assert class_id == 6
    assert min_ratio == 0.1
    assert min_pixels == 500
    assert total_patches == 20


def test_read_generation_config_uses_supported_defaults():
    satellite, label_source = _read_generation_config({})

    assert satellite == "sentinel2"
    assert label_source == "eurocrops"


def test_read_generation_config_accepts_explicit_supported_source():
    config = {
        "generation": {
            "satellite": "sentinel2",
            "label_source": "eurocrops",
        }
    }

    satellite, label_source = _read_generation_config(config)

    assert satellite == "sentinel2"
    assert label_source == "eurocrops"


def test_read_generation_config_rejects_unsupported_source():
    config = {
        "generation": {
            "satellite": "sentinel2",
            "label_source": "worldcover",
        }
    }

    try:
        _read_generation_config(config)
    except ValueError as exc:
        assert "Unsupported generation.label_source" in str(exc)
    else:
        raise AssertionError("Expected unsupported label source to raise ValueError")
