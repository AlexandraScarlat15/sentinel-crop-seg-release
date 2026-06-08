from pathlib import Path

import yaml

from src.data.eurocrops_mask_loader import EuroCropsMaskLoader


def make_loader_from_config_classes():
    config_path = Path(__file__).resolve().parents[1] / "config" / "classes.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = EuroCropsMaskLoader.__new__(EuroCropsMaskLoader)
    loader.config = config

    return loader


def test_eurocrops_class_mapping_matches_expected_project_taxonomy():
    loader = make_loader_from_config_classes()

    cases = {
        "grain maize": 1,
        "corn": 1,
        "oilseed rape": 2,
        "winter wheat": 3,
        "triticale": 3,
        "barley": 4,
        "oats": 5,
        "sunflower": 6,
        "sugar beet": 7,
        "soybean": 8,
        "potato": 9,
        "pasture": 10,
        "grassland": 10,
        "lucerne": 10,
        "alfalfa": 10,
    }

    for crop_name, expected_class_id in cases.items():
        assert loader.map_crop_to_class_id(crop_name) == expected_class_id


def test_eurocrops_class_mapping_respects_exclusions():
    loader = make_loader_from_config_classes()

    cases = {
        "buckwheat": 0,
        "fodder beet": 0,
        "beetroot": 0,
        "unknown crop": 0,
    }

    for crop_name, expected_class_id in cases.items():
        assert loader.map_crop_to_class_id(crop_name) == expected_class_id


def test_eurocrops_class_mapping_is_case_insensitive_and_handles_non_strings():
    loader = make_loader_from_config_classes()

    assert loader.map_crop_to_class_id("SUNFLOWER") == 6
    assert loader.map_crop_to_class_id("Winter Wheat") == 3
    assert loader.map_crop_to_class_id(None) == 0
    assert loader.map_crop_to_class_id(123) == 0
