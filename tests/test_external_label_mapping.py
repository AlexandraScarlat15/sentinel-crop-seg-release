from pathlib import Path

import yaml


PROJECT_CLASS_IDS = set(range(11))


def load_yaml(relative_path: str) -> dict:
    path = Path(__file__).resolve().parents[1] / relative_path

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dacia5_label_mapping_has_valid_project_class_ids_or_null():
    data = load_yaml("config/dacia5/label_mapping.yaml")
    cfg = data["dacia5"]

    source_labels = cfg["source_labels"]
    remap = cfg["remap_to_project_class_id"]

    assert set(source_labels.keys()) == set(remap.keys())

    for source_label, target_class_id in remap.items():
        assert source_label in source_labels

        if target_class_id is None:
            continue

        assert target_class_id in PROJECT_CLASS_IDS


def test_dacia5_label_mapping_matches_expected_semantics():
    data = load_yaml("config/dacia5/label_mapping.yaml")
    remap = data["dacia5"]["remap_to_project_class_id"]

    assert remap[1] == 3    # Common winter wheat -> wheat
    assert remap[2] == 3    # Common spring wheat -> wheat
    assert remap[3] == 1    # Corn -> maize
    assert remap[5] == 9    # Late potatoes -> potato
    assert remap[9] == 7    # Sugar beets -> sugar_beet
    assert remap[10] == 10  # Temporal grassland -> pasture
    assert remap[11] == 10  # Alfalfa_1 -> pasture / forage
    assert remap[12] == 10  # Permanent grassland -> pasture
    assert remap[13] == 1   # Corn silage -> maize
    assert remap[14] == 8   # Soybean -> soybean
    assert remap[15] == 10  # Alfalfa_2 -> pasture / forage
    assert remap[16] == 2   # Winter rapeseed -> rapeseed
    assert remap[17] == 6   # Sunflower -> sunflower

    assert remap[4] is None  # Peas are not in the current project taxonomy


def test_pastis_label_mapping_has_valid_project_class_ids_or_ignore_value():
    data = load_yaml("config/pastis/label_mapping.yaml")
    cfg = data["pastis"]

    ignore_value = cfg["ignore_value"]
    remap = cfg["remap_to_project_class_id"]

    assert ignore_value == 255

    allowed_values = PROJECT_CLASS_IDS | {ignore_value}

    for source_label, target_class_id in remap.items():
        assert isinstance(source_label, int)
        assert target_class_id in allowed_values


def test_pastis_label_mapping_matches_expected_semantics():
    data = load_yaml("config/pastis/label_mapping.yaml")
    cfg = data["pastis"]
    remap = cfg["remap_to_project_class_id"]
    ignore_value = cfg["ignore_value"]

    assert remap[0] == 0    # background
    assert remap[1] == 10   # meadow -> pasture
    assert remap[2] == 3    # soft winter wheat -> wheat
    assert remap[3] == 1    # corn -> maize
    assert remap[4] == 4    # winter barley -> barley
    assert remap[5] == 2    # winter rapeseed -> rapeseed
    assert remap[7] == 6    # sunflower -> sunflower
    assert remap[9] == 7    # beet -> sugar_beet
    assert remap[13] == 9   # potatoes -> potato
    assert remap[15] == 8   # soybeans -> soybean

    assert remap[8] == ignore_value
    assert remap[12] == ignore_value
    assert remap[16] == ignore_value
    assert remap[17] == ignore_value
    assert remap[18] == ignore_value
    assert remap[19] == ignore_value