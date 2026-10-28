from sentinel_crop_seg.core.data.sentinel_product_metadata import (
    infer_sentinel_scene_date,
    infer_sentinel_tile_id,
)


def test_infer_sentinel_scene_date_from_product_id():
    product_id = "S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939"

    assert infer_sentinel_scene_date(product_id) == "2021-08-10"


def test_infer_sentinel_tile_id_from_product_id():
    product_id = "S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939"

    assert infer_sentinel_tile_id(product_id) == "T33TWN"


def test_infer_sentinel_metadata_returns_empty_string_for_invalid_values():
    assert infer_sentinel_scene_date(None) == ""
    assert infer_sentinel_scene_date("invalid") == ""

    assert infer_sentinel_tile_id(None) == ""
    assert infer_sentinel_tile_id("invalid") == ""