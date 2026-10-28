"""Offline regressions for weighted phenology selection and immutable holdouts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentinel_crop_seg.core.quality.weighted_calendar_selection import (
    _fraction, build_weighted_calendar_selection,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "config/classes.yaml"
BINDING = ROOT / "config/calendars/class_calendar_binding_v1.yaml"


def fixtures(tmp_path):
    csv = tmp_path / "calendar.csv"
    csv.write_text(
        "country,crop,start,end\n"
        "AT,Maize,05-10,09-30\n"
        "AT,Rapeseed,04-01,07-20\n"
        "AT,Wheat,04-01,08-15\n"
        "AT,Barley,04-01,08-15\n"
        "AT,Oats,04-15,08-15\n"
        "AT,Sunflower,05-10,09-30\n"
        "AT,Sugar beet,04-20,09-30\n"
        "AT,Soybean,05-15,09-30\n"
        "AT,Potato,05-01,09-30\n"
        "AT,Grassland / pasture,04-01,09-30\n"
        "AT,Arable grass,04-15,09-30\n"
        "AT,Clover,04-15,09-30\n"
        "AT,Lucerne / alfalfa,04-15,09-30\n",
        encoding="utf-8"
    )
    def row(name, split, date, pixels, *, country="AT"):
        values = {
            "dataset": "AT_2021_v2", "sample_id": name, "cohort": "AT_2021",
            "country": country, "split": split, "partition": split,
            "acquisition_date": date, "img_path": f"images/{name}.npy",
            "mask_path": f"masks/{name}.npy",
        }
        values.update({f"p{i}": str(pixels.get(i, 0)) for i in range(11)})
        return values
    rows = [
        row("exact90", "train", "2021-04-20", {0: 10000, 3: 90, 1: 10}),
        row("below90", "train", "2021-04-20", {3: 899, 1: 101}),
        row("full", "train", "2021-06-20", {10: 35, 3: 65}),
        row("pasture_outside", "train", "2021-04-05", {10: 80, 3: 20}),
        row("unresolved", "train", "", {3: 95, 1: 5}),
        row("background", "train", "2021-06-20", {0: 65536}),
        row("val_keep", "val", "2021-01-01", {3: 100}),
        row("test_keep", "test", "2021-01-01", {3: 100}),
        row("excluded_keep", "excluded", "2021-01-01", {3: 100}),
    ]
    source = tmp_path / "source" / "manifest.csv"
    source.parent.mkdir()
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "derived"
    return source, csv, output


def call(source, csv, out, **kwargs):
    return build_weighted_calendar_selection(
        source, csv, TAXONOMY, BINDING, out, **kwargs
    )


def test_exact_inclusive_90_and_zero_background_in_denominator(tmp_path):
    source, csv, out = fixtures(tmp_path)
    result = call(source, csv, out)
    assert result["status"] == "validated_only"
    assert result["chosen_threshold"] == "0.90"
    assert result["train_input"] == 6
    assert result["train_selected"] == 2
    assert result["train_unresolved"] == 2
    assert result["train_no_agricultural_pixels"] == 1
    assert result["val_preserved"] == result["test_preserved"] == 1
    assert {x["threshold"]: x["train_retained"] for x in
            result["threshold_sensitivity_train_only"]} == {
                "0.75": 3, "0.90": 2, "1.00": 1,
            }
    mass = next(item for item in result["threshold_sensitivity_train_only"]
                if item["threshold"] == "0.90")["train_agricultural_pixel_mass_by_class"]
    assert mass["3"] == {"before_pixels": 1169, "after_pixels": 155}
    assert mass["10"] == {"before_pixels": 115, "after_pixels": 35}
    assert not out.exists()


def test_apply_preserves_holdout_and_rebases_paths(tmp_path):
    source, csv, out = fixtures(tmp_path)
    source_hash = source.read_bytes()
    result = call(source, csv, out, validate_only=False)
    assert result["status"] == "new_train_filtered_manifest"
    written = pd.read_csv(out / "calendar_selected_manifest.csv", dtype=str,
                          keep_default_na=False)
    assert written["sample_id"].tolist() == [
        "exact90", "full", "val_keep", "test_keep", "excluded_keep"
    ]
    assert written.loc[written["split"].eq("test"), "sample_id"].tolist() == ["test_keep"]
    path = (out / written.iloc[0]["img_path"]).resolve()
    assert path == (source.parent / "images/exact90.npy").resolve()
    audit = pd.read_csv(out / "calendar_train_audit.csv")
    assert len(audit) == 6
    exactly = audit.loc[audit["sample_id"].eq("exact90")].iloc[0]
    assert exactly["agricultural_pixels_total"] == 100
    assert exactly["background_pixels"] == 10000
    pasture = audit.loc[audit["sample_id"].eq("pasture_outside")].iloc[0]
    assert pasture["calendar_compatible_pixels"] == 20
    assert pasture["calendar_incompatible_pixels"] == 80
    report = json.loads((out / "calendar_selection_report.json").read_text())
    assert len(report["source_sha256"]["calendar"]) == 64
    assert source.read_bytes() == source_hash
    with pytest.raises(FileExistsError, match="already exists"):
        call(source, csv, out, validate_only=False)


def test_custom_threshold_is_effective_and_checked_as_exact_integer_ratio(tmp_path):
    source, csv, out = fixtures(tmp_path)
    r = call(source, csv, out, threshold="0.899", sweep=("0.9",))
    assert r["train_selected"] == 3
    assert _fraction("0.90") == (9, 10, "0.90")
    with pytest.raises(ValueError, match="Threshold"):
        call(source, csv, out, threshold="NaN")
    with pytest.raises(ValueError, match="Threshold"):
        call(source, csv, out, threshold="1.1")


@pytest.mark.parametrize("change,error", [
    ("duplicate", "duplicate"),
    ("negative_count", "Invalid integer"),
    ("float_count", "Invalid integer"),
    ("missing_column", "missing source class"),
    ("partition_mismatch", "split/partition"),
    ("wrong_country", "uppercase ISO-2"),
    ("invalid_split", "Unknown manifest split"),
    ("missing_date_field", "acquisition_date or scene_date"),
])
def test_bad_manifest_rejected_before_writes(tmp_path, change, error):
    source, csv, out = fixtures(tmp_path)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    if change == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    elif change == "negative_count":
        frame.loc[0, "p3"] = "-2"
    elif change == "float_count":
        frame.loc[0, "p3"] = "2.5"
    elif change == "missing_column":
        frame = frame.drop(columns=["p9"])
    elif change == "partition_mismatch":
        frame.loc[0, "partition"] = "val"
    elif change == "wrong_country":
        frame.loc[0, "country"] = "Austria"
    elif change == "invalid_split":
        frame.loc[0, "split"] = "irrelevant"
        frame.loc[0, "partition"] = "irrelevant"
    elif change == "missing_date_field":
        frame = frame.drop(columns=["acquisition_date"])
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match=error):
        call(source, csv, out, validate_only=False)
    assert not out.exists()


def test_missing_country_window_cannot_be_silently_accepted(tmp_path):
    source, csv, out = fixtures(tmp_path)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame.loc[0, "country"] = "BE"
    frame.to_csv(source, index=False)
    report = call(source, csv, out)
    assert report["train_selected"] == 1
    assert report["train_unresolved"] == 3


def test_new_taxonomy_without_matching_binding_fails_closed(tmp_path):
    source, csv, out = fixtures(tmp_path)
    edited = tmp_path / "classes.yaml"
    edited.write_text(TAXONOMY.read_text().replace("maize:", "corn:"), encoding="utf-8")
    with pytest.raises(ValueError, match="binding must cover exactly"):
        build_weighted_calendar_selection(source, csv, edited, BINDING, out)
