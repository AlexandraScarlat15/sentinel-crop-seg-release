"""Manifest-derived calendar coverage tests; no invented country crop windows."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from sentinel_crop_seg.core.quality.calendar_cohort_audit import (
    REQUIRED_CALENDAR_CROPS, audit_calendar_cohorts,
)


def fixture(tmp_path: Path, *, with_fr: bool = False):
    m = tmp_path / "manifest.csv"
    c = tmp_path / "calendar.csv"
    rows = [
        dict(country="AT", cohort="AT_2021_oats", split="train",
             partition="train", acquisition_date="2021-06-11"),
        dict(country="AT", cohort="AT_2021_oats", split="val",
             partition="val", acquisition_date="2021-07-11"),
        dict(country="DE", cohort="DE_LS_2021", split="test",
             partition="test", acquisition_date="2021-07-19"),
    ]
    if with_fr:
        rows.append(dict(country="FR", cohort="FR_2018", split="train",
                         partition="train", acquisition_date="2018-07-09"))
    pd.DataFrame(rows).to_csv(m, index=False)
    with c.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["country", "crop", "start", "end"])
        writer.writeheader()
        for country in ("AT", "DE"):
            for crop in REQUIRED_CALENDAR_CROPS:
                writer.writerow(dict(country=country, crop=crop, start="04-01", end="09-30"))
    return m, c


def test_counts_by_cohort_country_year_and_split(tmp_path):
    manifest, calendar = fixture(tmp_path)
    result = audit_calendar_cohorts(
        manifest, calendar, output_dir=tmp_path / "report",
        require_full_calendar=True,
    )
    assert result["complete_calendar"]
    assert result["countries"] == ["AT", "DE"]
    assert result["cohorts"] == ["AT_2021_oats", "DE_LS_2021"]
    assert result["missing_acquisition_dates"] == 0
    table = pd.read_csv(tmp_path / "report" / "manifest_cohort_partition_counts.csv")
    assert len(table) == 3
    assert set(table["year"]) == {2021}
    assert (tmp_path / "report" / "missing_calendar_pairs.csv").is_file()


def test_missing_fr_calendar_is_reported_but_not_invented(tmp_path):
    manifest, calendar = fixture(tmp_path, with_fr=True)
    result = audit_calendar_cohorts(manifest, calendar)
    assert not result["complete_calendar"]
    assert result["countries"] == ["AT", "DE", "FR"]
    assert len(result["missing_calendar_pairs"]) == len(REQUIRED_CALENDAR_CROPS)
    assert all(row["country"] == "FR" for row in result["missing_calendar_pairs"])
    with pytest.raises(ValueError, match="Missing 13 country/crop"):
        audit_calendar_cohorts(manifest, calendar, require_full_calendar=True)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(("column", "value", "message"), [
    ("cohort", "AT_2021_2023_oats", "unambiguous year"),
    ("country", "FR", "inconsistent with country"),
    ("acquisition_date", "2020-05-21", "disagree with declared cohort years"),
    ("partition", "val", "partition disagrees"),
])
def test_rejects_mismatched_country_year_or_partition(tmp_path, column, value, message):
    manifest, calendar = fixture(tmp_path)
    data = pd.read_csv(manifest)
    data.loc[0, column] = value
    data.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match=message):
        audit_calendar_cohorts(manifest, calendar)


def test_duplicate_calendar_country_crop_rejected(tmp_path):
    manifest, calendar = fixture(tmp_path)
    data = pd.read_csv(calendar)
    pd.concat([data, data.iloc[[0]]]).to_csv(calendar, index=False)
    with pytest.raises(ValueError, match="Duplicate calendar"):
        audit_calendar_cohorts(manifest, calendar)


def test_missing_date_is_reported_not_fabricated(tmp_path):
    manifest, calendar = fixture(tmp_path)
    data = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    data.loc[0, "acquisition_date"] = ""
    data.to_csv(manifest, index=False)
    result = audit_calendar_cohorts(manifest, calendar)
    assert result["missing_acquisition_dates"] == 1


def test_refuses_overwrite(tmp_path):
    manifest, calendar = fixture(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        audit_calendar_cohorts(manifest, calendar, output_dir=output)


def test_real_bundled_calendar_scope_is_explicit_not_inferred():
    path = Path(__file__).resolve().parents[1] / "config" / "crop_calendar_recalibrated_v0_3.csv"
    calendar = pd.read_csv(path)
    expected_countries = {"AT", "CZ", "DE", "FR", "SK"}
    assert set(calendar["country"]) == expected_countries
    assert len(calendar) == len(expected_countries) * len(REQUIRED_CALENDAR_CROPS)
    assert not calendar.duplicated(["country", "crop"]).any()
    for country in sorted(expected_countries):
        assert set(calendar.loc[calendar["country"] == country, "crop"]) == set(
            REQUIRED_CALENDAR_CROPS
        )
