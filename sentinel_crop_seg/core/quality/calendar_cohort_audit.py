"""Audit the exact countries, years and crop-calendar coverage of a manifest.

This is a validation/reporting tool, not an acquisition scheduler. Missing
country/crop windows are reported instead of fabricated or extrapolated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

# Use the documented 13 calendar crop names, including four pasture types.
REQUIRED_CALENDAR_CROPS = (
    "Maize", "Rapeseed", "Wheat", "Barley", "Oats",
    "Sunflower", "Sugar beet", "Soybean", "Potato",
    "Grassland / pasture", "Arable grass", "Clover", "Lucerne / alfalfa",
)
COUNTRY_ALIASES = {
    "AT": "AT", "AUSTRIA": "AT", "DE": "DE", "GERMANY": "DE",
    "CZ": "CZ", "CZECHIA": "CZ", "CZECH REPUBLIC": "CZ",
    "FR": "FR", "FRANCE": "FR", "SK": "SK", "SLOVAKIA": "SK",
    "BE": "BE", "BELGIUM": "BE",
}


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_calendar_cohorts(
    manifest_file: str | Path, calendar_file: str | Path, *,
    output_dir: str | Path | None = None, require_full_calendar: bool = False,
) -> dict:
    manifest = pd.read_csv(manifest_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    calendar = pd.read_csv(calendar_file, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required_manifest = {"country", "cohort", "split"}
    required_calendar = {"country", "crop", "start", "end"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"Manifest missing {sorted(missing)}")
    if missing := required_calendar - set(calendar.columns):
        raise ValueError(f"Calendar missing {sorted(missing)}")
    if manifest.empty or calendar.empty:
        raise ValueError("Manifest and calendar must be nonempty")
    if "partition" in manifest and not manifest["partition"].equals(manifest["split"]):
        raise ValueError("Manifest partition disagrees with split")
    if not manifest["split"].isin(["train", "val", "test", "excluded"]).all():
        raise ValueError("Unexpected partition in manifest")
    if any(manifest[k].str.strip().eq("").any() for k in required_manifest):
        raise ValueError("Blank country/cohort/split in manifest")

    normalized = manifest["country"].str.strip().str.upper().map(COUNTRY_ALIASES)
    if normalized.isna().any():
        raise ValueError("Unknown country name or ISO-2 code in manifest")
    manifest = manifest.assign(_country=normalized)
    years = []
    for cohort, country in zip(manifest["cohort"], normalized):
        cohort_text = str(cohort).strip().upper()
        if not cohort_text.startswith(country + "_"):
            raise ValueError(f"Cohort {cohort!r} inconsistent with country {country}")
        matches = re.findall(r"(?:^|_)(20\d{2})(?=_|$)", cohort_text)
        if len(matches) != 1:
            raise ValueError(f"Cohort {cohort!r} has no unambiguous year")
        years.append(int(matches[0]))
    manifest["_year"] = years
    date_column = ("acquisition_date" if "acquisition_date" in manifest.columns
                   else "scene_date" if "scene_date" in manifest.columns else None)
    missing_dates = 0
    if date_column:
        missing_dates = int(manifest[date_column].eq("").sum())
        present = manifest[date_column].ne("")
        parsed = pd.to_datetime(manifest.loc[present, date_column], errors="raise")
        if not all(parsed.dt.year.to_numpy() == manifest.loc[present, "_year"].to_numpy()):
            raise ValueError("Acquisition dates disagree with declared cohort years")

    calendar["_country"] = calendar["country"].str.strip().str.upper().map(COUNTRY_ALIASES)
    if calendar["_country"].isna().any():
        raise ValueError("Unrecognized calendar country code")
    calendar["_crop"] = calendar["crop"].str.strip()
    if calendar.duplicated(["_country", "_crop"]).any():
        raise ValueError("Duplicate calendar country/crop entries")
    for row in calendar.itertuples(index=False):
        for field in ("start", "end"):
            val = str(getattr(row, field))
            try:
                pd.Timestamp("2000-" + val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid calendar {field}={val!r}") from exc

    observed = (
        manifest.groupby(["_country", "_year", "cohort", "split"], dropna=False)
        .size().rename("samples").reset_index()
        .rename(columns={"_country": "country", "_year": "year"})
        .sort_values(["country", "year", "cohort", "split"])
    )
    covered = set(zip(calendar["_country"], calendar["_crop"]))
    countries = sorted(set(manifest["_country"]))
    missing = [
        {"country": country, "crop": crop}
        for country in countries for crop in REQUIRED_CALENDAR_CROPS
        if (country, crop) not in covered
    ]
    report = {
        "manifest_sha256": _sha(manifest_file),
        "calendar_sha256": _sha(calendar_file),
        "manifest_rows": len(manifest),
        "countries": countries,
        "cohorts": sorted(manifest["cohort"].unique().tolist()),
        "calendar_countries": sorted(calendar["_country"].unique().tolist()),
        "missing_calendar_pairs": missing,
        "complete_calendar": not missing,
        "date_column": date_column,
        "missing_acquisition_dates": missing_dates,
        "note": ("Manifest-derived cohort inventory only. Calendar is descriptive; "
                 "this audit does not constrain Sentinel catalog selection or change images."),
    }
    if require_full_calendar and missing:
        raise ValueError(f"Missing {len(missing)} country/crop calendar windows: {missing[:5]}")
    if output_dir is not None:
        dest = Path(output_dir)
        if dest.exists():
            raise FileExistsError(f"Output directory exists: {dest}")
        dest.mkdir(parents=True, exist_ok=False)
        observed.to_csv(dest / "manifest_cohort_partition_counts.csv", index=False)
        with (dest / "missing_calendar_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["country", "crop"])
            writer.writeheader()
            writer.writerows(missing)
        (dest / "cohort_calendar_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
