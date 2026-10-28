"""Optional acquisition-date gating from a versioned crop-calendar CSV.

A crop-union policy accepts a Sentinel-2 scene when its date falls inside at
least one explicitly selected crop window. It does not filter per-parcel labels.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

COUNTRY_CODE = re.compile(r"[A-Z]{2}\Z")
COLUMNS = {"country", "crop", "start", "end"}
FIELDS = {"mode", "file", "country", "year", "crops"}


def _calendar_day(raw: str, *, field: str) -> tuple[int, int]:
    if not isinstance(raw, str) or not re.fullmatch(r"\d{2}-\d{2}", raw):
        raise ValueError(f"Invalid calendar {field}: {raw!r}; expected MM-DD")
    try:
        parsed = date.fromisoformat(f"2000-{raw}")
    except ValueError as exc:
        raise ValueError(f"Invalid calendar {field}: {raw!r}") from exc
    return parsed.month, parsed.day


def _scene_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("Calendar requires a Sentinel product acquisition_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid Sentinel acquisition_date: {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"Invalid Sentinel acquisition_date: {value!r}")
    return parsed


@dataclass(frozen=True)
class AcquisitionCalendar:
    country: str
    year: int
    crops: tuple[str, ...]
    windows: tuple[tuple[str, date, date], ...]
    csv_path: Path
    csv_sha256: str

    @classmethod
    def from_config(cls, config: dict) -> "AcquisitionCalendar | None":
        section = config.get("catalog", {}).get("calendar")
        if section is None:
            return None
        if not isinstance(section, dict):
            raise ValueError("catalog.calendar must be a mapping")
        if unknown := set(section) - FIELDS:
            raise ValueError(f"Unknown catalog.calendar fields: {sorted(unknown)}")
        mode = section.get("mode", "none")
        if mode == "none":
            if set(section) != {"mode"}:
                raise ValueError("Disabled calendar policy accepts only mode: none")
            return None
        if mode != "selected_crop_union" or set(section) != FIELDS:
            raise ValueError("Calendar mode selected_crop_union requires file, country, year and crops")
        country, year, crops = section["country"], section["year"], section["crops"]
        if not isinstance(country, str) or not COUNTRY_CODE.fullmatch(country):
            raise ValueError("Calendar country must be uppercase ISO-2")
        if type(year) is not int or not 2015 <= year <= 2100:
            raise ValueError("Calendar year must be an integer between 2015 and 2100")
        if (not isinstance(crops, list) or not crops or
                any(not isinstance(c, str) or not c.strip() for c in crops)):
            raise ValueError("Calendar crops must be a nonempty list of crop names")
        keys = [c.strip().casefold() for c in crops]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate selected calendar crops")
        filename = section["file"]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Calendar file must be a nonempty path")
        csv_path = Path(filename).resolve()
        data = csv_path.read_bytes()
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or set(reader.fieldnames) != COLUMNS:
                raise ValueError("Calendar CSV requires exactly country,crop,start,end")
            entries = {}
            for row in reader:
                if None in row or any(not isinstance(row[c], str) for c in COLUMNS):
                    raise ValueError("Malformed calendar CSV row")
                code, crop = row["country"].strip(), row["crop"].strip()
                if not COUNTRY_CODE.fullmatch(code) or not crop:
                    raise ValueError("Invalid calendar country or crop")
                key = (code, crop.casefold())
                if key in entries:
                    raise ValueError(f"Duplicate calendar country/crop: {code}/{crop}")
                start = _calendar_day(row["start"].strip(), field="start")
                end = _calendar_day(row["end"].strip(), field="end")
                if start > end:
                    raise ValueError(f"Calendar window crosses the year boundary: {code}/{crop}")
                try:
                    entries[key] = (crop, date(year, *start), date(year, *end))
                except ValueError as exc:
                    raise ValueError(f"Invalid calendar date for {code}/{crop}/{year}") from exc
        missing = [c for c in crops if (country, c.strip().casefold()) not in entries]
        if missing:
            raise ValueError(f"Missing calendar windows for {country}: {missing}")
        windows = tuple(entries[(country, key)] for key in keys)
        return cls(country, year, tuple(crops), windows,
                   csv_path, hashlib.sha256(data).hexdigest())

    def accepts(self, acquisition_date: object) -> bool:
        observed = _scene_date(acquisition_date)
        return observed.year == self.year and any(
            start <= observed <= end for _, start, end in self.windows
        )

    def search_intervals(self, start: str, end: str) -> list[tuple[str, str]]:
        """Intersect and merge crop windows with the configured STAC range."""
        def bound(value: str, *, is_end: bool) -> datetime:
            if not isinstance(value, str):
                raise ValueError("STAC bounds must be ISO date/datetime strings")
            if len(value) == 10:
                try:
                    return datetime.combine(
                        date.fromisoformat(value), time.max if is_end else time.min,
                        tzinfo=timezone.utc,
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid STAC date: {value!r}") from exc
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid STAC datetime: {value!r}") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("STAC datetime bounds must include a UTC offset")
            return parsed.astimezone(timezone.utc)

        lower, upper = bound(start, is_end=False), bound(end, is_end=True)
        if lower > upper:
            raise ValueError("STAC start_date must precede end_date")
        ranges = sorted(
            (
                max(lower, datetime.combine(first, time.min, tzinfo=timezone.utc)),
                min(upper, datetime.combine(last, time.max, tzinfo=timezone.utc)),
            )
            for _, first, last in self.windows
        )
        merged: list[tuple[datetime, datetime]] = []
        for first, last in ranges:
            if first > last:
                continue
            if merged and first <= merged[-1][1] + timedelta(microseconds=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], last))
            else:
                merged.append((first, last))
        if not merged:
            raise ValueError("Selected crop calendar has no overlap with STAC acquisition dates")
        return [
            (first.isoformat().replace("+00:00", "Z"),
             last.isoformat().replace("+00:00", "Z"))
            for first, last in merged
        ]

    def provenance(self) -> dict:
        return {
            "mode": "selected_crop_union",
            "country": self.country,
            "year": self.year,
            "crops": list(self.crops),
            "calendar_csv": str(self.csv_path),
            "calendar_sha256": self.csv_sha256,
            "windows": [
                {"crop": name, "start": begin.isoformat(), "end": end.isoformat()}
                for name, begin, end in self.windows
            ],
            "semantics": "scene_date_in_selected_crop_window_union_not_parcel_filter",
        }
