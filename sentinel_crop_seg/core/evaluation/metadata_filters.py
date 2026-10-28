from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from pathlib import PurePosixPath


def month_is_allowed(month: int, month_start: int, month_end: int) -> bool:
    """
    Check whether a month is inside an inclusive month interval.

    Supports normal intervals, e.g. April-September: 4..9,
    and wrap-around intervals, e.g. November-March: 11..3.
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month must be in [1, 12], got {month}")

    if not (1 <= month_start <= 12):
        raise ValueError(f"month_start must be in [1, 12], got {month_start}")

    if not (1 <= month_end <= 12):
        raise ValueError(f"month_end must be in [1, 12], got {month_end}")

    if month_start <= month_end:
        return month_start <= month <= month_end

    return month >= month_start or month <= month_end


def parse_scene_month(scene_date: str) -> int:
    """
    Parse a scene date and return its month.

    Expected common format: YYYY-MM-DD.
    ISO datetime strings are also accepted because only the first 10 chars are used.
    """
    value = str(scene_date).strip()

    if not value:
        raise ValueError("scene_date is empty")

    try:
        return date.fromisoformat(value[:10]).month
    except ValueError as exc:
        raise ValueError(f"Could not parse scene_date={scene_date!r}") from exc


def resolve_sample_name_column(
    fieldnames: list[str] | None,
    preferred_column: str | None = None,
) -> str:
    """
    Resolve the column containing sample identifiers.

    Supported default aliases:
    - sample_name: used by Dacia5 metadata.csv
    - sample_id: used by Sentinel-EuroCrops patches_log.csv
    """
    if fieldnames is None:
        raise ValueError("metadata CSV has no header.")

    available = set(fieldnames)

    if preferred_column is not None:
        if preferred_column not in available:
            raise ValueError(
                f"metadata CSV is missing required sample column: {preferred_column!r}"
            )
        return preferred_column

    for candidate in ("sample_name", "sample_id"):
        if candidate in available:
            return candidate

    raise ValueError(
        "metadata CSV must contain one of these sample identifier columns: "
        "sample_name, sample_id"
    )


def read_allowed_sample_names_by_month(
    metadata_csv: str | Path,
    *,
    month_start: int,
    month_end: int,
    sample_name_column: str | None = None,
    scene_date_column: str = "scene_date",
) -> set[str]:
    """
    Read a metadata CSV and return sample identifiers whose scene_date month is allowed.

    The sample column is auto-detected:
    - sample_name for Dacia5 metadata.csv
    - sample_id for Sentinel-EuroCrops patches_log.csv
    """
    metadata_csv = Path(metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_csv}")

    allowed: set[str] = set()

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        resolved_sample_column = resolve_sample_name_column(
            reader.fieldnames,
            preferred_column=sample_name_column,
        )

        if reader.fieldnames is None:
            raise ValueError(f"metadata CSV has no header: {metadata_csv}")

        if scene_date_column not in set(reader.fieldnames):
            raise ValueError(
                f"metadata CSV is missing required scene date column: {scene_date_column!r}"
            )

        for row in reader:
            sample_name = str(row[resolved_sample_column]).strip()
            scene_date = str(row[scene_date_column]).strip()

            if not sample_name:
                continue

            month = parse_scene_month(scene_date)

            if month_is_allowed(
                month=month,
                month_start=month_start,
                month_end=month_end,
            ):
                allowed.add(sample_name)

    return allowed


def validate_month_filter_args(
    *,
    metadata_csv: str | Path | None,
    month_start: int | None,
    month_end: int | None,
) -> bool:
    """
    Validate optional month filtering arguments.

    Returns True when filtering is enabled.
    """
    enabled = month_start is not None or month_end is not None

    if not enabled:
        return False

    if metadata_csv is None:
        raise ValueError("--metadata-csv is required when filtering by month.")

    if month_start is None or month_end is None:
        raise ValueError("Both --month-start and --month-end must be provided.")

    if not (1 <= int(month_start) <= 12):
        raise ValueError("--month-start must be in [1, 12].")

    if not (1 <= int(month_end) <= 12):
        raise ValueError("--month-end must be in [1, 12].")

    return True


def normalize_sample_id(value: str) -> str:
    """Normalize slash style and an optional patch-file suffix for joins."""
    return PurePosixPath(str(value).replace("\\", "/")).stem


def read_allowed_manifest_sample_keys(
    filter_csv: str | Path,
    *,
    group: str,
) -> set[tuple[str, str]]:
    """Read ``dataset/sample_id`` keys from a read-only evaluation sidecar.

    The filter is intentionally identity-based, so it cannot redefine a split
    or change any manifest row.  The caller still selects ``--split`` from the
    original manifest.
    """
    filter_csv = Path(filter_csv)
    if not filter_csv.is_file():
        raise FileNotFoundError(f"sample filter CSV not found: {filter_csv}")
    wanted = str(group).strip()
    if not wanted:
        raise ValueError("--sample-filter-group must not be empty")
    allowed: set[tuple[str, str]] = set()
    with filter_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "sample_id", "phenology_group"}
        present = set(reader.fieldnames or ())
        missing = sorted(required - present)
        if missing:
            raise ValueError(f"sample filter CSV missing required columns: {missing}")
        for row in reader:
            if str(row["phenology_group"]).strip() != wanted:
                continue
            dataset = str(row["dataset"]).strip()
            sample_id = normalize_sample_id(str(row["sample_id"]).strip())
            if not dataset or not sample_id:
                raise ValueError("sample filter CSV has an empty dataset/sample_id in selected group")
            key = (dataset, sample_id)
            if key in allowed:
                raise ValueError(f"sample filter CSV has a duplicate selected identity: {key}")
            allowed.add(key)
    if not allowed:
        raise ValueError(f"sample filter group {wanted!r} selected zero identities")
    return allowed
