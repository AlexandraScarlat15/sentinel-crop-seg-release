"""Auditable pixel-weighted calendar selection for NEW TRAIN manifests.

This is separate from scene-date discovery and does not reproduce frozen B/B'
memberships: source class counts and their label versions must be audited.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import yaml

from sentinel_crop_seg.core.splitting.spatial_split import rebase_asset_paths

SPLITS = frozenset({"train", "val", "test", "excluded"})
COUNTRY = re.compile(r"[A-Z]{2}\Z")
INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
SCHEMA = {"country", "crop", "start", "end"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(threshold: str | float | Decimal) -> tuple[int, int, str]:
    try:
        decimal = Decimal(str(threshold))
    except InvalidOperation as exc:
        raise ValueError("Threshold must be a finite value between 0 and 1") from exc
    if not decimal.is_finite() or not Decimal(0) <= decimal <= Decimal(1):
        raise ValueError("Threshold must be a finite value between 0 and 1")
    numerator, denominator = decimal.as_integer_ratio()
    return numerator, denominator, str(decimal)


def _load_classes(taxonomy_file: Path, binding_file: Path) -> dict[int, list[str]]:
    taxonomy = yaml.safe_load(taxonomy_file.read_text(encoding="utf-8-sig"))
    binding = yaml.safe_load(binding_file.read_text(encoding="utf-8-sig"))
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("classes"), dict):
        raise ValueError("Invalid taxonomy YAML classes mapping")
    classes = taxonomy["classes"]
    if (not isinstance(binding, dict) or set(binding) != {"schema_version", "class_calendar"}
            or type(binding["schema_version"]) is not int or binding["schema_version"] != 1
            or not isinstance(binding["class_calendar"], dict)):
        raise ValueError("Invalid class/calendar binding schema")
    mapping = binding["class_calendar"]
    if set(mapping) != set(classes) - {"background"}:
        raise ValueError("Calendar binding must cover exactly the canonical non-background classes")
    ids: dict[int, list[str]] = {}
    for name, detail in classes.items():
        if not isinstance(detail, dict) or type(detail.get("id")) is not int:
            raise ValueError("Taxonomy contains an invalid class ID")
        class_id = detail["id"]
        if class_id in ids or class_id < 0:
            raise ValueError("Taxonomy contains duplicated or negative class IDs")
        if name == "background":
            if class_id != 0:
                raise ValueError("Taxonomy background must have ID zero")
            ids[0] = []
            continue
        crops = mapping[name]
        if (not isinstance(crops, list) or not crops or
                any(not isinstance(c, str) or not c.strip() for c in crops)
                or len({c.strip().casefold() for c in crops}) != len(crops)):
            raise ValueError(f"Invalid crop binding for class {name}")
        if len(crops) != 1 and name != "pasture":
            raise ValueError(f"Only aggregate pasture can bind to multiple crops: {name}")
        if name == "pasture" and len(crops) != 4:
            raise ValueError("Pasture binding requires all four constituent vegetation windows")
        ids[class_id] = [c.strip() for c in crops]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Taxonomy IDs must be contiguous")
    return ids


def _window_day(text: str, year: int) -> date:
    if not isinstance(text, str) or not re.fullmatch(r"[0-9]{2}-[0-9]{2}", text):
        raise ValueError(f"Invalid calendar MM-DD: {text!r}")
    try:
        return date.fromisoformat(f"{year}-{text}")
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date {text!r} for year {year}") from exc


def _load_calendar(calendar_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    table = pd.read_csv(calendar_path, dtype=str, keep_default_na=False,
                        encoding="utf-8-sig")
    if set(table.columns) != SCHEMA or table.empty:
        raise ValueError("Calendar CSV requires country,crop,start,end")
    mapping = {}
    for row in table.itertuples(index=False):
        country = row.country.strip()
        crop = row.crop.strip()
        if not COUNTRY.fullmatch(country) or not crop:
            raise ValueError("Invalid calendar country/crop")
        key = (country, crop.casefold())
        if key in mapping:
            raise ValueError(f"Duplicate calendar country/crop mapping: {key}")
        _window_day(row.start, 2000)
        _window_day(row.end, 2000)
        if row.start > row.end:
            raise ValueError(f"Calendar window crosses year: {key}")
        mapping[key] = (row.start, row.end)
    return mapping


def _read_manifest(path: Path, ids: dict[int, list[str]]) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    needed = {"dataset", "sample_id", "country", "split"}
    if missing := needed - set(table):
        raise ValueError(f"Manifest missing required identity fields: {sorted(missing)}")
    if table.empty or table.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Empty manifest or duplicate (dataset,sample_id) identities")
    if any(table[col].str.strip().eq("").any() for col in needed):
        raise ValueError("Blank manifest identity/country/split")
    if not table["split"].isin(SPLITS).all():
        raise ValueError("Unknown manifest split")
    if "partition" in table and not table["split"].equals(table["partition"]):
        raise ValueError("Manifest split/partition mismatch")
    if not table["country"].map(lambda x: bool(COUNTRY.fullmatch(x))).all():
        raise ValueError("Manifest countries must use uppercase ISO-2 codes")
    if not ("acquisition_date" in table or "scene_date" in table):
        raise ValueError("Manifest requires acquisition_date or scene_date")
    if "acquisition_date" in table and "scene_date" in table:
        both = table["acquisition_date"].ne("") & table["scene_date"].ne("")
        if not table.loc[both, "acquisition_date"].equals(table.loc[both, "scene_date"]):
            raise ValueError("Conflicting manifest acquisition/scene dates")
    missing_counts = [f"p{i}" for i in ids if f"p{i}" not in table]
    if missing_counts:
        raise ValueError(f"Manifest missing source class-pixel fields: {missing_counts}")
    return table


def _count(raw: str, key: str) -> int:
    if not isinstance(raw, str) or not INTEGER.fullmatch(raw):
        raise ValueError(f"Invalid integer class pixel count in {key}: {raw!r}")
    return int(raw)


def _date(raw: str) -> date | None:
    if not raw:
        return None
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid manifest acquisition date {raw!r}") from exc
    if parsed.isoformat() != raw:
        raise ValueError(f"Invalid manifest acquisition date {raw!r}")
    return parsed


def _evaluate_training(
    table: pd.DataFrame, ids: dict[int, list[str]],
    calendar: dict[tuple[str, str], tuple[str, str]],
) -> pd.DataFrame:
    records = []
    date_field = "acquisition_date" if "acquisition_date" in table else "scene_date"
    for row in table.loc[table["split"].eq("train")].to_dict("records"):
        pixels = {idx: _count(row[f"p{idx}"], f"p{idx}") for idx in ids}
        agricultural = sum(n for idx, n in pixels.items() if idx != 0)
        country = row["country"]
        observed = _date(row[date_field])
        inside = outside = unknown = 0
        for class_id, crop_names in ids.items():
            n = pixels[class_id]
            if class_id == 0 or n == 0:
                continue
            if observed is None:
                unknown += n
                continue
            windows = [calendar.get((country, name.casefold())) for name in crop_names]
            if any(window is None for window in windows):
                unknown += n
                continue
            starts = [_window_day(window[0], observed.year) for window in windows]
            ends = [_window_day(window[1], observed.year) for window in windows]
            first, last = max(starts), min(ends)
            if first > last:
                unknown += n
            elif first <= observed <= last:
                inside += n
            else:
                outside += n
        if inside + outside + unknown != agricultural:
            raise RuntimeError("Calendar agricultural pixel mass did not balance")
        complete = unknown == 0 and agricultural > 0
        records.append({
            "dataset": row["dataset"], "sample_id": row["sample_id"],
            "country": country, "cohort": row.get("cohort", ""),
            "split": "train", "acquisition_date": observed.isoformat() if observed else "",
            "background_pixels": pixels[0],
            "agricultural_pixels_total": agricultural,
            "calendar_compatible_pixels": inside,
            "calendar_incompatible_pixels": outside,
            "calendar_unresolved_pixels": unknown,
            "calendar_coverage_complete": complete,
            "calendar_coverage": inside / agricultural if complete else None,
            "unresolved_reason": (
                "no_agricultural_pixels" if agricultural == 0 else
                "missing_date_or_country_crop_window" if unknown else ""
            ),
        })
    if not records:
        raise ValueError("A training selection requires at least one original TRAIN sample")
    return pd.DataFrame.from_records(records)


def build_weighted_calendar_selection(
    manifest_file: str | Path, calendar_file: str | Path,
    taxonomy_file: str | Path, binding_file: str | Path,
    output_dir: str | Path, *, threshold: str = "0.90",
    sweep: tuple[str, ...] = ("0.75", "0.90", "1.00"),
    validate_only: bool = True,
) -> dict:
    """Audit calendar-compatible TRAIN pixels and optionally write a new manifest.

    Args:
        manifest_file: Source manifest containing split, date and class counts.
        calendar_file: Country/crop MM-DD window CSV.
        taxonomy_file: YAML definition of canonical class IDs.
        binding_file: YAML mapping classes to calendar crop names.
        output_dir: New destination for optional reports and derived manifest.
        threshold: Minimum eligible crop-pixel fraction, inclusive.
        sweep: Additional thresholds for TRAIN-only retention sensitivity.
        validate_only: When true, return a report without writing files.

    Returns:
        A JSON-compatible summary of selected TRAIN counts, threshold
        sensitivity, preserved validation/test membership and source hashes.

    Raises:
        FileExistsError: If a requested output destination already exists.
        ValueError: If input schemas, dates, counts or thresholds are invalid.
    """
    source, calendar_path = Path(manifest_file), Path(calendar_file)
    taxonomy, binding = Path(taxonomy_file), Path(binding_file)
    destination = Path(output_dir)
    if destination.exists() and not validate_only:
        raise FileExistsError(f"Output directory already exists: {destination}")
    ids = _load_classes(taxonomy, binding)
    mapping = _load_calendar(calendar_path)
    data = _read_manifest(source, ids)
    audit = _evaluate_training(data, ids, mapping)
    train = data.loc[data["split"].eq("train")]
    cut = _fraction(threshold)
    choices = list(dict.fromkeys([str(x) for x in (*sweep, threshold)]))
    sweep_output = []
    for raw in choices:
        numerator, denominator, label = _fraction(raw)
        keep = (
            audit["calendar_coverage_complete"] &
            audit["agricultural_pixels_total"].gt(0) &
            audit.apply(
                lambda row: int(row["calendar_compatible_pixels"]) * denominator >=
                int(row["agricultural_pixels_total"]) * numerator, axis=1
            )
        )
        class_retention = {
            str(class_id): {
                "before_pixels": sum(_count(v, f"p{class_id}") for v in
                                     train[f"p{class_id}"]),
                "after_pixels": sum(_count(v, f"p{class_id}") for v in
                                    train.loc[keep.to_numpy(), f"p{class_id}"]),
            }
            for class_id in ids if class_id != 0
        }
        sweep_output.append({
            "threshold": label, "train_retained": int(keep.sum()),
            "train_agricultural_pixel_mass_by_class": class_retention,
            "train_rejected": int(len(keep) - keep.sum()),
            "train_retained_percent": float(100 * keep.mean()),
            "train_retained_by_country": {
                str(k): int(v) for k, v in
                audit.loc[keep, "country"].value_counts().sort_index().items()
            },
            "train_retained_by_cohort": {
                str(k): int(v) for k, v in
                audit.loc[keep, "cohort"].value_counts().sort_index().items()
            },
        })
        if (numerator, denominator) == cut[:2]:
            selected = keep
    assert selected is not None
    train_ids = set(zip(audit.loc[selected, "dataset"], audit.loc[selected, "sample_id"]))
    identities = list(zip(data["dataset"], data["sample_id"]))
    preserve = data["split"].ne("train")
    chosen = data.loc[
        preserve | pd.Series([key in train_ids for key in identities], index=data.index)
    ].copy()
    base_holdout = [
        (r["dataset"], r["sample_id"], r["split"])
        for r in data.loc[data["split"].isin(["val", "test"])].to_dict("records")
    ]
    new_holdout = [
        (r["dataset"], r["sample_id"], r["split"])
        for r in chosen.loc[chosen["split"].isin(["val", "test"])].to_dict("records")
    ]
    if base_holdout != new_holdout:
        raise RuntimeError("Validation/test membership or order was modified")
    report = {
        "status": "validated_only" if validate_only else "new_train_filtered_manifest",
        "selection_rule": "complete_calendar_and_pixel_weighted_coverage_ge_threshold",
        "coverage_denominator": "sum(p1..pN); background p0 excluded",
        "class10_rule": "intersection_of_all_declared_pasture_component_windows",
        "chosen_threshold": cut[2],
        "train_input": int(data["split"].eq("train").sum()),
        "train_selected": len(train_ids),
        "train_unresolved": int((~audit["calendar_coverage_complete"]).sum()),
        "train_no_agricultural_pixels": int(audit["agricultural_pixels_total"].eq(0).sum()),
        "val_preserved": int(data["split"].eq("val").sum()),
        "test_preserved": int(data["split"].eq("test").sum()),
        "excluded_preserved": int(data["split"].eq("excluded").sum()),
        "threshold_sensitivity_train_only": sweep_output,
        "source_sha256": {
            "manifest": _sha(source), "calendar": _sha(calendar_path),
            "taxonomy": _sha(taxonomy), "class_calendar_binding": _sha(binding),
        },
        "notes": [
            "New experiment, not a reproduction of historical Model B or B-prime.",
            "Source p1..pN are input histogram counts, not independently verified corrected masks.",
            "All sensitivity counts are computed on TRAIN; VAL and TEST are preserved unchanged.",
            "90% is an experiment parameter, not a proven optimal universal threshold.",
            "No satellite data, imagery, masks, checkpoints or source CSV is modified.",
        ],
    }
    if validate_only:
        return report
    chosen = rebase_asset_paths(chosen, source, destination)
    destination.mkdir(parents=True, exist_ok=False)
    chosen.to_csv(destination / "calendar_selected_manifest.csv", index=False)
    audit.to_csv(destination / "calendar_train_audit.csv", index=False)
    (destination / "calendar_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
