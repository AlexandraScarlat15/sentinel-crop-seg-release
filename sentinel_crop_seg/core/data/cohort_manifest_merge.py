"""Merge separately adapted cohort manifests without moving or modifying assets.

Run after per-cohort patch-log adaptation and before one audited spatial split.
This never regenerates image patches and does not reconstruct frozen paper splits.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from sentinel_crop_seg.core.splitting.spatial_split import BOX, rebase_asset_paths

REQUIRED = frozenset({
    "dataset", "sample_id", "country", "cohort", "epsg", *BOX,
    "img_path", "mask_path", "band_order",
})
FORBIDDEN = frozenset({
    "split", "partition", "source_split", "source_partition",
    "cell_id", "exclusion_reason",
})
COUNTRY = re.compile(r"[A-Z]{2}\Z")
NONNEGATIVE_INTEGER = re.compile(r"(0|[1-9][0-9]*)\Z")
HISTOGRAM = re.compile(r"p(0|[1-9][0-9]*)\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_cohort_manifests(
    manifests: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    verify_assets: bool = False,
    validate_only: bool = True,
) -> dict:
    """Join distinct unsplit cohort manifests into one spatial-split input.

    Args:
        manifests: At least two separately adapted cohort manifest CSVs.
        output_dir: New output directory for combined_manifest.csv and audit JSON.
        verify_assets: Check selected image/mask file existence before merging.
        validate_only: Return a read-only report; writing requires explicit opt-in.

    Returns:
        Manifest-merge audit with input SHA-256 hashes, band/class information,
        counts and optional new output paths. No image/mask assets are copied.

    Raises:
        ValueError: If identities, provenance, pixel counts or band order conflict.
        FileNotFoundError: If an input manifest or checked asset is missing.
        FileExistsError: If a requested new output directory already exists.
    """
    if len(manifests) < 2:
        raise ValueError("Supply at least two independently adapted cohort manifests")
    sources = [Path(path).resolve() for path in manifests]
    if len(set(sources)) != len(sources):
        raise ValueError("Input manifest paths must be distinct")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {destination}")
    if any(destination == source or destination in source.parents for source in sources):
        raise ValueError("The output directory cannot contain an input manifest")

    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []
    dataset_ids: set[str] = set()
    cohort_ids: set[str] = set()
    baseline_band_order: str | None = None
    baseline_label_source: str | None = None
    histogram_columns: set[str] = set()

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing input manifest: {source}")
        data = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if data.empty:
            raise ValueError(f"{source.name}: input manifest is empty")
        missing = REQUIRED - set(data)
        if missing:
            raise ValueError(f"{source.name}: missing required columns {sorted(missing)}")
        present_forbidden = FORBIDDEN & set(data)
        if present_forbidden:
            raise ValueError(
                f"{source.name}: already split/derived columns {sorted(present_forbidden)}; "
                "merge only unsplit adapted manifests"
            )
        for col in REQUIRED:
            if data[col].astype(str).str.strip().eq("").any():
                raise ValueError(f"{source.name}: empty {col}")
        for key in ("dataset", "country", "cohort", "band_order"):
            values = data[key].unique()
            if len(values) != 1:
                raise ValueError(f"{source.name}: inconsistent {key} within one cohort manifest")
        dataset = str(data["dataset"].iloc[0])
        cohort = str(data["cohort"].iloc[0])
        country = str(data["country"].iloc[0])
        band_order = str(data["band_order"].iloc[0])
        if dataset in dataset_ids or cohort in cohort_ids:
            raise ValueError("Each input manifest must use a distinct dataset and cohort identity")
        if not COUNTRY.fullmatch(country):
            raise ValueError(f"{source.name}: country must be uppercase ISO-2")
        bands = [item.strip() for item in band_order.split(",")]
        if not bands or any(not item for item in bands) or len(bands) != len(set(bands)):
            raise ValueError(f"{source.name}: invalid band_order")
        if baseline_band_order is None:
            baseline_band_order = band_order
        elif band_order != baseline_band_order:
            raise ValueError("Incompatible band_order across input cohorts")
        if "label_source" in data:
            sources_label = {str(value).strip() for value in data["label_source"]}
            if len(sources_label) != 1 or not next(iter(sources_label)):
                raise ValueError(f"{source.name}: inconsistent label_source")
            label_source = next(iter(sources_label))
            if baseline_label_source is None:
                baseline_label_source = label_source
            elif label_source != baseline_label_source:
                raise ValueError("Incompatible label_source across input cohorts")
        elif baseline_label_source is not None:
            raise ValueError("Mixed declared/undeclared label_source across cohorts")

        if "label_source_ref" in data:
            refs = {str(value).strip() for value in data["label_source_ref"]}
            if len(refs) != 1 or not next(iter(refs)):
                raise ValueError(f"{source.name}: inconsistent label_source_ref")
            if "label_source" not in data:
                raise ValueError(f"{source.name}: label_source_ref needs label_source")

        if data.duplicated(["dataset", "sample_id"]).any():
            raise ValueError(f"{source.name}: repeated sample identity")
        if "acquisition_date" in data:
            dates = data["acquisition_date"].astype(str).str.strip()
            invalid = pd.to_datetime(dates[dates != ""], format="%Y-%m-%d", errors="coerce").isna()
            if invalid.any():
                raise ValueError(f"{source.name}: invalid acquisition_date")
        for column in ("epsg", *BOX):
            values = pd.to_numeric(data[column], errors="raise")
            if not pd.Series(values).map(lambda value: pd.notna(value) and abs(value) < float("inf")).all():
                raise ValueError(f"{source.name}: nonfinite {column}")
            if column == "epsg" and (
                (values <= 0).any() or (values != values.astype(int)).any()
            ):
                raise ValueError(f"{source.name}: invalid epsg")
        for col in ("bbox_minx", "bbox_miny"):
            maximum = "bbox_maxx" if col.endswith("x") else "bbox_maxy"
            if (pd.to_numeric(data[col]) >= pd.to_numeric(data[maximum])).any():
                raise ValueError(f"{source.name}: invalid projected bounds")
        cols = {col for col in data if HISTOGRAM.fullmatch(col)}
        if not cols:
            raise ValueError(f"{source.name}: missing p0..pN histogram columns")
        for col in cols:
            if not data[col].astype(str).map(lambda v: bool(NONNEGATIVE_INTEGER.fullmatch(v))).all():
                raise ValueError(f"{source.name}: invalid nonnegative integer histogram column {col}")
        histogram_columns.update(cols)
        # Keep original source-log columns as provenance. Only selected, consumable
        # img_path/mask_path are rebased, never source-log-relative asset columns.
        rebased = rebase_asset_paths(data, source, destination)
        if verify_assets:
            for col in ("img_path", "mask_path"):
                for value in rebased[col]:
                    resolved = Path(value)
                    if not resolved.is_absolute():
                        resolved = (destination / resolved).resolve()
                    if not resolved.is_file():
                        raise FileNotFoundError(f"{source.name}: unresolved {col}: {resolved}")
        frames.append(rebased)
        provenance.append({
            "manifest_file": source.name,
            "sha256": _sha256(source),
            "dataset": dataset,
            "cohort": cohort,
            "country": country,
            "samples": len(data),
            "label_source_ref": (
                str(data["label_source_ref"].iloc[0]) if "label_source_ref" in data
                and data["label_source_ref"].nunique() == 1 else None
            ),
        })
        dataset_ids.add(dataset)
        cohort_ids.add(cohort)

    # A declared label source must be present in EVERY input, not merely
    # in later files. Its country/year edition may differ by cohort.
    if any("label_source" in frame for frame in frames) and not all(
        "label_source" in frame for frame in frames
    ):
        raise ValueError("Mixed declared/undeclared label_source across cohorts")
    if any("label_source_ref" in frame for frame in frames) and not all(
        "label_source_ref" in frame for frame in frames
    ):
        raise ValueError("Mixed declared/undeclared label_source_ref across cohorts")
    all_classes = sorted(histogram_columns, key=lambda col: int(col[1:]))
    for frame in frames:
        for col in all_classes:
            if col not in frame:
                frame[col] = "0"
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Duplicate (dataset,sample_id) in combined manifest")
    report = {
        "status": "validated_only" if validate_only else "written",
        "samples": int(len(combined)),
        "datasets": len(dataset_ids),
        "cohorts": len(cohort_ids),
        "band_order": baseline_band_order,
        "class_ids_in_histograms": [int(k[1:]) for k in all_classes],
        "source_manifests": provenance,
        "verified_asset_paths": bool(verify_assets),
        "source_files_unchanged": True,
        "split_assigned": False,
        "notes": [
            "Use sentinel-crop-split-spatial to assign a new geographic train/val/test split.",
            "Country and cohort values are declarations, not verified geolocation.",
            "Original per-cohort source-log relative columns are retained as provenance.",
            "No historical paper partition or corrected-mask identity is inferred.",
        ],
    }
    if validate_only:
        return report
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "combined_manifest.csv"
    combined.to_csv(target, index=False)
    report["combined_manifest_sha256"] = _sha256(target)
    (destination / "merge_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
