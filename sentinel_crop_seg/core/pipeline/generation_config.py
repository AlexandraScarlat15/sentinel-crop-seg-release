"""Explicit, inspectable configuration boundary for dataset generation.

Legacy mode remains untouched. Isolated mode merges only the declared
generator YAML files, in explicit order; experiment-specific training YAML
files cannot silently override the generator's dataset settings.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sentinel_crop_seg.core.config_loader import apply_overrides, load_config
from sentinel_crop_seg.core.pipeline.acquisition_calendar import AcquisitionCalendar
from sentinel_crop_seg.core.pipeline.generation_recipe import GenerationRecipe, apply_generation_recipe


GENERATION_YAML_FILES = (
    "generation.yaml",
    "paths.yaml",
    "eurocrops.yaml",
    "catalog.yaml",
    "dataset.yaml",
    "classes.yaml",
    "sentinel.yaml",
)
CONFIG_MODES = ("legacy", "isolated")


def load_generation_config(
    config_dir: str | Path = "config",
    *,
    mode: str = "legacy",
    overrides: Sequence[str] | None = None,
    experiment_recipe: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve baseline YAML, an optional versioned recipe and overrides.

    Args:
        config_dir: Directory with baseline generator YAML files.
        mode: `isolated` for explicit generation YAMLs or `legacy` for
            historical configuration loading.
        overrides: Dotted-key CLI overrides applied to the resolved settings.
        experiment_recipe: Optional YAML recipe; requires isolated mode.

    Returns:
        A merged generator configuration. An enabled calendar additionally
        includes checked acquisition-calendar provenance.

    Raises:
        ValueError: If the mode, recipe or calendar configuration is invalid.
    """
    if mode not in CONFIG_MODES:
        raise ValueError(f"Unknown generation config mode {mode!r}; choose {CONFIG_MODES}")
    if experiment_recipe is not None and mode != "isolated":
        raise ValueError("Experiment recipes require isolated configuration mode")
    selected = GENERATION_YAML_FILES if mode == "isolated" else None
    config = load_config(config_dir, overrides=None if experiment_recipe else overrides, yaml_files=selected)
    if experiment_recipe is None:
        return config
    resolved = apply_generation_recipe(config, GenerationRecipe.from_yaml(experiment_recipe))
    apply_overrides(resolved, overrides)
    calendar = AcquisitionCalendar.from_config(resolved)
    if calendar is not None:
        stac = resolved["catalog"]["stac"]
        calendar.search_intervals(stac["start_date"], stac["end_date"])
        resolved["acquisition_calendar_provenance"] = calendar.provenance()
    return resolved


def generation_config_report(
    config_dir: str | Path = "config",
    *,
    mode: str = "isolated",
    overrides: Sequence[str] | None = None,
    experiment_recipe: str | Path | None = None,
) -> dict[str, Any]:
    """Report effective generator settings without retrieving satellite data.

    Args:
        config_dir: Directory with baseline generator YAML files.
        mode: Configuration loader mode; defaults to `isolated`.
        overrides: Dotted-key CLI overrides.
        experiment_recipe: Optional versioned experiment recipe.

    Returns:
        A JSON-serializable preflight report with configuration fingerprints,
        sampling settings and optional calendar provenance.

    Raises:
        ValueError: If any required setting fails validation.
    """
    config = load_generation_config(config_dir, mode=mode, overrides=overrides, experiment_recipe=experiment_recipe)
    generation = config.get("generation", {})
    paths = config.get("paths", {})
    catalog = config.get("catalog", {}).get("stac", {})
    dataset = config.get("dataset", {})
    roi = dataset.get("roi", {})
    if not all(isinstance(part, dict) for part in (
        generation, paths, catalog, dataset, roi
    )):
        raise ValueError("Generator configuration sections must be mappings")
    required = {
        "paths": ("sentinel_data_root", "dataset_output_dir"),
        "eurocrops": ("shapefile_path",),
        "dataset": ("patch", "bands", "roi"),
        "catalog": ("stac",),
    }
    missing = [
        f"{section}.{key}"
        for section, keys in required.items()
        if not isinstance(config.get(section), dict)
        for key in keys
    ]
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            continue
        missing.extend(f"{section}.{key}" for key in keys if key not in config[section])
    if missing:
        raise ValueError(f"Generator configuration missing sections or keys: {sorted(set(missing))}")

    bands = dataset["bands"]
    patch = dataset["patch"]
    if not isinstance(bands, dict) or not bands:
        raise ValueError("dataset.bands must contain at least one band")
    if not isinstance(patch, dict) or int(patch.get("size", 0)) <= 0 or int(patch.get("stride", 0)) <= 0:
        raise ValueError("dataset.patch.size and stride must be positive integers")
    if roi.get("strategy", "grid") not in ("grid", "hotcell"):
        raise ValueError("dataset.roi.strategy must be grid or hotcell")
    if roi.get("strategy") == "hotcell" and dataset.get("target", {}).get("class_id") is None:
        raise ValueError("Hotcell generation requires dataset.target.class_id")

    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           default=str, allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "config_mode": mode,
        "experiment_recipe": config.get("experiment_recipe"),
        "acquisition_calendar": config.get("acquisition_calendar_provenance"),
        "selected_config_files": list(GENERATION_YAML_FILES) if mode == "isolated" else
            [p.name for p in sorted(Path(config_dir).glob("*.yaml"))],
        "resolved_config_sha256": digest,
        "generator": {
            "satellite": generation.get("satellite", "sentinel2"),
            "label_source": generation.get("label_source", "eurocrops"),
        },
        "roi_strategy": roi.get("strategy", "grid"),
        "patch_size": int(patch["size"]),
        "stride": int(patch["stride"]),
        "bands": list(bands),
        "target_class_id": dataset.get("target", {}).get("class_id"),
        "max_patches_per_roi": dataset.get("max_patches_per_roi"),
        "target_total_patches": dataset.get("target", {}).get("total_patches"),
        "stac_max_cloud_cover": catalog.get("max_cloud_cover"),
        "notes": [
            "This is an offline configuration preflight, not proof that source data exists.",
            "Strict resume checks resolved configuration and source fingerprints; legacy run_tag alone is insufficient.",
            "Use a new output directory and run_tag for a changed generator recipe.",
        ],
    }
