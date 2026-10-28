"""Safe planning and execution of independent Sentinel-2/EuroCrops cohorts."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import yaml

from sentinel_crop_seg.core.pipeline.generation_config import generation_config_report, load_generation_config

IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}\Z")
COUNTRY = re.compile(r"[A-Z]{2}\Z")


def _read_plan(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "name", "cohorts"}:
        raise ValueError("Cohort plan requires schema_version, name and cohorts")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported cohort plan schema_version")
    if not isinstance(raw["name"], str) or not IDENTIFIER.fullmatch(raw["name"]):
        raise ValueError("Invalid cohort plan name")
    cohorts = raw["cohorts"]
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("cohorts must be a nonempty list")
    return raw


def _datetime(raw: object, label: str) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be an ISO-8601 datetime with UTC offset")
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 datetime") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return result


def plan_cohorts(plan_file: str | Path, *, config_dir: str | Path = "config") -> dict:
    """Validate independent cohort recipes without generating data.

    Args:
        plan_file: Versioned YAML plan containing cohort IDs and recipe paths.
        config_dir: Directory containing baseline generator YAML files.

    Returns:
        A validated preview with cohort identifiers, configuration hashes,
        source declarations, time bounds and output destinations.

    Raises:
        ValueError: If the plan, a recipe or its provenance is inconsistent.
    """
    source = Path(plan_file).resolve()
    plan = _read_plan(source)
    seen_ids: set[str] = set()
    destinations: list[Path] = []
    records = []
    for number, item in enumerate(plan["cohorts"], start=1):
        if not isinstance(item, dict) or set(item) != {
            "id", "country", "year", "region", "recipe"
        }:
            raise ValueError(f"Cohort #{number} needs id, country, year, region, recipe")
        cohort_id, country, year = item["id"], item["country"], item["year"]
        if not isinstance(cohort_id, str) or not IDENTIFIER.fullmatch(cohort_id):
            raise ValueError(f"Cohort #{number}: invalid id")
        if cohort_id in seen_ids:
            raise ValueError(f"Duplicate cohort id: {cohort_id}")
        seen_ids.add(cohort_id)
        if not isinstance(country, str) or not COUNTRY.fullmatch(country):
            raise ValueError(f"Cohort {cohort_id}: country must be ISO-2 uppercase")
        if type(year) is not int or not 2015 <= year <= 2100:
            raise ValueError(f"Cohort {cohort_id}: invalid year")
        region = item["region"]
        if not isinstance(region, str) or not region.strip():
            raise ValueError(f"Cohort {cohort_id}: region must be nonempty")
        recipe_name = item["recipe"]
        if not isinstance(recipe_name, str) or not recipe_name.strip():
            raise ValueError(f"Cohort {cohort_id}: recipe path must be nonempty")
        recipe_path = (source.parent / recipe_name).resolve()
        report = generation_config_report(
            config_dir, mode="isolated", experiment_recipe=recipe_path
        )
        resolved = load_generation_config(
            config_dir, mode="isolated", experiment_recipe=recipe_path
        )
        calendar = resolved.get("acquisition_calendar_provenance")
        if calendar is not None and (
            calendar["country"] != country or calendar["year"] != year
        ):
            raise ValueError(f"Cohort {cohort_id}: calendar country/year disagrees with plan")
        stac = resolved["catalog"]["stac"]
        start = _datetime(stac.get("start_date"), f"{cohort_id}.start_date")
        end = _datetime(stac.get("end_date"), f"{cohort_id}.end_date")
        if end <= start:
            raise ValueError(f"Cohort {cohort_id}: STAC end_date must follow start_date")
        vector = resolved["eurocrops"].get("shapefile_path")
        if not isinstance(vector, str) or not vector.strip():
            raise ValueError(f"Cohort {cohort_id}: missing EuroCrops shapefile path")
        raw_destination = resolved["paths"].get("dataset_output_dir")
        if not isinstance(raw_destination, str) or not raw_destination.strip():
            raise ValueError(f"Cohort {cohort_id}: missing dataset output directory")
        destination = Path(raw_destination).resolve()
        if any(destination == previous or destination in previous.parents or
               previous in destination.parents for previous in destinations):
            raise ValueError(f"Cohort {cohort_id}: overlapping cohort output directories")
        destinations.append(destination)
        records.append({
            "id": cohort_id, "country": country, "year": year, "region": region,
            "recipe_path": str(recipe_path),
            "recipe_name": report["experiment_recipe"]["name"],
            "recipe_sha256": report["experiment_recipe"]["sha256"],
            "resolved_config_sha256": report["resolved_config_sha256"],
            "vector_path": vector,
            "stac_start": start.isoformat(), "stac_end": end.isoformat(),
            "dataset_output_dir": str(destination),
            "run_tag": cohort_id,
            "acquisition_calendar": calendar,
        })
    return {
        "schema_version": 1,
        "plan_name": plan["name"],
        "mode": "validated_plan_only",
        "cohorts": records,
        "notes": [
            "Country, year and region are declared metadata, not verified geographic provenance.",
            "Each cohort uses an independent isolated recipe and strictly fingerprinted generation.",
            "This planner does not apply class-specific phenology calendars or combine output manifests.",
            "Frozen publication datasets and paper partitions are not changed.",
        ],
    }


def run_cohorts(plan_file: str | Path, *, config_dir: str | Path = "config",
                execute: bool = False) -> dict:
    """Preview a cohort plan, or explicitly execute each independent run.

    Args:
        plan_file: Versioned multi-cohort YAML plan.
        config_dir: Directory containing baseline generator YAML files.
        execute: Whether to start generation; defaults to a read-only preview.

    Returns:
        Validated plan metadata and, if executed, per-cohort run summaries.

    Raises:
        FileExistsError: If an execution destination already exists.
        ValueError: If the plan or a referenced recipe fails preflight.
    """
    report = plan_cohorts(plan_file, config_dir=config_dir)
    if not execute:
        return report
    for cohort in report["cohorts"]:
        if Path(cohort["dataset_output_dir"]).exists():
            raise FileExistsError(
                f"Cohort destination already exists: {cohort['dataset_output_dir']}"
            )
    from sentinel_crop_seg.core.pipeline.dataset_generation import generate_dataset
    results = []
    for cohort in report["cohorts"]:
        generated = generate_dataset(
            config_dir=config_dir, config_mode="isolated",
            resume_policy="strict", experiment_recipe=cohort["recipe_path"],
            run_tag=cohort["run_tag"],
        )
        results.append({
            "id": cohort["id"], "samples_generated": generated.total_samples,
            "selected_rois": generated.selected_rois,
            "output_dir": str(generated.output_dir),
        })
    report["mode"] = "executed"
    report["generation_results"] = results
    return report
