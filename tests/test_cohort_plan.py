"""Synthetic no-download tests for independent multi-cohort generation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sentinel_crop_seg.core.pipeline.cohort_plan import plan_cohorts, run_cohorts


def workspace(tmp_path: Path):
    cohorts = []
    for country, year in (("AT", 2021), ("FR", 2018)):
        key = f"{country}_{year}"
        recipe_file = tmp_path / f"{key}.yaml"
        recipe_file.write_text(yaml.safe_dump({
            "schema_version": 1, "name": f"{key.lower()}_demo",
            "settings": {
                "eurocrops": {"shapefile_path": f"missing/{key}.shp"},
                "catalog": {"stac": {
                    "start_date": f"{year}-04-01T00:00:00Z",
                    "end_date": f"{year}-09-30T23:59:59Z"}},
                "paths": {"dataset_output_dir": str(tmp_path / "generated" / key)},
            },
        }), encoding="utf-8")
        cohorts.append({
            "id": key, "country": country, "year": year, "region": "national",
            "recipe": recipe_file.name,
        })
    file = tmp_path / "multi.yaml"
    file.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "test_multicohort", "cohorts": cohorts
    }), encoding="utf-8")
    return file, cohorts


def test_plan_is_deterministic_read_only_and_carries_hashes(tmp_path):
    file, _ = workspace(tmp_path)
    before = file.read_bytes()
    first = plan_cohorts(file)
    second = run_cohorts(file)
    assert first == second
    assert first["mode"] == "validated_plan_only"
    assert len(first["cohorts"]) == 2
    assert [v["run_tag"] for v in first["cohorts"]] == ["AT_2021", "FR_2018"]
    assert len({v["dataset_output_dir"] for v in first["cohorts"]}) == 2
    assert all(len(v["resolved_config_sha256"]) == 64 for v in first["cohorts"])
    assert all(len(v["recipe_sha256"]) == 64 for v in first["cohorts"])
    assert file.read_bytes() == before
    assert not (tmp_path / "generated").exists()


def test_duplicate_ids_and_output_collision_fail_before_writes(tmp_path):
    file, cohorts = workspace(tmp_path)
    cohorts[1]["id"] = cohorts[0]["id"]
    file.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "demo", "cohorts": cohorts
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate cohort id"):
        plan_cohorts(file)
    cohorts[1]["id"] = "FR_2018"
    fr_path = tmp_path / cohorts[1]["recipe"]
    fr = yaml.safe_load(fr_path.read_text())
    fr["settings"]["paths"]["dataset_output_dir"] = str(tmp_path / "generated" / "AT_2021")
    fr_path.write_text(yaml.safe_dump(fr))
    file.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "demo", "cohorts": cohorts
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping cohort output"):
        plan_cohorts(file)
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("field,value,error", [
    ("country", "Austria", "ISO-2"),
    ("year", "2021", "invalid year"),
    ("year", True, "invalid year"),
    ("region", "", "region must be nonempty"),
    ("recipe", "", "recipe path must be nonempty"),
    ("id", "not allowed space", "invalid id"),
])
def test_invalid_cohort_metadata_rejected(tmp_path, field, value, error):
    file, cohorts = workspace(tmp_path)
    cohorts[0][field] = value
    file.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "demo", "cohorts": cohorts
    }), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        plan_cohorts(file)


def test_invalid_stac_range_rejected(tmp_path):
    file, _ = workspace(tmp_path)
    path = tmp_path / "AT_2021.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["settings"]["catalog"]["stac"]["start_date"] = "2021-12-01T00:00:00Z"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="end_date must follow"):
        plan_cohorts(file)


def test_execute_is_explicit_and_uses_strict_isolated_generation(tmp_path, monkeypatch):
    file, _ = workspace(tmp_path)
    calls = []
    def fake_generation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total_samples=3, selected_rois=1, output_dir="result")
    monkeypatch.setattr(
        "sentinel_crop_seg.core.pipeline.dataset_generation.generate_dataset", fake_generation
    )
    report = run_cohorts(file, execute=True)
    assert report["mode"] == "executed"
    assert len(calls) == len(report["generation_results"]) == 2
    assert all(c["config_mode"] == "isolated" and c["resume_policy"] == "strict"
               for c in calls)
    assert [c["run_tag"] for c in calls] == ["AT_2021", "FR_2018"]
    assert all(c["experiment_recipe"].endswith(f"{code}.yaml") for c, code
               in zip(calls, ["AT_2021", "FR_2018"]))
    assert not (tmp_path / "generated").exists()


def test_existing_cohort_destination_blocks_every_execution(tmp_path, monkeypatch):
    file, _ = workspace(tmp_path)
    (tmp_path / "generated" / "FR_2018").mkdir(parents=True)
    called = []
    monkeypatch.setattr(
        "sentinel_crop_seg.core.pipeline.dataset_generation.generate_dataset",
        lambda **kwargs: called.append(kwargs),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        run_cohorts(file, execute=True)
    assert called == []
