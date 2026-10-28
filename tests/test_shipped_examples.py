"""The shipped examples still satisfy the contracts their tools enforce.

A documented command that cannot be run is not documentation. These tests run
each example through the same validation the CLI uses, from the package root,
so a contract change cannot leave a stale example behind in the archive.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sentinel_crop_seg.core.data.cohort_manifest_merge import merge_cohort_manifests
from sentinel_crop_seg.core.data.patch_log_manifest import adapt_patch_log_file
from sentinel_crop_seg.core.quality.calendar_cohort_audit import audit_calendar_cohorts
from sentinel_crop_seg.core.quality.post_training_filter import review_and_filter
from sentinel_crop_seg.core.quality.weighted_calendar_selection import (
    build_weighted_calendar_selection,
)
from sentinel_crop_seg.core.splitting.spatial_oof_folds import (
    SpatialOOFConfig,
    plan_spatial_oof,
)
from sentinel_crop_seg.core.splitting.spatial_split import SpatialSplitConfig, split_dataframe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CONFIG = ROOT / "config"
CALENDAR = CONFIG / "crop_calendar_recalibrated_v0_3.csv"


def _unchanged(paths: list[Path]):
    return {path: path.read_bytes() for path in paths}


def test_every_example_is_documented_and_present():
    documented = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    shipped = sorted(
        path.relative_to(EXAMPLES).as_posix()
        for path in EXAMPLES.rglob("*.csv")
    )
    assert shipped, "the archive must ship runnable examples"
    for relative in shipped:
        assert relative in documented, f"{relative} is shipped but not documented"
    assert "synthetic" in documented.lower()


def test_examples_declare_themselves_synthetic_and_not_a_dataset():
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "research dataset" in text
    assert "no satellite pixels" in text


def test_patch_log_example_adapts_without_writing(tmp_path):
    source = EXAMPLES / "patch_log" / "patches_log.csv"
    before = _unchanged([source])
    audit = adapt_patch_log_file(
        source, tmp_path / "adapted.csv", dataset_id="DEMO_2021",
        country="AT", cohort="DEMO_2021", validate_only=True,
    )
    assert audit["samples"] == 6
    assert not (tmp_path / "adapted.csv").exists()
    assert _unchanged([source]) == before


def test_patch_log_example_has_exactly_the_v3_columns():
    """Guards the failure that a hand-written CSV hides: comma-bearing fields."""
    with (EXAMPLES / "patch_log" / "patches_log.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.reader(stream))
    assert len({len(row) for row in rows}) == 1, "ragged CSV: a field was not quoted"
    header = rows[0]
    band_order = rows[1][header.index("band_order")]
    assert band_order.count(",") == 9, "band_order must survive as one field"


def test_cohort_merge_example_previews_both_cohorts(tmp_path):
    inputs = sorted((EXAMPLES / "cohort_merge").glob("*_adapted_manifest.csv"))
    assert len(inputs) == 2
    before = _unchanged(inputs)
    preview = merge_cohort_manifests(
        inputs, tmp_path / "merged", verify_assets=False, validate_only=True
    )
    assert preview["status"] == "validated_only"
    assert preview["samples"] == 12
    assert not (tmp_path / "merged").exists()
    assert _unchanged(inputs) == before


def test_spatial_split_example_reports_a_clean_leakage_audit():
    config = SpatialSplitConfig.from_yaml(CONFIG / "examples" / "spatial_split.yaml")
    manifest = pd.read_csv(EXAMPLES / "spatial_split_demo_manifest.csv")
    _, summary, audit, _ = split_dataframe(manifest, config)
    assert summary["leakage_audit_clean"] is True
    # The demo rows repeat one footprint across dates; a clean audit means the
    # block split kept every date of a window in the same partition.
    assert audit["clean"] is True
    assert all(pair["violations"] == 0 for pair in audit["pairs"].values())


def test_oof_example_excludes_uncertifiable_rows_and_reserves_test():
    config = yaml.safe_load(
        (CONFIG / "examples" / "spatial_oof_folds.yaml").read_text(encoding="utf-8")
    )
    manifest = pd.read_csv(EXAMPLES / "oof_folds" / "split_manifest.csv")
    _, audit = plan_spatial_oof(manifest, SpatialOOFConfig(**config["oof_folds"]))
    assert audit["eligible_development_rows"] == 36
    assert audit["eligible_windows"] == 18
    # The two rows that cannot be certified must be reported, not planned.
    assert audit["excluded_development_rows"] == 2
    assert audit["test_rows_reserved_untouched"] == 4
    assert audit["all_eligible_samples_held_out_exactly_once"] is True
    assert audit["publication_ready"] is False


def test_calendar_audit_example_has_complete_country_coverage(tmp_path):
    report = audit_calendar_cohorts(
        EXAMPLES / "calendar_audit" / "cohort_split_manifest.csv",
        CALENDAR, output_dir=tmp_path / "audit",
    )
    assert report["manifest_rows"] == 8
    assert report["countries"] == ["AT", "FR"]
    assert report["missing_calendar_pairs"] == []
    assert report["missing_acquisition_dates"] == 0


def test_weighted_calendar_example_actually_rejects_something(tmp_path):
    """An example that retains 100% would demonstrate nothing."""
    source = EXAMPLES / "calendar_audit" / "cohort_split_manifest.csv"
    before = _unchanged([source])
    report = build_weighted_calendar_selection(
        source, CALENDAR, CONFIG / "classes.yaml",
        CONFIG / "calendars" / "class_calendar_binding_v1.yaml",
        tmp_path / "weighted", validate_only=True,
    )
    assert report["train_input"] == 4
    assert report["train_selected"] == 2
    assert report["val_preserved"] == 2
    assert report["test_preserved"] == 2
    assert _unchanged([source]) == before


def test_label_review_example_queues_only_high_priority_parcels(tmp_path):
    base = EXAMPLES / "label_review"
    sources = [
        base / "frozen_training_manifest.csv",
        base / "parcel_review_levels.csv",
        base / "parcel_admissibility.csv",
    ]
    before = _unchanged(sources)
    summary = review_and_filter(*sources, tmp_path / "queue")
    assert summary["review_parcels"] == 6
    assert summary["high_priority_parcels"] == 2
    assert summary["mapped_high_priority_patch_links"] == 3
    assert summary["applied_approved_exclusions"] is False
    assert summary["automatic_relabeling"] is False
    queue = pd.read_csv(tmp_path / "queue" / "human_review_queue.csv")
    assert set(queue["review_level"]) == {"HIGH_PRIORITY_REVIEW_CANDIDATE"}
    assert "test" not in set(queue["split"])
    assert _unchanged(sources) == before


def test_label_review_example_applies_only_approved_exclusions(tmp_path):
    base = EXAMPLES / "label_review"
    sources = [
        base / "frozen_training_manifest.csv",
        base / "parcel_review_levels.csv",
        base / "parcel_admissibility.csv",
    ]
    before = _unchanged(sources)
    summary = review_and_filter(
        *sources, tmp_path / "applied",
        decisions_file=base / "human_decisions.csv",
        apply_approved_exclusions=True,
    )
    assert summary["manifest_rows"] == 10
    assert summary["approved_patch_rows_to_remove"] == 2
    assert summary["rows_after_filter"] == 8
    assert summary["test_rows_unchanged"] is True
    filtered = pd.read_csv(tmp_path / "applied" / "filtered_training_manifest.csv")
    assert set(filtered["split"]) >= {"train", "val", "test"}
    assert _unchanged(sources) == before


@pytest.mark.parametrize("decision", ["relabel", "delete", ""])
def test_only_keep_and_exclude_patch_decisions_are_accepted(tmp_path, decision):
    base = EXAMPLES / "label_review"
    bad = tmp_path / "decisions.csv"
    pd.DataFrame([
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=1002", decision=decision)
    ]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        review_and_filter(
            base / "frozen_training_manifest.csv",
            base / "parcel_review_levels.csv",
            base / "parcel_admissibility.csv",
            tmp_path / "rejected",
            decisions_file=bad,
            apply_approved_exclusions=True,
        )
