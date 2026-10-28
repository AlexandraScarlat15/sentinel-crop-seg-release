"""Offline post-training uncertainty review tests with no model or historic data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentinel_crop_seg.core.quality.post_training_filter import review_and_filter


def make_fixture(tmp_path: Path):
    paths = {n: tmp_path / f"{n}.csv" for n in ("manifest", "review", "links", "decisions")}
    pd.DataFrame([
        dict(dataset="AT_2021", sample_id="a", cohort="AT_2021", partition="train",
             split="train", img_path="a.npy", mask_path="a_mask.npy"),
        dict(dataset="AT_2021", sample_id="b", cohort="AT_2021", partition="train",
             split="train", img_path="b.npy", mask_path="b_mask.npy"),
        dict(dataset="AT_2021", sample_id="v", cohort="AT_2021", partition="val",
             split="val", img_path="v.npy", mask_path="v_mask.npy"),
        dict(dataset="AT_2021", sample_id="t", cohort="AT_2021", partition="test",
             split="test", img_path="t.npy", mask_path="t_mask.npy"),
    ]).to_csv(paths["manifest"], index=False)
    pd.DataFrame([
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=42",
             partition="train", review_level="HIGH_PRIORITY_REVIEW_CANDIDATE",
             requires_human_review="true"),
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=43",
             partition="val", review_level="TEMPORAL_EXPLANATION",
             requires_human_review="false"),
    ]).to_csv(paths["review"], index=False)
    pd.DataFrame([
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=42", dataset="AT_2021",
             sample_id="a", partition="train", admissible_min_pixels="true"),
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=42", dataset="AT_2021",
             sample_id="b", partition="train", admissible_min_pixels="true"),
        dict(cohort="AT_2021", parcel_key="AT_2021|fid=43", dataset="AT_2021",
             sample_id="v", partition="val", admissible_min_pixels="true"),
    ]).to_csv(paths["links"], index=False)
    pd.DataFrame([dict(cohort="AT_2021", parcel_key="AT_2021|fid=42",
                       decision="exclude_patch")]).to_csv(paths["decisions"], index=False)
    return paths


def run(paths, tmp_path, **overrides):
    return review_and_filter(
        paths["manifest"], paths["review"], paths["links"],
        tmp_path / "output", **overrides,
    )


def test_default_produces_only_review_queue_and_audit(tmp_path):
    paths = make_fixture(tmp_path)
    before = paths["manifest"].read_bytes()
    summary = run(paths, tmp_path)
    out = tmp_path / "output"
    queue = pd.read_csv(out / "human_review_queue.csv")
    assert len(queue) == 2
    assert set(queue["sample_id"]) == {"a", "b"}
    assert not (out / "filtered_training_manifest.csv").exists()
    assert summary["automatic_relabeling"] is False
    assert summary["applied_approved_exclusions"] is False
    assert paths["manifest"].read_bytes() == before


def test_only_manual_approvals_exclude_whole_train_patches(tmp_path):
    paths = make_fixture(tmp_path)
    original_bytes = paths["manifest"].read_bytes()
    s = run(paths, tmp_path, decisions_file=paths["decisions"],
            apply_approved_exclusions=True)
    retained = pd.read_csv(tmp_path / "output" / "filtered_training_manifest.csv")
    assert retained["sample_id"].tolist() == ["v", "t"]
    assert retained["split"].tolist() == ["val", "test"]
    # The filtered manifest is in another directory; original relative
    # image/mask paths must continue to resolve to the same source assets.
    output = tmp_path / "output"
    for entry in retained.itertuples(index=False):
        for column in ("img_path", "mask_path"):
            assert (output / getattr(entry, column)).resolve() == (
                tmp_path / (
                    f"{entry.sample_id}.npy"
                    if column == "img_path" else f"{entry.sample_id}_mask.npy"
                )
            ).resolve()
    removed = pd.read_csv(tmp_path / "output" / "approved_patch_exclusions.csv")
    assert set(removed["sample_id"]) == {"a", "b"}
    assert s["approved_patch_rows_to_remove"] == 2
    assert paths["manifest"].read_bytes() == original_bytes
    audit = json.loads((tmp_path / "output" / "review_filter_audit.json").read_text())
    assert len(audit["source_sha256"]["manifest"]) == 64
    assert audit["test_rows_unchanged"]


def test_reviewed_decisions_only_dry_run_does_not_apply(tmp_path):
    paths = make_fixture(tmp_path)
    summary = run(paths, tmp_path, decisions_file=paths["decisions"])
    assert summary["approved_patch_rows_to_remove"] == 2
    assert summary["rows_after_filter"] is None
    assert not (tmp_path / "output" / "filtered_training_manifest.csv").exists()


@pytest.mark.parametrize(("tamper", "expected_error"), [
    ("test_review", "Review file must contain only original train/val OOF evidence"),
    ("test_links", "Parcel links must come only from development partitions"),
    ("mismatched_split", "Manifest partition and split disagree"),
])
def test_never_accepts_test_leakage_or_split_mismatch(tmp_path, tamper, expected_error):
    paths = make_fixture(tmp_path)
    if tamper == "test_review":
        data = pd.read_csv(paths["review"])
        data.loc[0, "partition"] = "test"
        data.to_csv(paths["review"], index=False)
    elif tamper == "test_links":
        data = pd.read_csv(paths["links"])
        data.loc[0, "partition"] = "test"
        data.to_csv(paths["links"], index=False)
    else:
        data = pd.read_csv(paths["manifest"])
        data.loc[0, "partition"] = "val"
        data.to_csv(paths["manifest"], index=False)
    with pytest.raises(ValueError, match=expected_error):
        run(paths, tmp_path)
    assert not (tmp_path / "output").exists()


def test_rejects_unapproved_exclusion_decision(tmp_path):
    paths = make_fixture(tmp_path)
    bad = pd.read_csv(paths["decisions"])
    bad.loc[0, "parcel_key"] = "AT_2021|fid=43"
    bad.to_csv(paths["decisions"], index=False)
    with pytest.raises(ValueError, match="high-priority"):
        run(paths, tmp_path, decisions_file=paths["decisions"],
            apply_approved_exclusions=True)
    assert not (tmp_path / "output").exists()


def test_rejects_val_links_for_approved_exclusion(tmp_path):
    paths = make_fixture(tmp_path)
    links = pd.read_csv(paths["links"])
    links.loc[len(links)] = {
        "cohort": "AT_2021", "parcel_key": "AT_2021|fid=42",
        "dataset": "AT_2021", "sample_id": "v", "partition": "val",
        "admissible_min_pixels": "true",
    }
    links.to_csv(paths["links"], index=False)
    with pytest.raises(ValueError, match="OOF review partition"):
        run(paths, tmp_path, decisions_file=paths["decisions"],
            apply_approved_exclusions=True)


@pytest.mark.parametrize("tamper", ["duplicate_review", "missing_link", "false_flag"])
def test_ambiguous_oof_provenance_fails_closed(tmp_path, tamper):
    paths = make_fixture(tmp_path)
    if tamper == "duplicate_review":
        df = pd.read_csv(paths["review"])
        pd.concat([df, df.iloc[[0]]]).to_csv(paths["review"], index=False)
    elif tamper == "missing_link":
        df = pd.read_csv(paths["links"])
        df.loc[0, "sample_id"] = "not-in-manifest"
        df.to_csv(paths["links"], index=False)
    else:
        df = pd.read_csv(paths["review"])
        df.loc[0, "requires_human_review"] = False
        df.to_csv(paths["review"], index=False)
    with pytest.raises(ValueError, match="Duplicate|absent|disagree"):
        run(paths, tmp_path)


def test_output_never_overwrites_and_missing_approvals_refused(tmp_path):
    paths = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires human decisions"):
        run(paths, tmp_path, apply_approved_exclusions=True)
    (tmp_path / "output").mkdir()
    with pytest.raises(FileExistsError):
        run(paths, tmp_path)


def test_all_dates_of_same_parcel_are_reviewed_under_same_identity(tmp_path):
    paths = make_fixture(tmp_path)
    df = pd.read_csv(paths["links"])
    df.loc[len(df)] = df.iloc[0].to_dict()
    df.to_csv(paths["links"], index=False)
    report = run(paths, tmp_path)
    # Repeated parcel/window views do not double count a patch identity.
    assert report["mapped_high_priority_patch_links"] == 2


def test_review_and_parcel_link_partitions_must_match_before_queue(tmp_path):
    paths = make_fixture(tmp_path)
    review = pd.read_csv(paths["review"])
    review.loc[0, "partition"] = "val"
    review.to_csv(paths["review"], index=False)
    with pytest.raises(ValueError, match="OOF review partition"):
        run(paths, tmp_path)
    assert not (tmp_path / "output").exists()


def test_review_partition_mismatch_refuses_dry_run_with_decisions(tmp_path):
    paths = make_fixture(tmp_path)
    review = pd.read_csv(paths["review"])
    review.loc[0, "partition"] = "val"
    review.to_csv(paths["review"], index=False)
    with pytest.raises(ValueError, match="OOF review partition"):
        run(paths, tmp_path, decisions_file=paths["decisions"])
    assert not (tmp_path / "output").exists()
