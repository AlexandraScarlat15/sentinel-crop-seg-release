"""Post-training label-review filtering with explicit human-approved exclusions.

Consumes the existing six-fold out-of-fold parcel review levels and the
parcel-admissibility join, never recalculates confidence or calls a model.
No candidate is automatically treated as an incorrect label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sentinel_crop_seg.core.splitting.spatial_split import rebase_asset_paths

HIGH_PRIORITY = "HIGH_PRIORITY_REVIEW_CANDIDATE"
LEVELS = frozenset({
    "NO_FLAG", "INSUFFICIENT_TEMPORAL_DEPTH", "TEMPORAL_EXPLANATION",
    "TEMPORALLY_UNSTABLE", "CONSISTENT_ALTERNATIVE_CLASS", HIGH_PRIORITY,
})
REQUIRED_MANIFEST = {"dataset", "sample_id", "split", "cohort"}
REQUIRED_REVIEW = {"cohort", "parcel_key", "review_level", "requires_human_review", "partition"}
REQUIRED_LINK = {"cohort", "parcel_key", "dataset", "sample_id", "partition", "admissible_min_pixels"}


def _read(path: str | Path, required: set[str], *, allow_empty: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if frame.empty and not allow_empty:
        raise ValueError(f"{path} is empty")
    for field in required:
        if frame[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path} has blank {field}")
    return frame


def _unique(frame: pd.DataFrame, keys: list[str], *, label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {label} identity: {keys}")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    normal = frame[column].astype(str).str.strip().str.lower()
    if not normal.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Invalid Boolean values in {column}")
    return normal.isin(["true", "1"])


def _sha(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as src:
        for piece in iter(lambda: src.read(1024 * 1024), b""):
            hasher.update(piece)
    return hasher.hexdigest()


def review_and_filter(
    manifest_file: str | Path,
    review_file: str | Path,
    parcel_links_file: str | Path,
    output_dir: str | Path,
    *,
    decisions_file: str | Path | None = None,
    apply_approved_exclusions: bool = False,
) -> dict:
    """Create an OOF human-review queue and optionally exclude approved TRAIN patches.

    This operation removes whole patch rows, never relabels pixels, and
    preserves original validation and test membership.

    Args:
        manifest_file: Original partitioned dataset manifest.
        review_file: Precomputed parcel-level OOF review levels.
        parcel_links_file: Admissible parcel-to-patch identity associations.
        output_dir: Fresh output directory for audit reports and review queue.
        decisions_file: Optional CSV of human `keep` or `exclude_patch` decisions.
        apply_approved_exclusions: Whether to write a derived TRAIN-filtered
            manifest; defaults to review-only mode.

    Returns:
        A summary of review candidates, approved exclusions, source hashes
        and output membership counts.

    Raises:
        ValueError: If identities, partition roles or approvals disagree.
        FileExistsError: If the output directory already exists.
    """
    dest = Path(output_dir)
    if dest.exists():
        raise FileExistsError(f"Output directory already exists: {dest}")
    if apply_approved_exclusions and decisions_file is None:
        raise ValueError("Applying exclusions requires human decisions CSV")

    manifest = _read(manifest_file, REQUIRED_MANIFEST)
    review = _read(review_file, REQUIRED_REVIEW)
    links = _read(parcel_links_file, REQUIRED_LINK)
    _unique(manifest, ["dataset", "sample_id"], label="manifest sample")
    _unique(review, ["cohort", "parcel_key"], label="OOF parcel review")

    if "partition" in manifest.columns and not manifest["partition"].equals(manifest["split"]):
        raise ValueError("Manifest partition and split disagree; cannot safely filter")
    if not manifest["split"].isin(["train", "val", "test", "excluded"]).all():
        raise ValueError("Manifest contains unknown split labels")
    if not review["partition"].isin(["train", "val"]).all():
        raise ValueError("Review file must contain only original train/val OOF evidence")
    if not links["partition"].isin(["train", "val"]).all():
        raise ValueError("Parcel links must come only from development partitions")
    if not review["review_level"].isin(LEVELS).all():
        raise ValueError("Unrecognized parcel review level")
    flagged = _bool(review, "requires_human_review")
    if not flagged.equals(review["review_level"].eq(HIGH_PRIORITY)):
        raise ValueError("OOF review levels disagree with requires_human_review")
    admissible = _bool(links, "admissible_min_pixels")
    if not admissible.all():
        links = links.loc[admissible].copy()
    # Existing join can contain repeated views; deduplicate exact associations.
    keys = ["cohort", "parcel_key", "dataset", "sample_id"]
    links = links.drop_duplicates(keys)
    joined = links.merge(
        manifest[["dataset", "sample_id", "cohort", "split"]],
        on=["dataset", "sample_id"], how="left", validate="many_to_one",
        suffixes=("_link", "_manifest"), indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("Parcel link references sample absent from the manifest")
    # Drop pandas' temporary merge indicator before the later review-decision
    # join, which needs its own independent unmatched-identity check.
    joined = joined.drop(columns="_merge")
    if not joined["cohort_link"].eq(joined["cohort_manifest"]).all():
        raise ValueError("Parcel cohort does not match manifest cohort")
    if not joined["partition"].eq(joined["split"]).all():
        raise ValueError("Parcel link partition does not match manifest split")
    # Review can contain other cohorts or different admissibility subsets;
    # never infer links for a parcel without explicit provenance.
    pairs = ["cohort", "parcel_key"]
    observations = joined.rename(columns={"cohort_link": "cohort"}).merge(
        review[pairs + ["partition", "review_level", "requires_human_review"]].rename(
            columns={"partition": "review_partition"}
        ),
        on=pairs, how="inner", validate="many_to_one",
    )
    if not observations["review_partition"].eq(observations["partition"]).all():
        raise ValueError("OOF review partition does not match admissible parcel-link partition")
    queue = observations.loc[observations["review_level"].eq(HIGH_PRIORITY)].copy()
    queue = queue[["cohort", "parcel_key", "dataset", "sample_id", "split", "review_level"]]
    queue = queue.sort_values(["cohort", "parcel_key", "dataset", "sample_id"]).reset_index(drop=True)

    removed = pd.DataFrame(columns=["dataset", "sample_id", "cohort", "parcel_key"])
    decisions = None
    if decisions_file is not None:
        decisions = _read(decisions_file, {"cohort", "parcel_key", "decision"})
        _unique(decisions, pairs, label="review decision")
        if not decisions["decision"].isin(["keep", "exclude_patch"]).all():
            raise ValueError("decision must be keep or exclude_patch")
        unknown = decisions.merge(review[pairs], on=pairs, how="left", indicator=True)
        if unknown["_merge"].ne("both").any():
            raise ValueError("Decision references parcel not in OOF review")
        exclude = decisions.loc[decisions["decision"].eq("exclude_patch"), pairs]
        if not exclude.empty:
            approved = exclude.merge(review[pairs + ["review_level"]],
                                     on=pairs, validate="one_to_one")
            if not approved["review_level"].eq(HIGH_PRIORITY).all():
                raise ValueError("Cannot exclude parcels without a high-priority OOF review")
            mapped = exclude.merge(joined.rename(columns={"cohort_link": "cohort"}),
                                   on=pairs, how="left", indicator=True)
            if mapped["_merge"].ne("both").any():
                raise ValueError("Approved parcel has no admissible patch links")
            if not mapped["split"].eq("train").all():
                raise ValueError("Approved exclusion intersects validation/test; only train is filterable")
            removed = mapped[["dataset", "sample_id", "cohort", "parcel_key"]].drop_duplicates()
    if apply_approved_exclusions:
        approved_samples = removed[["dataset", "sample_id"]].drop_duplicates()
        mark = manifest.merge(approved_samples.assign(_remove=True),
                              on=["dataset", "sample_id"], how="left", validate="one_to_one")
        filtered = mark.loc[mark["_remove"].ne(True)].drop(columns="_remove")
        if not manifest.loc[manifest["split"].ne("train"), ["dataset", "sample_id"]].reset_index(drop=True).equals(
            filtered.loc[filtered["split"].ne("train"), ["dataset", "sample_id"]].reset_index(drop=True)
        ):
            raise RuntimeError("Refusing any change to held-out or validation partition")
        # The new manifest lives in a NEW directory. Convert relative image
        # and mask paths from the original manifest's base to the output's
        # base, without copying or modifying any patch file.
        filtered = rebase_asset_paths(filtered, Path(manifest_file), dest)
    else:
        filtered = None

    report = {
        "protocol": "six-fold OOF parcel review; post-training approved patch exclusions",
        "automatic_relabeling": False,
        "automatic_exclusion": False,
        "applied_approved_exclusions": bool(apply_approved_exclusions),
        "source_sha256": {
            "manifest": _sha(manifest_file), "parcel_review": _sha(review_file),
            "parcel_links": _sha(parcel_links_file),
            "decisions": _sha(decisions_file) if decisions_file else None,
        },
        "manifest_rows": len(manifest),
        "review_parcels": len(review),
        "high_priority_parcels": int(flagged.sum()),
        "mapped_high_priority_patch_links": len(queue),
        "approved_patch_rows_to_remove": int(len(removed[["dataset", "sample_id"]].drop_duplicates())),
        "rows_after_filter": len(filtered) if filtered is not None else None,
        "test_rows_unchanged": True,
        "warning": "Excluding a whole patch may also remove correctly labelled parcels; no pixels are relabeled.",
    }
    dest.mkdir(parents=True, exist_ok=False)
    queue.to_csv(dest / "human_review_queue.csv", index=False)
    removed.sort_values(["dataset", "sample_id", "parcel_key"]).to_csv(
        dest / "approved_patch_exclusions.csv", index=False
    )
    if filtered is not None:
        filtered.to_csv(dest / "filtered_training_manifest.csv", index=False)
    (dest / "review_filter_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
