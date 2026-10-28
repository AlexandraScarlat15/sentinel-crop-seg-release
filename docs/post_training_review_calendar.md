# Post-training label uncertainty: reviewed filter and cohort/calendar audit

The supervisors requested a **post-training filter based on label uncertainty**.
The internal research protocol defines a **six-fold, one-repeat geographically held-out**
parcel-review procedure. Its experimental implementation and original
fold-prediction artifacts are **not included** in this curated package. It is **not** a ground-truth correction algorithm:
model disagreement is a **human review candidate**, not proof of an incorrect
EuroCrops label. The test partition must never participate in this selection.

## Inputs from the EXISTING research workflow

The new curated utility **does not retrain the six fold models**, reconstruct
parcels or reimplement their thresholds. It consumes their precomputed outputs:

1. `parcel_review_levels.csv` from the project OOF aggregation stage (must have
   `cohort,parcel_key,review_level,requires_human_review,partition`).
2. `parcel_admissibility.csv` from the existing parcel join stage (must have
   `cohort,parcel_key,dataset,sample_id,partition,admissible_min_pixels`).
   This connects parcel IDs to **specific** sample IDs without guessing.
3. The frozen manifest (`dataset,sample_id,cohort,split`; if it has
   `partition` too, both columns must agree).

Only original `train` and `val` OOF evidence is accepted. Duplicate parcel
identities in review output are rejected, not silently merged. For every
*supplied* admissible parcel-to-patch link, the sample must exist in the
manifest and the country/cohort identity and original partition must agree.
For every linked reviewed parcel, the OOF review partition must also match
the parcel-link partition. These checks prevent contradictory joins; they
**cannot prove the input OOF files are complete or were produced by the
original six-fold protocol**. That requires a separately approved provenance
record. Reconstructing the historical fold predictions is outside this
software release.

## File contract and provenance boundary

| Input | Mandatory columns | Interpretation |
|---|---|---|
| Manifest | `dataset,sample_id,split,cohort` | Original sample identity; optional `partition` must equal `split` |
| Parcel review | `cohort,parcel_key,review_level,requires_human_review,partition` | Precomputed OOF review level, **not** automatically verified ground truth |
| Parcel links | `cohort,parcel_key,dataset,sample_id,partition,admissible_min_pixels` | Existing admissible parcel/sample associations |
| Human decisions (optional) | `cohort,parcel_key,decision` | Only `keep` and `exclude_patch`; approvals must be separately documented |

The review code does not train fold models, calculate per-pixel uncertainty,
load Sentinel-2 images, label individual parcels or generate human decisions.
It validates the *provided* CSV contracts and protects original TRAIN/VAL/TEST
membership. The input hashes in `review_filter_audit.json` identify the
exact files supplied, but do not establish who approved them.

## Step 1 — Produce a review queue, WITHOUT filtering

```bash
sentinel-crop-filter-post-training \
  --manifest path/to/frozen_training_manifest.csv \
  --parcel-review path/to/parcel_review_levels.csv \
  --parcel-links path/to/parcel_admissibility.csv \
  --output-dir outputs/label_review_queue
```

Writes `human_review_queue.csv`, `approved_patch_exclusions.csv` (empty
until decisions are supplied) and `review_filter_audit.json`. The latter
records SHA-256 hashes for every input and the number of review candidates.
Nothing touches the source files or the `test` partition.

## Step 2 — After human adjudication, optionally exclude TRAIN patches

An explicitly reviewed decision CSV has these columns:

```csv
cohort,parcel_key,decision
AT_2021,AT_2021|fid=42,exclude_patch
```

Only decisions `keep` or `exclude_patch` are permitted. The latter requires
a `HIGH_PRIORITY_REVIEW_CANDIDATE` level, a valid admissible parcel-to-sample
link and **only train samples**. Test and validation samples cannot be
excluded, even when the same parcel touches another partition.

First inspect the audit without applying:

```bash
sentinel-crop-filter-post-training \
  --manifest path/to/frozen_training_manifest.csv \
  --parcel-review path/to/parcel_review_levels.csv \
  --parcel-links path/to/parcel_admissibility.csv \
  --decisions path/to/human_decisions.csv \
  --output-dir outputs/reviewed_dryrun
```

Then, only when approved for this experiment, choose a **new** output dir:

```bash
sentinel-crop-filter-post-training \
  --manifest path/to/frozen_training_manifest.csv \
  --parcel-review path/to/parcel_review_levels.csv \
  --parcel-links path/to/parcel_admissibility.csv \
  --decisions path/to/human_decisions.csv \
  --apply-approved-exclusions \
  --output-dir outputs/reviewed_train_manifest
```

`filtered_training_manifest.csv` contains original manifest columns with
approved **whole train patch** rows excluded; `approved_patch_exclusions.csv`
records the parcel reasons. Relative `img_path`/`mask_path` values are
rebased to the **new manifest directory**, so the existing DataLoader still
resolves the original NPY/GeoTIFF files without copying or editing them.
This is deliberately **not pixel-level label repair**. Dropping an entire
patch can remove correctly annotated neighbouring parcels too, so review
the proposed exclusions before using a derived manifest in any downstream
work. The model's original labels/weights never change. **For the present
Zenodo packaging work, no retraining or new filtered dataset production
is planned.**

**Not yet proven for the article:** final trained six-fold checkpoints,
approved human decisions, and the generated filtering artifacts have not been
bundled or validated as manuscript artifacts. This script gives a safe,
testable application mechanism only. It is not evidence that filtering has
already happened.

## Calendar and targeted country/year coverage

`config/crop_calendar_recalibrated_v0_3.csv` currently includes
**AT, CZ, DE, FR and SK**, each with 13 crop/forage calendar entries.
This is calendar **table coverage**, not proof that every cohort's acquisition
dates were filtered by this calendar or that these rules reproduce the
article's final selection policy. A manifest-driven audit is still required.

With a real manifest, run the read-only coverage check:

```bash
sentinel-crop-audit-cohorts \
  --manifest path/to/frozen_training_manifest.csv \
  --calendar config/crop_calendar_recalibrated_v0_3.csv \
  --output-dir outputs/cohort_calendar_audit
```

It writes `manifest_cohort_partition_counts.csv`, `missing_calendar_pairs.csv`
and `cohort_calendar_audit.json`, including cohort/year and per-partition
counts and hashes of both inputs. If an experiment **requires complete
calendar coverage**, use `--require-full-calendar` to reject missing pairs.

The **cohort audit tool** does not apply acquisition filters, reconstruct crop
phenology, or invent country/crop dates. An optional, separate
[scene-level acquisition-calendar policy](acquisition_calendar.md) is now
available for *new* generation experiments; it is not evidence that the
article's historical data followed that policy.
Per-country vector selection and STAC acquisition ranges must be frozen in
the actual experiment configuration before a publication claim is made.

## Reproducibility boundaries

- The original six-fold label-audit computations stay on the development
  branch; the curated software release exposes only the non-destructive
  post-training application and dataset/calendar audit tools.
- The original article train/val/test membership, review flags and masks
  remain immutable. New filtered manifests are separately named experiments.
- Frozen test data must never guide uncertainty thresholds or selection.
- Neither tool replaces a full checkpoint, a complete OOF artifact inventory,
  human-reviewed decisions, or an exact historical data split.
