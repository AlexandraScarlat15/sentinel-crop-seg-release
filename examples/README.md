# Runnable examples

Every documented CSV contract in this package has a synthetic example here, so
each command below can be executed from the package root **without any
Sentinel-2 product, EuroCrops vector, dataset or checkpoint**.

**These files are synthetic.** They contain no satellite pixels, no real parcel
geometry and no real parcel identifiers. They exist to demonstrate column
contracts and to let you verify that your installation works. They are **not**
a research dataset, and no number produced from them is a scientific result.

Every command below is non-destructive: each refuses to overwrite an existing
output directory, and none modifies its inputs.

| Example | Demonstrates |
|---|---|
| `patch_log/patches_log.csv` | the v3 generator log, before manifest adaptation |
| `cohort_merge/*_adapted_manifest.csv` | two adapted cohorts, ready for cross-country aggregation |
| `spatial_split_demo_manifest.csv` | a georeferenced manifest for block splitting |
| `oof_folds/split_manifest.csv` | an already-split manifest with certified provenance, for OOF planning |
| `calendar_audit/cohort_split_manifest.csv` | per-class histograms and acquisition dates, for both calendar tools |
| `label_review/*.csv` | the four files of the post-training label-uncertainty contract |

## 1. Adapt a generator log into a manifest

```bash
sentinel-crop-adapt-patch-log \
  --patch-log examples/patch_log/patches_log.csv \
  --output outputs/example_adapted.csv \
  --dataset-id DEMO_2021 --country AT --cohort DEMO_2021 \
  --validate-only
```

Reports the derived columns and preserves every original one. Drop
`--validate-only` to write the CSV. `--country` and `--cohort` are explicit
because they are **declarations**, never inferred from the data.

## 2. Aggregate two cohorts before splitting

```bash
sentinel-crop-merge-cohorts \
  --manifest examples/cohort_merge/AT_2021_adapted_manifest.csv \
  --manifest examples/cohort_merge/FR_2018_adapted_manifest.csv \
  --output-dir outputs/example_merged
```

Read-only by default; add `--apply` to write `combined_manifest.csv`. The
preview validates dataset and cohort identity, band order, label sources and
class histograms, and reports a SHA-256 per input. Asset files are only
checked with `--verify-assets`, which these path-only examples do not satisfy.

## 3. Create a geographically blocked split

```bash
sentinel-crop-split-spatial \
  --config config/examples/spatial_split.yaml \
  --manifest examples/spatial_split_demo_manifest.csv \
  --out-dir outputs/example_split --validate-only
```

The demo rows repeat the same footprint at several dates, which is exactly the
leakage a block split must absorb: `leakage_audit_clean` reports whether it
did.

## 4. Plan out-of-fold folds

```bash
sentinel-crop-split-oof \
  --config config/examples/spatial_oof_folds.yaml \
  --manifest examples/oof_folds/split_manifest.csv \
  --out-dir outputs/example_oof --validate-only
```

The example has 36 eligible development rows over 18 window groups, 4 reserved
TEST rows and **2 rows that are deliberately not certifiable** — one missing a
`window_id`, one with an uncertified spatial tier. Both appear in the
exclusion report rather than being planned into a fold, which is the point:
the planner refuses to invent provenance. The audit also reports
`all_eligible_samples_held_out_exactly_once` and
`test_rows_reserved_untouched`.

This planner is **not** the frozen article six-fold run, and it trains no
models and computes no uncertainty scores.

## 5. Audit country/year and calendar coverage

```bash
sentinel-crop-audit-cohorts \
  --manifest examples/calendar_audit/cohort_split_manifest.csv \
  --calendar config/crop_calendar_recalibrated_v0_3.csv \
  --output-dir outputs/example_cohort_audit
```

A manifest-derived inventory. The calendar is descriptive here: this audit
does not constrain scene selection and changes no image.

## 6. TRAIN-only pixel-weighted calendar selection

```bash
sentinel-crop-select-weighted-calendar \
  --manifest examples/calendar_audit/cohort_split_manifest.csv \
  --calendar config/crop_calendar_recalibrated_v0_3.csv \
  --taxonomy config/classes.yaml \
  --binding config/calendars/class_calendar_binding_v1.yaml \
  --output-dir outputs/example_weighted
```

The example keeps 2 of 4 TRAIN rows at the default 0.90 threshold: the second
row of each cohort is dominated by potato on a date **before** that country's
potato window opens. Validation and test rows are preserved untouched. Add
`--apply` to write a new TRAIN-filtered manifest; the source manifest is never
modified. A new selection is not a reproduction of any historical model.

## 7. Post-training label-uncertainty review

This is the contract that matters most to read carefully. The tool **consumes**
precomputed out-of-fold review evidence; it does not train fold models,
compute uncertainty, read images or decide anything about a label.

Step 1 — produce a review queue, filtering nothing:

```bash
sentinel-crop-filter-post-training \
  --manifest examples/label_review/frozen_training_manifest.csv \
  --parcel-review examples/label_review/parcel_review_levels.csv \
  --parcel-links examples/label_review/parcel_admissibility.csv \
  --output-dir outputs/example_review_queue
```

Of 6 reviewed parcels, 2 are `HIGH_PRIORITY_REVIEW_CANDIDATE` and map to 3
TRAIN patches. The other four levels are present on purpose to show what is
**not** queued: a disagreement with a temporal explanation, one unstable
across dates, one with a consistent alternative class, and an unflagged
parcel.

Step 2 — after a human adjudicates, apply only approved exclusions:

```bash
sentinel-crop-filter-post-training \
  --manifest examples/label_review/frozen_training_manifest.csv \
  --parcel-review examples/label_review/parcel_review_levels.csv \
  --parcel-links examples/label_review/parcel_admissibility.csv \
  --decisions examples/label_review/human_decisions.csv \
  --apply-approved-exclusions \
  --output-dir outputs/example_review_applied
```

The manifest goes from 10 to 8 rows and `test_rows_unchanged` stays true. Only
`HIGH_PRIORITY_REVIEW_CANDIDATE` parcels may carry a decision, and the only
accepted values are `keep` and `exclude_patch`. No pixel is ever relabeled:
excluding a patch can also remove correctly labelled parcels inside it, which
the audit states explicitly.

**What these examples cannot give you.** `parcel_review_levels.csv` and
`parcel_admissibility.csv` are the outputs of the historical six-fold
out-of-fold procedure, which this package does not re-run. The files here show
the column contract only. The real evidence, including the persistent
parcel-disagreement records and the human decisions, belongs to the dataset
record described in `docs/post_training_review_calendar.md`; running this tool
on synthetic input proves the contract, not a scientific finding.

## Regenerating these files

They are generated deterministically by `release/build_example_inputs.py` in
the development repository, which is not distributed. A packaged test checks
that each example still satisfies the contract its tool enforces, so a changed
contract cannot leave a stale example behind.
