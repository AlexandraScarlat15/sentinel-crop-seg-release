# Configurable geography-safe OOF fold manifests (NEW experiments)

**Purpose:** Generate **K spatially held-out folds**, inner training/validation
manifests and explicit audit/role sidecars as inputs for *future* OOF model
training and parcel-uncertainty review. The planner **never trains models,
estimates label uncertainty, changes masks, or alters frozen A′ partitions**.

The published A′ research protocol originally used `K=6`, `R=1`, 30-km
EPSG:3035 cells and a 1,280-m guard. The historic cell-balanced, parcel-
weighted, class-specific exact research results and completed fold membership
are documented in the original internal research protocol (not bundled
with this software-only package).
**This new, generic manifest generator is NOT an exact reimplementation
of that historical experiment**; use the frozen paper's authoritative OOF
files when reproducing it.

## Configuration

```yaml
oof_folds:
  folds: 6                     # integer 2..32; requires at least K+1 units
  seed: 42
  block_size_m: 30000
  guard_distance_m: 1280
  crs: EPSG:3035
  inner_val_fraction: 0.10
  require_spatial_tier: true
  require_window_id: true
```

Example: `config/examples/spatial_oof_folds.yaml`. To generate a new
3-fold or 8-fold plan, create a new YAML with only `folds` changed,
**not** by changing the frozen article's six-fold report.

The input manifest requires
`dataset,sample_id,cohort,split,img_path,mask_path,epsg,bbox_minx,bbox_miny,bbox_maxx,bbox_maxy`.
By default it also needs `window_id` and `final_spatial_tier`, so
inferred historical geometry cannot silently be promoted to certified.
Allowed spatial tiers: `EXACT`, `EXACT_TEMPORAL`,
`HIGH_CONFIDENCE`, `CERTIFIED`. Other train/val rows are retained
in `oof_ineligible_development.csv` with an explicit reason, never
silently corrected. Unclear geography is not acceptable OOF evidence.
Use the frozen A′ manifest with recovered provenance, not the obsolete
67,337-sample manifest.

## Usage

Preview all folds without writing files:

```bash
python -m sentinel_crop_seg.cli.split_oof \
  --config config/examples/spatial_oof_folds.yaml \
  --manifest path/to/frozen_A_prime_manifest.csv \
  --out-dir outputs/oof_k6_new \
  --validate-only
```

After reviewing the preview, use a **fresh** destination and remove
`--validate-only` to write manifests. To strengthen grouping where
actual physical-parcel association data is available, provide:

```bash
python -m sentinel_crop_seg.cli.split_oof \
  --config config/examples/spatial_oof_folds.yaml \
  --manifest path/to/frozen_A_prime_manifest.csv \
  --parcel-links path/to/parcel_admissibility.csv \
  --out-dir outputs/oof_k6_with_links
```

Links must identify `cohort,parcel_key,dataset,sample_id` and may have
`partition` restricted to train/val. A parcel linking multiple spatial
cells makes those cells a **single indivisible allocation unit**.
Be aware the audited admissibility links may not cover every physical
EuroCrops parcel; the summary never claims complete parcel-level
proof. To prove no same-parcel leakage, supply *complete* parcel linkage
and separately audit it. A geographic footprint guard alone does not
prove distant windows never intersect the same very large field.

## Outputs and meaning

- `fold_assignments.csv`: every qualified train/val patch, outer fold
  ID, original source split, cell ID and linked allocation unit.
- `fold_XX_training_manifest.csv`: original image/mask references and
  spatial/temporal fields, but only the **inner training** `split=train`
  and geographically held-out **early stopping** `split=val`.
  The held-out OOF fold and guard-excluded windows do not occur here.
- `fold_XX_held_out_manifest.csv`: the fold's OOF evaluation population,
  `split=held_out_fold` and `oof_role=held_out_fold`. The special
  label deliberately prevents accidental use as U-Net early-stopping
  validation or as the final frozen test. OOF inference needs a
  corresponding explicit held-out adapter/CLI and checkpoint pairing.
- `fold_XX_roles.csv`: every qualified train/val sample with role
  `held_out_fold`, `inner_train`, `inner_val`,
  `excluded_outer_guard` or `excluded_inner_guard`.
- `oof_ineligible_development.csv`: uncertified/missing-geometry
  population excluded **from this new OOF run only**.
- `oof_fold_audit.json`: hash of source manifest, configuration, and
  optional links; counts, K, 1 repetition, per-fold guard removals and
  final validation status.

All paths are rebased to the output manifest's location; source image
and mask arrays are not copied or modified. All output is created in a
new staging directory, never overwriting existing results.

## Leakage and model-selection rules

1. **TEST is never in the OOF population**. The planner only records its
   row count; its spatial coordinates and labels do not enter fold
   assignment or model selection.
2. Multiple acquisition dates with the same georeferenced footprint
   remain one window and one fold. Whole 30-km cells remain indivisible.
   Optional parcel links merge cells belonging to the same parcel.
3. A projected, densified-rectangle footprint gap of <1,280 m
   (and every overlap/touching footprint) causes the **training** side
   to be dropped. The held-out population is never pruned to conceal
   a conflict. The guard is checked after dropping.
4. Outer OOF held-out samples are never used to train or early-stop
   their own model. Early stopping is selected from spatial units
   remaining inside that outer fold's training population, and a
   second guard protects inner train/val.
5. Every eligible sample must be held out **exactly once** across K
   folds (R=1), with no shared cell/allocation unit between heldout
   and training or internal train/val. Invalid/impossible configurations
   fail rather than weaken the guard.
6. K varies the **number of trainings**, not the number of model votes
   each sample receives: with one repetition, each eligible window gets
   only one OOF judgment. For multiple votes use a deliberately
   pre-registered repeated-CV experiment with its own safeguards.

## What is not automated yet

This component produces only *audited fold manifests*. It does NOT
automatically train `K` U-Nets, ensure the correct fold-specific
checkpoint is applied to every held-out window, compute parcel-level
OOF disagreement, aggregate temporal consistency, or adjudicate/relabel
anything. The separate post-training review filter can **consume
previously generated** OOF parcel-review outputs plus human decisions.
It must not be misrepresented as an automatic mislabel detector.

Before calling this a full self-contained Zenodo research protocol:
validate the new fold generator in a clean staged environment,
integrate the held-out inference/OOF review adapter and verify
whole-parcel nonleakage from authoritative vector identities.
