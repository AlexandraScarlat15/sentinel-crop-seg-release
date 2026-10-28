# Configuration reference — curated research software

This page describes the **current public software draft**, not the frozen
manuscript acquisition recipe. To reproduce a previously published data
selection, obtain the original source inventory, calendar, random seeds,
metadata and run logs: a new recipe is **not** that historical experiment.

## Configuration entry points

The generator reads exactly these seven root configuration files when
`--config-mode isolated` is selected, in this order:

1. `generation.yaml` — Sentinel-2 and EuroCrops generator selection
2. `paths.yaml` — SAFE cache and dataset output paths
3. `eurocrops.yaml` — input vector and crop-name field
4. `catalog.yaml` — Sentinel-2 L2A STAC search range and metadata cloud cap
5. `dataset.yaml` — patch size, quality filters, sampling, ROI and bands
6. `classes.yaml` — class ID and EuroCrops-label matching
7. `sentinel.yaml` — cache and SAFE retention

For **new experiments**, select a version-1 YAML recipe from
`config/experiments/examples/` with `--experiment`. The resulting
configuration is baseline files → recipe → explicit `--override` arguments.
Unknown recipe keys are rejected. The recipe SHA-256 is recorded. CLI
overrides use dotted keys and YAML value syntax.

**An `--override` path must already exist in the configuration.** A dotted
path used to create whatever it named, so `dataset.targe.total_patches=4`
built a section nothing reads while `total_patches` kept its configured value
and the run reported success. Every component is now checked — intermediate
sections and the final key alike — before anything is written, so a rejected
override leaves the configuration untouched, and the error lists the keys that
do exist at the level where the path broke. A path that runs through a value
which is not a section is rejected as well, since no setting can live inside a
scalar. Inspect the resolved configuration anyway to catch a *valid* key given
an unintended value, which no key check can detect.

```powershell
sentinel-crop-audit-generation-config --config-dir config --experiment config/experiments/examples/at_2021_calendar.yaml
```

This preflight is offline: it does not verify satellite products or local
EuroCrops availability. Recipe-based generation defaults to isolated mode
and strict resume; historical commands without `--experiment` retain legacy
merging and resume behavior. **Do not copy other experiment YAML files into
the root `config/` directory.**

## Source and acquisition settings

| YAML path | Meaning | Caveat |
|---|---|---|
| `generation.satellite` | Currently `sentinel2` | No alternative backend |
| `generation.label_source` | Currently `eurocrops` | Does not establish source rights |
| `eurocrops.shapefile_path` | Source vector for one cohort | Supply your own verified vector |
| `eurocrops.crop_column` | Crop-name field, baseline `EC_trans_n` | Depends on source edition |
| `paths.sentinel_data_root` | SAFE/cache/processed-state root | Keep distinct from generated output |
| `paths.dataset_output_dir` | Image, mask, log and ledger output | Fresh path for each new experiment |
| `catalog.stac.start_date/end_date` | Inclusive requested STAC search period | ISO dates/timestamps |
| `catalog.stac.max_cloud_cover` | Scene-level metadata cloud cap | Different from patch SCL quality |
| `catalog.stac.limit` | STAC search item limit | Does **not** guarantee full acquisition coverage |
| `sentinel.cache.max_size_gb` | Local product cache target size | Storage policy, not a data filter |
| `sentinel.cache.keep_safe_after_processing` | Retain downloaded SAFE data | Consider storage and licensing |

For multiple countries/years, use the
[multi-cohort plan](multi_cohort_generation.md). Each cohort has its own
recipe, EuroCrops source, STAC dates, target output and SHA-256 report.
Declared country, region and year are metadata, **not independent geographic
verification**. The planner is read-only unless `--execute` is given.

## Sampling and quality settings

These are keys under `dataset`, with the **currently tracked baseline**
values; do not treat them as optimal hyperparameters for other regions.

| Key | Current baseline | Meaning |
|---|---:|---|
| `patch.size` / `patch.stride` | 256 / 128 | Output patch width and sampling stride in pixels |
| `min_foreground_ratio` | 0.20 | Required annotated (non-background) fraction |
| `max_nodata_fraction` | 0.05 | Maximum all-zero multi-band pixel fraction |
| `max_cloud_fraction` | 0.01 | Maximum dilated bad-SCL fraction across the patch |
| `max_cloud_on_foreground_fraction` | 0.005 | Maximum bad-SCL fraction over annotated foreground |
| `bad_scl_classes` | 0,1,2,3,7,8,9,10,11 | SCL categories treated as unsuitable |
| `cloud_dilate_radius` | 2 | Mask expansion in pixels |
| `use_b02_brightness_filter` | false | Enable optional brightness rejection |
| `b02_bright_threshold` | 2000 | B02 threshold if the optional filter is enabled |
| `max_b02_bright_fraction` | 0.02 | Global bright-pixel limit |
| `max_b02_bright_on_foreground_fraction` | 0.01 | Foreground bright-pixel limit |
| `max_patches_per_roi` | 1000 | Maximum accepted samples per selected ROI |
| `target.class_id` | 8 | Baseline targeted crop class (soybean); set null for untargeted |
| `target.min_target_ratio` | 0.05 | Minimum target-class fraction |
| `target.min_target_pixels` | 800 | Minimum target-class pixel count |
| `target.total_patches` | 10000 | Accepted patch quota/upper bound |

ROI selection uses `dataset.roi.strategy`: `hotcell` (current tracked
baseline) or `grid`. Grid selection also uses `size_km`,
`top_n_rois`, `min_polygon_count`, `min_total_area_m2`, and
`min_coverage`; hotcell uses the nested `dataset.roi.hotcell` keys.
The opt-in `area_mode: union` corrects overlapping-polygon area accounting,
while historical `legacy_sum` remains available. See
[ROI and cloud quality](roi_sampling_cloud_quality.md). Hotcell requires a
non-null target class; untargeted generation needs an appropriate grid recipe.

The ten configured band channels, **in order**, are B02, B03, B04, B08,
B05, B06, B07, B8A, B11, B12. The four 10 m channels precede six
20 m channels that are resampled to the reference patch grid. Changing this
order requires matching model, normalization, and checkpoint metadata.

## Calendar policies — do not conflate them

**Scene acquisition:** a recipe can opt into `catalog.calendar`:

```yaml
catalog:
  calendar:
    mode: selected_crop_union
    file: config/crop_calendar_recalibrated_v0_3.csv
    country: AT
    year: 2021
    crops: [Wheat]
```

These settings belong inside the recipe's `settings` mapping. The
calendar file has `country,crop,start,end` rows with MM-DD windows.
The generator queries scenes in the UNION of the selected crop windows;
this does **not** enforce a separate date window for every parcel. To
disable it, omit `catalog.calendar` or set `mode: none` where supported.
See [scene acquisition calendar](acquisition_calendar.md).

**Pixel-weighted compatibility (separate, post-generation):** the
[weighted calendar tool](weighted_calendar_selection.md) reads an existing
manifest with per-class histograms `p0..pN`, checks country/crop window
coverage, and optionally writes a **new TRAIN-only filtered manifest**.
Its default 0.90 threshold is configurable, excludes background from the
denominator, and does not change validation or test. A new training
selection is not evidence that the historical Model B/B′ was reproduced.

## Taxonomy, splits and provenance

`config/classes.yaml` maps EuroCrops text labels to class IDs 0–10:
background, maize, rapeseed, wheat, barley, oats, sunflower, sugar beet,
soybean, potato, pasture. Custom recipes must specify the **entire**
replacement taxonomy and maintain contiguous IDs with background 0.
Pasture in the historical taxonomy is a combined label; alternative class
definitions are new experiments, not corrections to frozen article masks.

The generator's `patches_log.csv` v3 records product/date, bands, sample
bounds and pixel counts. Adapt it using the
[patch log adapter](patch_log_adapter.md), then optionally use
[spatial splitting](spatial_splitting.md) and
[OOF fold planning](spatial_oof_folds.md). Neither reproduces the frozen
article's partition membership by default. Distinct spatial sampling
strategies require leakage audits.

Strict generation resumes bind the resolved recipe, source vector file
components and selected code fingerprints; a mismatched or interrupted
run is refused. See [strict resume](generation_resume_guard.md).

## Before a new user's real run

Provide authentic, licensed local EuroCrops files and Copernicus access,
confirm selected source edition, location and coordinate system, audit the
calendar choices and output destination, verify free storage and check
offline configuration first. Tests with mock STAC data do **not**
establish end-to-end real-data reproducibility. Nothing here modifies the
paper's existing dataset or requires retraining its checkpoint.
