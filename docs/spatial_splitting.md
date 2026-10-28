# Configurable spatial splitting

This is a **new, independent workflow**. It does not depend on or modify
the original research-only splitting scripts or historical frozen manifests.
The included example CSV is synthetic and contains no satellite images.

## Basic usage

From the repository root (after installing the project):

Try the **included synthetic demonstration dataset**, which requires no satellite
images or historical files:

```bash
python -m sentinel_crop_seg.cli.split_spatial --config config/examples/spatial_split.yaml --manifest examples/spatial_split_demo_manifest.csv --out-dir outputs/spatial_demo --validate-only
python -m sentinel_crop_seg.cli.split_spatial --config config/examples/spatial_split.yaml --manifest examples/spatial_split_demo_manifest.csv --out-dir outputs/spatial_demo
```

Or supply your own georeferenced manifest:

```bash
python -m sentinel_crop_seg.cli.split_spatial --config config/examples/spatial_split.yaml --manifest path/to/manifest.csv --out-dir outputs/example_split --validate-only
python -m sentinel_crop_seg.cli.split_spatial --config config/examples/spatial_split.yaml --manifest path/to/manifest.csv --out-dir outputs/example_split
```

The installed command is `sentinel-crop-split-spatial`. All paths above are
interpreted relative to your current working directory. The output directory
**must not exist**; rerunning with the same directory fails rather than
overwriting a published split.

To reuse metadata created by the Sentinel-2 / EuroCrops generator, first
convert its `patches_log.csv` v3 using the [patch-log adapter](patch_log_adapter.md).
That conversion retains product IDs, SAFE IDs, dates, CRS, source coordinates,
band order and original log fields, then generates the columns expected below.

### Required input columns

- `sample_id` and optional `dataset` (defaults to `default`); their combination must be unique.
- `epsg`: integer EPSG code for the **source patch coordinate system**.
- `bbox_minx, bbox_miny, bbox_maxx, bbox_maxy`: source-CRS patch footprint.
- Optional `country`, `cohort` for stratification (otherwise `unspecified`).
- Optional `window_id`: if provided, a window ID must not refer to different bounding boxes.
- Optional configured `parcel_id_column`: globally unique, non-null parcel identity.
- Optional class histograms `p0..pN` or `c0..cN` to compute class-distribution CSV.

The source and splitting CRS must be **projected**, and the splitting CRS must
use metres (default European LAEA EPSG:3035). The algorithm groups all
observations of the same footprint across acquisition dates into one spatial
window. For globally consistent parcel grouping across distinct windows,
provide `grouping.parcel_id_column`; no unsupported parcel-identity guarantee
is claimed when it is null.

### How the separation works

1. Reproject the footprint boundary into the requested metric CRS using densified
   edges (21 intermediate samples per edge).
2. Conservatively enclose each reprojected patch with an axis-aligned rectangle.
3. Assign whole configurable blocks to `train`, `val`, or `test`. If a
   parcel ID column is configured, connected blocks sharing that parcel form
   one indivisible allocation unit.
4. Sample `candidate_seeds` deterministic allocations. For each, detect
   footprint intersections and every inter-partition pair with **gap strictly
   below** `guard_distance_m`. Touching/overlapping footprints always conflict.
5. Keep test intact, exclude validation conflicts against test, then exclude
   train conflicts against surviving validation or test. Choose the candidate
   that discards the fewest samples; break ties by realized ratio deviation and
   seed.
6. Audit the **minimum true gap between the conservative rectangles** for each
   retained partition pair (not merely the distance of the closest centres).
   Fail closed if any violation remains.

Because blocks and guard exclusions constrain the data, requested partition
ratios are **targets rather than guarantees**. The report gives actual ratios,
excluded samples and missing class statistics if no histograms were provided.
The algorithm cannot always construct three nonempty partitions; in that case
it raises an error instead of silently loosening the guard.

## Outputs

- `split_manifest.csv`: all input rows, including `excluded`; new
  `partition`, `split`, `cell_id`, and `exclusion_reason` (`guard_distance`
  for excluded windows). An existing input `partition` is preserved as
  `source_partition`; an existing `split` is preserved as `source_split`.
  If these provenance names already exist alongside their source columns, the
  command refuses to overwrite them.
- `split_assignments.csv`: lightweight sample-to-partition mapping.
- `split_summary.json`: counts, desired and achieved proportions, excluded
  observations, selected seed, SHA-256 hashes of input manifest and config.
- `leakage_audit.json`: guard settings, cross-partition minimum rectangle gaps,
  and a validation result.
- `class_distribution.csv`: emitted only when class histogram columns exist.
- `spatial_split_map.png`: optional scatter map of retained and excluded windows
  when `splitting.outputs.save_split_map: true` (requires matplotlib).

### Image and mask path compatibility

When the source CSV includes `img_path` and/or `mask_path`, the command
rebases **relative** paths from the original manifest directory to the new
output directory. The loader can therefore resolve them from the generated
`split_manifest.csv` without a manual `base_dir` override. Existing absolute
paths are preserved unchanged. If source and destination are on different
Windows drives, relative paths cannot span drives, so a resolved absolute
path is emitted. Relative Windows-style separators are also recognized on
POSIX. No image or mask file is moved, copied, deleted or edited.

Paths still refer to the original data files: a moved or unmounted dataset
must be reconfigured separately. Empty path values are rejected (including in `--validate-only`) before an
output directory is created. The command does not verify that the data files
exist or are readable: that requires a DataLoader integration check.
The in-memory `split_dataframe` API does not rebase paths because it has no
source/output directory context.

For training, use only rows with `split` in `train,val,test`; excluded rows
are retained for traceability, **not** training. Split generation does not
move or alter source patches, masks, or annotations. The newly assigned
`partition`/`split` columns are a **new experiment**, not a continuation
of the original `source_partition` or `source_split` membership. Do not
replace a frozen manuscript test partition without a separate approved and
scientifically justified experimental protocol.

## Scientific scope and limitations

The method guarantees separation under its **conservative projected rectangle**
model and the selected geographic grouping assumptions. It does not prove
physical parcel separation without stable parcel identifiers, does not
guarantee any particular crop balance, and has not been benchmarked against the
exact historical paper split. It does not implement a spatial cross-validation
K-fold workflow; it creates a single train/validation/test split.

Do not claim numerical equivalence with the manuscript experiments without
a separate end-to-end comparison against their authoritative manifests.
