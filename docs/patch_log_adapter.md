# DataEngine patch-log v3 → spatial/training manifest

This adapter connects the **existing** Sentinel-2 / EuroCrops patch generator
to the reusable Spatial Splitting CLI. It is a standalone, non-destructive
conversion: no generator code, original patch file, mask, or historical split is
changed. It never downloads Sentinel-2 data or reruns rasterization.

## The important distinction

DataEngine writes images/masks as NPY training caches and also writes
georeferenced GeoTIFF images/masks. Its `patches_log.csv` (schema v3) retains
per-patch source metadata separately from the NPY arrays:

- `product_id`, `safe_id`, `scene_date`, `tile_id`;
- `crs`, source-grid bounds (`min_x/min_y/max_x/max_y`), affine transform
  coefficients, WGS84 bounding values;
- image/mask paths for **both** NPY and GeoTIFF, band order,
  `class_pixel_counts`, SCL/cloud/nodata and selection quality statistics.

The adapter retains **every input column** and adds:

| New column | Derivation |
|---|---|
| `dataset` | Explicit `--dataset-id`; use a distinct value for each independently generated log |
| `epsg` | Resolved from the logged projected `crs`, without guessing |
| `bbox_minx/bbox_miny/bbox_maxx/bbox_maxy` | Directly from the logged native projected bounds |
| `img_path/mask_path` | NPY or GeoTIFF paths selected by `--format`, rewritten relative to the new manifest directory if needed |
| `acquisition_date` | The logged `scene_date` only if present and ISO-valid; missing dates stay blank |
| `p0, p1, ...` | Per-class mask pixel counts from the logged `class_pixel_counts` JSON; absent class IDs become zero |
| `country/cohort` | Included **only** if passed explicitly |
| `label_source/label_source_ref` | Included **only** if passed explicitly; not guessed from a product name or country |

Original `image_npy_path`, `mask_npy_path`, `image_tif_path`, and
`mask_tif_path` columns are preserved verbatim as **original-log-relative
provenance**. For training and splitting, use the newly generated `img_path`
and `mask_path` columns.

The adapter does not reconstruct a missing EuroCrops source vector/version or
infer missing Sentinel-2 product IDs or dates. Use `--label-source` and
`--label-source-ref` if their actual identity is independently verified.

## Run without modifying any files

From the repository root:

```bash
python -m sentinel_crop_seg.cli.adapt_patch_log \
  --patch-log path/to/generated/patches_log.csv \
  --output path/to/manifests/AT_2021_v3_manifest.csv \
  --dataset-id AT_2021 \
  --country AT --cohort AT_2021 \
  --format npy \
  --verify-assets --validate-only
```

The `--validate-only` option performs all log, geometry, provenance,
histogram and selected-file existence checks, but does not write an output.
`--verify-assets` is optional and checks existence, not contents.

When ready, use the same command without `--validate-only`.
It writes a **new** CSV and a matching `*.provenance.json` sidecar recording
the original SHA-256, log schema, sample/class counts, missing metadata
statistics and the declared provenance. Output collision is refused.

To select georeferenced GeoTIFF images/masks instead, use `--format geotiff`.
Both formats use the **same** logged spatial bounding boxes; no CRS or
coordinate information is inferred from a bare NPY array.

## Then create a spatial split

```bash
python -m sentinel_crop_seg.cli.split_spatial \
  --config config/examples/spatial_split.yaml \
  --manifest path/to/manifests/AT_2021_v3_manifest.csv \
  --out-dir outputs/at_2021_split \
  --validate-only
```

When validated, repeat without `--validate-only`. The split manifest retains
the original provenance columns, plus spatial partition assignments. NPY or
GeoTIFF loader paths are rebased **again** to remain valid relative to the
final split manifest. Do not use a newly allocated split as a substitute for
the paper's historical frozen split without separately validating the
experimental protocol.

## Supported input and exclusions

- **Supported:** complete `patches_log.csv` schema **v3** produced by the
  current `sentinel_crop_seg/core/data_engine.py`, with one unique `sample_id` per log and
  a recognized projected EPSG CRS.
- **Rejected:** legacy metadata-free NPY folders, v1/v2 log layouts,
  contradictory/missing geometry, corrupt class-count JSON, missing paths for
  the selected image format, and attempts to overwrite existing outputs.
- **Missing optional provenance:** recorded as missing; it is never fabricated.
- **Mixed datasets:** adapt each v3 log separately with a distinct dataset ID,
  then use the independent `sentinel-crop-merge-cohorts` command to check
  identities, band order, histogram compatibility and rebased paths.
  This adapter itself does not concatenate multiple logs. See
  [cohort manifest merge](cohort_manifest_merge.md).
- **Full provenance:** the source vector's exact revision, taxonomy mapping,
  and generator configuration are not reconstructed from the v3 log alone.
  Record them in separate experiment-level provenance metadata before release.

For generator output structure see [Dataset Generation](dataset_generation.md).
For guards, geographic leakage checks, and split semantics see
[Configurable Spatial Splitting](spatial_splitting.md).
