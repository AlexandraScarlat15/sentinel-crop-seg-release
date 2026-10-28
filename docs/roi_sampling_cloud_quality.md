# ROI sampling and cloud-quality audit

This is an **opt-in, behavior-preserving** scientific improvement in this
package. The frozen manuscript split, legacy ROI settings, Sentinel-2 products,
EuroCrops files, class taxonomy, NPY/GeoTIFF writer and patch log v3 are
unchanged.

## ROI scoring: legacy sum versus geometric union

**Problem:** Summing the clipped areas of parcels double-counts spatially
overlapping annotations. Two identical parcels each occupying a complete
100m × 100m ROI produce a legacy coverage score of `2.0` (200%) although the
actual covered area is 10,000m² (100%). This can incorrectly influence ROI
rankings and class-directed hotcell thresholds.

**Old behavior remains the default** to reproduce previous selection:

```yaml
dataset:
  roi:
    area_mode: legacy_sum
    hotcell:
      area_mode: legacy_sum
```

For a **new scientific dataset**, opt in to union scoring:

```yaml
dataset:
  roi:
    strategy: grid
    area_mode: union
```

Or for rare classes with hotcell:

```yaml
dataset:
  roi:
    strategy: hotcell
    hotcell:
      area_mode: union
```

`grid.area_mode` controls how clipped crop polygons contribute to
`total_crop_area_m2`, `coverage`, `min_coverage` and ROI ranking.
`hotcell.area_mode` controls how same-class overlapping polygons contribute
to `target_area_m2`, cell scores and `min_target_area_m2`. In `union`
mode the areas are computed from the **union of clipped geometries**.
The count of intersecting source polygons remains a count of records
and is not deduplicated automatically.

In union mode the grid selector refuses a geographic or non-metre CRS.
Hotcell (which has always required projected data) now also checks that
its projected units are **metres**, not feet. Reproject to an appropriate
metric CRS *before* invoking the selector. The legacy grid fallback
(`roi_size_km/111` with unprojected data) is retained for reproducibility
but is not appropriate for scientific area metrics. No CRS conversion is
silently done during ROI selection.

The resulting grid/hotcell metadata records `area_mode`. Switching to
`union` can change ROI scores, which ROIs pass thresholds, and how many
patches get generated. **It does not reproduce the published dataset split**.
Use a fresh `run_tag`, output directory, and strict resume policy.

### What union area does *not* solve

- It does not eliminate overlapping ROI windows or repeated patches from
  different Sentinel products/acquisition dates.
- It does not adjudicate contradictory crop labels on overlapping parcels.
- It does not balance all crop classes, seasons or countries automatically.
- It does not normalize for spatial sampling bias or replace split leakage
  auditing. Those remain independent dataset-quality tasks.

## Cloud and pixel-quality rules: verified behavior

Patch acceptance in `sentinel_crop_seg/core/data_engine.py` remains **unchanged** in this
increment. Synthetic regression tests now exercise its actual NPY/GeoTIFF
writing and `patches_log.csv` v3 pathway, including these checks:

| Rule | Existing behavior |
|---|---|
| Nodata | Reject if proportion of pixels with all bands equal to zero exceeds `max_nodata_fraction` |
| Foreground | Reject when non-background mask fraction is below `min_foreground_ratio` |
| SCL quality | By default, reject SCL categories `0,1,2,3,7,8,9,10,11` according to two cloud thresholds |
| SCL dilation | Square morphological dilation radius `cloud_dilate_radius` pixels, applied to the bad-class mask |
| Global cloud threshold | Reject if dilated bad-SCL fraction exceeds `max_cloud_fraction` |
| Cloud over crops | Reject if bad-SCL pixels overlapping labeled foreground exceed `max_cloud_on_foreground_fraction` of foreground pixels |
| B02 optional brightness | If enabled, apply both global and foreground brightness thresholds; absent B02 turns this test off |
| Target class | Require `target.class_id` pixel share/count when target sampling is enabled |

A mismatched SCL patch shape is now rejected before writing any accepted
sample: SCL must align with the rasterized mask, rather than relying on
NumPy broadcasting. The existing SAFE loader also checks its SCL output shape.

SCL classes `4` (vegetation), `5` (bare soils) and `6` (water) are not
in the default bad-class list. `SentinelSafeDataLoader.read_scl()` uses
**nearest-neighbor** resampling to align categorical 20m SCL to the
reference 10m grid. These settings are configurable and explicitly affect
dataset composition.

**Limitations:** The current log records cloud/nodata/brightness metrics for
*accepted* patches. It does not persist all rejected patch candidates and
their reasons; console summaries are per ROI. A structured rejection audit
and cross-ROI/product duplication audit should be added before claiming
full quality provenance. Tests are synthetic and must be followed by a
small real Sentinel-2 + EuroCrops smoke run.

## Offline verification

```bash
python -m pytest tests/test_roi_area_modes.py tests/test_generation_cloud_quality.py -v
python -m pytest tests/ -q
```

These tests need no downloaded satellite product or real shapefile. Never run
the generator on the frozen article dataset solely to compare the new union
mode with legacy scores.
