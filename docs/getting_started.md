# Getting started

What you need before the generator will run, in the order you need it. Read
this before `docs/dataset_generation.md`, which documents the configuration in
full but assumes you already have access to the two external sources.

## 1. Install

```bash
python -m pip install .
```

Python 3.11 or newer. The exact versions the test suite last ran on are pinned
in [`requirements-tested.txt`](../requirements-tested.txt); see
[environment provenance](reproducibility_environment.md) for what that file
does and does not guarantee.

## 2. Verify the install with no external data at all

Every documented command in [`examples/README.md`](../examples/README.md) runs
against synthetic inputs shipped in this archive. They need no Sentinel-2
product, no EuroCrops file, no credentials and no network:

```bash
python -m pytest tests -q
sentinel-crop-split-spatial --config config/examples/spatial_split.yaml \
  --manifest examples/spatial_split_demo_manifest.csv \
  --out-dir /tmp/example_split --validate-only
```

If those pass, the package is installed correctly. Nothing beyond this point is
needed for the splitting, calendar, OOF-planning or label-review tools — they
consume manifests, not satellite data.

## 3. A Copernicus account, for the generator only

Sentinel-2 Level-2A products are retrieved from the Copernicus Data Space
Ecosystem. **Registration is free.** Create an account at
<https://dataspace.copernicus.eu/> and export two environment variables:

```bash
export COPERNICUS_USERNAME="your-cdse-login"
export COPERNICUS_PASSWORD="your-cdse-password"
```

```powershell
$env:COPERNICUS_USERNAME = "your-cdse-login"
$env:COPERNICUS_PASSWORD = "your-cdse-password"
```

The package reads them from the environment and never stores or logs them. If
they are absent the generator stops with
`Copernicus credentials not found in environment variables`.

Endpoints used, so you can check them against your institution's network
policy: `https://catalogue.dataspace.copernicus.eu/stac` for search,
`https://zipper.dataspace.copernicus.eu/odata/v1/` for download and
`https://identity.dataspace.copernicus.eu/...` for the OAuth2 token.

Already have a local Sentinel-2 cache? Point `paths.sentinel_data_root` at it
and no credentials are needed.

## 4. A EuroCrops shapefile, which this archive does not contain

Download the harmonized vectors for the country and year you want from the
[EuroCrops repository](https://github.com/maja601/EuroCrops), then set two
values in `config/eurocrops.yaml`:

```yaml
eurocrops:
  shapefile_path: "data/external/eurocrops/AT_2021/AT_2021_EC21.shp"
  crop_column: "EC_trans_n"
```

`crop_column` must name the **harmonized** crop-name column; `EC_trans_n` is
the usual one, but check your own file, because naming differs between
editions. A column that does not exist raises a `KeyError` immediately. The
failure to watch for is subtler: naming a column that *does* exist but holds
untranslated national crop names means the class rules match almost nothing
and you get a nearly all-background mask with no error at all. Inspect the
class-pixel counts of your first run before trusting them.

Before using any labels you derive, read
[attribution and rights](attribution_and_rights.md): the EuroCrops license is
edition-specific and a share-alike obligation may apply to your outputs.

## 5. Check the configuration before spending bandwidth

This reads your configuration, resolves the calendar and reports what a run
would do. It downloads nothing:

```bash
sentinel-crop-audit-generation-config --config-dir config \
  --experiment config/experiments/examples/at_2021_calendar.yaml
```

## 6. A first small run

Start deliberately small and confirm the outputs before scaling:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag first_run \
  --config-mode isolated \
  --resume-policy strict \
  --override eurocrops.shapefile_path="data/external/eurocrops/AT_2021/AT_2021_EC21.shp" \
  --override paths.dataset_output_dir="data/processed/first_run/generated" \
  --override dataset.target.total_patches=100 \
  --override dataset.max_patches_per_roi=20 \
  --override catalog.stac.limit=5
```

**One trap worth knowing before you override a date.** Override values are
parsed as YAML, so `catalog.stac.start_date=2021-09-01T00:00:00Z` becomes a
`datetime` object, not a string, and the catalog rejects it with
`invalid datetime component`. Quote it *inside* the value so YAML keeps it a
string:

```bash
  --override 'catalog.stac.start_date="2021-09-01T00:00:00Z"'
```

The same applies to any value you need to stay textual — a version like `1.0`
would otherwise become a float. Paths and plain words need no inner quotes.

`--config-mode isolated` reads only the generator YAMLs in declared order, and
`--resume-policy strict` binds the run to a fingerprint of the code and
configuration that produced it, so a later resume cannot silently mix
settings. Both are recommended for new work; the defaults preserve historical
behaviour instead.

You should get patch arrays plus a `patches_log.csv` recording, per patch, the
product ID, SAFE ID, acquisition date, MGRS tile, native CRS, patch bounds and
affine, WGS84 coordinates, band order, quality statistics and output paths.
Convert that log into a provenance-carrying manifest with
`sentinel-crop-adapt-patch-log`, then continue with the splitting tools.

## How far this has actually been tested

Worth knowing before you trust a run, because the automated suite and the
real-source check cover different things.

The packaged test suite is large and runs offline: it exercises the loaders,
the class mapping, ROI selection, the quality filters, the resume guard, the
splitting and calendar tools and every CSV contract, using a mock catalogue
rather than the network. It never downloads a Sentinel-2 product.

Separately, the pipeline above has been executed **once** against genuine
sources: one Copernicus Sentinel-2 L2A product and the real EuroCrops AT_2021
shapefile, on Windows with Python 3.13, producing four patches whose arrays,
masks, coordinate systems, per-class pixel counts and GeoTIFFs were all
checked, and whose log the adapter then converted into a provenance manifest.
The quality filters rejected 321 of 323 candidate windows in that run, so they
were exercised rather than bypassed.

That single run does **not** cover other countries or EuroCrops editions, the
grid ROI strategy, calendar-controlled selection, multi-cohort runs, or Linux.
It is not part of continuous integration, because it needs credentials and
about a gigabyte of download. Treat it as evidence that the path works, not as
a guarantee for your configuration.

## Expectations, stated plainly

**The generator produces new datasets; it does not reproduce the published
benchmark.** The 67,337-patch benchmark behind the article was generated by an
earlier pipeline that did not write a complete georeferenced log, and its
spatial and temporal provenance was reconstructed forensically afterwards. The
generator here writes that provenance at generation time, which is the
improvement — but a run of this code produces *your* dataset, not a byte-equal
copy of the historical one. Treat the published benchmark as data to obtain
from the dataset record, not as something to regenerate.

**Downloads are large and slow.** Sentinel-2 L2A products are hundreds of
megabytes to several gigabytes each. `sentinel.cache.max_size_gb` caps the
local cache and `keep_safe_after_processing` controls whether SAFE archives are
retained. Plan disk space before a multi-cohort run.

**Results are not bit-reproducible across environments.** Geospatial wheels
bundle their own GDAL/PROJ, PROJ datum grids are versioned, and GPU builds of
torch differ. See [environment provenance](reproducibility_environment.md).

**Before interpreting any metric**, read
[taxonomy and label quality](taxonomy_and_label_quality.md). The shipped class
map has measured, quantified defects — a pasture label omission and a ryegrass
overcapture — and per-pixel metrics are not comparable across countries.

## What this archive does not include

No Sentinel-2 products, no EuroCrops vectors, no generated dataset, no trained
weights and no out-of-fold review evidence. A combined dataset-and-model Zenodo record is planned but not yet published.
The label-review tool here ships with a synthetic example of its input
contract rather than the real review evidence.

## Where to go next

| You want to | Read |
|---|---|
| understand every configuration key | [configuration reference](configuration_reference.md) |
| generate one cohort | [dataset generation](dataset_generation.md) |
| generate several countries/years | [multi-cohort generation](multi_cohort_generation.md) |
| avoid spatial leakage in a split | [spatial splitting](spatial_splitting.md) |
| restrict scenes to a crop calendar | [acquisition calendar](acquisition_calendar.md), [weighted calendar selection](weighted_calendar_selection.md) |
| plan out-of-fold folds | [spatial OOF folds](spatial_oof_folds.md) |
| train, evaluate or predict | [U-Net evaluation and prediction](unet_evaluation_prediction.md) |
| review labels after training | [post-training review](post_training_review_calendar.md) |
