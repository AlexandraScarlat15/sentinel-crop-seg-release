# Independent country/year generation plans

Multi-cohort plans group **versioned single-cohort recipes** without
modifying the frozen article A′ dataset or relying on the legacy YAML merge.

## Offline planning (default)

```bash
sentinel-crop-generate-cohorts --config-dir config \
  --plan config/experiments/examples/multi_cohort.yaml
```

The command reads all referenced recipes, validates their generator settings,
requires unique cohort IDs and disjoint output directories, and reports
each cohort's vector source, STAC interval, recipe SHA-256, resolved recipe
SHA-256, country, year and region. It does **not** download products, inspect
EuroCrops vectors, create output folders or build a combined manifest.

A version-1 plan contains:

```yaml
schema_version: 1
name: central_europe_demo
cohorts:
  - id: AT_2021
    country: AT
    year: 2021
    region: national
    recipe: at_2021.yaml
  - id: FR_2018
    country: FR
    year: 2018
    region: national
    recipe: fr_2018.yaml
```

Recipe paths are resolved from the **plan YAML directory**. Each recipe
independently defines its EuroCrops shapefile, STAC acquisition interval,
sampling thresholds, target classes, taxonomy and dedicated output directory.
The ISO country, year and region in the plan are metadata declarations,
**not independent validation of the vector source or satellite footprint**.

Before real generation, replace the example EuroCrops vector filenames with
the correct locally available sources. Plan validation is not proof that
those sources exist. Use the same `--config-dir` for all cohorts.

## Explicit execution

Only after reviewing the plan, opt into data processing:

```bash
sentinel-crop-generate-cohorts --config-dir config \
  --plan config/experiments/examples/multi_cohort.yaml --execute
```

Execution preflights **every cohort** before starting and refuses *all*
execution if any cohort output directory already exists. It invokes the
existing generator per cohort using `config_mode=isolated`,
`resume_policy=strict`, and a distinct `run_tag`. If a later cohort fails,
previously generated cohorts remain on disk for inspection; this workflow
does not claim multi-cohort transactional rollback. To resume or extend an
existing cohort, use the separately documented strict single-cohort workflow
after checking its run ledger.

**Optional acquisition-calendar control:** each cohort recipe may declare
`catalog.calendar.mode: selected_crop_union`, with a versioned calendar CSV,
country/year and a nonempty crop list. The generator then queries and filters
Sentinel-2 scenes inside the selected crop-window union. See
[calendar acquisition policy](acquisition_calendar.md). Plan validation
rejects conflicting calendar country/year metadata.

**Limitations:** this generator does not provide per-parcel phenology filtering,
automatically merge output manifests, ensure consistent source licenses, or assert
the paper's historic date policy. An acquisition interval
from a recipe is distinct from a phenology window applied to each crop.
The five-country CSV is now optionally enforced at the **scene acquisition**
boundary; its calendar windows are not per-parcel label-validity assertions. Per-cohort acquisition decisions remain independent of the
standalone, audited manifest-merging step.

To create a common multi-country manifest, first adapt each cohort's
`patches_log.csv` with an explicit dataset/country/cohort identity, then use
the independent `sentinel-crop-merge-cohorts` command in
[cohort manifest aggregation](cohort_manifest_merge.md). The merged manifest
can be processed by `sentinel-crop-split-spatial` to create one audited
train/validation/test partition across all countries. This path does not
assert exact paper split reproduction.

Do not use multi-cohort output as a substitute for the article's immutable
train/validation/test manifests. New dataset splits require separate
provenance and leakage audits.
