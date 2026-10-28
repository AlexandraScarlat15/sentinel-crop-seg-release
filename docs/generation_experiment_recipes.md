# Versioned generation experiment recipes

Recipes are for **new experiments**. They do not modify the paper's frozen
A′ dataset, corrected masks or checkpoint lineage.

Inspect an example recipe offline:

```bash
sentinel-crop-audit-generation-config --config-dir config --experiment config/experiments/examples/at_2021.yaml
```

After supplying a matching EuroCrops vector source, start a fresh dataset:

```bash
sentinel-crop-generate --config-dir config --experiment config/experiments/examples/at_2021.yaml --run-tag at_2021_demo
```

The command defaults to isolated configuration and strict resume when a recipe
is selected. Existing commands without `--experiment` retain legacy defaults.
Configuration precedence is: the seven named generator YAML files, the explicit
recipe settings, then `--override` CLI values.

Schema:

```yaml
schema_version: 1
name: my_experiment
settings:
  eurocrops:
    shapefile_path: data/external/eurocrops/AT_2021/AT_2021_EC21.shp
  dataset:
    patch:
      size: 256
      stride: 128
    max_cloud_fraction: 0.01
```

Existing keys under `generation`, `paths`, `eurocrops`, `catalog`,
`dataset`, and `sentinel` can be changed. Unknown keys are rejected.
An optional `settings.classes` must contain the **complete** replacement
taxonomy, with background ID 0 and contiguous unique class IDs. Training,
checkpoint output channels and evaluation must agree with the new taxonomy;
this mechanism never changes the historical masks.

The resolved configuration records recipe name and raw recipe SHA-256.
Preflight reports the resolved configuration SHA-256 without accessing source
imagery. The strict run ledger also fingerprints configuration and local
input/source files. Do not mix experiments in a preexisting output directory.

Multi-country orchestration is available through
`sentinel-crop-generate-cohorts`. Optional, versioned crop-window selection
for STAC acquisition is documented in [acquisition calendar](acquisition_calendar.md).
It is a *scene-level crop-window union*, not a per-parcel phenology filter.

**Not implemented here:** automatic K-fold training, parcel-specific
phenology eligibility, or parcel uncertainty scoring. The
separate spatial and OOF split commands remain available; they do not
reproduce or overwrite the historical paper split automatically.
