# Sentinel-2 Crop Segmentation — curated research software

**Version 2.0.0: software-only distribution.** The historical research dataset and trained model are not included. A combined dataset-and-model Zenodo record is planned separately, subject to dataset provenance, attribution and model approval.

This selectively staged package demonstrates the **Sentinel-2 L2A + EuroCrops**
dataset generation workflow and four-level U-Net training. It does **not**
include the full historical research workspace or the Random Forest baseline.

## Where to start

| If you want to | Go to |
|---|---|
| **run anything at all** | [`docs/getting_started.md`](docs/getting_started.md) — prerequisites, a no-network install check, and a first small run |
| try the commands with no data of your own | [`examples/README.md`](examples/README.md) — every documented command, on synthetic inputs |
| understand a configuration key | [`docs/configuration_reference.md`](docs/configuration_reference.md) |
| know what the labels are worth | [`docs/taxonomy_and_label_quality.md`](docs/taxonomy_and_label_quality.md) — **read before interpreting any metric** |
| evaluate on another dataset | [`docs/external_dataset_conversion.md`](docs/external_dataset_conversion.md) — PASTIS and DACIA5 converters |

Each pipeline stage is a separate command, on purpose: you supply your own
dataset and your own checkpoint, and nothing runs implicitly.

## Included core functionality

- EuroCrops label mapping into an 11-class segmentation taxonomy;
- Sentinel-2 scene discovery, SAFE loading, SCL/nodata/cloud filtering;
- grid/hotcell ROI selection, NPY and GeoTIFF image-mask generation;
- patch-level Sentinel product and geographic provenance (patch log v3);
- conversion to a georeferenced manifest, spatial splitting, leakage auditing;
- NEW geographic OOF fold manifest planning with configurable K, outer-guard and inner validation (no model training);
- independent, versioned country/year recipe planning with opt-in per-cohort execution;
- optional validated crop-calendar windows applied to Sentinel-2 scene acquisition, not to individual parcel labels;
- independent TRAIN-only pixel-weighted calendar selection and threshold sensitivity (default 90%); untouched validation/test;
- U-Net training, inference, IoU-based evaluation, standalone training/evaluation plotting and selected synthetic tests;
- source configuration files, including a versioned five-country crop-calendar CSV;
- audited country/year coverage and a human-approved post-training review filter.
- PASTIS and DACIA5 label conversion for independent cross-dataset evaluation.

The five-country calendar CSV is a **versioned reference**. New, explicitly
configured recipes can enforce crop-window UNION selection on the dates of
Sentinel-2 scenes. This does **not** prove the frozen article dataset used
that selection, or that each parcel passed a crop-specific temporal test.

## Install (fresh environment)

Use Python >=3.11, then:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_generation_cloud_quality.py tests/test_roi_area_modes.py -q
```

Inspect available commands:

```bash
sentinel-crop-generate --help
sentinel-crop-generate-cohorts --help
sentinel-crop-select-weighted-calendar --help
sentinel-crop-audit-generation-config --help
sentinel-crop-adapt-patch-log --help
sentinel-crop-merge-cohorts --help
sentinel-crop-split-spatial --help
sentinel-crop-split-oof --help
sentinel-crop-train-unet --help
sentinel-crop-evaluate-unet --help
sentinel-crop-predict-unet --help
sentinel-crop-plot-training --help
sentinel-crop-plot-evaluation --help
sentinel-crop-audit-cohorts --help
sentinel-crop-filter-post-training --help
```

For a trusted U-Net checkpoint and a versioned-label manifest (defaults to
historical labels when `--label-config` is omitted):

```bash
sentinel-crop-evaluate-unet --checkpoint path/to/unet.pt \
  --manifest-csv path/to/split_manifest.csv --split test \
  --output-dir outputs/unet_evaluation --device cpu

sentinel-crop-predict-unet --checkpoint path/to/unet.pt \
  --manifest-csv path/to/split_manifest.csv --split test \
  --output-dir outputs/unet_predictions --save-color --device cpu
```

These commands use the checkpoint's exact band indices, normalization and
optional calendar channels by default, and refuse to overwrite existing
output directories. They do not include Random Forest.

Checkpoints are read with PyTorch's restricted reader, which accepts tensors,
plain containers and primitives. A checkpoint that also stores custom pickled
objects is **refused**, because loading one executes code from whoever wrote
the file. If you can vouch for its origin, pass `--allow-unsafe-checkpoint`.
Still load checkpoints **only from sources you trust**: the restricted reader
narrows the risk, it does not establish provenance.

Existing training and evaluation reports can be plotted without a model or
dataset: `sentinel-crop-plot-training --run-dir ... --out-dir ...` and
`sentinel-crop-plot-evaluation --eval-dir ... --out-dir ...`. They produce
PNG/PDF/SVG figures, class-level tables and percent-normalized confusion
matrices as appropriate. For input contracts, custom class labels and
non-destructive output handling see
[`docs/independent_plotting.md`](docs/independent_plotting.md).

**Start here:** [`docs/getting_started.md`](docs/getting_started.md). It states
what you need before the generator will run — a free Copernicus Data Space
account and two environment variables, a EuroCrops shapefile this archive does
not contain — how to verify the install with no external data at all, and a
first small run. It also says plainly that this generator produces new
datasets and does not reproduce the published benchmark.

For the full list of settings, see
[`docs/configuration_reference.md`](docs/configuration_reference.md).

**Before interpreting any metric produced with this package**, read
[`docs/taxonomy_and_label_quality.md`](docs/taxonomy_and_label_quality.md). The
shipped class map in `config/classes.yaml` was checked against an independent
crop hierarchy over 12,073,602 parcels with a definitive expectation, and nine
of its ten crop classes reach 99.836-100% recall. Three measured exceptions
matter when reading a result: pasture labels omit alpine and French extensive
grazing surfaces entirely, so part of any pasture error is a missing label
rather than a wrong prediction; the wheat rule also captures ryegrass; and one
French source code cannot distinguish sugar beet from chard.

That document states the population behind every figure — which is a set of
EuroCrops source files, not the published benchmark — names the artifact each
number came from, and explains why per-pixel metrics are not comparable across
countries. The supporting evidence files are not distributed, so the numbers
are reported measurements with stated scope rather than claims you can
reproduce from this archive alone.
Developers extending the public API should follow
[`docs/code_documentation_style.md`](docs/code_documentation_style.md):
concise docstrings with `Args`/`Returns` and comments reserved for
scientific assumptions and non-obvious safeguards.
For Python, Sentinel-2 and EuroCrops provenance requirements, see
[`docs/reproducibility_environment.md`](docs/reproducibility_environment.md).
Read [`docs/attribution_and_rights.md`](docs/attribution_and_rights.md)
before proposing any public upload. It records unresolved software ownership,
source-edition attribution and derived-data/weights redistribution conditions;
the presence of `LICENSE.md` is not authorization for every included or
externally supplied research asset.
For new generation experiments, see `docs/generation_experiment_recipes.md`.
The `--experiment` option uses an explicit versioned recipe, defaults to
isolated configuration and strict resume, and preserves legacy defaults when
omitted. The source EuroCrops dataset,
Sentinel-2 SAFE products and output data must be supplied separately.

A no-download configuration check (run from this package root) is:

```bash
sentinel-crop-generate-cohorts --config-dir config \
  --plan config/experiments/examples/multi_cohort_calendar.yaml
```

It reports `mode: validated_plan_only`. These example vector paths are
illustrative. Do not add `--execute` without confirming the real local data,
source rights, and output destinations.

**What happens after generation:** `sentinel-crop-generate` writes the
original image/mask assets and a v3 `patches_log.csv`. Convert a completed
cohort log with `sentinel-crop-adapt-patch-log` (specifying its unique
`--dataset-id`, `--country` and `--cohort`) and create geographical
train/validation/test partitions with the separate
`sentinel-crop-split-spatial` command. Multi-cohort execution produces **separate** output directories; to merge
compatible adapted manifests, use `sentinel-crop-merge-cohorts` with repeated
`--manifest` arguments. It validates dataset/cohort identities, the band
order, source labels, class counts, and asset references before writing
`combined_manifest.csv` to a fresh directory. Then apply one audited
`sentinel-crop-split-spatial` operation. See
[cohort manifest aggregation](docs/cohort_manifest_merge.md).

**Included fully offline split demonstration** (no satellite images required):

```bash
sentinel-crop-split-spatial --config config/examples/spatial_split.yaml \
  --manifest examples/spatial_split_demo_manifest.csv \
  --out-dir outputs/spatial_demo --validate-only
```

Remove `--validate-only` and use a fresh output directory to emit the
synthetic split reports. These demonstration rows are not a research dataset.

The configurable OOF manifest planner documents new experiments in
`docs/spatial_oof_folds.md`; it is **not** the historical article
six-fold run and it does **not** train K models or calculate
uncertainty scores.

## Scope and known limitations

- Broad genuine-source validation. The generator **has** been run once against real Copernicus Sentinel-2 L2A and a real EuroCrops shapefile, producing four verified patches, and the repeat was byte-identical; see [`docs/getting_started.md`](docs/getting_started.md). What is still missing is coverage: one cohort, one product, one date, one platform, and nothing on other countries or EuroCrops editions, the grid ROI strategy, calendar-controlled selection, multi-cohort runs or Linux. It is not part of CI, which needs no credentials and downloads nothing;
- A model-checkpoint smoke test; no weights are included or approved;
- final **OOF review outputs and human decisions** for the post-training filter
  (the safe application utility is included; it does not train six fold models);
- exact paper train/val/test samples and access/redistribution contract;
- final U-Net checkpoint, checksum and model card;
- frozen country/year and calendar **selection policy** matching the article
  (the included calendar has AT/CZ/DE/FR/SK rows; use in the final acquisition
  recipe has not been independently verified);
- exact historical training-environment reconstruction and the independent dataset/model record licenses (outside this software-only release).

This software distribution includes `CITATION.cff` with the reserved
software version DOI `10.5281/zenodo.23714448`. The associated manuscript is submitted
and has no article DOI; it is not described as accepted or published.

Source-code tests are not equivalent to broad real-data validation.
In particular, this software-only package does not reproduce the
historical article dataset without separately obtained source inputs,
frozen manifests and provenance. The planned dataset record will package the
frozen research data and an approved U-Net checkpoint together, subject to
separate verification of their provenance and redistribution rights.

The source inventory and approval conditions are maintained in the private
development repository and are not part of this public software distribution.
The Zenodo software archive contains its own build report and SHA-256 checksums.
