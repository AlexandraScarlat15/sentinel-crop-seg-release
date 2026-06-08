# Project Architecture

## Purpose

Sentinel Crop Segmentation is a research-oriented pipeline for generating crop segmentation datasets from Sentinel-2 imagery and EuroCrops parcel annotations, training baseline models, and evaluating crop segmentation predictions.

The current implemented backend focuses on:

* satellite source: Sentinel-2 Level-2A;
* label source: EuroCrops;
* dataset format: aligned image-mask patches saved as NumPy arrays and GeoTIFF files;
* model baseline: four-level U-Net;
* classical baseline: Random Forest;
* evaluation and prediction through installed command-line entrypoints.

The repository is organized as source code, configuration, tests, and documentation. Large external datasets, generated datasets, checkpoints, and experiment outputs are intentionally excluded from version control.

## High-level workflow

The main workflow is:

1. Load configuration from YAML files.
2. Apply optional runtime configuration overrides.
3. Load EuroCrops parcel annotations.
4. Map heterogeneous crop labels to a reduced crop taxonomy.
5. Select regions of interest using a configured ROI strategy.
6. Search Sentinel-2 Level-2A products through STAC.
7. Prepare Sentinel-2 SAFE products through the cache layer.
8. Load and align Sentinel-2 spectral bands.
9. Rasterize EuroCrops polygons into segmentation masks.
10. Save aligned image-mask patches as NumPy and GeoTIFF files.
11. Train a U-Net model or Random Forest baseline.
12. Evaluate model checkpoints.
13. Export predictions and optional visualization panels.

## Main layers

The project is separated into several layers:

```text
sentinel_crop_seg/        Public package API and installed CLI wrappers
src/                      Core implementation
  catalog/                Sentinel-2 STAC catalog search
  data/                   EuroCrops and Sentinel-2 loading/cache utilities
  pipeline/               Reusable dataset generation pipeline
  selection/              ROI selection strategies
  training/               Dataset loaders, U-Net model, loss, training loop
  utils/                  Spatial and helper utilities
scripts/                  Public utility scripts, including E2E smoke test
config/                   YAML configuration files
docs/                     Project documentation
tests/                    Test suite
```

## Public package API

Location:

```text
sentinel_crop_seg/
```

The package exposes the dataset generation API:

```python
from sentinel_crop_seg import generate_dataset
```

This API calls the reusable implementation under `src/pipeline/`.

The public package also provides installed command-line entrypoints through `pyproject.toml`.

Main installed commands include:

```text
sentinel-crop-generate
sentinel-crop-convert-pastis
sentinel-crop-convert-dacia5
sentinel-crop-train-unet
sentinel-crop-train-rf
sentinel-crop-evaluate
sentinel-crop-predict
```

## Dataset generation pipeline

Location:

```text
src/pipeline/dataset_generation.py
```

This module contains the reusable dataset generation function. It is used by both:

* the Python package API;
* the installed `sentinel-crop-generate` command.

The function reads configuration, validates the generation source, selects ROIs, searches Sentinel-2 products, prepares cached SAFE products, and delegates patch extraction to the data engine.

The current supported generation source is:

```yaml
generation:
  satellite: "sentinel2"
  label_source: "eurocrops"
```

Other generation backends are not part of the current implementation.

## Configuration loading and overrides

Location:

```text
src/config_loader.py
```

The configuration loader reads all YAML files from a configuration directory and merges them into a single nested dictionary.

This allows the project configuration to be split into focused files such as:

```text
generation.yaml
paths.yaml
eurocrops.yaml
catalog.yaml
dataset.yaml
classes.yaml
sentinel.yaml
```

The configuration system supports runtime overrides using dotted keys:

```bash
--override dataset.target.total_patches=8
--override paths.dataset_output_dir="data/processed/example/generated"
--override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp"
```

Override values are parsed with YAML syntax, so values such as `true`, `false`, `null`, integers, floats, and lists are converted automatically.

This makes it possible to run local experiments, small smoke tests, or path-specific runs without modifying tracked configuration files.

## EuroCrops mask loading

Location:

```text
src/data/eurocrops_mask_loader.py
```

This component loads EuroCrops vector data, maps crop names to project class IDs, and rasterizes parcel geometries into segmentation masks aligned with Sentinel-2 image windows.

The class mapping is controlled by the reduced taxonomy declared in `config/classes.yaml`.

## Sentinel-2 data access

Locations:

```text
src/catalog/
src/data/sentinel_cache_manager.py
src/data/sentinel_safe_loader.py
```

These modules are responsible for:

* searching Sentinel-2 Level-2A products through STAC;
* managing local Sentinel product cache state;
* loading spectral bands from SAFE products;
* preparing image data used by the dataset generation engine.

External service availability, network connectivity, and local cache state can affect online generation runs.

## Data engine

Location:

```text
src/data_engine.py
```

The data engine performs patch extraction.

It receives:

* a Sentinel loader;
* a mask loader;
* selected ROIs;
* configured spectral bands;
* filtering thresholds;
* optional target-class sampling constraints.

It produces aligned image-mask samples suitable for semantic segmentation training and evaluation.

## ROI selection

Locations:

```text
src/selection/
src/selection/roi_selectors/
```

ROI selection controls where patches are extracted from.

The project currently supports two strategies:

* `grid`;
* `hotcell`.

The grid strategy selects fixed-size regions based on crop polygon coverage.

The hotcell strategy focuses ROI selection around a target crop class and is useful for improving rare-class representation.

## Training components

Location:

```text
src/training/
```

The training package contains:

* root-folder dataset loader;
* manifest-based dataset loader;
* U-Net model definition;
* combined cross-entropy and Dice loss;
* U-Net training loop.

The public U-Net training command is:

```bash
sentinel-crop-train-unet --config config/train_unet.yaml
```

Training supports runtime configuration overrides:

```bash
sentinel-crop-train-unet \
  --config config/train_unet.yaml \
  --override training.epochs=1 \
  --override data.train_limit=4
```

## Evaluation and prediction

The public evaluation and prediction commands are:

```bash
sentinel-crop-evaluate
sentinel-crop-predict
```

The evaluation pipeline computes segmentation metrics and writes:

```text
summary.json
summary.txt
metrics_per_class.csv
confusion_matrix_counts.npy
confusion_matrix_row_norm.npy
evaluated_samples.txt
```

The prediction pipeline can export:

```text
prediction_samples.csv
prediction_summary.json
pred_masks_npy/
pred_masks_color/
gt_masks_color/
rgb/
panels/
```

These outputs support both quantitative evaluation and qualitative inspection.

## External dataset conversion utilities

The repository includes installed conversion entrypoints for external datasets:

```text
sentinel-crop-convert-pastis
sentinel-crop-convert-dacia5
```

These are intended to convert external datasets into formats compatible with the project pipeline. They are separate from the main Sentinel-2 + EuroCrops generation backend.

## Minimal E2E smoke test

Location:

```text
scripts/run_e2e_smoke_test.py
```

The E2E smoke test validates the complete installed pipeline:

```text
dataset generation
-> U-Net training
-> evaluation
-> prediction
```

It uses the normal configuration files plus runtime overrides. It does not require copied smoke-test configuration files.

Example:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

The E2E test is an online smoke test because it uses Sentinel-2 catalog search and data retrieval.

## Tests and CI

Location:

```text
tests/
```

The test suite checks important components of the pipeline without requiring the full external-data generation process.

Covered areas include:

* configuration loading and overrides;
* public package API;
* installed CLI entrypoints;
* CLI help availability;
* model forward pass;
* root-folder dataset loading;
* manifest dataset loading;
* loss behavior;
* evaluation utilities;
* prediction utilities;
* EuroCrops class mapping;
* external label mappings;
* spatial utilities;
* dataset generation API behavior.

Run the test suite with:

```bash
python -m pytest -q
```

## Current architecture summary

The current architecture separates:

1. public package and CLI interface;
2. reusable pipeline logic;
3. geospatial data loading and caching;
4. ROI selection;
5. model training;
6. evaluation and prediction;
7. tests and documentation.

This separation makes the project easier to test, document, extend, and archive as research software.

## Scope and future extensions

The current implementation focuses on Sentinel-2 + EuroCrops. Potential future extensions include:

* additional label sources;
* additional ROI sampling strategies;
* improved rare-class sampling;
* additional model architectures;
* expanded external benchmark workflows;
* richer experiment tracking.

These extensions are not required for the current public pipeline and should be treated as future work rather than current functionality.
