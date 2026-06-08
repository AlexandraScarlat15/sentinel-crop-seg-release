# Project Showcase

## Project summary

Sentinel Crop Segmentation is a research-oriented software pipeline for agricultural crop segmentation from Sentinel-2 satellite imagery and EuroCrops parcel annotations.

The project combines remote sensing data, geospatial processing, configurable dataset generation, semantic segmentation training, evaluation, prediction export, testing, packaging, and documentation.

The current implementation focuses on:

* Sentinel-2 Level-2A multispectral imagery;
* EuroCrops parcel-level crop annotations;
* configurable dataset generation through YAML files;
* runtime configuration overrides through CLI arguments;
* ROI-based patch extraction;
* grid and hotcell ROI sampling strategies;
* U-Net baseline training;
* Random Forest baseline training;
* evaluation and prediction through installed CLI tools;
* a minimal end-to-end smoke test;
* pytest-based automated tests.

## Main technical contributions

### Configurable dataset generation

The project can generate segmentation-ready image-mask pairs from Sentinel-2 imagery and EuroCrops vector labels.

The generation process is controlled by YAML configuration files. Runtime overrides can be supplied through CLI arguments, which makes local experiments and smoke tests possible without modifying tracked configuration files.

Example:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp"
```

### ROI sampling strategies

The project supports two ROI selection strategies:

* `grid`: broad coverage-based dataset generation;
* `hotcell`: target-class-focused generation for rare or underrepresented crops.

This makes the dataset generation process more flexible than simple random patch extraction.

### Reduced crop taxonomy

Raw EuroCrops labels are mapped to a reduced project taxonomy. This reduces ambiguity between heterogeneous agricultural labels and provides a consistent set of segmentation classes for model training and evaluation.

### U-Net segmentation baseline

The project includes a four-level U-Net baseline for multispectral crop segmentation.

The training setup supports:

* selected Sentinel-2 bands;
* scale-based normalization;
* class weighting;
* combined cross-entropy and Dice loss;
* augmentation;
* learning rate scheduling;
* early stopping;
* checkpoint and metric export.

### Random Forest baseline

A Random Forest baseline is included as a classical machine learning comparison.

It can be trained from manifest-based datasets and evaluated through the same unified evaluation CLI used by the segmentation model.

### Evaluation and prediction pipeline

The project provides unified CLI tools for evaluation and prediction:

```text
sentinel-crop-evaluate
sentinel-crop-predict
```

Evaluation exports summary metrics, per-class metrics, and confusion matrices.

Prediction can export raw predicted masks, color masks, RGB previews, and panel visualizations.

### Minimal end-to-end smoke test

The repository includes a minimal online E2E smoke test:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

The smoke test runs:

```text
dataset generation
-> U-Net training
-> evaluation
-> prediction
```

It is intended to validate that the installed pipeline works end to end. It does not replace full experiments and does not aim to reproduce final research results.

### Package-style usage

The project can be installed locally as a Python package:

```bash
python -m pip install -e ".[dev]"
```

The dataset generation API can be used directly:

```python
from sentinel_crop_seg import generate_dataset

result = generate_dataset(
    config_dir="config",
    run_tag="example_run",
    overrides=[
        "eurocrops.shapefile_path=path/to/AT_2021_EC21.shp",
        "dataset.target.total_patches=100",
    ],
)
```

The main installed CLI tools are:

```text
sentinel-crop-generate
sentinel-crop-convert-pastis
sentinel-crop-convert-dacia5
sentinel-crop-train-unet
sentinel-crop-train-rf
sentinel-crop-evaluate
sentinel-crop-predict
```

## Software engineering practices

The project includes several software engineering improvements:

* modular source code organization;
* public package API;
* installed command-line entrypoints;
* editable package configuration through `pyproject.toml`;
* YAML-based configuration;
* runtime configuration overrides;
* pytest-based test suite;
* minimal E2E smoke test;
* GitHub Actions CI support;
* README documentation;
* dedicated architecture and workflow documentation;
* strict exclusion of generated datasets, raw data, checkpoints, and output artifacts from version control.

## What this project demonstrates

This project demonstrates the ability to:

* design a real geospatial data processing pipeline;
* work with Sentinel-2 satellite imagery and vector parcel annotations;
* transform heterogeneous crop labels into a consistent segmentation taxonomy;
* generate aligned multispectral image-mask patches;
* train and evaluate semantic segmentation models;
* include a classical machine learning baseline;
* export quantitative metrics and qualitative prediction visualizations;
* structure a research project as maintainable software;
* add packaging, tests, documentation, and reproducibility checks.

## Current limitations

The current implementation is research-oriented and has several limitations:

* the main generation backend currently supports Sentinel-2 + EuroCrops;
* full dataset generation depends on external data availability, internet access, STAC service availability, and local cache state;
* large-scale training requires external compute resources;
* generated datasets, Sentinel-2 products, checkpoints, and outputs are not included in the repository;
* external dataset conversion utilities require the corresponding external datasets to be available locally.

## Future directions

Potential future improvements include:

* adding additional label sources;
* improving rare-class sampling;
* adding more model architectures;
* expanding external benchmark workflows;
* improving experiment tracking;
* adding richer result visualizations;
* preparing clearer release metadata and citation files for archived versions.

These are future directions and are not required for the current public pipeline.
