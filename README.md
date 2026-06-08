# Sentinel Crop Segmentation

Research-oriented pipeline for generating Sentinel-2 crop segmentation datasets from EuroCrops parcel annotations, training baseline crop segmentation models, and evaluating model predictions.

The project focuses on a Sentinel-2 + EuroCrops workflow:

```text
EuroCrops parcel annotations
+ Sentinel-2 Level-2A imagery
-> image-mask patch generation
-> model training
-> evaluation
-> prediction export and visualization
```

The repository is intended to support reproducible research code. Large raw datasets, generated datasets, Sentinel-2 products, model checkpoints, and experiment outputs are not included in the repository.

## Implemented features

The current implementation includes:

* Sentinel-2 / EuroCrops dataset generation;
* EuroCrops crop-label harmonization into a reduced crop taxonomy;
* grid-based and hotcell-based ROI selection;
* Sentinel-2 STAC search and local product caching;
* export of aligned image-mask patches as NumPy arrays and GeoTIFF files;
* U-Net training on root-folder or manifest-based datasets;
* Random Forest baseline training;
* unified evaluation for U-Net and Random Forest models;
* prediction export with optional color masks and visualization panels;
* command-line configuration overrides through `--override key=value`;
* a minimal end-to-end smoke test for validating the full pipeline.

## Repository structure

```text
sentinel_crop_seg/        Public package API and installed CLI entrypoints
src/                      Core implementation
  catalog/                Sentinel-2 catalog search
  data/                   EuroCrops and Sentinel-2 loading/cache utilities
  pipeline/               Dataset generation pipeline
  selection/              ROI selection strategies
  training/               Dataset loaders, U-Net model, loss, training logic
  utils/                  Spatial and helper utilities
scripts/                  Public utility scripts and E2E smoke test
config/                   YAML configuration files
docs/                     Additional documentation
tests/                    Unit and integration-style tests
```

## Installation

The package requires Python 3.11 or newer.

From the repository root, create and activate a virtual environment.

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

For a normal installation without development dependencies:

```bash
python -m pip install -e .
```

The main Python dependencies are declared in `pyproject.toml`. The repository also includes `requirements.txt` for users who prefer a pinned environment.

After installation, the public command-line entrypoints should be available:

```bash
sentinel-crop-generate --help
sentinel-crop-train-unet --help
sentinel-crop-train-rf --help
sentinel-crop-evaluate --help
sentinel-crop-predict --help
```

## Data requirements

This repository does not include external geospatial datasets or generated artifacts.

To run the full data generation pipeline, you need:

* EuroCrops shapefiles available locally;
* internet access for Sentinel-2 STAC search and product retrieval, unless the required Sentinel-2 products are already cached locally;
* enough local disk space for Sentinel-2 SAFE products, generated patches, checkpoints, and outputs.

The EuroCrops shapefile path is configured in `config/eurocrops.yaml`:

```yaml
eurocrops:
  shapefile_path: "data/external/eurocrops/AT_2021/AT_2021_EC21.shp"
  crop_column: "EC_trans_n"
```

For local runs, you can either edit the YAML file locally or use CLI overrides without changing tracked configuration files.

Example:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp"
```

## Environment variables

If Sentinel-2 product download requires Copernicus Data Space Ecosystem credentials, set them as environment variables before running the generation pipeline.

On Linux/macOS:

```bash
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"
```

On Windows PowerShell:

```powershell
$env:COPERNICUS_USERNAME = "your_username"
$env:COPERNICUS_PASSWORD = "your_password"
```

The repository includes `.env.example` as a template. Do not commit real credentials.

## Configuration

Dataset generation uses a configuration directory containing YAML files. The default configuration directory is:

```text
config/
```

Important configuration files include:

```text
config/generation.yaml
config/paths.yaml
config/eurocrops.yaml
config/catalog.yaml
config/dataset.yaml
config/classes.yaml
config/sentinel.yaml
config/train_unet.yaml
```

`paths.dataset_output_dir` is used by dataset generation to write newly generated image-mask patches.

`paths.dataset_root` is a generic dataset root used by training, evaluation, prediction, and smoke-test workflows when working with already generated datasets.

The implemented generation source is:

```yaml
generation:
  satellite: "sentinel2"
  label_source: "eurocrops"
```

The project supports runtime configuration overrides through repeated `--override` arguments:

```bash
--override dataset.target.total_patches=8
--override catalog.stac.limit=5
--override paths.dataset_output_dir="data/processed/example/generated"
```

Override values are parsed using YAML syntax, so values such as `true`, `false`, `null`, integers, floats, and lists are converted automatically.

## Target classes

The reduced taxonomy currently used by the project includes:

```text
0   background
1   maize
2   rapeseed
3   wheat
4   barley
5   oats
6   sunflower
7   sugar beet
8   soybean
9   potato
10  pasture
```

The mapping from raw EuroCrops labels to these classes is configured in `config/classes.yaml`.

## Dataset generation

Run dataset generation through the installed CLI:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag experiment_01
```

A local EuroCrops path can be supplied without modifying committed configuration files:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag experiment_01 \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp"
```

A smaller generation run can be requested with additional overrides:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag small_test \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp" \
  --override paths.dataset_output_dir="data/processed/small_test/generated" \
  --override dataset.target.total_patches=100 \
  --override dataset.max_patches_per_roi=20 \
  --override catalog.stac.limit=5
```

The generated dataset is saved under the configured `paths.dataset_output_dir` and typically contains:

```text
generated/
  images/
  masks/
  images_tif/
  masks_tif/
  patches_log.csv
```

### ROI strategies

The dataset generation pipeline supports two ROI selection strategies.

Grid strategy:

```yaml
dataset:
  roi:
    strategy: "grid"
```

The grid strategy selects fixed-size regions based on crop polygon coverage.

Hotcell strategy:

```yaml
dataset:
  target:
    class_id: 8
    min_target_ratio: 0.05
    min_target_pixels: 800
    total_patches: 10000

  roi:
    strategy: "hotcell"
```

The hotcell strategy focuses ROI selection around a target crop class and requires `dataset.target.class_id` to be set.

More details are available in:

```text
docs/dataset_generation.md
```

## U-Net training

Train the U-Net baseline with:

```bash
sentinel-crop-train-unet \
  --config config/train_unet.yaml
```

The training configuration controls:

* dataset source;
* manifest or root-folder loading;
* selected Sentinel-2 bands;
* normalization;
* model size;
* dropout;
* class weights;
* augmentation;
* learning rate;
* scheduler;
* early stopping;
* output directory.

Training outputs are saved under the configured `paths.out_dir`, for example:

```text
outputs/experiments/<experiment_name>/
  best.pt
  last.pt
  last_state.pt
  training_metrics.csv
  training_summary.json
  model_info.json
  metrics.jsonl
  resolved_config.json
```

For small test runs, use CLI overrides instead of editing the training config:

```bash
sentinel-crop-train-unet \
  --config config/train_unet.yaml \
  --override paths.manifest_csv=null \
  --override paths.train_root="data/processed/example/generated" \
  --override paths.val_root="data/processed/example/generated" \
  --override paths.out_dir="outputs/example_unet" \
  --override data.train_limit=4 \
  --override data.val_limit=4 \
  --override training.epochs=1 \
  --override training.batch_size=1 \
  --override loader.num_workers=0 \
  --override loader.prefetch_factor=null
```

## Random Forest baseline

The repository also provides a Random Forest baseline CLI:

```bash
sentinel-crop-train-rf --help
```

A typical run uses a manifest CSV and writes the model and associated outputs to an experiment directory:

```bash
sentinel-crop-train-rf \
  --manifest data/processed/sentinel_eurocrops/indexes/manifest.csv \
  --out-dir outputs/experiments/random_forest_spectral \
  --feature-mode spectral
```

## Evaluation

Use the unified evaluation CLI:

```bash
sentinel-crop-evaluate --help
```

Example for evaluating a U-Net checkpoint on a root-folder dataset:

```bash
sentinel-crop-evaluate \
  --model-type unet \
  --data-source root \
  --checkpoint outputs/experiments/example/best.pt \
  --dataset-root data/processed/sentinel_eurocrops/generated \
  --sample-format npy \
  --output-dir outputs/evaluation/example_root \
  --batch-size 1 \
  --device cpu
```

Example for evaluating a U-Net checkpoint on a manifest split:

```bash
sentinel-crop-evaluate \
  --model-type unet \
  --data-source manifest \
  --checkpoint outputs/experiments/example/best.pt \
  --manifest-csv data/processed/sentinel_eurocrops/indexes/manifest.csv \
  --split test \
  --output-dir outputs/evaluation/example_manifest \
  --batch-size 1 \
  --device cpu
```

The evaluation output includes:

```text
summary.json
summary.txt
metrics_per_class.csv
confusion_matrix_counts.npy
confusion_matrix_row_norm.npy
evaluated_samples.txt
```

## Prediction and visualization

Use the prediction CLI:

```bash
sentinel-crop-predict --help
```

Example:

```bash
sentinel-crop-predict \
  --model-type unet \
  --data-source root \
  --checkpoint outputs/experiments/example/best.pt \
  --dataset-root data/processed/sentinel_eurocrops/generated \
  --sample-format npy \
  --output-dir outputs/predictions/example_root \
  --batch-size 1 \
  --device cpu \
  --save-color \
  --save-panels
```

Prediction outputs can include:

```text
prediction_samples.csv
prediction_summary.json
pred_masks_npy/
pred_masks_color/
gt_masks_color/
rgb/
panels/
```

## Minimal end-to-end smoke test

A minimal end-to-end smoke test is provided for validating the complete installed pipeline:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

A smaller validation run can be requested with:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp" \
  --run-tag zenodo_smoke_001 \
  --target-patches 8 \
  --limit 4 \
  --device cpu
```

The smoke test runs:

```text
dataset generation
-> U-Net training
-> evaluation
-> prediction
```

It uses the default project configuration plus temporary runtime overrides. It does not require separate copied smoke-test configuration files.

The test creates temporary outputs under:

```text
data/processed/e2e_smoke_dataset/
outputs/e2e_smoke/
```

The generated smoke-test dataset contains:

```text
data/processed/e2e_smoke_dataset/generated/
  images/
  masks/
  images_tif/
  masks_tif/
  patches_log.csv
```

The `outputs/e2e_smoke/` directory contains the temporary training, evaluation, and prediction outputs produced by the smoke test.

The EuroCrops shapefile can also be provided through an environment variable:

```bash
export EUROCROPS_SHAPEFILE_PATH="path/to/AT_2021_EC21.shp"
python scripts/run_e2e_smoke_test.py
```

On Windows PowerShell:

```powershell
$env:EUROCROPS_SHAPEFILE_PATH = "path/to/AT_2021_EC21.shp"
python scripts\run_e2e_smoke_test.py
```

The smoke test is an online validation step because it uses Sentinel-2 catalog search and data retrieval. It may fail if external services, internet access, authentication, or local cache state are unavailable.

Additional details are available in:

```text
docs/e2e_smoke_test.md
```

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

The tests cover important parts of the codebase, including:

* configuration loading and overrides;
* package entrypoints;
* public CLI availability;
* dataset loading;
* manifest loading;
* model forward pass;
* loss behavior;
* evaluation utilities;
* prediction utilities;
* dataset generation API behavior.

The full online dataset generation process is not run as part of the normal unit test suite.

## Documentation

Additional documentation is available under `docs/`:

```text
docs/architecture.md
docs/dataset_generation.md
docs/training_and_evaluation.md
docs/e2e_smoke_test.md
```

Some documents describe implementation details and research workflow decisions. The README is intended to reflect the current public usage path.

## Files not included in the repository

The following are intentionally excluded from version control:

```text
data/raw/
data/external/
data/processed/
outputs/
checkpoints/
models/
Sentinel-2 SAFE products
generated datasets
model checkpoints
evaluation outputs
prediction outputs
```

This keeps the repository focused on source code, configuration, documentation, and tests.

## Validation status

Before preparing the public archive, the following validation checks were run successfully:

```text
python -m pytest -q
```

and a minimal end-to-end smoke test was run successfully on a small generated dataset:

```text
dataset generation -> U-Net training -> evaluation -> prediction
```

These checks validate the main public pipeline, but they do not replace full-scale experimental reproduction.

## License

This project is released under the MIT License. See `LICENSE.md` for details.

## Citation

If this repository is archived through Zenodo, cite the archived release DOI associated with the version used in your work.

A formal citation file can be added once the release metadata and DOI are finalized.
