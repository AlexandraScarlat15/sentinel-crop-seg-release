# Minimal End-to-End Smoke Test

This repository includes a minimal end-to-end smoke test for validating the complete Sentinel-2 crop segmentation pipeline.

The smoke test runs the following stages:

1. Sentinel-2 and EuroCrops-based dataset generation
2. U-Net training on a small generated dataset
3. Model evaluation
4. Prediction and visualization export

The purpose of this script is not to reproduce the full experimental results, but to verify that the installation, command-line interfaces, configuration system, data generation pipeline, training loop, evaluation code, and prediction pipeline work together correctly.

## Requirements

Before running the smoke test, make sure that:

* the package is installed in editable mode;
* the Python environment contains the required dependencies;
* a EuroCrops shapefile is available locally;
* internet access is available for Sentinel-2 STAC search and data retrieval, unless the required Sentinel-2 products are already cached locally;
* enough disk space is available for temporary Sentinel-2 cache files and generated outputs.

Install the package from the project root:

```powershell
python -m pip install -e ".[dev]"
```

For a normal non-development installation, use:

```powershell
python -m pip install -e .
```

If Sentinel-2 product download requires Copernicus Data Space Ecosystem credentials, set them as environment variables before running the smoke test.

On Windows PowerShell:

```powershell
$env:COPERNICUS_USERNAME = "your_username"
$env:COPERNICUS_PASSWORD = "your_password"
```

On Linux or macOS:

```bash
export COPERNICUS_USERNAME="your_username"
export COPERNICUS_PASSWORD="your_password"
```

Do not commit real credentials to the repository.

## Running the smoke test

Run the script from the project root:

```powershell
python scripts\run_e2e_smoke_test.py --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

A smaller validation run can be requested with:

```powershell
python scripts\run_e2e_smoke_test.py `
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp" `
  --run-tag zenodo_smoke_001 `
  --target-patches 8 `
  --limit 4 `
  --device cpu
```

This command requests a small generated dataset, trains U-Net for a minimal validation run, evaluates a small number of samples, and exports predictions.

Alternatively, the EuroCrops shapefile path can be provided through an environment variable.

On Windows PowerShell:

```powershell
$env:EUROCROPS_SHAPEFILE_PATH = "path/to/AT_2021_EC21.shp"
python scripts\run_e2e_smoke_test.py
```

On Linux or macOS:

```bash
export EUROCROPS_SHAPEFILE_PATH="path/to/AT_2021_EC21.shp"
python scripts/run_e2e_smoke_test.py
```

## Useful options

The script exposes several options for small validation runs:

```text
--run-tag              Run tag used by dataset generation.
--target-patches       Number of generated patches requested from the generator.
--limit                Sample limit used for training, evaluation, and prediction.
--device               Device used for evaluation and prediction: cpu, cuda, or auto.
--skip-generate        Skip dataset generation and reuse an existing generated dataset.
--skip-predict         Skip the prediction step.
--no-clean             Do not remove previous E2E outputs before running.
```

Inspect all available options with:

```powershell
python scripts\run_e2e_smoke_test.py --help
```

## Expected outputs

The script creates temporary outputs under:

```text
data/processed/e2e_smoke_dataset/
outputs/e2e_smoke/
```

The generated dataset is saved under:

```text
data/processed/e2e_smoke_dataset/generated/
  images/
  masks/
  images_tif/
  masks_tif/
  patches_log.csv
```

The training, evaluation, and prediction outputs are saved under:

```text
outputs/e2e_smoke/
  unet_generated/
  eval_generated/
  predict_generated/
```

If the smoke test succeeds, the final message should be:

```text
[E2E] Smoke test passed.
```

The generated outputs include:

* generated NumPy image and mask patches;
* generated GeoTIFF image and mask patches;
* patch-level metadata;
* a trained U-Net checkpoint;
* training metrics and summary files;
* evaluation metrics and confusion matrices;
* prediction masks;
* RGB previews;
* color masks;
* visualization panels.

## Example successful validation

A small successful smoke test should confirm that:

```text
dataset generation -> U-Net training -> evaluation -> prediction
```

runs without errors.

For example, a small validation run may generate:

```text
8 NumPy image patches
8 NumPy mask patches
8 GeoTIFF image patches
8 GeoTIFF mask patches
```

and then train, evaluate, and export predictions using a limited number of samples.

The exact number of generated patches depends on the requested `--target-patches`, ROI selection, cloud filtering, target-class filtering, product availability, and local cache state.

## Notes

This is an online smoke test. It depends on the availability of external Sentinel-2 catalog and data services. Failures can occur because of network issues, remote service availability, rate limits, authentication problems, or local cache problems.

The smoke-test outputs are temporary validation artifacts and are intentionally excluded from version control. Do not commit generated files from:

```text
data/raw/
data/processed/
outputs/
```

For full experiments, use the standard modular CLI commands and configuration files instead of this minimal smoke test.
