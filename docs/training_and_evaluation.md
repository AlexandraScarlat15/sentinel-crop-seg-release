# Training, Evaluation, and Prediction

## Purpose

This document summarizes how the project trains, evaluates, and exports predictions for crop segmentation models.

The current training and evaluation workflow supports:

* U-Net semantic segmentation training;
* Random Forest baseline training;
* evaluation on root-folder datasets or manifest-based datasets;
* prediction export with optional color masks and visualization panels.

The main public entrypoints are installed as command-line tools:

```text
sentinel-crop-train-unet
sentinel-crop-train-rf
sentinel-crop-evaluate
sentinel-crop-predict
```

## Dataset formats

The training and evaluation code can work with two dataset access patterns:

1. root-folder datasets;
2. manifest-based datasets.

## Root-folder datasets

A root-folder dataset contains image and mask folders under the same dataset root.

Typical structure:

```text
dataset_root/
  images/
    sample_000000.npy
    sample_000001.npy
  masks/
    sample_000000.npy
    sample_000001.npy
```

For generated Sentinel-2 / EuroCrops datasets, the root folder usually corresponds to the configured `paths.dataset_output_dir`.

Example:

```text
data/processed/sentinel_eurocrops/generated/
  images/
  masks/
  images_tif/
  masks_tif/
  patches_log.csv
```

The NumPy `images/` and `masks/` folders are used for model training and evaluation. The GeoTIFF folders are useful for geospatial inspection and external analysis.

## Manifest-based datasets

A manifest CSV can explicitly define the dataset split and sample paths.

A manifest-based setup is useful when:

* train, validation, and test splits are stored in a single index file;
* samples come from multiple datasets;
* experiments require reproducible sample selection;
* metadata such as dataset name, sample ID, or split must be tracked.

The U-Net training pipeline supports both root-folder and manifest-based loading through the training YAML configuration.

## U-Net training

The U-Net baseline is trained through:

```bash
sentinel-crop-train-unet --config config/train_unet.yaml
```

The training configuration controls:

* experiment name;
* dataset roots or manifest CSV;
* output directory;
* selected Sentinel-2 bands;
* normalization mode;
* input channel count;
* number of classes;
* model base width;
* dropout settings;
* loss configuration;
* class weights;
* augmentation;
* learning rate;
* scheduler behavior;
* early stopping;
* dataloader settings.

## Training configuration

A typical training configuration includes sections such as:

```yaml
experiment:
  name: "unet_experiment"

paths:
  manifest_csv: null
  train_root: "data/processed/sentinel_eurocrops/generated"
  val_root: "data/processed/sentinel_eurocrops/generated"
  test_root: "data/processed/sentinel_eurocrops/generated"
  out_dir: "outputs/experiments/unet_experiment"
  resume_ckpt: null

data:
  train_limit: 0
  val_limit: 0
  in_channels: 10
  num_classes: 11
  band_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  normalize_mode: "scale_10000"
  scale_divisor: 10000.0
  sample_format: "npy"
```

If `paths.manifest_csv` is set, training uses the manifest-based dataset loader.

If `paths.manifest_csv` is `null`, training uses the root-folder dataset loader.

## Runtime configuration overrides

Training supports runtime configuration overrides through repeated `--override key=value` arguments.

This makes it possible to run small experiments or smoke tests without editing tracked YAML files.

Example:

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

Override values are parsed using YAML syntax, so values such as `true`, `false`, `null`, integers, floats, and lists are converted automatically.

## Model

The current U-Net baseline is:

```text
UNet4LevelsGN
```

It is a four-level U-Net variant using group normalization. It is used as a practical semantic segmentation baseline for Sentinel-2 crop mapping experiments.

## Loss function

The U-Net training pipeline uses a combined cross-entropy and Dice loss:

```text
CrossEntropyDiceLoss
```

The loss configuration supports:

* class weights;
* optional ignore index;
* configurable Dice contribution;
* optional exclusion of the background class from the Dice term.

This is useful for crop segmentation because crop classes are often imbalanced.

## Training outputs

A training run saves outputs under the configured `paths.out_dir`.

Typical output structure:

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

The most important outputs are:

* `best.pt`: model checkpoint from the best validation mIoU epoch;
* `last.pt`: model checkpoint from the last completed epoch;
* `last_state.pt`: full training state for resuming training;
* `training_metrics.csv`: epoch-level metrics;
* `training_summary.json`: compact summary of the run;
* `resolved_config.json`: final resolved configuration after applying overrides.

## Random Forest baseline

The project also includes a Random Forest baseline.

Inspect the CLI options with:

```bash
sentinel-crop-train-rf --help
```

Example run:

```bash
sentinel-crop-train-rf \
  --manifest data/processed/sentinel_eurocrops/indexes/manifest.csv \
  --out-dir outputs/experiments/random_forest_spectral \
  --feature-mode spectral
```

The Random Forest baseline is useful as a classical machine learning comparison against the U-Net segmentation model.

## Evaluation

Model evaluation is handled by the unified evaluation CLI:

```bash
sentinel-crop-evaluate --help
```

The evaluation pipeline supports:

* U-Net checkpoints;
* Random Forest checkpoints;
* root-folder datasets;
* manifest-based datasets.

## Evaluating a U-Net checkpoint on a root-folder dataset

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

## Evaluating a U-Net checkpoint on a manifest split

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

## Evaluating a Random Forest model

```bash
sentinel-crop-evaluate \
  --model-type random_forest \
  --data-source manifest \
  --checkpoint outputs/experiments/random_forest_spectral/model.joblib \
  --manifest-csv data/processed/sentinel_eurocrops/indexes/manifest.csv \
  --split test \
  --feature-mode spectral \
  --output-dir outputs/evaluation/random_forest_spectral
```

## Evaluation outputs

The evaluation output directory typically contains:

```text
summary.json
summary.txt
metrics_per_class.csv
confusion_matrix_counts.npy
confusion_matrix_row_norm.npy
evaluated_samples.txt
```

The evaluation metrics include:

* mean IoU over all classes;
* mean IoU over classes present in the evaluated samples;
* pixel accuracy;
* mean class accuracy;
* per-class IoU;
* per-class accuracy;
* confusion matrices.

## Prediction export

Predictions are exported with:

```bash
sentinel-crop-predict --help
```

Example for exporting U-Net predictions:

```bash
sentinel-crop-predict \
  --model-type unet \
  --data-source root \
  --checkpoint outputs/experiments/example/best.pt \
  --dataset-root data/processed/sentinel_eurocrops/generated \
  --sample-format npy \
  --output-dir outputs/predictions/example_root \
  --batch-size 1 \
  --device cpu
```

## Prediction visualizations

Color masks and panel visualizations can be exported with:

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

These outputs are useful for qualitative inspection of model behavior.

## Minimal end-to-end smoke test

The repository provides a minimal end-to-end smoke test that runs:

```text
dataset generation
-> U-Net training
-> evaluation
-> prediction
```

Run it with:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

The smoke test uses the normal project configuration plus runtime overrides. It is intended to validate the installed pipeline, not to reproduce full experimental results.

Additional details are available in:

```text
docs/e2e_smoke_test.md
```

## Testing strategy

The full training process is not run inside the normal unit test suite because it is computationally expensive and depends on external data.

Instead, the test suite checks smaller but important components, including:

* configuration loading and overrides;
* public package imports;
* installed CLI entrypoints;
* CLI help availability;
* dataset loading;
* manifest dataset loading;
* model forward pass;
* loss behavior;
* evaluation utilities;
* prediction utilities;
* dataset generation API behavior.

Run the tests with:

```bash
python -m pytest -q
```

## Practical notes

Training and evaluation outputs are intentionally excluded from version control.

Before publishing or archiving the repository, avoid committing:

```text
data/
outputs/
checkpoints/
models/
*.pt
*.joblib
*.npy
*.tif
```

These files should be regenerated or stored separately from the source-code repository.
