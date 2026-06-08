# Dataset Generation

## Purpose

The dataset generation pipeline creates segmentation-ready image-mask pairs from Sentinel-2 Level-2A imagery and EuroCrops parcel annotations.

The currently implemented generation backend is:

```yaml
generation:
  satellite: "sentinel2"
  label_source: "eurocrops"
```

This means that Sentinel-2 provides the multispectral imagery, while EuroCrops provides the vector crop parcel annotations used to create pixel-level segmentation masks.

The generated samples can be used by the training, evaluation, and prediction pipelines.

## Inputs

The generation pipeline requires:

* a configuration directory containing YAML files;
* a local EuroCrops shapefile;
* access to Sentinel-2 Level-2A products through STAC search and download, or an existing local Sentinel-2 cache;
* class mapping rules;
* ROI selection settings;
* patch extraction and filtering settings.

The repository does not include EuroCrops source files, Sentinel-2 SAFE products, generated datasets, or output artifacts.

## Configuration directory

The default configuration directory is:

```text
config/
```

A typical generation configuration is split across files such as:

```text
config/
  generation.yaml
  paths.yaml
  eurocrops.yaml
  catalog.yaml
  dataset.yaml
  classes.yaml
  sentinel.yaml
```

The configuration files are loaded and merged into a single configuration dictionary.

Runtime overrides can be provided through repeated `--override key=value` arguments. This is useful for local paths, small test runs, or temporary output directories without modifying tracked configuration files.

Example:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp" \
  --override paths.dataset_output_dir="data/processed/example/generated" \
  --override dataset.target.total_patches=100
```

Override values are parsed using YAML syntax, so values such as `true`, `false`, `null`, integers, floats, and lists are converted automatically.

## Main configuration sections

### Generation source

```yaml
generation:
  satellite: "sentinel2"
  label_source: "eurocrops"
```

The code currently validates this generation source explicitly. Other satellite or label backends are not part of the current implementation.

### Paths

The path configuration defines where local caches and generated outputs are located.

Example:

```yaml
paths:
  sentinel_data_root: "data/raw/sentinel"
  dataset_output_dir: "data/processed/sentinel_eurocrops/generated"
```

`sentinel_data_root` is the local root used for Sentinel-2 cache files and generation state files.

`dataset_output_dir` is the directory where generated image-mask samples are written.

For local runs, the EuroCrops shapefile path can be supplied through an override instead of editing committed configuration files:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
  --override eurocrops.shapefile_path="path/to/AT_2021_EC21.shp"
```

### EuroCrops source

```yaml
eurocrops:
  shapefile_path: "data/external/eurocrops/AT_2021/AT_2021_EC21.shp"
  crop_column: "EC_trans_n"
```

The crop column is used for mapping raw EuroCrops crop labels to the reduced class taxonomy defined in `config/classes.yaml`.

For public or shared runs, it is recommended to keep only a generic relative path in the tracked configuration and provide the real local path through `--override`.

### Sentinel-2 catalog search

```yaml
catalog:
  stac:
    start_date: "2021-04-01T00:00:00Z"
    end_date: "2021-09-30T23:59:59Z"
    max_cloud_cover: 10
    limit: 50
```

These settings control the Sentinel-2 product search window, the maximum accepted cloud cover, and the STAC query limit.

The generation pipeline uses STAC search to find Sentinel-2 products intersecting selected ROIs. Product availability and download behavior depend on external services, network connectivity, and local cache state.

The delay between product downloads is controlled at runtime through the `--sleep-seconds` CLI argument, not through the YAML configuration.

Example:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
  --sleep-seconds 1.5
```

### Patch settings

```yaml
dataset:
  patch:
    size: 256
    stride: 128

  max_patches_per_roi: 1000
```

Patch size controls the spatial size of each extracted sample. Stride controls overlap between candidate patches.

`max_patches_per_roi` limits how many accepted patches can be extracted from a single ROI.

### Filtering settings

The dataset configuration also controls foreground, nodata, cloud, and target-class filtering.

Example:

```yaml
dataset:
  min_foreground_ratio: 0.20
  max_nodata_fraction: 0.05

  max_cloud_fraction: 0.01
  max_cloud_on_foreground_fraction: 0.005
  bad_scl_classes: [0, 1, 2, 3, 7, 8, 9, 10, 11]
  cloud_dilate_radius: 2
```

These thresholds are applied during patch extraction to reject unsuitable samples.

## ROI sampling strategies

ROI selection decides where the pipeline should search for image-mask patches.

The project currently supports two ROI strategies:

* `grid`
* `hotcell`

### Grid strategy

```yaml
dataset:
  roi:
    strategy: "grid"
    size_km: 30
    top_n_rois: 200
    min_polygon_count: 20
    min_total_area_m2: 5000000
    min_coverage: 0.1
```

The grid strategy builds fixed-size regions over the EuroCrops study area and keeps regions that satisfy coverage and polygon-count thresholds.

This strategy is useful when the goal is to generate a broader dataset across the available study area.

### Hotcell strategy

```yaml
dataset:
  target:
    class_id: 8
    min_target_ratio: 0.05
    min_target_pixels: 800
    total_patches: 10000

  roi:
    strategy: "hotcell"
    size_km: 30
    hotcell:
      cell_km: 5
      top_k_cells: 2000
      rois_from_cells: 300
      min_target_area_m2: 200000
      class_id_column: "class_id"
```

The hotcell strategy focuses ROI selection around a selected target crop class.

This is useful when a crop class is rare or underrepresented. For example, if soybean is class `8`, hotcell selection can focus generation around areas where soybean polygons are present.

The hotcell strategy requires:

```yaml
dataset:
  target:
    class_id: <target class id>
```

## Target classes

The reduced taxonomy currently used by the project is defined in `config/classes.yaml`.

The current class IDs are:

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

Raw EuroCrops labels are mapped to this taxonomy before rasterization.

## Output format

The generated dataset is saved under `paths.dataset_output_dir`.

A typical output structure is:

```text
dataset_output_dir/
  images/
    sample_000000.npy
    sample_000001.npy
  masks/
    sample_000000.npy
    sample_000001.npy
  images_tif/
    sample_000000.tif
    sample_000001.tif
  masks_tif/
    sample_000000.tif
    sample_000001.tif
  patches_log.csv
```

Each image sample is a multispectral tensor stored in `C x H x W` format.

Each mask sample is a 2D segmentation mask in `H x W` format, with integer class IDs.

The NumPy outputs are used by the training and evaluation code. The GeoTIFF outputs preserve geospatial information for inspection and external analysis.

## Command-line usage

The installed command-line entrypoint is:

```bash
sentinel-crop-generate --config-dir config --run-tag example_run
```

A local EuroCrops shapefile path can be provided through `--override`:

```bash
sentinel-crop-generate \
  --config-dir config \
  --run-tag example_run \
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

## Python API usage

After installing the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

the generation pipeline can be called programmatically:

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

print(result.output_dir)
print(result.total_samples)
print(result.selected_rois)
print(result.processed_products)
print(result.strategy)
print(result.satellite)
print(result.label_source)
```

## Processed state

The generation pipeline keeps a processed-state file for each strategy and run tag:

```text
processed_<strategy>_<run_tag>.json
```

This prevents the same ROI/product pair from being processed repeatedly during repeated runs with the same `run_tag`.

Examples:

```text
processed_grid_example_run.json
processed_hotcell_soybean_run.json
```

The processed-state file is stored under `paths.sentinel_data_root`.

For fresh test runs, use a new `--run-tag` or remove the corresponding processed-state file.

## Minimal E2E generation check

The repository includes a minimal end-to-end smoke test that runs dataset generation, training, evaluation, and prediction:

```bash
python scripts/run_e2e_smoke_test.py \
  --eurocrops-shapefile "path/to/AT_2021_EC21.shp"
```

This script uses the normal configuration files plus runtime overrides. It is intended to verify that the installed pipeline works end to end on a small local run.

More details are available in:

```text
docs/e2e_smoke_test.md
```

## Practical notes

The full generation process can depend on:

* local EuroCrops files;
* Sentinel-2 STAC availability;
* internet connectivity;
* local Sentinel-2 cache state;
* disk space;
* cloud filtering thresholds;
* selected ROI strategy;
* product availability for the selected date range.

For this reason, automated unit tests do not run the full external-data pipeline. Instead, tests cover reusable code paths, loaders, mappings, model components, package API behavior, configuration overrides, and CLI availability.
