# Project Diagrams

This document contains project-specific diagrams for the Sentinel Crop Segmentation project.

The diagrams are written in Mermaid so they can be rendered directly by GitHub Markdown and exported later as PNG/SVG figures for documentation, thesis writing, or presentations.

---

## 1. High-level project architecture

This diagram shows the main parts of the project without implementation-level details.

```mermaid
flowchart LR
    CFG["Configuration files<br/>config/*.yaml<br/>+ CLI overrides"]

    subgraph Input["Input data"]
        EC["EuroCrops<br/>parcel crop annotations"]
        S2["Sentinel-2 Level-2A<br/>multispectral imagery"]
    end

    GEN["Dataset generation<br/>sentinel-crop-generate<br/>generate_dataset()"]

    subgraph OutputDataset["Generated dataset"]
        IMG["images/*.npy<br/>multispectral patches"]
        MSK["masks/*.npy<br/>segmentation masks"]
        GTIF["images_tif/ + masks_tif/<br/>GeoTIFF outputs"]
        LOG["patches_log.csv<br/>metadata"]
    end

    TRAIN["Model training<br/>sentinel-crop-train-unet<br/>sentinel-crop-train-rf"]
    CKPT["Model artifacts<br/>best.pt / last.pt<br/>model.joblib"]

    subgraph Results["Evaluation and prediction"]
        EVAL["Evaluation<br/>sentinel-crop-evaluate<br/>metrics + confusion matrices"]
        PRED["Prediction<br/>sentinel-crop-predict<br/>masks + visualization panels"]
        E2E["E2E smoke test<br/>run_e2e_smoke_test.py"]
    end

    CFG --> GEN
    EC --> GEN
    S2 --> GEN

    GEN --> IMG
    GEN --> MSK
    GEN --> GTIF
    GEN --> LOG

    IMG --> TRAIN
    MSK --> TRAIN
    TRAIN --> CKPT

    CKPT --> EVAL
    CKPT --> PRED
    GEN --> E2E
    TRAIN --> E2E
    EVAL --> E2E
    PRED --> E2E
```

### Explanation

The project starts from Sentinel-2 imagery and EuroCrops parcel annotations. Configuration files and runtime overrides define paths, class mappings, ROI strategy, Sentinel search settings, and training parameters.

The dataset generation pipeline produces aligned image-mask pairs saved as NumPy files and GeoTIFF files. The NumPy samples are used for model training, evaluation, and prediction. The GeoTIFF outputs are useful for inspection and geospatial analysis.

---

## 2. Dataset generation pipeline

This diagram focuses on the dataset generation stage.

```mermaid
flowchart TB
    CFG["Load YAML configuration<br/>load_config(config_dir)"]
    OVR["Apply optional overrides<br/>--override key=value"]

    subgraph EuroCropsSide["EuroCrops branch"]
        EC1["Read EuroCrops shapefile"]
        EC2["Map crop names<br/>to project class IDs"]
        EC3["Rasterize parcels<br/>into masks"]
    end

    subgraph ROISide["ROI branch"]
        ROI1["Select ROI strategy<br/>grid or hotcell"]
        ROI2["Generate candidate ROIs"]
    end

    subgraph SentinelSide["Sentinel-2 branch"]
        S1["Convert ROI to WGS84"]
        S2["Search Sentinel-2 products<br/>STAC catalog"]
        S3["Check EuroCrops overlap"]
        S4["Use cache / prepare SAFE product"]
        S5["Load selected Sentinel-2 bands"]
    end

    ENGINE["DataEngine.save_dataset()"]

    OUT1["Save image patch<br/>images/*.npy"]
    OUT2["Save mask patch<br/>masks/*.npy"]
    OUT3["Save GeoTIFF patch<br/>images_tif/ + masks_tif/"]
    OUT4["Save metadata<br/>patches_log.csv"]
    STATE["Update processed state<br/>processed_strategy_runTag.json"]

    CFG --> OVR
    OVR --> EC1
    OVR --> ROI1
    OVR --> S2

    EC1 --> EC2
    EC2 --> EC3

    ROI1 --> ROI2
    ROI2 --> S1

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5

    EC3 --> ENGINE
    ROI2 --> ENGINE
    S5 --> ENGINE

    ENGINE --> OUT1
    ENGINE --> OUT2
    ENGINE --> OUT3
    ENGINE --> OUT4
    ENGINE --> STATE
```

### Explanation

The dataset generation pipeline combines three branches.

The EuroCrops branch loads parcel annotations, maps heterogeneous crop names to the reduced project taxonomy, and generates raster masks. The ROI branch decides where samples should be extracted, using either the grid strategy or the hotcell strategy. The Sentinel-2 branch searches satellite products, uses the cache to avoid unnecessary repeated downloads, and loads the required spectral bands.

The `DataEngine` combines these inputs and saves aligned image-mask patches.

---

## 3. Main software components

This diagram shows the main components used by `generate_dataset()` and the public CLI tools.

```mermaid
flowchart TB
    CLI["Installed CLI entrypoints<br/>sentinel-crop-*"]
    API["Public package API<br/>sentinel_crop_seg"]
    GENAPI["generate_dataset()<br/>src/pipeline/dataset_generation.py"]

    subgraph ConfigLayer["Configuration"]
        LOADCFG["load_config()"]
        OVR["apply_overrides()"]
        YAML["config/*.yaml"]
    end

    subgraph LabelLayer["Label processing"]
        ECLOADER["EuroCropsMaskLoader"]
        CLASSMAP["map_crop_to_class_id()"]
        MASKGEN["generate_mask()"]
    end

    subgraph RoiLayer["ROI selection"]
        CONTEXT["RoiSelectionContext"]
        GRID["GridCoverageSelector"]
        HOTCELL["HotcellSelector"]
    end

    subgraph SentinelLayer["Sentinel-2 processing"]
        SEARCHER["STACSentinelCatalogSearcher"]
        CACHE["SentinelCacheManager"]
        SAFELOADER["SentinelSafeDataLoader"]
        SPATIAL["tile_intersects_eurocrops()"]
    end

    subgraph PatchLayer["Patch extraction"]
        ENGINE["DataEngine"]
        SAVE["save_dataset()"]
    end

    subgraph TrainingLayer["Training"]
        ROOTDS["SegmentationPatchDataset"]
        MANIFESTDS["ManifestSegmentationDataset"]
        MODEL["UNet4LevelsGN"]
        LOSS["CrossEntropyDiceLoss"]
        RF["Random Forest baseline"]
    end

    subgraph EvalPredLayer["Evaluation and prediction"]
        EVAL["sentinel-crop-evaluate"]
        PRED["sentinel-crop-predict"]
    end

    YAML --> LOADCFG
    LOADCFG --> OVR
    OVR --> GENAPI

    CLI --> GENAPI
    API --> GENAPI

    GENAPI --> ECLOADER
    ECLOADER --> CLASSMAP
    ECLOADER --> MASKGEN

    GENAPI --> CONTEXT
    CONTEXT --> GRID
    CONTEXT --> HOTCELL

    GENAPI --> SEARCHER
    SEARCHER --> SPATIAL
    SEARCHER --> CACHE
    CACHE --> SAFELOADER

    MASKGEN --> ENGINE
    GRID --> ENGINE
    HOTCELL --> ENGINE
    SAFELOADER --> ENGINE

    ENGINE --> SAVE
    SAVE --> ROOTDS
    SAVE --> MANIFESTDS

    ROOTDS --> MODEL
    MANIFESTDS --> MODEL
    MODEL --> LOSS
    MANIFESTDS --> RF

    MODEL --> EVAL
    RF --> EVAL
    MODEL --> PRED
```

### Explanation

`generate_dataset()` is the orchestration function for dataset generation. It coordinates specialized modules instead of doing all work directly.

`EuroCropsMaskLoader` handles label mapping and mask generation. `GridCoverageSelector` and `HotcellSelector` choose regions of interest. `STACSentinelCatalogSearcher`, `SentinelCacheManager`, and `SentinelSafeDataLoader` handle Sentinel-2 product discovery, caching, and band loading. `DataEngine` extracts final aligned image-mask patches.

The generated dataset is then used by the training and evaluation components.

---

## 4. Training, evaluation, and prediction flow

```mermaid
flowchart LR
    DATA["Generated dataset<br/>images + masks"]
    DS["Dataset loaders<br/>root-folder or manifest"]
    UNET["U-Net training<br/>sentinel-crop-train-unet"]
    RF["Random Forest baseline<br/>sentinel-crop-train-rf"]

    UCKPT["U-Net checkpoints<br/>best.pt / last.pt"]
    RFCKPT["Random Forest model<br/>model.joblib"]

    subgraph Evaluation["Evaluation"]
        EVAL["sentinel-crop-evaluate"]
        METRICS["summary metrics<br/>per-class metrics<br/>confusion matrices"]
    end

    subgraph Prediction["Prediction"]
        PRED["sentinel-crop-predict"]
        MASKS["predicted masks"]
        PANELS["RGB / GT / prediction panels"]
    end

    DATA --> DS
    DS --> UNET
    DS --> RF

    UNET --> UCKPT
    RF --> RFCKPT

    UCKPT --> EVAL
    RFCKPT --> EVAL
    EVAL --> METRICS

    UCKPT --> PRED
    PRED --> MASKS
    PRED --> PANELS
```

### Explanation

After dataset generation, the NumPy patches are loaded by root-folder or manifest-based dataset loaders. The U-Net model predicts pixel-level crop classes, while the Random Forest baseline provides a classical machine learning comparison.

The evaluation CLI computes metrics and confusion matrices. The prediction CLI exports masks and optional visualization panels for qualitative inspection.

---

## 5. Minimal E2E smoke test

```mermaid
flowchart TB
    START["run_e2e_smoke_test.py"]
    SHP["EuroCrops shapefile path<br/>argument or environment variable"]
    GEN["Generate small dataset<br/>sentinel-crop-generate + overrides"]
    VERIFY1["Verify images/ and masks/"]
    TRAIN["Train one-epoch U-Net<br/>sentinel-crop-train-unet + overrides"]
    VERIFY2["Verify checkpoint and summaries"]
    EVAL["Evaluate checkpoint<br/>sentinel-crop-evaluate"]
    VERIFY3["Verify evaluation outputs"]
    PRED["Export predictions<br/>sentinel-crop-predict"]
    VERIFY4["Verify prediction outputs"]
    PASS["Smoke test passed"]

    START --> SHP
    SHP --> GEN
    GEN --> VERIFY1
    VERIFY1 --> TRAIN
    TRAIN --> VERIFY2
    VERIFY2 --> EVAL
    EVAL --> VERIFY3
    VERIFY3 --> PRED
    PRED --> VERIFY4
    VERIFY4 --> PASS
```

### Explanation

The E2E smoke test validates that the installed pipeline works end to end. It uses the normal project configuration plus temporary runtime overrides. It is not intended to reproduce full experimental results.

The test checks that dataset generation, training, evaluation, and prediction can run together through the public CLI commands.
