# U-Net-only evaluation and prediction

These are **new curated release commands**, separate from the historical unified
Random Forest / U-Net CLIs. The development repository retains old experiment
paths; the Zenodo draft stages a dedicated RF-free evaluation initializer.

## Evaluation on an approved manifest

```bash
python -m sentinel_crop_seg.cli.evaluate_unet \
  --checkpoint path/to/approved_checkpoint.pt \
  --manifest-csv path/to/frozen_split_manifest.csv \
  --split test --output-dir outputs/eval_unet --device cpu
```

The command uses the checkpoint architecture, band indices, preprocessing
(`normalize_mode`, `scale_divisor`) and optional DOY calendar features.
It uses the manifest's test split by default, without shuffling.
For corrected masks, supply the exact versioned `--label-config` used by the
approved experiment. Omitting it means historical labels.

The evaluation writes `summary.json`, `summary.txt`,
`confusion_matrix_counts.npy`, `confusion_matrix_row_norm.npy`,
`metrics_per_class.csv` and `evaluated_samples.txt`. All-class and
present-class mIoU are reported, alongside pixel accuracy.
A fresh output directory is required: existing reports are never overwritten.

## Predict masks using the same model

```bash
python -m sentinel_crop_seg.cli.predict_unet \
  --checkpoint path/to/approved_checkpoint.pt \
  --manifest-csv path/to/frozen_split_manifest.csv \
  --split test --output-dir outputs/predict_unet --save-color
```

For each sample the command writes a predicted mask to
`pred_masks_npy/`. With `--save-color` it also renders `pred_color/` PNGs
using the 11-class project palette (only when checkpoint classes fit this
palette). A `prediction_samples.csv` links output filenames to the original
sample identifiers and source image paths; `prediction_summary.json`
includes checkpoint and input-manifest hashes.

For a simple root-folder dataset instead of a manifest, pass
`--data-source root --dataset-root ... --sample-format npy` (or
`geotiff`). Root-folders cannot supply calendar acquisition dates or the
identity-based versioned label resolver.

Both commands accept `--batch-size`, `--num-workers`, `--device`,
`--limit`, `--band-indices`, `--normalize-mode-override` and
`--scale-divisor-override`. Prefer checkpoint defaults; overrides can
invalidate comparability with the published model. If the inferred number of
channels does not match the checkpoint, evaluation/prediction fail before
creating output files. Standardized checkpoints require matching normalization
statistics, which are not reconstructed from an arbitrary manifest; verify the
final approved checkpoint before attempting to reproduce the paper.

## Independent plotting of existing model results

The release includes two optional, checkpoint-free plotting commands:
`sentinel-crop-plot-training` for existing `training_metrics.csv` and
`training_summary.json`, and `sentinel-crop-plot-evaluation` for
`confusion_matrix_counts.npy` and `summary.json`. These render
PNG/PDF/SVG figures and per-class tables **without** training, inference
or regenerating a dataset. Class labels can be supplied from a JSON file.
See [independent plotting](independent_plotting.md).

## Scientific cautions

- The release does not **yet include** approved U-Net weights or the exact
  paper test manifest. The commands are functional software, not independently
  verified reproduced article scores.
- `torch.load` can execute unsafe deserialization for legacy checkpoints.
  Use checkpoint files **only from a trusted source**.
- The U-Net CLIs do not automatically select the manuscript-specific
  calendar, cohort subset, label-uncertainty review, or boundary erosion
  protocol. Reproducing those experiments requires explicitly frozen inputs.
- Predicted masks are class-ID arrays, not confidence maps or
  post-training label-uncertainty filtering results.
- No original sample file, mask, manifest, or checkpoint is changed.

Validate the included synthetic inference tests:

```bash
python -m pytest tests/test_unet_release_cli.py -v
```
