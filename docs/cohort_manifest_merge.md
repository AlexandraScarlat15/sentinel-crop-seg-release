# Merge multiple Sentinel-2/EuroCrops cohort manifests safely

Each independently generated country/year cohort has its own image/mask
folder and version-3 `patches_log.csv`. **The generator does not assign
train/val/test splits and the multi-cohort planner does not concatenate
patch logs or manifests.** Use the steps below to combine cohorts before
running one geographic split.

## 1. Adapt each cohort log independently

Use the documented `sentinel-crop-adapt-patch-log` command with a unique
`--dataset-id`, an explicit `--country` and `--cohort`. Prefer
`--verify-assets --validate-only` on the first attempt, then produce
the manifest to a new path. Example:

```bash
sentinel-crop-adapt-patch-log --patch-log data/AT_2021/patches_log.csv \
  --output outputs/manifests/AT_2021.csv --dataset-id AT_2021 \
  --country AT --cohort AT_2021 --format npy --verify-assets

sentinel-crop-adapt-patch-log --patch-log data/FR_2018/patches_log.csv \
  --output outputs/manifests/FR_2018.csv --dataset-id FR_2018 \
  --country FR --cohort FR_2018 --format npy --verify-assets
```

These paths are **illustrative**: real input logs and assets are not shipped
in the software package. The adapter does not infer provenance.

## 2. Merge the adapted manifests (preview first)

```bash
sentinel-crop-merge-cohorts --manifest outputs/manifests/AT_2021.csv \
  --manifest outputs/manifests/FR_2018.csv \
  --output-dir outputs/at_fr_combined --verify-assets
```

The default produces a **JSON preview only**. When it succeeds, repeat
with `--apply` to write `combined_manifest.csv` and `merge_audit.json`
to a **new** output directory. Existing outputs cannot be overwritten.

This tool validates the manifest schema, unique `dataset` and `cohort`
identities, per-row `sample_id`, explicit country codes, compatible
`band_order`, optional consistent label-source names, projected footprint
fields, strict nonnegative class histograms and referenced assets if
`--verify-assets` is specified. Histograms for crop classes absent from one
cohort are padded with zero when another cohort contains that class.
The source manifests and image/mask files stay untouched.

The audit records the source manifest names, SHA-256 digests, declared
country/cohort, row counts, band order and observed class histogram IDs.
**This does not prove that every EuroCrops edition uses the same exact class
taxonomy**, that histograms were computed from corrected masks, or that
declared country/year values match external source metadata. Inspect the
original generator configuration and source label provenance independently.
The source-log columns (e.g. `image_npy_path`) remain unchanged as
historical provenance; the consumable `img_path` and `mask_path`
references are rebased to the output manifest directory.

## 3. Make one shared geographic split

```bash
sentinel-crop-split-spatial \
  --config config/examples/spatial_split.yaml \
  --manifest outputs/at_fr_combined/combined_manifest.csv \
  --out-dir outputs/at_fr_spatial_split --validate-only
```

Repeat with a **fresh** output folder and without `--validate-only` to
write `split_manifest.csv` and geographic leakage audit files.
Desired 70/15/15 shares in the example are **targets**, not promises:
overlapping windows and the geographic guard may exclude rows. A split is
new experiment metadata and does **not** replace the original A′ partitions
in the article.

The training CLI reads the resulting split manifest through
`paths.manifest_csv`; provide an appropriate training YAML configuration.
Prediction/evaluation and plotting remain separate checkpoint/report
workflows. No automatic training or re-training is performed by merging.

## Scope and safety

- At least two **unsplit adapted** manifests are required. Inputs already
  containing train/test partition columns are rejected.
- Source patch assets are never copied or modified. `--verify-assets` checks
  existence only, not TIFF/NPY array contents.
- Repeating the command into the same destination fails rather than
  overwriting any scientific evidence.
- Mixed band order, incompatible label-source names, corrupt class counts,
  duplicate dataset/cohort IDs and conflicting schema are rejected.
- A new merged experiment is not an exact reproduction of the paper split,
  original temporal selection policy or six-fold OOF label review.

Run the synthetic regression suite:

```bash
python -m pytest tests/test_cohort_manifest_merge.py -q
```
