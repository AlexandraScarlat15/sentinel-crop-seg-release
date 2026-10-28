# External dataset conversion

Two converters bring other labelled crop datasets into this project's
11-class taxonomy and NPY layout, so a model trained on the Sentinel-2 /
EuroCrops data can be evaluated on an independent source.

This is the cleanest kind of external validation available here: the
geography, the label authority and the acquisition pipeline all differ, so
good numbers cannot come from leakage within one benchmark.

**Neither converter downloads anything.** You supply an already-obtained
dataset; the converter reads it and writes patches.

## What they produce

Both write the same layout the training and evaluation commands already read:
`images/`, `masks/` and a metadata CSV, with masks remapped to the project's
class IDs. The class map is **declared in YAML, never hard-coded**, so a
conversion is auditable and you can change the mapping without touching code.

Two contract details decide whether a cross-dataset number means anything.

**Unmapped crops become 255, the ignore value — not background.** Background
is class 0 and is a real prediction target. Pixels written as 255 are
**excluded from the loss and from the metrics**, so the model is not scored on
crops this taxonomy does not model. That is the opposite of being penalized
for them, and confusing the two inverts how you read external recall.

**Reflectance is written at the source scale, not normalized.** Both
converters leave scaling to the consumer, exactly as the project's own
generator does, and evaluation applies the checkpoint's `normalize_mode`. So
the normalization happens once. If you convert with some other tool that
already divides by 10,000, pass
`--normalize-mode-override none` to the evaluation command, or the values will
be scaled twice and come out 10,000x too small with no error. The DACIA5
converter records what it did in `conversion_contract.json` beside its output.

**One conversion per output directory.** Both converters refuse to write into
a directory that already holds a conversion. `--overwrite` authorizes replacing
one, and then genuinely clears it — `images/`, `masks/` and the report files —
so you cannot end up with the previous run's surplus patches sitting beside the
new `metadata.csv`, which an evaluator reading the directory would score. If
the directory holds anything no converter wrote, the run refuses rather than
deleting it: convert into a directory of its own.

## PASTIS

[PASTIS](https://github.com/VSainteuf/pastis-benchmark) is a French
Sentinel-2 time-series benchmark. The converter collapses a time series into
single-date patches, which is what this project's U-Net consumes.

```bash
sentinel-crop-convert-pastis \
  --pastis-root /path/to/PASTIS \
  --mapping-yaml config/pastis/label_mapping.yaml \
  --output-dir outputs/pastis_converted \
  --month-start 4 --month-end 9
```

The mapping in [`config/pastis/label_mapping.yaml`](../config/pastis/label_mapping.yaml)
is explicit about every correspondence, for example `meadow -> pasture`,
`soft winter wheat -> wheat`, `spring barley -> barley`. Read it before
trusting a cross-dataset number: PASTIS distinguishes crops this taxonomy
merges, and the merge direction is a scientific choice recorded in that file,
not an implementation detail.

Because a time series must be reduced to one date per patch, the converter
exposes date-selection and quality controls: `--month-start` / `--month-end`
bound the season, `--select-best-per-patch` picks the best available date per
patch instead of the first, and the cloud, brightness and vegetation options
(`--max-cloud-score`, `--max-very-bright-ratio`, `--min-vegetation-ratio`,
`--ndvi-threshold`) reject unusable dates. The conversion reorders PASTIS's bands into this project's order **by
default**, deriving the permutation from the two band-name lists; pass
`--band-reorder none` only for a copy already in project order, and the
converter prints a loud warning whenever the output order differs from the
project's. `--clip-reflectance` with
`--clip-min` / `--clip-max` bounds reflectance values. Defaults are visible
with `--help`; **state the ones you used** when reporting a result, because
they change which dates a patch is scored on.

Use `--fold`, `--limit-samples` and `--limit-patches` for a small trial run
before converting the whole benchmark.

## DACIA5

A Romanian Sentinel-2 dataset supplied as GeoTIFF scenes with separate masks.
The converter tiles scenes into patches and remaps the labels.

```bash
sentinel-crop-convert-dacia5 \
  --images-root /path/to/Dataset_DACIA5/Images_Sentinel2_12bands_GeoTIFF \
  --masks-root /path/to/Dataset_DACIA5/Masks_and_legend \
  --mapping-path config/dacia5/label_mapping.yaml \
  --output-dir outputs/dacia5_converted \
  --years 2020 2021 2024
```

[`config/dacia5/label_mapping.yaml`](../config/dacia5/label_mapping.yaml)
lists the source label names alongside the project class they map to. Note
that the source separates potato categories (late, seed, other) that this
taxonomy merges into a single `potato` class, and that it carries crops with
no counterpart here, which the mapping sends to the ignore value. Those pixels
are dropped from the score rather than counted against the model, so an
external metric describes only the crops this taxonomy models. **Report the
ignored share alongside any number**: a high ignore fraction means the metric
covers a small part of the scene, which is a different claim from good
performance over it.

`--patch-size` and `--stride` control tiling; `--drop-all-ignore` and
`--drop-all-background` discard patches with nothing to learn from;
`--min-valid-pixels` and `--min-nonbackground-pixels` set the admission
thresholds.

## Reading a cross-dataset result honestly

- **Taxonomy merges are not free.** Both mappings collapse distinctions the
  source made. A class that performs well may be easier on the external
  source simply because fewer confusable neighbours survive the merge.
- **Unmapped labels become ignore, not background**, so they are excluded
  from the metric rather than scored as errors. The number therefore describes
  only the modelled crops, over however much of the scene survived.
- **Per-pixel metrics are not comparable across sources** for the same reason
  they are not comparable across countries here: parcel geometry differs, and
  boundary-pixel share with it. See
  [taxonomy and label quality](taxonomy_and_label_quality.md).
- **A single date is not a time series.** The PASTIS conversion discards the
  temporal dimension that benchmark exists to test, so a result here is not
  comparable to PASTIS's own published time-series numbers.

## What is not included

No PASTIS or DACIA5 data. You obtain those from their own sources, under
their own licenses, which this package does not grant — see
[attribution and rights](attribution_and_rights.md).
