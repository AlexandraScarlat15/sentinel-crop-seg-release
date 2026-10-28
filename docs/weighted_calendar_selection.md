# Pixel-weighted calendar compatibility for NEW training experiments

This is a separate stage from Sentinel-2 **scene acquisition filtering**
([acquisition calendar](acquisition_calendar.md)). Both use country/crop
calendar windows, but they answer different questions:

- Scene-level `catalog.calendar`: *when may a satellite product be retrieved?*
- Pixel-weighted TRAIN selection: *what fraction of a patch's annotated
  agricultural pixels are within their respective crop-calendar windows?*

## Audited 90% experiment definition

The historical Model B used `calendar_coverage >= 0.90` on the training
population, with **background class 0 excluded** from the denominator.
A sample is eligible only if every agricultural pixel can be accounted for
by a known calendar window and an acquisition date is present:

```text
denominator = sum(source_p1, ..., source_pN)
compatible  = sum(source_p_i for class i with acquisition_date INSIDE its window)
coverage    = compatible / denominator, only when unresolved_pixels == 0
eligible    = complete AND denominator > 0 AND compatible >= threshold * denominator
```

Threshold checks use exact integer arithmetic, inclusive at exactly 90%.
The 10-class reference binding lives in
`config/calendars/class_calendar_binding_v1.yaml`, matched against the
canonical class IDs in `config/classes.yaml`. Class 10 is an *aggregate*
pasture class with several possible vegetation subtypes. It is eligible
only when the acquisition date lies in the **intersection** of the
Grassland / pasture, Arable grass, Clover, and Lucerne / alfalfa windows.
Missing or non-overlapping mappings make those pixels unresolved rather than
guessing the subtype.

`p0..pN` come from the input manifest. If labels were corrected afterward,
the stored counts may no longer correspond to the corrected masks.
This command does not open images/masks, reconstruct those counts or claim
historical B/B′ membership equivalence. For publication, provide a
versioned source-statistics audit and explicit label provenance.

## Read-only threshold sensitivity (recommended first)

First create or adapt a manifest carrying: `dataset, sample_id, country,
split, acquisition_date` (or `scene_date`), and `p0..pN`.
`partition` is optional; if present, it must equal `split`.
All dates are ISO `YYYY-MM-DD`. Countries are uppercase ISO-2.

```powershell
python -m sentinel_crop_seg.cli.select_weighted_calendar `
  --manifest path/to/new_manifest.csv `
  --calendar config/crop_calendar_recalibrated_v0_3.csv `
  --taxonomy config/classes.yaml `
  --binding config/calendars/class_calendar_binding_v1.yaml `
  --threshold 0.90 `
  --sweep 0.75 0.90 1.00 `
  --output-dir outputs/calendar90_preview
```

The default is **preview-only**: it prints how many TRAIN samples each
threshold would retain, including country/cohort breakdowns, and the hashes
of all four source files. It writes nothing, even to the supplied output dir.
No TEST or VAL labels are used to choose or optimize the threshold.

When the preview and input histogram provenance have been reviewed, run the
same command with `--apply` and a **new empty output path** to create:

- `calendar_selected_manifest.csv`: selected TRAIN plus **all original**
  validation, test and excluded rows, with split membership/order preserved
- `calendar_train_audit.csv`: individual TRAIN eligibility and pixel counts
- `calendar_selection_report.json`: thresholds, retention, hashes and limits

The original manifest, images and labels are unchanged. Relative image/mask
paths are rebased into the new manifest's directory without copying files.

## How to choose the threshold scientifically

**90% is an experimental hypothesis, not a universal optimum.** Compare at
least 0.75, 0.90 and 1.00 on TRAIN retention; inspect differences by
country/cohort and by crop representation before considering retraining.
For a fair model experiment, hold architecture, spatial split, validation,
test, optimizer, seed and corrected-label version constant. Training-only
selection is intentional; it avoids narrowing TEST to visually convenient
samples and does not create a post-hoc better-looking benchmark.

A meaningful model-performance comparison requires independent training and
evaluation and, ideally, repeated seeds with per-country/class results.
**Do not tune the calendar threshold based on TEST metrics.**

The historical Model B's threshold applies at patch level; the unrelated
Model B′ analysis also evaluated weighted membership over other partitions.
Neither historical protocol is silently reconstructed or overwritten by
this standalone mechanism.
