# Opt-in crop-calendar acquisition filtering

The five-country reference `config/crop_calendar_recalibrated_v0_3.csv` can
now constrain Sentinel-2 *acquisition dates* in **new, versioned** experiments.
Legacy generation and frozen A′/paper artifacts are unchanged.

## Preview before download

```bash
sentinel-crop-audit-generation-config --config-dir config \
  --experiment config/experiments/examples/at_2021_calendar.yaml
sentinel-crop-generate-cohorts --config-dir config \
  --plan config/experiments/examples/multi_cohort_calendar.yaml
```

Both commands are offline. The output includes the selected crop names, calendar
source SHA-256, exact day windows, and the effective generator recipe hash.
Missing CSVs, country/crop coverage, invalid dates and windows not intersecting
the configured STAC bounds fail *before* imagery processing begins.

Example excerpt inside a versioned generation recipe:

```yaml
settings:
  catalog:
    calendar:
      mode: selected_crop_union
      file: config/crop_calendar_recalibrated_v0_3.csv
      country: AT
      year: 2021
      crops: [Wheat]
```

The `file` path is resolved relative to the process working directory, like
existing source paths; run from the repository root or use an absolute path.
The CSV schema is exactly `country,crop,start,end` (MM-DD boundaries,
inclusive). Crop names match CSV entries case-insensitively, but spelling
must match; missing or duplicated rows are rejected. Windows are restricted
to one calendar year in this version.

## Selection semantics

`selected_crop_union` merges the selected windows into disjoint UTC
intervals, intersects them with the recipe's STAC acquisition period and
queries only those intervals. Returned product acquisition dates are also
checked locally before any SAFE download or patch extraction. The STAC
`limit` still applies **per query interval** and is not an assurance of
complete scene coverage; tune and audit search results for a real release.

**Important scientific limit:** crop windows control *scene* acquisition,
not individual parcel pixels. If a patch contains several crops, all mapped
classes can appear in its mask. Setting `dataset.target.class_id` constrains
target-class patch sampling, but does not rewrite or temporally validate other
classes. Selecting multiple crops means the UNION of their scene-date
intervals, not simultaneous suitability of every crop in a mixed patch.
For parcel-specific phenology eligibility and per-class calendar evaluation,
a separate label-aware algorithm and independent scientific validation
are still required.

The policy is opt-in: omit `catalog.calendar` entirely for historical
selection behavior. The calendar CSV bytes, selected windows and names are
recorded in the strictly fingerprinted resolved recipe, and the relevant
calendar-processing source is included in the strict code fingerprint.
Modifying the CSV or settings invalidates a strict resume.

For a *separate* post-generation training selection based on at least 90%
pixel-weighted crop compatibility, see
[weighted calendar selection](weighted_calendar_selection.md). That metric
uses per-class annotated pixel counts and cannot be replaced by this
scene-level date search.

The `country/year` in `catalog.calendar` must match the parent multi-cohort
plan when present. The plan's declared `region` remains metadata, not proof
of polygon geography. Real-run provenance and distribution rights still
need checking prior to publication.

## New dry-run example

```bash
python -m sentinel_crop_seg.cli.generate_cohorts \
  --config-dir config \
  --plan config/experiments/examples/multi_cohort_calendar.yaml
```

This remains `validated_plan_only`. Do **not** append `--execute` until
the correct local EuroCrops files, storage directories and policies have
been reviewed. No command here reproduces historical A′ calendar selection.
