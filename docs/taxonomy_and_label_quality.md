# Taxonomy and label quality

This package ships the class map in [`config/classes.yaml`](../config/classes.yaml).
That file decides what every mask pixel means, so this document states what the
map was validated against and where it is known to be wrong. Shipping a class
map without its measured limits would let a reader treat label noise as model
error.

**Which configuration these results describe.** Every measurement below was
produced against the class map in `config/classes.yaml`, internally identified
as `MAP_a45fa755ef`. It is rule-identical to two archived historical copies of
the same map. **If you edit `config/classes.yaml`, none of these numbers
describes your labels any more.**

**Status of the evidence.** These are measurements taken inside this project,
each from a named artifact. They are **not independently reproducible from this
archive**: the underlying evidence files are several gigabytes of audit output
and staged masks that are not distributed, and they depend on specific
EuroCrops national editions and on locally staged historical masks. The
provenance table at the end names the artifact behind every figure so a
reviewer with access to the full evidence set can check each one. Treat the
numbers as reported measurements with stated scope, not as universally
reproducible results.

## What the map does

Classes are assigned by matching EuroCrops crop-name strings against include
and exclude terms. Eleven semantic classes, of which class 0 is background:

| id | class | id | class |
|---|---|---|---|
| 0 | background | 6 | sunflower |
| 1 | maize | 7 | sugar_beet |
| 2 | rapeseed | 8 | soybean |
| 3 | wheat | 9 | potato |
| 4 | barley | 10 | pasture |
| 5 | oats | | |

Two rules matter when comparing against another taxonomy: class 10 includes
`clover`, `lucerne` and `alfalfa`, and class 2 matches `rapeseed` / `rape seed`
/ `winter rape` / `oilseed rape` while **excluding** `grape`.

**Parse this file; never retype the map.** Tools in this package accept a
`--class-map` argument and archive the parsed YAML as evidence. A hard-coded
copy has already silently dropped an exclusion in this project's own history.

## Independent validation against HCAT

The matching rule was checked against the HCAT crop hierarchy, which the rule
never consults.

**What was measured, precisely.** This is a **source-vector audit**: it
compares the rule's assignment of EuroCrops *parcel attributes* against HCAT's
expectation for the same parcels. It is **not** a measurement of the
benchmark's rasterized masks, and the two populations differ:

| | |
|---|---|
| Country-year files examined | 9: AT_2021, BE_VLG_2021, BE_WAL_2021, CZ_2023, DE_BB_2023, DE_LS_2021, DE_NRW_2021, FR_2018, SK_2021 |
| Parcels examined | 15,654,725 |
| Parcels with a definitive HCAT expectation | 12,073,602 |
| Parcels HCAT places outside the modelled crops | 3,510,452 (no expectation asserted) |
| Parcels deliberately left arguable | 70,671 |

One scope caveat a reader needs. **BE_VLG_2021 and BE_WAL_2021 contribute
parcels to this audit but no samples to the benchmark**: neither has a
generated patch log, and neither appears as a subset in any training
configuration. `BE_VLG_2021` does appear in the shipped
`config/taxonomy_mask_corrections_v1.json`, as `pasture_omission` override
rules produced by this same audit — rules, not data, and no Belgian patch is
part of any split. So the per-class percentages below describe the *rule
applied to these nine source files*, not the composition of the published
benchmark. Per-class figures are **parcel counts, not pixels or patches**.

**A naming discrepancy, and one claim retracted.** So the list above is not
misread as a different population.
This audit, the generated patch logs and every training configuration call the
North Rhine-Westphalia cohort `DE_NRW_2021`; parts of the project's internal
forensic notes call the same cohort `DE_NRW_2023`.

An earlier version of this document asserted that the audit used a 2021 cohort
*while the benchmark uses a 2023 one*, implying two populations. **That claim
is retracted: it was not supported.** There is one cohort, not two.

What the artifacts show: the cohort's cached product inventory holds 136
Sentinel-2 products, and **all 45 that lie on a North Rhine-Westphalia tile
are 2021 acquisitions**. The 91 products dated 2023 are on other tiles
entirely and belong to other cohorts' cache. So no 2023 acquisition on this
cohort's own territory exists in the inventory at all, and the year label
`2023` has no supporting product.

The limit, stated rather than glossed: none of those 45 products appears in
the cohort's processed state, whose record is truncated to a single key, and
this cohort's temporal provenance is deliberately `UNRESOLVED` for every one
of its samples. The cache may also be incomplete. So *which* 2021 dates
produced which samples remains unresolved, and must stay so — what is settled
is only that every candidate product on the cohort's tiles is from 2021.

The consequence is small in practice: this cohort contributes no samples to
the calendar-eligible subset precisely because its temporal provenance is
unresolved, so no calendar or phenology result depends on its year label.

The forensic evidence is not distributed with this package, and the records
and directory names it uses keep the original `local_audit/DE_NRW_2023/`
spelling deliberately: renaming them is a provenance decision, not a
documentation fix, and it would invalidate paths already cited as evidence.

**Definitions, because the last column is not what it looks like.** Per class,
over parcels:

- **recall** = correct / HCAT-truth parcels for the class.
- **precision** = correct / parcels the rule assigned to the class.
- **precision excluding contested** = 1 − (wrongly assigned *and* judged a rule
  defect) / parcels assigned. It does **not** remove the contested parcels from
  the denominator; it keeps them and counts them as correct, i.e. it accepts
  the rule's reading of a genuinely ambiguous source code. For every class with
  no contested parcels the two precision columns coincide by construction.

| class | HCAT truth | rule assigned | recall | precision | precision excl. contested |
|---|---|---|---|---|---|
| maize | 1,460,170 | 1,460,478 | 100.000% | 99.978% | 99.978% |
| rapeseed | 355,881 | 355,363 | 99.854% | 100.000% | 100.000% |
| wheat | 1,784,324 | 1,874,073 | 99.929% | **95.143%** | 95.143% |
| barley | 620,316 | 620,144 | 99.972% | 100.000% | 100.000% |
| oats | 85,907 | 85,766 | 99.836% | 100.000% | 100.000% |
| sunflower | 148,041 | 148,031 | 99.993% | 100.000% | 100.000% |
| sugar_beet | 64,246 | 130,138 | 100.000% | **49.368%** | 99.778% |
| soybean | 83,887 | 83,866 | 99.975% | 100.000% | 100.000% |
| potato | 155,363 | 155,512 | 99.999% | 99.904% | 99.904% |
| pasture | 6,581,460 | 5,937,598 | **90.217%** | 100.000% | 100.000% |

Read the counts, not a single headline:

- **On recall**, nine of the ten crop classes fall in 99.836-100.000%. Pasture
  is the one exception.
- **On precision**, eight of the ten fall in 99.904-100.000%. Wheat and
  sugar_beet are the exceptions.
- **On both at once**, seven of the ten clear 99.8%. Background (class 0) is
  not evaluated here at all, so no statement is made about it.

That is the quantitative justification for the class merges, and it is the main
reason the map is usable. The three exceptions are specific and explainable,
and two of them share one cause.

## Exception 1: pasture recall is 90.217%, mostly label omission

Of the 643,862 pasture-truth parcels the rule did not assign to class 10:

- **552,842 went to background.** Alpine `FORAGE AREA` and French
  `Pastoral area` match no include term, so those surfaces are **missing from
  the labels entirely** rather than labelled incorrectly.
- **91,020 went to wheat.** This is the ryegrass defect of Exception 2, seen
  from the other side. The two exceptions are not independent: the same
  parcels appear in both.

The consequence of the omission is asymmetric and easy to misread: **part of
the published pasture error is not a prediction failure.** A model that
correctly identifies alpine forage is scored against a mask that does not
contain it.

**Pixel-level impact of the omission, with its own and much narrower scope.**
Measured only on 34,260 exact-provenance patches with directly observed staged
masks — not on the whole benchmark — the omitted pasture accounts for
4,776,841 pixels:

| denominator | share |
|---|---|
| of all pixels evaluated (2,245,263,360) | 0.213% |
| of foreground pixels (1,529,838,285) | 0.312% |
| of current class-10 pixels (168,273,798) | **2.839%** |

Do not compare 2.839% with the 90.217% above: one is pixels on exact-provenance
patches, the other is parcels on nine source files.

A separate cross-country issue affects the same class: in CZ_2023, 69.24% of
pasture is `temporary_grass`, a rotational crop. `pasture` is therefore not the
same target in Czechia as in Austria or France, and pooling the class across
those countries pools different things.

## Exception 2: wheat precision is 95.143%, from a known rule defect

The `wheat` class deliberately merges wheat, rye and triticale. The include
term `rye` additionally captures `Lolium` ryegrass (HCAT `3301090205`), a
fodder grass.

Two different measurements exist, with different populations. Both are given
because neither substitutes for the other.

**Parcel counts from the source-vector probe:**

| quantity | value |
|---|---|
| ryegrass parcels captured as wheat | 90,905 (FR_2018 89,821, CZ_2023 1,084) |
| parcels wrongly assigned to wheat per the HCAT arbiter | 91,020 |
| other cosmetic term variants | 111 total: `corn`/cornflowers 73, `beet`/beetroot 29, `potato`/sweet potatoes 9 |

The 90,905 and 91,020 are **not the same measurement**: the first counts
ryegrass parcels found by the term probe, the second counts every parcel the
arbiter judged wrongly assigned to wheat across all nine files. They agree to
about 0.1% and are reported separately rather than reconciled into one number.

**Pixel and sample impact, scoped to exact-geography samples in two cohorts
only** (FR_2018 and CZ_2023, the only cohorts where this was measured):

| cohort | proven samples | containing ryegrass | mislabelled pixels | wheat-class pixels |
|---|---|---|---|---|
| FR_2018 | 8,253 | 931 | 335,142 | 141,439,646 |
| CZ_2023 | 6,566 | 250 | 101,399 | 98,155,777 |
| **total** | **14,819** | **1,181 (8.0%)** | **436,541** | **239,595,423 (0.182%)** |

So: **0.182% of wheat-class pixels, touching 8.0% of samples.** The pixel share
is small; the sample share is not, because one affected parcel contaminates
every patch overlapping it. FR_2018 figures rest on full 256x256 pixel identity
between reconstruction and historical mask; CZ_2023 figures are read directly
from staged historical masks, with 304 threshold-tier samples excluded. An
additional 47 FR samples in the weaker threshold tier contain ryegrass and are
reported separately in the source artifact as indicative only.

No other cohort was measured for this defect.

## Two figures that are *not* defects

**`sugar_beet` raw precision 49.368%.** This is an artefact of the validation
arbiter, not of the labels. A single French LPIS code covers both non-forage
beet and Swiss chard; the separate `Fodder beet` code is correctly left as
background. HCAT resolves the merged code to `beetroot_beets`, but France's
non-forage beet area is overwhelmingly sugar beet, so class 7 is agronomically
defensible. 65,603 of the 65,892 parcels counted against precision are this one
contested code, leaving 289 genuine rule defects.

Accepting the rule's reading of that code — counting the 65,603 as correct
while keeping them in the denominator — gives
(130,138 − 289) / 130,138 = **99.778%**. Removing them from both numerator and
denominator instead gives 64,246 / 64,535 = **99.552%**. Both are reported
because the choice between them is the contested judgement itself, not a
calculation. Neither reading can be proven from the attributes. This is the
cleanest example in this benchmark of
quantified, irreducible source ambiguity, and should be reported as such rather
than as a labelling error.

**70,671 `arguable` parcels excluded by design.** Chiefly `cereal` 39,375,
winter spelt 21,430, spelt 6,612, unspecified cereals 1,871 and smaller groups
down to switchgrass at 5. These are deliberately not assigned to a class. That
is a taxonomy decision, open to disagreement, and not a bug.

## Residual ambiguity that no available evidence resolves

The shipped map cannot be distinguished from a map differing **only** in class
2, under which a bare `rape` term would also capture `Summer rape`. No proven
patch contains a parcel that separates the two. This is recorded as a limit,
not as a preference for either variant.

## Rasterization

Masks are `uint8`, written with `all_touched=False` (centre-based
rasterization). Measured on **2,000 CZ_2023 parcels only**, total rasterized
area is within **+0.008%** of true polygon area, so in aggregate it is
area-unbiased. It does distort individual small parcels: 2 of those 2,000
(0.10%) were affected materially.

**Do not claim that centre-based rasterization shrank the labels.** It did not,
and that claim has been checked and rejected in this project. The supporting
measurement covers one cohort and 2,000 parcels, so it is an aggregate result,
not a per-parcel guarantee.

## Why per-pixel metrics are not comparable across countries

Boundary pixels are a function of parcel geometry, which differs by country.
Pooled over 1.048 billion foreground pixels, the share of mask pixels within
k pixels of a class boundary is 26.98% at k=1, 46.08% at k=2 and 58.87% at k=3.

Per cohort, and this four-fold spread is the actual finding:

| cohort | masks | k=1 | mean component (px) |
|---|---|---|---|
| AT_2021_oats | 9,682 | 42.20% | 96 |
| AT_2021_sunflower | 9,300 | 26.01% | 191 |
| CZ_2023 | 6,877 | 10.78% | 853 |
| AT_2021_soybean | 20 | 24.12% | 214 |
| DE_BB_2023 | 20 | 11.15% | 1,160 |

The boundary share is almost perfectly inversely ordered by mean connected-
component size. **The last two rows are 20-mask control sets and are
indicative only.**

**Coverage limit.** FR_2018, SK_2021 and DE_LS_2021 are **not measured**,
because their masks were not staged locally. FR_2018 matters most: it is the
largest cohort in the benchmark, so the pooled figures above are not
benchmark-representative.

Two consequences for any evaluation built on this map:

- Part of an internal-versus-external performance gap is parcel geometry, not
  generalization. Report cross-country results **stratified by boundary
  distance**, or report interior-only metrics alongside full-mask metrics.
- Within a single cohort, minority classes carry two to three times the
  boundary share of majority classes, so unequal per-class performance is
  partly geometric.

Boundary-distance erosion is an **evaluation-only** device and must never be
the headline result: boundary pixels are real land that a deployed model has to
classify. If you implement it, preserve both the 4-connected structuring
element and `border_value=1`; omitting the latter inflates every reported share
by a spurious one-pixel frame.

## Provenance of every figure above

None of these artifacts is distributed with this package. They are named so a
reviewer with the full evidence set can verify each claim, and so no number
here is untraceable.

| figures | source artifact | population |
|---|---|---|
| per-class recall/precision table; 12,073,602 definitive parcels; 70,671 arguable; 65,603 contested | `local_audit/taxonomy/taxonomy_hcat_arbiter_performance.csv`, `taxonomy_hcat_arbiter_summary.json`, `taxonomy_hcat_arbiter_report.md` | 9 EuroCrops country-year files, 15,654,725 parcels |
| 90,905 ryegrass parcels; FR/CZ split; 111 cosmetic variants | `local_audit/taxonomy/taxonomy_overcapture_audit_summary.md`, `taxonomy_substring_overcapture_probe.csv` | same 9 source files |
| 0.182% of wheat pixels; 8.0% of samples; FR and CZ rows | `local_audit/taxonomy/FR_2018_ryegrass_pixel_impact_summary.json`, `CZ_2023_ryegrass_pixel_impact_summary.json`, `FR_2018_wheat_class_pixels_summary.json` | 14,819 exact-geography samples, FR_2018 + CZ_2023 only |
| pasture omission 4,776,841 pixels and its three denominators | `local_audit/taxonomy/pasture_omission_pixel_impact_summary.json`, `pasture_omission_pixel_impact_report.md` | 34,260 exact-provenance patches |
| CZ_2023 pasture 69.24% temporary grass | `local_audit/taxonomy/taxonomy_pasture_heterogeneity_by_cohort.csv` | CZ_2023 |
| boundary shares, mean components, `all_touched` +0.008% | `local_audit/boundary_geometry/boundary_geometry_report.md` | 5 cohorts for boundaries (3 substantive), 2,000 CZ parcels for area |

## What this document does not cover

Per-cohort provenance tiers, the spatial and temporal evidence behind each
sample, and the full taxonomy audit tables are not part of the software record.
They belong to the separate **dataset** record and to the article, both reached
through this record's `relatedIdentifiers` metadata on its archive landing
page. See also [attribution and rights](attribution_and_rights.md) for the
EuroCrops and Copernicus terms that govern redistribution of any labels derived
with this map.
