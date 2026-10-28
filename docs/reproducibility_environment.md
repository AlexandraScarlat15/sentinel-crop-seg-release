# Environment and source provenance — publication checklist

This is a **software draft contract**, not a claim that the paper's original
runtime or Sentinel-2 / EuroCrops inputs have been independently recovered.
Record exact evidence **before** declaring any release reproducible.
No new satellite retrieval, model training or dataset generation is required
to fill in historical metadata when original records are available.

## Python and scientific dependencies

The curated package declares Python **>=3.11** and unpinned scientific
dependencies in `pyproject.toml` (numpy, pandas, PyYAML, geopandas,
rasterio, shapely, pystac-client, torch, matplotlib, requests, scipy,
pyproj). The existing CI runs on **Python 3.11**, but that is not proof
that the original experiments used precisely 3.11.

Because unpinned declarations alone are not a reproducibility record, the
archive also ships **`requirements-tested.txt`**: the direct dependencies and
their applicable closure, pinned at the versions present on the machine that
last ran the package's test suite. Its header names that machine's Python
version, implementation and platform. Install the exact tested set with:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e .
```

Read the file's own header before relying on it. It is a **tested-environment
record**, not a lockfile: it carries no hashes, so it does not guarantee an
identical install, and it is **not** the environment that produced the
historical checkpoint. If its Python version differs from yours, the pins may
not all be installable.

To document the environment you actually used, run:

```bash
python --version
python -m pip freeze --all
```

The **development repository only** regenerates the shipped record with a
read-only helper, which also fails if a declared dependency is missing:

```bash
python release/report_tested_environment.py \
  --pyproject release/templates/pyproject.toml --require-all \
  --emit-requirements requirements-tested.txt
```

The helper itself is intentionally **not** included in the Zenodo software
package, so the standalone instructions above do not depend on it. Re-run it
whenever `pyproject.toml` dependencies change; a packaged test fails if a
declared dependency has no pin.

### Binary dependencies that are not captured by a version pin

These are the common reasons a third-party install fails or behaves
differently, and none of them is recorded by a version number alone:

| Component | What the pin does not say | What to record |
|---|---|---|
| GDAL / PROJ | `rasterio` and `pyproj` binary wheels **bundle their own** GDAL and PROJ. A system GDAL of a different version, or a source build linking against it, changes behavior | Output of `python -c "import rasterio; print(rasterio.__gdal_version__)"` and `python -c "import pyproj; print(pyproj.proj_version_str)"` |
| PROJ datum grids | Reprojection accuracy can depend on whether optional transformation grids were downloaded | `pyproj.datadir.get_data_dir()` contents, and whether network grid access was enabled |
| torch build | The CUDA variant is selected by package index URL, **not** by version string, so `torch==2.x` does not distinguish CPU from GPU builds | `torch.__version__` in full (it encodes `+cpu`/`+cu###`), `torch.version.cuda`, `torch.cuda.is_available()`, driver version |
| Nondeterminism | Pinning does not make GPU training reproducible | Seeds, `torch.use_deterministic_algorithms` state, cuDNN benchmark flag |

Install geospatial dependencies from wheels where possible, and do not mix
`pip`-installed and `conda`-installed GDAL stacks in one environment.

For any environment that actually performed a historical run, capture
read-only details:

```powershell
python --version
python -m pip --version
python -m pip freeze --all
python -c "import torch, rasterio, geopandas, numpy, pandas, pyproj; print('torch', torch.__version__); print('rasterio', rasterio.__version__); print('geopandas', geopandas.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('pyproj', pyproj.__version__)"
```

Record these together with OS, machine type, available CPU/GPU and CUDA
runtime where relevant. If the original executed environment is missing,
label the new snapshot as a **tested reconstruction**, not the environment
that produced the original checkpoint. The unpinned project requirements
must not be described as a full dependency lock.

For reproducibility, retain a versioned, reviewed environment report or
package lock derived from verified installed versions. Never fabricate
missing historical dependencies or edit checkpoint metadata to match a
new environment.

## Sentinel-2 source contract

The generator searches the Copernicus Data Space STAC catalogue using
collection `sentinel-2-l2a`, i.e. Sentinel-2 Level-2A surface reflectance.
The configured ten bands are B02, B03, B04, B08 at 10 m and B05, B06,
B07, B8A, B11, B12 at native 20 m, aligned onto the reference patch grid.
The default quality filter uses Scene Classification Layer (SCL) categories
for invalid/cloud/shadow/snow/cirrus rejection and optionally B02
brightness. STAC scene-level `eo:cloud_cover` is distinct from patch-level
SCL checks.

The identity of a **historically used product** must come from actual
`product_id` / `safe_id` / original product inventory and processing
baseline (when verifiable). STAC search parameters, collection name and
acquisition dates **alone** do not prove which scene processing baseline
was downloaded or that future catalogue queries return identical items.
The product inventory should record observed product ID, acquisition date,
tile, processing baseline, source platform, retrieval endpoint, and any
asset checksums present in the original evidence. Mark missing values
`not verified` rather than substituting expected defaults.

## EuroCrops source contract

EuroCrops is used as the **parcel-annotation source**, not as a satellite
product. A filename alone does not establish the dataset edition or reuse
permission. For each country/year/region included in the manuscript or in a
new release recipe, preserve:

| Required provenance field | Status at this software-only checkpoint |
|---|---|
| Country / region / reference year | Declared in experiment/cohort metadata; independently verify |
| Official EuroCrops edition and release identifier | Not frozen in the curated software metadata |
| Official source URL / institutional attribution | Requires review against the original acquired files |
| Source license and redistribution conditions | Requires explicit legal/data-source review |
| Vector file identifiers and sidecar checksums | Strict runs can hash local shapefile components |
| Coordinate reference system and crop-name column | Runtime loader uses the source CRS and configured field |
| Harmonization/taxonomy version | `config/classes.yaml`; corrected masks are a separate versioned artifact |

**Do not upload raw EuroCrops source vectors by default.** Distinguish rights
to distribute original vectors, derived rasters and metadata. Keep provenance
per source cohort; do not call every country-year file one unified edition.

## License and attribution boundaries

See [attribution and rights review](attribution_and_rights.md) before
distributing any original/derived input data or weights. The current
software root license is CC BY 4.0, but code authorship and whether the
authors want that software license remain approval-dependent. The official
EuroCrops repository and one distribution catalogue currently describe
different Creative Commons variants; the actual downloaded country/year
edition must be checked before stating terms. Sentinel-2 also has its own
source/modified-data notices. None of these are resolved by software tests.

## Dataset / checkpoint boundary

The A′ manifest and pre-existing checkpoints are separate prospective
research artifacts, **not included in this curated software draft**.
Dataset packing/compression, distribution rights, train/val/test membership
verification and model approval are deliberately parked pending author and
supervisor decisions.

When a paper-aligned checkpoint is selected, store its original executed
training configuration, label-version resolver, model architecture, band
order/normalization, class IDs, input manifest checksum and evaluation
protocol. A read-only load/inference smoke test may validate the delivered
weight file without retraining or changing results.

The uncertainty-review utility consumes **precomputed** parcel-review
evidence and human decisions; it does not regenerate the original six-fold
checkpoints and must not be described as having already corrected labels.

## Publication gate

Before uploading, check the complete staged package in a clean Python
environment on a new machine, confirm CLI entrypoints and checksum
verification, review upstream source licenses and record the accurate
paper status in `CITATION.cff`. If a fact cannot be verified, preserve
the explicit unknown. The current draft is intentionally marked
`DO_NOT_PUBLISH_DRAFT.txt`.
