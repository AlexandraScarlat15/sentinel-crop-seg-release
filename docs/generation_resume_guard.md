# Strict generation resume guard

This is an **opt-in safety mechanism for NEW generated datasets**. It does
not rewrite or adopt any historical revision or paper dataset, or its state.
The default remains `--resume-policy legacy` and retains historic semantics.

## New experiment

Use a fresh dataset directory, fresh `run_tag` and an explicit isolated
generator configuration:

```bash
python -m sentinel_crop_seg.cli.generate --config-dir config \
  --config-mode isolated --resume-policy strict \
  --run-tag AT_2021_experiment_001 \
  --override eurocrops.shapefile_path=path/to/AT_2021_EC21.shp \
  --override paths.dataset_output_dir=data/processed/AT_2021_experiment_001
```

For safety, the first strict run refuses if its output directory contains
any preexisting files or its processed-state file already exists without a
matching strict ledger.

The guard writes `generation_run_guard.json` into the output directory.
The ledger includes:

- SHA-256 of the resolved configuration, plus a separate class-taxonomy
  digest;
- SHA-256 of generator Python modules that affect selection, filtering,
  rasterization and writing, plus the Python version;
- SHA-256 for the EuroCrops source vector; for Shapefiles, fingerprints
  `.shp`, `.shx`, `.dbf`, `.prj` and any existing `.cpg`;
- run tag, output location and processed-state location;
- clean/running status and a snapshot of the generated file inventory;
- the `patches_log.csv` SHA-256, sample count, processed-state SHA-256.

The source data must exist and the vector components must be complete
before the run starts. The original EuroCrops files and historical
configurations are never written to or updated.

## Resuming a *clean* strict run

Reissue the **identical** command with unchanged source files and the same
output directory and run tag. The guard verifies that the saved configuration,
taxonomy, source vector, file inventory, patch log and processed-state file
still match. Only then does normal generator processing start.

A positive `dataset.target.total_patches` in strict mode limits the
**cumulative** number of accepted samples recorded in the guarded dataset,
including prior clean sessions. If the target has already been met, no
additional products are processed. The quota is still an upper bound,
not a guarantee that enough suitable patches can be found.

The historical `legacy` mode continues to apply its quota only to the
current invocation. A new strict run cannot claim or reinterpret the
historic processed state.

## Interrupted or unsafe run

The ledger becomes `running` before any generation begins, and becomes
`clean` only after normal completion and a successful inventory check.
A process crash, Ctrl+C, partial patch write, corrupted/missing output,
changed source or changed recipe causes any later strict run to **refuse
automatic resume** rather than duplicate or mix samples.

In this initial strict implementation, an interrupted run has **no automatic
rollback or recovery procedure**. Preserve it for inspection, and start a
new run with a new output directory and run tag after resolving the issue.
Never manually relabel `running` as `clean`.

A `.generation_run_guard.lock` file rejects concurrent invocations.
An unclean process termination can leave the lock behind; in that case the
tool also refuses restart. Inspect the output before any manual action.

The inventory check compares relative filenames and sizes, verifies all
four output sample directories against patch-log sample IDs, and hashes
the log and processed-state file. A non-empty generated dataset must also
have a valid processed-state checkpoint with at least one completed product.
It **does not** recompute the byte hashes
of all multispectral data and masks (potentially hundreds of GB), and
cannot certify unchanged image content if file names and sizes are
unchanged. Final scientific publication still needs independent integrity
checks and a complete real-data smoke test.

## Compatibility and limitations

- Require `--config-mode isolated --resume-policy strict` together.
- No automatic migration of old NPY/GeoTIFF datasets or processed
  `processed_<strategy>_<tag>.json` files.
- Provenance is bound to resolved configuration and primary EuroCrops
  components. Ancillary external dependencies such as mutable STAC
  catalogue responses, downloaded SAFE contents, external code version or
  arbitrary extra source files are not fully captured by this guard.
- The generated `patches_log.csv` schema remains v3; the separate
  [adapter](patch_log_adapter.md) remains compatible.
- Legacy generation, historical paper splits and existing training
  checkpoints are intentionally unchanged.

This guard is a **fail-closed consistency protection**, not a full
transactional or exactly-once distributed processing system.
