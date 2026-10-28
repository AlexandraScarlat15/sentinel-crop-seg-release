"""Opt-in, fail-closed generation-run ledger.

A strict run is bound to the fully resolved recipe, the exact EuroCrops vector
components, the output directory, processed-state file and run tag.
An interrupted in-flight run is *never* silently resumed. There is no
automatic rollback or deletion of source files, generated patches or caches.

Legacy runs do not use this ledger and retain their historical behavior.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

LEDGER_NAME = "generation_run_guard.json"
LOCK_NAME = ".generation_run_guard.lock"
# Version 2 relocated the fingerprinted sources from a top-level ``src``
# directory into this package. ``generator_code_sha256`` is keyed by path, so
# its keys changed: a version-1 ledger is refused rather than compared against
# renamed keys. That fail-closed refusal is intended. Only opt-in strict
# resumes of NEW experiments are affected; legacy runs carry no ledger.
LEDGER_VERSION = 2

# Files whose implementation affects samples, filters, classification or IO.
# Changing these invalidates a strict resume even when YAML is unchanged.
# Paths are relative to this package directory so that the fingerprint is
# identical in a source checkout and in an installed distribution.
RECIPE_CODE_PATHS = (
    "pipeline/dataset_generation.py",
    "pipeline/generation_config.py",
    "pipeline/generation_recipe.py",
    "pipeline/acquisition_calendar.py",
    "pipeline/generation_run_guard.py",
    "config_loader.py",
    "data_engine.py",
    "selection/roi_area.py",
    "selection/roi_selectors/grid_coverage_selector.py",
    "selection/roi_selectors/hotcell_selector.py",
    "selection/roi_grid_builder.py",
    "data/eurocrops_mask_loader.py",
    "data/sentinel_safe_loader.py",
    "data/roi_image_cache.py",
    "data/roi_mask_cache.py",
    "geospatial/geotiff_writer.py",
)


def _sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def vector_sources(path: str | Path) -> dict[str, str]:
    """Fingerprint the required source and its semantic shapefile sidecars."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EuroCrops vector source not found: {source}")
    if source.suffix.lower() == ".shp":
        related = [source.with_suffix(x) for x in (".shp", ".shx", ".dbf", ".prj")]
        optional = [source.with_suffix(".cpg")]
        for candidate in related:
            if not candidate.is_file():
                raise FileNotFoundError(f"Incomplete EuroCrops shapefile: {candidate}")
        related.extend(p for p in optional if p.is_file())
    else:
        related = [source]
    return {p.name: _sha_file(p) for p in related}


def run_identity(
    config: dict[str, Any], *, run_tag: str, output_dir: Path,
    processed_path: Path, config_mode: str,
) -> dict[str, Any]:
    if config_mode != "isolated":
        raise ValueError("Strict resume guard requires --config-mode isolated")
    if not isinstance(run_tag, str) or not run_tag.strip():
        raise ValueError("Strict mode requires a nonempty run_tag")
    source_cfg = config.get("eurocrops", {})
    if not isinstance(source_cfg, dict) or not source_cfg.get("shapefile_path"):
        raise ValueError("Strict mode requires eurocrops.shapefile_path")
    sources = vector_sources(source_cfg["shapefile_path"])
    # Resolve against this package, not a presumed repository root: the
    # fingerprint must be computable from an installed distribution too.
    package_root = Path(__file__).resolve().parents[1]
    code_sha256 = {
        name: _sha_file(package_root / name) for name in RECIPE_CODE_PATHS
    }
    return {
        "schema_version": LEDGER_VERSION,
        "config_mode": config_mode,
        "run_tag": run_tag,
        "config_sha256": _canonical_digest(config),
        "taxonomy_sha256": _canonical_digest(config.get("classes", {})),
        "generator_code_sha256": code_sha256,
        "python_version": list(sys.version_info[:3]),
        "vector_files_sha256": sources,
        "source_path": str(Path(source_cfg["shapefile_path"]).resolve()),
        "output_dir": str(output_dir.resolve()),
        "processed_state_path": str(processed_path.resolve()),
    }


def output_snapshot(output_dir: Path, processed_path: Path) -> dict[str, Any]:
    """Track accepted file inventory and log/state integrity at clean checkpoints."""
    inventory: list[tuple[str, int]] = []
    for p in sorted(output_dir.rglob("*")):
        if not p.is_file() or p.name in (LEDGER_NAME, LOCK_NAME):
            continue
        inventory.append((p.relative_to(output_dir).as_posix(), p.stat().st_size))
    log = output_dir / "patches_log.csv"
    count = 0
    if log.exists():
        with log.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError("Invalid generator patches_log.csv header")
            seen = set()
            for row in reader:
                sample_id = row.get("sample_id")
                if not sample_id or sample_id in seen:
                    raise ValueError("Duplicate or missing sample_id in patches_log.csv")
                seen.add(sample_id)
                count += 1
    assets = ("images", "masks", "images_tif", "masks_tif")
    if count:
        if not processed_path.is_file():
            raise RuntimeError("Generated patches have no processed-state checkpoint")
        try:
            state = json.loads(processed_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Invalid processed-state checkpoint") from exc
        if (not isinstance(state, dict)
                or not any(value is True for value in state.values())):
            raise RuntimeError("No completed ROI/product in processed-state checkpoint")
        expected = {
            "images": (".npy",), "masks": (".npy",),
            "images_tif": (".tif",), "masks_tif": (".tif",)
        }
        for directory in assets:
            files = sorted(p for p in (output_dir / directory).glob("*") if p.is_file())
            if len(files) != count or any(p.suffix.lower() not in expected[directory] for p in files):
                raise RuntimeError(f"Incomplete patch assets under {directory}")
            if {p.stem for p in files} != seen:
                raise RuntimeError(f"Patch asset identities disagree with log in {directory}")
    elif any((output_dir / d).exists() and any((output_dir / d).iterdir()) for d in assets):
        raise RuntimeError("Generated patch files exist without patch log entries")
    return {
        "sample_count": count,
        "inventory_sha256": _canonical_digest(inventory),
        "inventory_files": len(inventory),
        "log_sha256": _sha_file(log) if log.exists() else None,
        "processed_state_sha256": _sha_file(processed_path) if processed_path.is_file() else None,
    }


class GenerationRunGuard:
    """A clean ledger may be resumed; an in-flight or mismatched one may not."""

    def __init__(self, config: dict, run_tag: str, output_dir: Path,
                 processed_path: Path, *, config_mode: str):
        self.output_dir = Path(output_dir).resolve()
        self.processed_path = Path(processed_path).resolve()
        self.identity = run_identity(
            config, run_tag=run_tag, output_dir=self.output_dir,
            processed_path=self.processed_path, config_mode=config_mode,
        )
        self.ledger_path = self.output_dir / LEDGER_NAME
        self.lock_path = self.output_dir / LOCK_NAME
        self.initial_samples = 0
        self._locked = False

    def _write(self, record: dict[str, Any]) -> None:
        temp = self.output_dir / (LEDGER_NAME + ".tmp")
        if temp.exists():
            raise RuntimeError("Stale generation ledger temp file: manual review required")
        try:
            with temp.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.ledger_path)
        finally:
            # A failure in os.replace is treated as unsafe; leave temporary
            # evidence behind instead of silently deleting it.
            pass

    def __enter__(self) -> "GenerationRunGuard":
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=False)
        if not self.output_dir.is_dir():
            raise RuntimeError("Generator output must be a directory")
        try:
            with self.lock_path.open("x", encoding="utf-8") as lock:
                lock.write("An active or interrupted strict generation session owns this lock.\n")
                lock.flush()
                os.fsync(lock.fileno())
        except FileExistsError as exc:
            raise RuntimeError(
                "Existing generation lock: concurrent or interrupted run; manual review required"
            ) from exc
        self._locked = True
        try:
            if self.ledger_path.exists():
                if (self.output_dir / (LEDGER_NAME + ".tmp")).exists():
                    raise RuntimeError("Stale ledger temporary file; manual review required")
                with self.ledger_path.open(encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("identity") != self.identity:
                    raise RuntimeError("Generation recipe, taxonomy, vector or run identity changed")
                if previous.get("status") != "clean":
                    raise RuntimeError("Incomplete strict generation run; refusing unsafe resume")
                current = output_snapshot(self.output_dir, self.processed_path)
                if current != previous.get("snapshot"):
                    raise RuntimeError("Generated files or processed-state file changed since checkpoint")
                self.initial_samples = int(current["sample_count"])
            else:
                preexisting = [
                    p for p in self.output_dir.iterdir()
                    if p.name != LOCK_NAME
                ]
                if preexisting or self.processed_path.exists():
                    raise RuntimeError(
                        "Existing unguarded output or processed state: cannot adopt a legacy run"
                    )
            self._write({"identity": self.identity, "status": "running",
                         "baseline_samples": self.initial_samples})
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def finish(self) -> dict:
        if not self._locked:
            raise RuntimeError("Strict guard must be locked before finish")
        snapshot = output_snapshot(self.output_dir, self.processed_path)
        if snapshot["sample_count"] < self.initial_samples:
            raise RuntimeError("Patch rows disappeared during strict run")
        self._write({"identity": self.identity, "status": "clean", "snapshot": snapshot})
        return snapshot

    def __exit__(self, exc_type, exc, tb) -> None:
        # The running status survives errors and requires manual inspection.
        if self._locked:
            self.lock_path.unlink()
            self._locked = False
