"""Synthetic fail-closed resume tests; no network or real EuroCrops needed."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentinel_crop_seg.core.pipeline.generation_run_guard import (
    GenerationRunGuard, LEDGER_NAME, LOCK_NAME, output_snapshot, run_identity,
)


def make_recipe(tmp_path: Path):
    root = tmp_path / "input"
    root.mkdir()
    shp = root / "AT_2021.shp"
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        shp.with_suffix(suffix).write_bytes(f"synthetic {suffix}".encode())
    recipe = {
        "eurocrops": {"shapefile_path": str(shp)},
        "classes": {"maize": {"id": 1, "include": ["maize"]}},
        "dataset": {
            "patch": {"size": 4, "stride": 4},
            "max_cloud_fraction": 0.01,
            "bands": {"B02": {"resolution": 10}},
            "target": {"total_patches": 2},
        },
        "paths": {
            "dataset_output_dir": str(tmp_path / "out"),
            "sentinel_data_root": str(tmp_path / "cache"),
        },
    }
    output = tmp_path / "out"
    state = tmp_path / "cache" / "processed_grid_trial.json"
    return recipe, output, state, shp


def new_guard(recipe, out, state):
    return GenerationRunGuard(
        recipe, "trial", out, state, config_mode="isolated",
    )


def emit_patches(out: Path, count: int):
    for folder, ext in (("images", ".npy"), ("masks", ".npy"),
                        ("images_tif", ".tif"), ("masks_tif", ".tif")):
        d = out / folder
        d.mkdir(exist_ok=True)
        for idx in range(count):
            (d / f"sample_{idx:06d}{ext}").write_bytes(b"synthetic bytes")
    with (out / "patches_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "product_id"])
        writer.writeheader()
        for idx in range(count):
            writer.writerow({"sample_id": f"sample_{idx:06d}", "product_id": "fake"})


def test_initial_strict_run_creates_clean_ledger_and_accepts_identical_resume(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        assert session.initial_samples == 0
        emit_patches(out, 2)
        state.parent.mkdir()
        state.write_text('{"roi:product": true}', encoding="utf-8")
        finished = session.finish()
        assert finished["sample_count"] == 2
    assert not (out / LOCK_NAME).exists()
    ledger = json.loads((out / LEDGER_NAME).read_text(encoding="utf-8"))
    assert ledger["status"] == "clean"
    assert ledger["snapshot"] == finished
    with new_guard(config, out, state) as resume:
        assert resume.initial_samples == 2
        resume.finish()
    assert not (out / LOCK_NAME).exists()


@pytest.mark.parametrize("change", ["quality", "taxonomy", "vector", "sidecar", "run_tag"])
def test_strict_resume_rejects_changed_recipe_or_source(tmp_path, change):
    config, out, state, shp = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        session.finish()
    if change == "quality":
        config["dataset"]["max_cloud_fraction"] = 0.02
    elif change == "taxonomy":
        config["classes"]["maize"]["id"] = 9
    elif change == "vector":
        shp.write_bytes(b"DIFFERENT")
    elif change == "sidecar":
        shp.with_suffix(".dbf").write_bytes(b"changed annotations")
    else:
        with pytest.raises(RuntimeError, match="identity changed"):
            with GenerationRunGuard(
                config, "another_run_tag", out, state, config_mode="isolated"
            ):
                pass
        return
    with pytest.raises(RuntimeError, match="identity changed"):
        with new_guard(config, out, state):
            pass
    assert not (out / LOCK_NAME).exists()


@pytest.mark.parametrize("mode", ["legacy", "anything"])
def test_strict_fingerprint_rejects_nonisolated_config(tmp_path, mode):
    config, out, state, _ = make_recipe(tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        GenerationRunGuard(config, "trial", out, state, config_mode=mode)
    assert not out.exists()


@pytest.mark.parametrize("existing", ["output", "state"])
def test_refuses_unguarded_existing_files(tmp_path, existing):
    config, out, state, _ = make_recipe(tmp_path)
    if existing == "output":
        out.mkdir()
        (out / "patches_log.csv").write_text("sample_id\na\n", encoding="utf-8")
    else:
        state.parent.mkdir()
        state.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unguarded output or processed state"):
        with new_guard(config, out, state):
            pass
    assert not (out / LOCK_NAME).exists()


def test_interrupted_run_is_left_running_and_cannot_be_adopted(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with pytest.raises(ZeroDivisionError):
        with new_guard(config, out, state):
            emit_patches(out, 1)
            1 / 0
    assert json.loads((out / LEDGER_NAME).read_text())["status"] == "running"
    with pytest.raises(RuntimeError, match="Incomplete strict generation"):
        with new_guard(config, out, state):
            pass


@pytest.mark.parametrize("mutation", ["log", "asset", "processed"])
def test_modified_outputs_or_state_prevent_clean_resume(tmp_path, mutation):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        emit_patches(out, 1)
        state.parent.mkdir()
        state.write_text('{"roi:product":true}', encoding="utf-8")
        session.finish()
    if mutation == "log":
        with (out / "patches_log.csv").open("a", encoding="utf-8") as stream:
            stream.write("sample_123456,fake\n")
    elif mutation == "asset":
        (out / "images" / "sample_000000.npy").write_bytes(b"more bytes")
    else:
        # Keep a valid completed product while changing the state contents:
        # this must exercise checkpoint fingerprint mismatch rather than
        # the separate "no completed product" guard.
        state.write_text(
            '{"roi:product":true,"unexpected:product":true}', encoding="utf-8"
        )
    with pytest.raises((RuntimeError, ValueError), match="changed since checkpoint|Incomplete patch|identities disagree"):
        with new_guard(config, out, state):
            pass


def test_refuses_active_lock_without_changing_ledger(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        with pytest.raises(RuntimeError, match="Existing generation lock"):
            with new_guard(config, out, state):
                pass
        session.finish()
    assert not (out / LOCK_NAME).exists()


def test_strict_snapshot_catches_file_count_disagreements(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state):
        emit_patches(out, 2)
        # The scenario tests asset consistency, so supply a valid product
        # checkpoint first; otherwise the guard correctly fails earlier.
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text('{"roi:product":true}', encoding="utf-8")
        (out / "masks" / "sample_000001.npy").unlink()
        with pytest.raises(RuntimeError, match="Incomplete patch assets"):
            output_snapshot(out, state)
    assert json.loads((out / LEDGER_NAME).read_text())["status"] == "running"


def test_source_vector_must_exist_and_be_complete(tmp_path):
    config, out, state, shp = make_recipe(tmp_path)
    shp.with_suffix(".shx").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete EuroCrops"):
        new_guard(config, out, state)
    assert not out.exists()


def test_guard_refuses_to_certify_patches_without_processed_checkpoint(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        emit_patches(out, 1)
        with pytest.raises(RuntimeError, match="processed-state checkpoint"):
            session.finish()
    assert json.loads((out / LEDGER_NAME).read_text())["status"] == "running"


def test_code_fingerprint_is_recorded_for_scientific_traceability(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    identity = run_identity(
        config, run_tag="trial", output_dir=out, processed_path=state,
        config_mode="isolated",
    )
    hashes = identity["generator_code_sha256"]
    # Keys are package-relative so the fingerprint is identical in a source
    # checkout and in an installed distribution. Ledger schema 2 renamed
    # them; a schema-1 ledger is refused rather than silently compared.
    assert identity["schema_version"] == 2
    assert "data_engine.py" in hashes
    assert "pipeline/dataset_generation.py" in hashes
    assert not any(key.startswith("src/") for key in hashes)
    assert all(len(digest) == 64 for digest in hashes.values())
    assert len(identity["taxonomy_sha256"]) == 64


def test_incomplete_processed_state_is_rejected_even_when_assets_match(tmp_path):
    config, out, state, _ = make_recipe(tmp_path)
    with new_guard(config, out, state) as session:
        emit_patches(out, 1)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text('{"roi:product":false}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="No completed ROI/product"):
            session.finish()
    assert json.loads((out / LEDGER_NAME).read_text())["status"] == "running"
