"""Geographically separated K-fold OOF manifest planning; no model execution.

This is a generic NEW-EXPERIMENT planner, not a recreation of the frozen
six-fold article experiment. Test is never rotated, read for geometry, or
exported. Original labels/images/masks and manuscript splits are immutable.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sentinel_crop_seg.core.splitting.spatial_split import (
    SpatialSplitConfig,
    build_windows,
    conflict_pairs,
    prepare_manifest,
    rebase_asset_paths,
)

CERTIFIED_TIERS = frozenset({"EXACT", "EXACT_TEMPORAL", "HIGH_CONFIDENCE", "CERTIFIED"})
DEVELOPMENT = frozenset({"train", "val"})
SPLITS = DEVELOPMENT | {"test", "excluded"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(seed: int, *items: object) -> str:
    return hashlib.sha256(("|".join(map(str, (seed, *items)))).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpatialOOFConfig:
    folds: int = 6
    seed: int = 42
    block_size_m: float = 30000.0
    guard_distance_m: float = 1280.0
    crs: str = "EPSG:3035"
    inner_val_fraction: float = 0.10
    require_spatial_tier: bool = True
    require_window_id: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpatialOOFConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"oof_folds"} or not isinstance(raw["oof_folds"], dict):
            raise ValueError("Expected only an oof_folds YAML mapping")
        section = raw["oof_folds"]
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown oof_folds keys: {sorted(unknown)}")
        value = cls(**section)
        value.validate()
        return value

    def validate(self) -> None:
        if type(self.folds) is not int or not 2 <= self.folds <= 32:
            raise ValueError("folds must be an integer in [2,32]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("block_size_m", "guard_distance_m", "inner_val_fraction"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
        if self.block_size_m <= 0 or self.guard_distance_m < 0:
            raise ValueError("invalid block_size_m or guard_distance_m")
        if not 0 < self.inner_val_fraction < 0.5:
            raise ValueError("inner_val_fraction must be in (0,0.5)")
        if type(self.require_spatial_tier) is not bool or type(self.require_window_id) is not bool:
            raise ValueError("provenance requirements must be Boolean")
        SpatialSplitConfig(
            block_size_m=self.block_size_m, guard_distance_m=self.guard_distance_m,
            crs=self.crs
        ).validate()

    def spatial_config(self) -> SpatialSplitConfig:
        return SpatialSplitConfig(
            block_size_m=self.block_size_m, guard_distance_m=self.guard_distance_m,
            crs=self.crs
        )


def _development_population(source: pd.DataFrame, cfg: SpatialOOFConfig):
    required = {"dataset", "sample_id", "split", "cohort", "img_path", "mask_path", "epsg",
                "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"OOF manifest missing columns: {sorted(missing)}")
    if cfg.require_spatial_tier and "final_spatial_tier" not in source:
        raise ValueError("Missing certified final_spatial_tier; cannot assert OOF geographic validity")
    if cfg.require_window_id and "window_id" not in source:
        raise ValueError("Missing window_id; cannot group historical acquisition windows")
    if source.duplicated(["dataset", "sample_id"]).any() or source[["dataset", "sample_id"]].isna().any().any():
        raise ValueError("Duplicate or missing sample identity")
    if not source["split"].isin(SPLITS).all():
        raise ValueError("Unknown source split; TEST must remain reserved")
    if "partition" in source and not source["partition"].equals(source["split"]):
        raise ValueError("source split/partition disagree")
    if any(field in source.columns for field in (
        "source_split", "source_partition", "role", "outer_fold_id", "cell_id", "unit_id", "_window"
    )):
        raise ValueError("Input already has reserved OOF output fields")
    dev = source[source["split"].isin(DEVELOPMENT)].copy()
    if dev.empty:
        raise ValueError("No development train/val rows")
    reason = pd.Series("", index=dev.index, dtype=object)
    if "window_id" in dev:
        no_window = dev["window_id"].fillna("").astype(str).str.strip().isin(["", "nan", "None"])
        reason.loc[no_window] = "missing_window_id"
    if "final_spatial_tier" in dev:
        good = dev["final_spatial_tier"].fillna("").astype(str).str.upper().str.strip().isin(CERTIFIED_TIERS)
        reason.loc[(~good) & reason.eq("")] = "uncertified_spatial_provenance"
    for key in ("epsg", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"):
        invalid = pd.to_numeric(dev[key], errors="coerce").isna()
        reason.loc[invalid & reason.eq("")] = "missing_georeferenced_footprint"
    excluded = dev[reason.ne("")].copy()
    excluded["oof_exclusion_reason"] = reason[reason.ne("")]
    eligible = dev[reason.eq("")].copy()
    if eligible.empty:
        raise ValueError("No spatially qualified development rows for OOF")
    # This is the only point at which train+val spatial coordinates are used.
    return eligible, excluded, int(source["split"].eq("test").sum())


def _linked_units(data: pd.DataFrame, windows: pd.DataFrame,
                  links: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    cells = sorted(windows["cell_id"].unique())
    parent = {c: c for c in cells}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    linked_parcels = 0
    covered_samples = 0
    if links is not None:
        if links.empty:
            raise ValueError("Explicit parcel-links file is empty; refusing an unaudited grouping")
        required = {"cohort", "parcel_key", "dataset", "sample_id"}
        if required - set(links):
            raise ValueError(f"parcel links missing columns: {sorted(required - set(links))}")
        if "partition" in links and not links["partition"].isin(DEVELOPMENT).all():
            raise ValueError("Parcel link table contains TEST/nondevelopment evidence")
        if links[list(required)].isna().any().any():
            raise ValueError("Parcel links cannot have null identities")
        lookup = data[["dataset", "sample_id", "cohort", "_window"]].merge(
            windows[["_window", "cell_id"]], on="_window", validate="many_to_one"
        )
        links = links.merge(lookup, on=["dataset", "sample_id"], how="inner",
                            suffixes=("_link", "_data"), validate="many_to_one")
        if links.empty:
            raise ValueError(
                "Parcel-links file has no eligible matching development samples; "
                "cannot establish parcel grouping"
            )
        if "cohort_link" in links:
            if not links["cohort_link"].eq(links["cohort_data"]).all():
                raise ValueError("Parcel link cohort disagrees with eligible sample cohort")
            links["cohort"] = links["cohort_link"]
        covered_samples = len(links[["dataset", "sample_id"]].drop_duplicates())
        for _, group in links.groupby(["cohort", "parcel_key"], sort=True):
            linked_parcels += 1
            touched = sorted(group["cell_id"].unique())
            if touched:
                root = find(touched[0])
                for cell in touched[1:]:
                    parent[find(cell)] = root

    result = windows.copy()
    result["unit_id"] = result["cell_id"].map(find)
    return result, {
        "parcel_link_rows_on_eligible_development": int(len(links)) if links is not None else 0,
        "eligible_samples_with_parcel_link": int(covered_samples),
        "unique_linked_parcels": int(linked_parcels),
        "parcel_level_coverage_complete": False,  # Links may include only audited parcels.
    }


def _allocation(windows: pd.DataFrame, cfg: SpatialOOFConfig) -> dict[str, int]:
    weights = windows.groupby("unit_id")["sample_count"].sum().to_dict()
    if len(weights) < cfg.folds + 1:
        raise ValueError(
            f"Need at least folds+1 independent spatial units for an inner validation split; "
            f"found {len(weights)} for {cfg.folds} folds"
        )
    # Seed-dependent unit order, then class-blind greedy balancing.
    ordered = sorted(weights, key=lambda u: _stable(cfg.seed, "outer_order", u))
    loads = [0] * cfg.folds
    assigned = {}
    for index, unit in enumerate(ordered):
        # Seed first K distinct units in separate folds to avoid empty folds.
        fold = index if index < cfg.folds else min(
            range(cfg.folds), key=lambda f: (loads[f], _stable(cfg.seed, "outer_tie", unit, f))
        )
        assigned[unit] = fold
        loads[fold] += int(weights[unit])
    return assigned


def _guard_drop(roles: np.ndarray, pairs: np.ndarray, *, held: str, training: str) -> np.ndarray:
    dropped = np.zeros(len(roles), dtype=bool)
    for left, right in pairs:
        if roles[left] == held and roles[right] == training:
            dropped[right] = True
        elif roles[right] == held and roles[left] == training:
            dropped[left] = True
    return dropped


def _check_pairs(roles: np.ndarray, pairs: np.ndarray, left_role: str, right_role: str):
    for left, right in pairs:
        if ((roles[left] == left_role and roles[right] == right_role) or
                (roles[right] == left_role and roles[left] == right_role)):
            raise ValueError(f"Geographic guard violation: {left_role} vs {right_role}")


def plan_spatial_oof(
    source: pd.DataFrame, cfg: SpatialOOFConfig, *,
    parcel_links: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return validated CSV-ready tables and audit, WITHOUT touching the filesystem."""
    cfg.validate()
    eligible, excluded, test_count = _development_population(source, cfg)
    if parcel_links is not None:
        required = {"dataset", "sample_id"}
        if not required.issubset(parcel_links):
            raise ValueError("Parcel links must identify dataset and sample_id")
        reserved_test = set(map(tuple, source.loc[
            source["split"].eq("test"), ["dataset", "sample_id"]
        ].to_numpy()))
        linked_test = set(map(tuple, parcel_links[["dataset", "sample_id"]].to_numpy()))
        if reserved_test & linked_test:
            raise ValueError("Parcel-links file references a reserved TEST sample (TEST/nondevelopment evidence)")
    spatial = cfg.spatial_config()
    data = prepare_manifest(eligible, spatial)
    windows = build_windows(data, spatial)
    windows, parcel_stats = _linked_units(data, windows, parcel_links)
    units = _allocation(windows, cfg)
    windows["outer_fold_id"] = windows["unit_id"].map(units).astype(int)
    indexed = data.merge(
        windows[["_window", "cell_id", "unit_id", "outer_fold_id"]],
        on="_window", validate="many_to_one"
    )
    pairs = conflict_pairs(windows, cfg.guard_distance_m)
    sample_keys = list(map(tuple, indexed[["dataset", "sample_id"]].to_numpy()))
    all_keys = set(sample_keys)
    if len(all_keys) != len(sample_keys):
        raise ValueError("Eligible sample identity collision")
    tables: dict[str, pd.DataFrame] = {}
    tables["fold_assignments.csv"] = indexed[[
        "dataset", "sample_id", "cohort", "source_split" if "source_split" in indexed else "split",
        "cell_id", "unit_id", "outer_fold_id"
    ]].copy().rename(columns={"split": "source_split"})
    summary_folds = []
    all_held = []
    for fold in range(cfg.folds):
        held = windows["outer_fold_id"].to_numpy() == fold
        outer = np.where(held, "held_out_fold", "outer_train").astype(object)
        outer_drop = _guard_drop(outer, pairs, held="held_out_fold", training="outer_train")
        outer[outer_drop] = "excluded_outer_guard"
        _check_pairs(outer, pairs, "held_out_fold", "outer_train")
        train_units = windows.loc[outer == "outer_train", "unit_id"].unique()
        if len(train_units) < 2:
            raise ValueError(f"fold {fold}: fewer than two independent inner-training units")
        unit_weights = (windows.loc[outer == "outer_train"]
                        .groupby("unit_id")["sample_count"].sum().to_dict())
        target = sum(unit_weights.values()) * cfg.inner_val_fraction
        candidates = sorted(train_units, key=lambda u: (
            unit_weights[u], _stable(cfg.seed, "inner_order", fold, u)
        ))
        chosen = []
        accumulated = 0
        for unit in candidates:
            if accumulated >= target:
                break
            chosen.append(unit)
            accumulated += unit_weights[unit]
        if not chosen or len(chosen) == len(train_units):
            raise ValueError(f"fold {fold}: cannot form independent inner validation units")
        inner = np.where(
            outer != "outer_train", outer,
            np.where(windows["unit_id"].isin(chosen), "inner_val", "inner_train")
        ).astype(object)
        inner_drop = _guard_drop(inner, pairs, held="inner_val", training="inner_train")
        inner[inner_drop] = "excluded_inner_guard"
        _check_pairs(inner, pairs, "inner_train", "inner_val")
        masks = {
            role: indexed["_window"].isin(windows.loc[inner == role, "_window"]).to_numpy()
            for role in ("held_out_fold", "inner_train", "inner_val", "excluded_outer_guard",
                         "excluded_inner_guard")
        }
        heldout = indexed.loc[masks["held_out_fold"]].copy()
        training = indexed.loc[masks["inner_train"] | masks["inner_val"]].copy()
        if heldout.empty or training.empty or not masks["inner_val"].any() or not masks["inner_train"].any():
            raise ValueError(f"fold {fold}: empty OOF, inner train or inner validation")
        # Guard conflicts are tested using projected, densified rectangles; no
        # same window or same linked physical-parcel UNIT crosses live roles.
        if set(training["unit_id"]) & set(heldout["unit_id"]):
            raise ValueError(f"fold {fold}: parcel/cell allocation unit leaked")
        training["source_split"] = training["split"]
        if "partition" in training:
            training["source_partition"] = training["partition"]
        training["split"] = np.where(training["unit_id"].isin(chosen), "val", "train")
        training["partition"] = training["split"]
        training["oof_role"] = np.where(training["split"].eq("val"), "inner_val", "inner_train")
        heldout["source_split"] = heldout["split"]
        if "partition" in heldout:
            heldout["source_partition"] = heldout["partition"]
        heldout["split"] = "held_out_fold"
        heldout["partition"] = "held_out_fold"
        heldout["oof_role"] = "held_out_fold"
        roles = indexed.copy()
        role_lookup = dict(zip(windows["_window"], inner))
        roles["oof_role"] = roles["_window"].map(role_lookup)
        roles["source_split"] = roles["split"]
        if "partition" in roles:
            roles["source_partition"] = roles["partition"]
        roles["outer_fold_id"] = fold
        for table in (training, heldout, roles):
            table.drop(columns=["_window"], inplace=True)
            table["outer_fold_id"] = fold
        stem = f"fold_{fold:02d}"
        tables[f"{stem}_training_manifest.csv"] = training
        tables[f"{stem}_held_out_manifest.csv"] = heldout
        tables[f"{stem}_roles.csv"] = roles
        held_keys = set(map(tuple, heldout[["dataset", "sample_id"]].to_numpy()))
        all_held.extend(held_keys)
        summary_folds.append({
            "fold": fold, "held_out_samples": len(heldout),
            "inner_train_samples": int(training["split"].eq("train").sum()),
            "inner_val_samples": int(training["split"].eq("val").sum()),
            "outer_guard_dropped_windows": int(outer_drop.sum()),
            "inner_guard_dropped_windows": int(inner_drop.sum()),
            "held_out_units": int(windows.loc[held, "unit_id"].nunique()),
            "training_units": int(training["unit_id"].nunique()),
        })
    if len(all_held) != len(all_keys) or len(set(all_held)) != len(all_keys) or set(all_held) != all_keys:
        raise ValueError("OOF held-out coverage or uniqueness failed")
    if set(map(tuple, source.loc[source["split"].eq("test"), ["dataset", "sample_id"]].to_numpy())) & set(all_held):
        raise ValueError("TEST sample found in OOF held-out membership")
    tables["oof_ineligible_development.csv"] = excluded
    audit = {
        "status": "validated_fold_manifests_only_NO_OOF_predictions",
        "folds": cfg.folds, "repeats": 1, "test_rows_reserved_untouched": test_count,
        "eligible_development_rows": len(indexed), "excluded_development_rows": len(excluded),
        "eligible_windows": len(windows), "spatial_units": len(units),
        "all_eligible_samples_held_out_exactly_once": True,
        "geographic_guard_checked_for_every_fold": True,
        "parcel_linkage": parcel_stats,
        "config": asdict(cfg),
        "fold_summaries": summary_folds,
        "publication_ready": False,
        "notes": [
            "The original frozen train/val/test split is not rewritten.",
            "Held-out manifests are NOT trainer validation; inner validation is separate.",
            "Parcel-level nonleakage is only as complete as the supplied parcel links.",
            "Predictions, checkpoints, label uncertainty and human decisions are not generated.",
            "This is a new generic planner; not a reproduction of the frozen historical K6 experiment.",
        ],
    }
    return tables, audit


def run_spatial_oof(
    config_path: str | Path, manifest_path: str | Path, out_dir: str | Path, *,
    parcel_links_path: str | Path | None = None,
    validate_only: bool = False,
) -> dict:
    cfg = SpatialOOFConfig.from_yaml(config_path)
    source_path = Path(manifest_path)
    links = None
    if parcel_links_path is not None:
        links = pd.read_csv(parcel_links_path, low_memory=False)
    tables, audit = plan_spatial_oof(
        pd.read_csv(source_path, low_memory=False), cfg, parcel_links=links
    )
    audit["input_sha256"] = {
        "manifest": _sha256(source_path), "config": _sha256(Path(config_path)),
        "parcel_links": _sha256(Path(parcel_links_path)) if parcel_links_path else None,
    }
    dest = Path(out_dir)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing OOF output: {dest}")
    # Validate path rebasing BEFORE creating an output directory.
    for name, table in tables.items():
        tables[name] = rebase_asset_paths(table, source_path, dest)
    if validate_only:
        return audit
    partial = dest.with_name(dest.name + ".tmp")
    if partial.exists():
        raise FileExistsError(f"Temporary OOF output exists: {partial}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        for name, table in tables.items():
            table.to_csv(partial / name, index=False)
        (partial / "oof_fold_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        partial.rename(dest)
    except Exception:
        # Retain an interrupted staging directory for forensic inspection.
        raise
    return audit
