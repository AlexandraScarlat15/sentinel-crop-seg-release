"""Reusable spatial-block splitting with audited footprint separation.

No historical experiment files are imported, rewritten or required. Geometries
are conservatively enclosed in projected axis-aligned rectangles.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pyproj
import yaml
from scipy.spatial import cKDTree

PARTITIONS = ("train", "val", "test")
BOX = ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")


@dataclass(frozen=True)
class SpatialSplitConfig:
    block_size_m: float = 30000.0
    guard_distance_m: float = 1280.0
    crs: str = "EPSG:3035"
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    random_seed: int = 42
    candidate_seeds: int = 200
    stratify_by: str = "country"
    parcel_id_column: str | None = None
    save_split_map: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpatialSplitConfig":
        with Path(path).open(encoding="utf-8") as stream:
            root = yaml.safe_load(stream)
        if not isinstance(root, dict) or not isinstance(root.get("splitting"), dict):
            raise ValueError("Expected a splitting YAML mapping")
        settings = root["splitting"]
        allowed = {"strategy", "spatial", "grouping", "partitions", "allocation", "quality", "outputs"}
        if set(settings) - allowed:
            raise ValueError(f"Unknown splitting keys: {sorted(set(settings) - allowed)}")
        if settings.get("strategy", "spatial_block") != "spatial_block":
            raise ValueError("Only spatial_block is supported")
        allowed_nested = {
            "spatial": {"crs", "block_size_m", "guard_distance_m"},
            "grouping": {"same_window_same_partition", "parcel_id_column"},
            "partitions": set(PARTITIONS),
            "allocation": {"random_seed", "candidate_seeds", "stratify_by", "priority"},
            "quality": {"verify_leakage", "fail_on_leakage"},
            "outputs": {"save_split_map"},
        }
        sections = {}
        for name, allowed_keys in allowed_nested.items():
            section = settings.get(name, {})
            if not isinstance(section, dict) or set(section) - allowed_keys:
                raise ValueError(f"Invalid or unknown splitting.{name} fields")
            sections[name] = section
        spatial, grouping, ratios, allocation, quality, outputs = (
            sections[k] for k in ("spatial", "grouping", "partitions", "allocation", "quality", "outputs")
        )
        if grouping.get("same_window_same_partition", True) is not True:
            raise ValueError("Window atomicity cannot be disabled")
        if quality.get("verify_leakage", True) is not True or quality.get("fail_on_leakage", True) is not True:
            raise ValueError("Leakage auditing and fail-closed behavior cannot be disabled")
        if allocation.get("priority", ["test", "val", "train"]) != ["test", "val", "train"]:
            raise ValueError("Only priority [test, val, train] is supported")
        config = cls(
            block_size_m=spatial.get("block_size_m", 30000.0),
            guard_distance_m=spatial.get("guard_distance_m", 1280.0),
            crs=spatial.get("crs", "EPSG:3035"),
            train=ratios.get("train", 0.70), val=ratios.get("val", 0.15),
            test=ratios.get("test", 0.15),
            random_seed=allocation.get("random_seed", 42),
            candidate_seeds=allocation.get("candidate_seeds", 200),
            stratify_by=allocation.get("stratify_by", "country"),
            parcel_id_column=grouping.get("parcel_id_column"),
            save_split_map=outputs.get("save_split_map", False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for key in ("block_size_m", "guard_distance_m", "train", "val", "test"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
        if self.block_size_m <= 0 or self.guard_distance_m < 0:
            raise ValueError("block_size_m must be positive; guard_distance_m cannot be negative")
        if min(self.train, self.val, self.test) <= 0 or not math.isclose(
            self.train + self.val + self.test, 1, abs_tol=1e-8
        ):
            raise ValueError("Positive partition shares must sum to 1")
        if (not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool)
                or not isinstance(self.candidate_seeds, int) or isinstance(self.candidate_seeds, bool)
                or self.candidate_seeds <= 0 or self.random_seed < 0):
            raise ValueError("random_seed must be integer and candidate_seeds positive integer")
        if not isinstance(self.save_split_map, bool):
            raise ValueError("save_split_map must be boolean")
        if self.stratify_by not in ("country", "cohort", "none"):
            raise ValueError("stratify_by must be country, cohort, or none")
        target = pyproj.CRS.from_user_input(self.crs)
        if not target.is_projected or any(
            abs(axis.unit_conversion_factor - 1) > 1e-8 for axis in target.axis_info[:2]
        ):
            raise ValueError("Target CRS must be projected with metre units")


def prepare_manifest(manifest: pd.DataFrame, config: SpatialSplitConfig) -> pd.DataFrame:
    config.validate()
    if manifest.empty:
        raise ValueError("Empty manifest")
    required = {"sample_id", "epsg", *BOX}
    if required - set(manifest):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(manifest))}")
    data = manifest.copy().reset_index(drop=True)
    if "dataset" not in data:
        data["dataset"] = "default"
    if data[["dataset", "sample_id"]].isna().any().any() or data.duplicated(["dataset", "sample_id"]).any():
        raise ValueError("Each (dataset, sample_id) must be unique and non-null")
    for column in ("epsg", *BOX):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {column} in manifest")
    if not (data["epsg"] == data["epsg"].astype(int)).all():
        raise ValueError("epsg must be an integer code")
    if not ((data["bbox_maxx"] > data["bbox_minx"]) & (
        data["bbox_maxy"] > data["bbox_miny"]
    )).all():
        raise ValueError("All patch bounding boxes must have positive dimensions")
    for column in ("country", "cohort"):
        if column not in data:
            data[column] = "unspecified"
        data[column] = data[column].fillna("unspecified").astype(str)
    if config.parcel_id_column:
        c = config.parcel_id_column
        if c not in data or data[c].isna().any():
            raise ValueError(f"parcel_id_column={c!r} must be present and non-null")
        data[c] = data[c].astype(str)
    data["_window"] = data["epsg"].astype(int).astype(str) + ":" + data[list(BOX)].apply(
        lambda row: ":".join(f"{float(v):.6f}" for v in row), axis=1
    )
    if "window_id" in data and (
        data["window_id"].isna().any() or data.groupby("window_id")["_window"].nunique().gt(1).any()
    ):
        raise ValueError("window_id is null or has conflicting footprints")
    return data


def build_windows(data: pd.DataFrame, config: SpatialSplitConfig) -> pd.DataFrame:
    transformers = {}
    rows = []
    for window, g in data.groupby("_window", sort=True):
        sample = g.iloc[0]
        epsg = int(sample["epsg"])
        if epsg not in transformers:
            source = pyproj.CRS.from_epsg(epsg)
            if not source.is_projected:
                raise ValueError("Source EPSG must be projected")
            transformers[epsg] = pyproj.Transformer.from_crs(source, config.crs, always_xy=True)
        x1, y1, x2, y2 = (float(sample[k]) for k in BOX)
        # Densify the box edges before reprojection: corner-only projection
        # can miss extrema along curved transformed edges.
        lo_x, lo_y, hi_x, hi_y = transformers[epsg].transform_bounds(
            x1, y1, x2, y2, densify_pts=21
        )
        if not np.isfinite([lo_x, lo_y, hi_x, hi_y]).all():
            raise ValueError("Non-finite reprojected coordinates")
        x, y = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        rows.append({
            "_window": window, "x": x, "y": y,
            "half_x": (hi_x - lo_x) / 2, "half_y": (hi_y - lo_y) / 2,
            "cell_id": f"{math.floor(x / config.block_size_m)}:{math.floor(y / config.block_size_m)}",
            "sample_count": len(g),
        })
    return pd.DataFrame(rows)


def footprint_gaps(windows: pd.DataFrame, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = windows["x"].to_numpy(); y = windows["y"].to_numpy()
    hx = windows["half_x"].to_numpy(); hy = windows["half_y"].to_numpy()
    dx = np.maximum(0.0, abs(x[left] - x[right]) - hx[left] - hx[right])
    dy = np.maximum(0.0, abs(y[left] - y[right]) - hy[left] - hy[right])
    return np.hypot(dx, dy)


def conflict_pairs(windows: pd.DataFrame, distance_m: float) -> np.ndarray:
    if len(windows) < 2:
        return np.empty((0, 2), dtype=int)
    points = windows[["x", "y"]].to_numpy()
    size = max(windows["half_x"].max(), windows["half_y"].max())
    candidates = cKDTree(points).query_pairs(
        math.sqrt(2) * (2 * size + distance_m) + 1e-7, output_type="ndarray"
    )
    if not len(candidates):
        return np.empty((0, 2), dtype=int)
    gaps = footprint_gaps(windows, candidates[:, 0], candidates[:, 1])
    # Touching or overlapping footprints always conflict, including guard=0.
    return candidates[(gaps < distance_m) | (gaps <= 1e-8)]


def allocation_units(data: pd.DataFrame, windows: pd.DataFrame,
                     config: SpatialSplitConfig) -> pd.DataFrame:
    cells = sorted(windows["cell_id"].unique())
    parent = {c: c for c in cells}

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    if config.parcel_id_column:
        linked = data[["_window", config.parcel_id_column]].merge(
            windows[["_window", "cell_id"]], on="_window", validate="many_to_one"
        )
        for _, group in linked.groupby(config.parcel_id_column, sort=True):
            touched = sorted(group["cell_id"].unique())
            for c in touched[1:]:
                parent[find(c)] = find(touched[0])
    windows = windows.copy()
    windows["unit_id"] = windows["cell_id"].map(find)
    return windows


def assign_units(windows: pd.DataFrame, data: pd.DataFrame,
                 config: SpatialSplitConfig, seed: int) -> dict[str, str]:
    labels = data[["_window", "country", "cohort"]].groupby("_window", sort=True).agg(
        country=("country", lambda s: sorted(s)[0]),
        cohort=("cohort", lambda s: sorted(s)[0]),
    )
    marked = windows.join(labels, on="_window")
    units = marked.groupby("unit_id", sort=True).agg(
        sample_count=("sample_count", "sum"),
        country=("country", "first"), cohort=("cohort", "first")
    ).reset_index()
    if len(units) < 3:
        raise ValueError("Fewer than three independent spatial allocation units")
    rng = np.random.default_rng(seed)
    groups = ([units] if config.stratify_by == "none" else
              [group for _, group in units.groupby(config.stratify_by, sort=True)])
    ratios = {p: getattr(config, p) for p in PARTITIONS}
    result = {}
    for group in groups:
        got = {p: 0 for p in PARTITIONS}
        total = group["sample_count"].sum()
        for j in rng.permutation(len(group)):
            unit = group.iloc[int(j)]
            chosen = max(PARTITIONS, key=lambda p: (ratios[p] - got[p] / total, p == "train", p))
            result[unit["unit_id"]] = chosen
            got[chosen] += int(unit["sample_count"])
    return result


def exclude_conflicts(parts: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    dropped = np.zeros(len(parts), dtype=bool)
    if not len(pairs):
        return dropped
    a, b = pairs[:, 0], pairs[:, 1]
    val_test = (((parts[a] == "val") & (parts[b] == "test")) |
                ((parts[b] == "val") & (parts[a] == "test")))
    dropped[np.where(parts[a[val_test]] == "val", a[val_test], b[val_test])] = True
    active = ~dropped
    cross = (parts[a] != parts[b]) & active[a] & active[b]
    train_eval = cross & ((((parts[a] == "train") & (parts[b] != "train"))) |
                          (((parts[b] == "train") & (parts[a] != "train"))))
    dropped[np.where(parts[a[train_eval]] == "train", a[train_eval], b[train_eval])] = True
    return dropped


def minimum_footprint_gap(windows: pd.DataFrame, left: np.ndarray,
                          right: np.ndarray) -> float | None:
    """Exact minimum over rectangle pairs, not a nearest-centroid approximation."""
    if not len(left) or not len(right):
        return None
    points = windows[["x", "y"]].to_numpy()
    hx = windows["half_x"].to_numpy(); hy = windows["half_y"].to_numpy()
    tree = cKDTree(points[right])
    _, nearest = tree.query(points[left], k=1)
    nearest_ids = right[np.asarray(nearest).reshape(-1)]
    best = float(footprint_gaps(windows, left, nearest_ids).min())
    max_right = float(max(hx[right].max(), hy[right].max()))
    for i in left:
        radius = best + math.sqrt(2) * (max(hx[i], hy[i]) + max_right) + 1e-7
        candidates = tree.query_ball_point(points[i], radius)
        if candidates:
            other = right[np.asarray(candidates, dtype=int)]
            best = min(best, float(footprint_gaps(
                windows, np.full(len(other), i, dtype=int), other
            ).min()))
    return best


def split_dataframe(manifest: pd.DataFrame, config: SpatialSplitConfig):
    """Compute (manifest, summary, audit, class_distribution), without I/O."""
    data = prepare_manifest(manifest, config)
    windows = allocation_units(data, build_windows(data, config), config)
    if windows["cell_id"].nunique() < 3:
        raise ValueError("At least three geographical cells are needed")
    pairs = conflict_pairs(windows, config.guard_distance_m)
    weights = windows["sample_count"].to_numpy(dtype=int)
    best = None
    for n in range(config.candidate_seeds):
        seed = config.random_seed + n
        unit_parts = assign_units(windows, data, config, seed)
        parts = windows["unit_id"].map(unit_parts).to_numpy(dtype=object)
        dropped = exclude_conflicts(parts, pairs)
        counts = {p: int(weights[(parts == p) & ~dropped].sum()) for p in PARTITIONS}
        retained = sum(counts.values())
        if min(counts.values()) == 0:
            continue
        error = sum(abs(counts[p] / retained - getattr(config, p)) for p in PARTITIONS)
        score = (int(weights[dropped].sum()), round(error, 12), seed)
        if best is None or score < best[0]:
            best = (score, parts, dropped, counts, seed)
    if best is None:
        raise ValueError("No valid nonempty train/val/test split found")
    _, parts, dropped, counts, seed = best
    windows["partition"] = np.where(dropped, "excluded", parts)
    # Preserve historical membership without allowing pandas merge suffixes
    # to silently change the meaning of either partition column.
    for original, preserved in (
        ("partition", "source_partition"),
        ("split", "source_split"),
    ):
        if original in data:
            if preserved in data:
                raise ValueError(
                    f"Cannot preserve existing {original}: {preserved} already exists"
                )
            data = data.rename(columns={original: preserved})
    for reserved in ("cell_id", "exclusion_reason"):
        if reserved in data:
            raise ValueError(f"Input manifest already contains reserved column {reserved}")
    result = data.merge(windows[["_window", "cell_id", "partition"]],
                        on="_window", validate="many_to_one").drop(columns="_window")
    result["split"] = result["partition"]
    result["exclusion_reason"] = np.where(
        result["partition"] == "excluded", "guard_distance", ""
    )
    audit = {"clean": True, "crs": config.crs, "guard_distance_m": config.guard_distance_m,
             "method": "densified_projected_footprint_bounding_rectangle", "pairs": {}}
    for p, q in (("train", "val"), ("train", "test"), ("val", "test")):
        left = np.flatnonzero(windows["partition"].to_numpy() == p)
        right = np.flatnonzero(windows["partition"].to_numpy() == q)
        gap = minimum_footprint_gap(windows, left, right)
        violation = gap is None or gap < config.guard_distance_m - 1e-8 or gap <= 1e-8
        audit["pairs"][f"{p}_vs_{q}"] = {
            "minimum_footprint_gap_m": gap, "violations": int(violation)
        }
        audit["clean"] = audit["clean"] and not violation
    if config.parcel_id_column and result.groupby(config.parcel_id_column)["partition"].apply(
        lambda s: len(set(s) - {"excluded"}) > 1
    ).any():
        audit["clean"] = False
    if not audit["clean"]:
        raise RuntimeError("Spatial leakage audit failed; refusing to publish split")
    retained = sum(counts.values())
    summary = {
        "samples_input": len(result), "samples_retained": retained,
        "samples_excluded_guard": len(result) - retained,
        "windows": len(windows), "cells": int(windows["cell_id"].nunique()),
        "block_size_m": config.block_size_m, "selected_seed": seed,
        "candidate_seeds": config.candidate_seeds,
        "partitions": {
            p: {"samples": counts[p], "target_share": getattr(config, p),
                "actual_share": counts[p] / retained} for p in PARTITIONS
        }, "leakage_audit_clean": True,
    }
    histogram_cols = [c for c in result.columns if len(c) > 1
                      and c[0] in ("p", "c") and c[1:].isdigit()]
    distribution = pd.DataFrame(columns=["partition", "class_id", "source_column", "pixel_count"])
    if histogram_cols:
        hist = result[histogram_cols].apply(pd.to_numeric, errors="raise")
        if hist.isna().any().any() or (hist < 0).any().any():
            raise ValueError("Class histogram counts must be nonnegative and non-null")
        distribution = pd.DataFrame([
            {"partition": p, "class_id": int(c[1:]), "source_column": c,
             "pixel_count": float(hist.loc[result["partition"] == p, c].sum())}
            for p in PARTITIONS for c in histogram_cols
        ])
    else:
        summary["class_coverage_note"] = "Not computed: missing p0..pN/c0..cN histograms"
    return result, summary, audit, distribution


def rebase_asset_paths(manifest: pd.DataFrame, source_manifest: Path,
                       output_dir: Path) -> pd.DataFrame:
    """Keep img_path/mask_path references valid when a CSV moves directories.

    Relative paths in the input are relative to the original manifest directory.
    Express them relative to the new output directory; keep absolute paths as
    supplied. On Windows, cross-drive relative paths cannot be represented,
    so write the resolved absolute target instead. No assets are copied.
    """
    rebased = manifest.copy()
    source_dir = source_manifest.resolve().parent
    destination = output_dir.resolve()

    for column in ("img_path", "mask_path"):
        if column not in rebased:
            continue

        def convert(value: object) -> str:
            if pd.isna(value) or not str(value).strip():
                raise ValueError(f"{column} contains an empty asset path")
            raw = str(value)
            original = Path(raw)
            if original.is_absolute() or PureWindowsPath(raw).is_absolute() or raw.startswith("\\\\"):
                # Keep native or foreign-platform absolute references unchanged.
                return raw
            if os.name != "nt" and "\\" in raw:
                # Accept relative paths authored on Windows when running on POSIX.
                original = Path(*PureWindowsPath(raw).parts)
            target = (source_dir / original).resolve()
            try:
                return os.path.relpath(target, start=destination)
            except ValueError:
                # os.path.relpath cannot cross drive letters on Windows.
                return str(target)

        rebased[column] = rebased[column].map(convert)
    return rebased


def run_from_files(config_path: str | Path, manifest_path: str | Path,
                   output_dir: str | Path, *, validate_only: bool = False) -> dict:
    config = SpatialSplitConfig.from_yaml(config_path)
    source = Path(manifest_path)
    result, summary, audit, distribution = split_dataframe(
        pd.read_csv(source, low_memory=False), config
    )
    summary["manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary["config_sha256"] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    output = Path(output_dir)
    if not validate_only and output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    # Check all path values even for a dry-run, without accessing or altering data.
    result = rebase_asset_paths(result, source, output)
    if validate_only:
        return summary
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "split_manifest.csv", index=False)
    result[["dataset", "sample_id", "cell_id", "partition"]].to_csv(
        output / "split_assignments.csv", index=False
    )
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not distribution.empty:
        distribution.to_csv(output / "class_distribution.csv", index=False)
    if config.save_split_map:
        write_split_map(result, config, output / "spatial_split_map.png")
    return summary


def write_split_map(manifest: pd.DataFrame, config: SpatialSplitConfig,
                    destination: str | Path) -> None:
    """Render selected/excluded footprint centres in the configured metric CRS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = prepare_manifest(manifest, config)
    windows = build_windows(data, config)
    partitions = data.groupby("_window", sort=True)["partition"].first()
    windows["partition"] = windows["_window"].map(partitions)
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in ("train", "val", "test", "excluded"):
        group = windows[windows["partition"] == label]
        if not group.empty:
            ax.scatter(group["x"], group["y"], s=6, alpha=0.65,
                       label=f"{label} ({len(group)} windows)")
    ax.set_xlabel(f"X ({config.crs}; metres)")
    ax.set_ylabel(f"Y ({config.crs}; metres)")
    ax.set_title("Geographically separated patch windows")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
