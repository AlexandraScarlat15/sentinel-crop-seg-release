"""Versioned, read-only mask-path resolution for manifest training datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


HISTORICAL_VERSION = "historical"
CORRECTED_V1_VERSION = "taxonomy-mask-corrections-v1"
SUPPORTED_VERSIONS = {HISTORICAL_VERSION, CORRECTED_V1_VERSION}


def _portable_parts(value: str | Path) -> tuple[str, ...]:
    """Split a Windows or POSIX path without binding resolution to either host."""
    return tuple(part for part in str(value).replace("\\", "/").split("/") if part)


def sample_id_basename(value: str | Path) -> str:
    return PurePosixPath(str(value).replace("\\", "/")).stem


def normalized_mask_identity(dataset: str, sample_id: str | Path, mask_path: str | Path) -> tuple[str, str, str]:
    """Return a portable dataset/mask/sample key and reject inconsistent rows."""
    dataset = str(dataset)
    sample_stem = sample_id_basename(sample_id)
    parts = _portable_parts(mask_path)
    if len(parts) < 3 or parts[-3] != dataset or parts[-2] != "masks":
        raise ValueError(
            "Manifest mask path must end with '<dataset>/masks/<sample>.npy': "
            f"dataset={dataset!r}, mask_path={mask_path!s}"
        )
    filename = parts[-1]
    if sample_id_basename(filename) != sample_stem:
        raise ValueError(
            "Manifest sample_id and mask filename disagree: "
            f"sample_id={sample_id!s}, mask_path={mask_path!s}"
        )
    return dataset, f"{dataset}/masks/{filename}", sample_stem


def _corrected_relative_path(raw_path: str | Path) -> Path:
    parts = _portable_parts(raw_path)
    positions = [index for index, value in enumerate(parts) if value == "masks"]
    if not positions:
        raise ValueError(f"Corrected manifest path has no masks component: {raw_path!s}")
    return Path(*parts[positions[-1]:])


@dataclass(frozen=True)
class MaskResolution:
    path: Path
    corrected: bool


@dataclass(frozen=True)
class MaskResolverSettings:
    """Portable configuration for a label-version selection."""

    version: str = HISTORICAL_VERSION
    corrected_manifest: Path | None = None
    corrected_mask_root: Path | None = None
    subset_to_dataset: dict[str, str] | None = None
    expected_corrected_records: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MaskResolverSettings":
        data = dict(value or {})
        version = str(data.get("version", HISTORICAL_VERSION))
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported labels.version={version!r}")
        manifest = data.get("corrected_manifest")
        root = data.get("corrected_mask_root")
        mapping = {str(key): str(item) for key, item in (data.get("subset_to_dataset", {}) or {}).items()}
        expected = data.get("expected_corrected_records")
        if version == CORRECTED_V1_VERSION and (not manifest or not root or not mapping):
            raise ValueError(
                "taxonomy-mask-corrections-v1 requires corrected_manifest, "
                "corrected_mask_root, and subset_to_dataset"
            )
        return cls(
            version=version,
            corrected_manifest=Path(manifest) if manifest else None,
            corrected_mask_root=Path(root) if root else None,
            subset_to_dataset=mapping,
            expected_corrected_records=None if expected is None else int(expected),
        )

    def provenance(self) -> dict[str, object]:
        return {
            "label_version": self.version,
            "corrected_manifest": None if self.corrected_manifest is None else str(self.corrected_manifest),
            "corrected_mask_root": None if self.corrected_mask_root is None else str(self.corrected_mask_root),
            "corrected_record_count": self.expected_corrected_records if self.version == CORRECTED_V1_VERSION else 0,
        }


def load_mask_resolver_settings(label_config: str | Path | None) -> MaskResolverSettings:
    """Load the ``labels`` block from a YAML config; None means historical."""
    if label_config is None:
        return MaskResolverSettings()
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required to read --label-config") from error
    with Path(label_config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError("Label config must be a YAML mapping")
    return MaskResolverSettings.from_mapping(config.get("labels", config))


def build_mask_resolver(settings: MaskResolverSettings) -> "VersionedMaskResolver":
    return VersionedMaskResolver(
        version=settings.version,
        corrected_manifest=settings.corrected_manifest,
        corrected_mask_root=settings.corrected_mask_root,
        subset_to_dataset=settings.subset_to_dataset,
        expected_corrected_records=settings.expected_corrected_records,
    )


def require_manifest_data_source(settings: MaskResolverSettings, data_source: str) -> None:
    """Prevent unsafe corrected-label use where manifest identity is unavailable."""
    if settings.version == CORRECTED_V1_VERSION and data_source != "manifest":
        raise ValueError("taxonomy-mask-corrections-v1 requires manifest-based data loading")


class VersionedMaskResolver:
    """Resolve only manifest-listed labels; never copy or modify a mask file."""

    def __init__(
        self,
        *,
        version: str,
        corrected_manifest: str | Path | None = None,
        corrected_mask_root: str | Path | None = None,
        subset_to_dataset: Mapping[str, str] | None = None,
        expected_corrected_records: int | None = None,
    ) -> None:
        self.version = str(version)
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported label version: {self.version!r}")
        self._index: dict[tuple[str, str, str], Path] = {}
        if self.version == HISTORICAL_VERSION:
            return
        if corrected_manifest is None or corrected_mask_root is None or not subset_to_dataset:
            raise ValueError(
                "taxonomy-mask-corrections-v1 requires corrected_manifest, "
                "corrected_mask_root, and an explicit subset_to_dataset mapping"
            )
        manifest_path = Path(corrected_manifest)
        root = Path(corrected_mask_root)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("revision") != CORRECTED_V1_VERSION:
            raise ValueError("Corrected manifest revision is not taxonomy-mask-corrections-v1")
        for record in payload.get("patches", []):
            subset = str(record["subset"])
            if subset not in subset_to_dataset:
                raise ValueError(f"Missing subset_to_dataset entry for {subset!r}")
            dataset = str(subset_to_dataset[subset])
            sample_id = str(record["sample_id"])
            filename = f"{sample_id}.npy"
            key = (dataset, f"{dataset}/masks/{filename}", sample_id_basename(sample_id))
            if key in self._index:
                raise ValueError(f"Ambiguous corrected-mask key: {key}")
            self._index[key] = root / _corrected_relative_path(record["corrected_path"])
        if expected_corrected_records is not None and len(self._index) != int(expected_corrected_records):
            raise ValueError(
                f"Corrected record count mismatch: expected {expected_corrected_records}, got {len(self._index)}"
            )

    @property
    def corrected_record_count(self) -> int:
        return len(self._index)

    def resolve(self, *, dataset: str, sample_id: str | Path, original_mask_path: str | Path) -> MaskResolution:
        original = Path(original_mask_path)
        if self.version == HISTORICAL_VERSION:
            return MaskResolution(path=original, corrected=False)
        key = normalized_mask_identity(dataset, sample_id, original)
        corrected = self._index.get(key)
        return MaskResolution(path=original if corrected is None else corrected, corrected=corrected is not None)
