from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.

    If both base[key] and incoming[key] are dicts, merge them recursively.
    Otherwise, incoming[key] overwrites base[key].
    """
    for key, value in incoming.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_override_value(raw_value: str) -> Any:
    """
    Parse a CLI override value using YAML syntax.

    Examples:
        "1" -> 1
        "0.5" -> 0.5
        "true" -> True
        "false" -> False
        "null" -> None
        "[0, 1, 2]" -> [0, 1, 2]
        "data/path" -> "data/path"
    """
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def apply_override(config: MutableMapping[str, Any], override: str) -> None:
    """
    Apply one dotted-key override to a nested configuration dictionary.

    Example:
        apply_override(config, "dataset.target.total_patches=8")
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid override '{override}'. Expected format: key.subkey=value"
        )

    raw_key, raw_value = override.split("=", 1)
    key = raw_key.strip()

    if not key:
        raise ValueError(f"Invalid override '{override}'. Empty key.")

    parts = key.split(".")
    value = parse_override_value(raw_value.strip())

    current: MutableMapping[str, Any] = config

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}

        if not isinstance(current[part], MutableMapping):
            raise ValueError(
                f"Cannot apply override '{override}'. "
                f"'{part}' already exists and is not a dictionary."
            )

        current = current[part]

    current[parts[-1]] = value


def apply_overrides(
    config: MutableMapping[str, Any],
    overrides: Sequence[str] | None,
) -> MutableMapping[str, Any]:
    """
    Apply multiple dotted-key overrides to a configuration dictionary.
    """
    if not overrides:
        return config

    for override in overrides:
        apply_override(config, override)

    return config


def load_config(
    config_dir: str | Path,
    overrides: Sequence[str] | None = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    config_path = Path(config_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    for yaml_file in sorted(config_path.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            _deep_merge(config, data)

    apply_overrides(config, overrides)

    return config