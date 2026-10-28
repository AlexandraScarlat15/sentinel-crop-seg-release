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


def _available(container: MutableMapping[str, Any]) -> str:
    """Render a level's keys for an error message, bounded in length."""
    keys = sorted(str(k) for k in container)
    shown = ", ".join(keys[:12])
    return shown + (", ..." if len(keys) > 12 else "") if keys else "(none)"


def apply_override(
    config: MutableMapping[str, Any],
    override: str,
    *,
    allow_new_keys: bool = False,
    known_paths: frozenset[str] | None = None,
) -> None:
    """Apply one dotted-key override to a nested configuration dictionary.

    Every path component must already exist. A dotted path used to create
    whatever it named, so a mistyped `--override` built a subtree nothing
    reads and the run continued on the default - a plausible artifact with a
    silently wrong setting, which is the failure mode this project has had to
    chase repeatedly.

    The path is checked in full before anything is written, so a rejected
    override leaves the configuration exactly as it was and cannot deposit a
    half-built subtree.

    Args:
        config: Already-loaded configuration to modify in place.
        override: One `key.subkey=value` string.
        allow_new_keys: Permit introducing paths the configuration does not
            define. For a caller that deliberately extends a configuration,
            never for CLI input.
        known_paths: Dotted paths that are valid even when the configuration
            file omits them. Needed where a consumer reads a setting with a
            default, so the file is a subset of what the setting space allows
            and the file alone would reject a legitimate override.

    Raises:
        ValueError: If the format is invalid, the path is unknown while
            `allow_new_keys` is false, or the path traverses a value that is
            not a mapping.

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
    if any(not part for part in parts):
        raise ValueError(
            f"Invalid override '{override}'. Empty path component in '{key}'."
        )
    value = parse_override_value(raw_value.strip())
    permitted = allow_new_keys or (known_paths is not None and key in known_paths)

    # Phase one: walk the whole path without writing. Nothing is created
    # until the entire path is known to be valid. A path that traverses a
    # non-mapping is rejected whatever the permissions, because no value can
    # live inside a scalar.
    cursor: MutableMapping[str, Any] = config
    for depth, part in enumerate(parts[:-1]):
        if part not in cursor:
            if not permitted:
                raise ValueError(
                    f"Unknown configuration path in override '{override}': "
                    f"'{'.'.join(parts[: depth + 1])}' is not defined. "
                    f"Available keys at this level: {_available(cursor)}"
                )
            break
        if not isinstance(cursor[part], MutableMapping):
            raise ValueError(
                f"Cannot apply override '{override}'. "
                f"'{'.'.join(parts[: depth + 1])}' is a "
                f"{type(cursor[part]).__name__}, not a section, so "
                f"'{key}' cannot name a value inside it."
            )
        cursor = cursor[part]
    else:
        if parts[-1] not in cursor and not permitted:
            raise ValueError(
                f"Unknown configuration key in override '{override}': "
                f"'{key}' is not defined. "
                f"Available keys at this level: {_available(cursor)}"
            )

    # Phase two: assign, creating intermediate sections only when authorized.
    current: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(
    config: MutableMapping[str, Any],
    overrides: Sequence[str] | None,
    *,
    allow_new_keys: bool = False,
    known_paths: frozenset[str] | None = None,
) -> MutableMapping[str, Any]:
    """Apply multiple dotted-key overrides to a configuration dictionary.

    Args:
        config: Already-loaded configuration to modify in place.
        overrides: `key.subkey=value` strings, applied in order.
        allow_new_keys: See :func:`apply_override`.
        known_paths: See :func:`apply_override`.
    """
    if not overrides:
        return config

    for override in overrides:
        apply_override(
            config, override,
            allow_new_keys=allow_new_keys, known_paths=known_paths,
        )

    return config


def load_config(
    config_dir: str | Path,
    overrides: Sequence[str] | None = None,
    *,
    yaml_files: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Load YAML configuration.

    The default is the exact legacy behavior: merge every root-level *.yaml
    alphabetically. An explicit filename list instead provides deterministic
    selection/precedence and excludes unrelated experiment configurations.
    """
    config: Dict[str, Any] = {}
    config_path = Path(config_dir)

    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if yaml_files is None:
        selected = sorted(config_path.glob("*.yaml"))
    else:
        if not yaml_files:
            raise ValueError("Explicit yaml_files cannot be empty")
        names = list(yaml_files)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate filenames in explicit yaml_files")
        for name in names:
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in {".", ".."}
                or not name.endswith(".yaml")
                or "\\" in name
            ):
                raise ValueError(f"Invalid explicit YAML filename: {name!r}")
        selected = [config_path / name for name in names]
        missing = [p.name for p in selected if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Required config YAML files not found: {missing}")

    for yaml_file in selected:
        with yaml_file.open("r", encoding="utf-8-sig") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration YAML must be a mapping: {yaml_file}")
        _deep_merge(config, data)

    apply_overrides(config, overrides)
    return config