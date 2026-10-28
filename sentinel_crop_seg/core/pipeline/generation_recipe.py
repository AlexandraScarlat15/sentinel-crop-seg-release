"""Explicit, versioned configuration overlays for new generation experiments."""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SECTIONS = frozenset({"generation", "paths", "eurocrops", "catalog", "dataset", "classes", "sentinel"})
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


@dataclass(frozen=True)
class GenerationRecipe:
    name: str
    settings: dict[str, Any]
    sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationRecipe":
        data = Path(path).read_bytes()
        raw = yaml.safe_load(data)
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "name", "settings"}:
            raise ValueError("Recipe requires schema_version, name and settings")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("Unsupported recipe schema_version")
        name = raw["name"]
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError("Recipe name must use ASCII letters, numbers, '_' or '-'")
        settings = raw["settings"]
        if not isinstance(settings, dict) or not settings:
            raise ValueError("Recipe settings must be a nonempty mapping")
        if unexpected := set(settings) - SECTIONS:
            raise ValueError(f"Unknown recipe sections: {sorted(unexpected)}")
        if any(not isinstance(value, dict) or not value for value in settings.values()):
            raise ValueError("Each recipe section must be a nonempty mapping")
        return cls(name, settings, hashlib.sha256(data).hexdigest())


def _overlay_known(base: dict, changes: dict, prefix: str) -> None:
    for key, value in changes.items():
        if prefix == "catalog" and key == "calendar":
            if not isinstance(value, dict) or not value:
                raise ValueError("catalog.calendar must be a nonempty mapping")
            base["calendar"] = deepcopy(value)
            continue
        if not isinstance(key, str) or key not in base:
            raise ValueError(f"Unknown experiment setting: {prefix}.{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{prefix}.{key} must be a nonempty mapping")
            _overlay_known(base[key], value, f"{prefix}.{key}")
        elif isinstance(value, dict):
            raise ValueError(f"{prefix}.{key} must not be a mapping")
        else:
            base[key] = value


def _validate_classes(classes: dict) -> None:
    ids = []
    for name, rule in classes.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Taxonomy names must be nonempty strings")
        if not isinstance(rule, dict) or set(rule) != {"id", "include", "exclude"}:
            raise ValueError(f"Invalid taxonomy rule for {name!r}")
        if type(rule["id"]) is not int or rule["id"] < 0:
            raise ValueError(f"Invalid class id for {name!r}")
        for field in ("include", "exclude"):
            if not isinstance(rule[field], list) or any(
                not isinstance(item, str) or not item.strip() for item in rule[field]
            ):
                raise ValueError(f"Invalid {name!r}.{field} taxonomy list")
        ids.append(rule["id"])
    if sorted(ids) != list(range(len(ids))) or classes.get("background", {}).get("id") != 0:
        raise ValueError("Taxonomy requires background=0 and contiguous unique class ids")


def apply_generation_recipe(base: dict[str, Any], recipe: GenerationRecipe) -> dict[str, Any]:
    """Return an isolated resolved configuration without mutating the base."""
    resolved = deepcopy(base)
    for section, changes in recipe.settings.items():
        if section not in resolved or not isinstance(resolved[section], dict):
            raise ValueError(f"Missing generator config section: {section}")
        if section == "classes":
            _validate_classes(changes)
            resolved["classes"] = deepcopy(changes)
        else:
            _overlay_known(resolved[section], changes, section)
    resolved["experiment_recipe"] = {"name": recipe.name, "sha256": recipe.sha256}
    return resolved
