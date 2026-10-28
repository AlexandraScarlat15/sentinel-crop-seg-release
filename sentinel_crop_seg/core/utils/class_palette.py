from __future__ import annotations

from typing import Final

# Project class IDs are kept unchanged.
# Colors are adapted from the JRC EUCROPMAP classification palette for visualization only.

CLASS_INFO: Final[list[tuple[int, str, str]]] = [
    (0, "background", "#000000"),
    (1, "maize", "#ffd300"),
    (2, "rapeseed", "#d1ff00"),
    (3, "wheat", "#a57000"),
    (4, "barley", "#e2007c"),
    (5, "oats", "#a05989"),
    (6, "sunflower", "#ffff00"),
    (7, "sugar_beet", "#a800e2"),
    (8, "soybean", "#267000"),
    (9, "potato", "#dda50a"),
    (10, "pasture", "#e8ffbf"),
]

IGNORE_LABEL: Final[int] = 255
IGNORE_COLOR: Final[str] = "#d3d3d3"
IGNORE_NAME: Final[str] = "ignore"

CLASS_ID_TO_NAME: Final[dict[int, str]] = {
    class_id: name for class_id, name, _ in CLASS_INFO
}

CLASS_ID_TO_COLOR: Final[dict[int, str]] = {
    class_id: color for class_id, _, color in CLASS_INFO
}

CLASS_NAMES: Final[list[str]] = [
    name for _, name, _ in CLASS_INFO
]

NUM_CLASSES: Final[int] = len(CLASS_INFO)