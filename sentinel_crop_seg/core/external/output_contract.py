"""One output-directory contract for both external dataset converters.

A converter writes `images/`, `masks/` and a few top-level report files. Three
ways of getting this wrong were found in use, and all three produce a directory
that loads without error and scores the wrong samples:

* permitting a second conversion to land on top of the first, so the newer
  patches replace some of the older ones and the rest survive;
* clearing only `images/` and `masks/` on an authorized overwrite, leaving the
  previous run's patches in place while the metadata CSV describes only the new
  ones;
* judging emptiness from the subdirectories alone, so a directory holding a
  stale `metadata.csv` counted as unused.

The rule here is therefore stated over the whole directory, not over parts of
it: a conversion either starts from a directory that holds nothing, or it is
explicitly authorized to replace what an earlier conversion left. Anything
present that no converter wrote is never deleted — it makes the run refuse
instead, because a directory holding unrelated files is not one whose contents
a later reader can attribute.
"""
from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

#: Top-level files the converters emit beside `images/` and `masks/`. Listed in
#: one place so an overwrite clears exactly what a conversion can create.
CONVERSION_ARTIFACTS: tuple[str, ...] = (
    "metadata.csv",
    "dataset_info.json",
    "quality_report.csv",
    "conversion_contract.json",
)

_SUBDIRS: tuple[str, ...] = ("images", "masks")


def prepare_conversion_output(
    output_dir: Path,
    *,
    overwrite: bool = False,
    artifacts: Sequence[str] = CONVERSION_ARTIFACTS,
) -> tuple[Path, Path]:
    """Return the `images/` and `masks/` directories, ready to be written to.

    Args:
        output_dir: Directory the conversion will populate.
        overwrite: Authorize replacing a previous conversion's output.
        artifacts: Top-level filenames a conversion may have written. Only
            these are removed on an overwrite.

    Returns:
        The `images/` and `masks/` paths, both existing and both empty.

    Raises:
        FileExistsError: If the directory holds anything and `overwrite` is
            false, or if it holds entries no converter wrote.
    """
    output_dir = Path(output_dir)
    recognized = set(_SUBDIRS) | set(artifacts)
    existing = sorted(p.name for p in output_dir.glob("*")) if output_dir.exists() else []

    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains results: {output_dir} "
            f"(found {', '.join(existing)}). Choose a new --output-dir, or "
            f"pass --overwrite to replace a previous conversion."
        )

    foreign = [name for name in existing if name not in recognized]
    if foreign:
        # Refusing is the safe branch: deleting files this tool never wrote
        # would be worse than stopping, and leaving them in place would let an
        # evaluator read them as part of the conversion.
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: it holds entries no "
            f"conversion wrote ({', '.join(foreign)}). Move them aside, or "
            f"convert into a directory of its own."
        )

    if overwrite:
        for name in _SUBDIRS:
            subdir = output_dir / name
            if subdir.exists():
                shutil.rmtree(subdir)
        for name in artifacts:
            stale = output_dir / name
            if stale.exists():
                stale.unlink()

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir
