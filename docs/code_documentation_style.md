# Python comments and docstrings — curated software conventions

These conventions apply to public functions, classes and modules shipped with
the Sentinel-2/EuroCrops research software. They are **documentation rules**,
not changes to the manuscript or experimental protocol. Keep the scientific
behavior, class/band order, correction provenance and frozen splits unchanged.

## Where docstrings are useful

- **Module:** one sentence explaining the responsibility; add constraints
  only when important (for example that an audit never executes a model).
  Avoid file names, decorative underlines and changelog-style text.
- **Public function or method:** one imperative/active summary sentence,
  followed by Google-style **Args**, **Returns** and **Raises** when meaningful.
  Document units, input shape, side effects and failure modes rather than
  rewriting the signature in prose.
- **Class:** describe its invariant and important constructor parameters
  in the class docstring. A redundant `__init__` docstring is unnecessary
  when the class already documents these arguments.
- **Private, tiny helper:** a one-line docstring only when the intent is
  genuinely unclear. Trivial private getters, wrappers and obvious
  implementation details need no extra docstring.
- **CLI `main()`:** its module description and `--help` may be enough.
  Document surprising side effects in the user guide instead.
- **Tests:** give modules a short purpose; individual test docstrings
  are usually unnecessary when their names explain the assertions.

Example:

```python
def read_window(window: tuple[int, int, int, int], bands: list[str]) -> np.ndarray:
    """Read selected spectral bands on the reference grid.

    Args:
        window: Pixel bounds as (column, row, width, height).
        bands: Names of requested channels in output order.

    Returns:
        Channel-first array with shape (channels, height, width).

    Raises:
        KeyError: If a requested band is unavailable.
    """
```

`Returns:` is omitted for a `None`-returning command that only writes
files when no meaningful return value exists; document **side effects**.
`Raises:` is needed for meaningful validation and public exceptions, not
every possible error in third-party libraries.

## Comments: explain *why*, not *what*

**Keep** comments that prevent scientific mistakes or unsafe recovery:
- STAC scene dates versus parcel-specific phenology eligibility
- Nearest-neighbor SCL resampling rather than continuous interpolation
- Train/validation/test leakage prevention and geographic guard logic
- Strict generation resume checks and preserved execution evidence
- Class/taxonomy and corrected-mask compatibility, temporal rules and
  numerical edge cases

**Remove** comments that only narrate adjacent code: `# Encoder`,
`# Extract`, `# ROI size`, `# RETRY LOGIC`, `# Useful debug output`.
Do not commit dead `# print(...)` fragments, decorative file headers,
AI-agent chat commentary, or unchecked `TODO` claims into the curated
software. Avoid blanket deletion of inline scientific explanations.

## Automated regression boundary

`tests/test_release_api_documentation.py` checks the core **public**
generation, catalog, raster, manifest, calendar and review entrypoints for
a meaningful summary and structured input/output documentation. It also
checks for specific old debug/comment fragments in these modules.

This is intentionally **not a mandate to document every private helper**.
The complete repo retains historical tests and research-only code that are
not included in the curated release; tests for documentation style are
scoped to the public software surface. Existing OOF and trained-model
artifacts are independent release-approval gates.
