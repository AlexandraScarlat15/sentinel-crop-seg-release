"""Protect the documentation contract of curated public pipeline APIs.

Source-only checks: no satellite access, model execution or data generation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_APIS = {
    "sentinel_crop_seg/core/catalog/stac_sentinel_catalog_searcher.py": [
        "STACSentinelCatalogSearcher.search",
    ],
    "sentinel_crop_seg/core/catalog/copernicus_safe_downloader.py": [
        "CopernicusSafeDownloader.download_safe",
    ],
    "sentinel_crop_seg/core/data/eurocrops_mask_loader.py": [
        "EuroCropsMaskLoader.map_crop_to_class_id",
        "EuroCropsMaskLoader.generate_mask",
    ],
    "sentinel_crop_seg/core/data/sentinel_safe_loader.py": [
        "SentinelSafeDataLoader.read_window",
        "SentinelSafeDataLoader.read_scl",
    ],
    "sentinel_crop_seg/core/data/patch_log_manifest.py": [
        "adapt_patch_log",
        "adapt_patch_log_file",
    ],
    "sentinel_crop_seg/core/pipeline/cohort_plan.py": [
        "plan_cohorts",
        "run_cohorts",
    ],
    "sentinel_crop_seg/core/pipeline/generation_config.py": [
        "load_generation_config",
        "generation_config_report",
    ],
    "sentinel_crop_seg/core/pipeline/dataset_generation.py": [
        "generate_dataset",
        "roi_to_wgs84",
        "make_roi_key",
    ],
    "sentinel_crop_seg/core/quality/post_training_filter.py": ["review_and_filter"],
    "sentinel_crop_seg/core/quality/weighted_calendar_selection.py": [
        "build_weighted_calendar_selection",
    ],
    "sentinel_crop_seg/core/selection/roi_selectors/hotcell_selector.py": [
        "HotcellSelector.select",
    ],
    "sentinel_crop_seg/core/data/sentinel_product_metadata.py": [
        "infer_sentinel_scene_date",
        "infer_sentinel_tile_id",
    ],
}


def _find_public_function(tree: ast.Module, qualified_name: str) -> ast.FunctionDef:
    nodes = tree.body
    for name in qualified_name.split("."):
        found = next(
            (node for node in nodes
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name == name),
            None,
        )
        if found is None:
            raise AssertionError(f"Missing curated public API: {qualified_name}")
        nodes = found.body
    if not isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise AssertionError(f"Curated API is not a function: {qualified_name}")
    return found


@pytest.mark.parametrize(
    ("relative_path", "name"),
    [
        (path, name)
        for path, names in PUBLIC_APIS.items()
        for name in names
    ],
)
def test_public_api_documents_arguments_results_and_purpose(relative_path, name):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))
    fn = _find_public_function(tree, name)
    doc = ast.get_docstring(fn)
    assert doc, f"{relative_path}:{name} has no API docstring"
    assert doc.splitlines()[0].endswith("."), f"{name}: summary must be a sentence"
    assert "\nArgs:\n" in "\n" + doc, f"{name}: document input arguments"
    assert "\nReturns:\n" in "\n" + doc, f"{name}: document output value"
    assert "param" not in doc.lower().splitlines()[0], "Avoid signature restatements"


@pytest.mark.parametrize("relative_path", list(PUBLIC_APIS))
def test_curated_public_modules_have_no_leftover_debug_comments(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in source.splitlines()]
    assert not any(line.startswith("# print(") for line in lines)
    assert not any(line in {"# Encoder", "# Decoder", "# RETRY LOGIC"}
                   for line in lines)


def test_shipped_class_map_travels_with_its_measured_limits():
    """The archive ships `config/classes.yaml`, so its known defects must ship too.

    Without this document a reader treats measured label noise as model error.
    The specific numbers are asserted because a vaguer rewrite that dropped
    them would leave the caveat unusable.

    This test is itself part of the distribution, so it reads the files beside
    it rather than the release allowlist, which is deliberately not shipped.
    Inside a staged archive it therefore checks that the document really
    arrived, not merely that it was selected.
    """
    root = Path(__file__).resolve().parents[1]
    assert (root / "config" / "classes.yaml").is_file()
    doc = root / "docs" / "taxonomy_and_label_quality.md"
    assert doc.is_file(), "the class map ships without its limitations"

    # Phrases are checked against whitespace-normalised text, because the
    # document is hard-wrapped and a sentence may straddle a line break.
    text = " ".join(doc.read_text(encoding="utf-8").split())
    # The configuration the numbers belong to, and the honest status of the
    # evidence. Without these two the figures look universal.
    assert "MAP_a45fa755ef" in text
    assert "not independently reproducible from this archive" in text

    # The audited population is not the benchmark: two Belgian source files
    # contribute parcels but no samples. Dropping this overstates the scope.
    assert "BE_VLG_2021" in text and "BE_WAL_2021" in text
    assert "no samples to the benchmark" in text
    # The DE_NRW naming differs between the operational artifacts and the
    # forensic notes. The document must flag that rather than silently pick
    # one, and must not present it as two cohorts.
    assert "DE_NRW_2021" in text and "DE_NRW_2023" in text
    # The retracted claim must be marked as retracted, with what the evidence
    # actually supports, not quietly reworded.
    assert "That claim is retracted" in text
    assert "not supported" in text
    assert "There is one cohort, not two" in text
    assert "has no supporting product" in text
    # The year is settled; the per-sample dates are not, and the document
    # must keep that distinction instead of upgrading one into the other.
    assert "cache may also be incomplete" in text
    assert "remains unresolved, and must stay so" in text
    assert "no calendar or phenology result depends on its year label" in text
    assert "forensic evidence is not distributed" in text
    # The Belgian cohorts appear in a shipped config as rules, not data.
    assert "pasture_omission` override" in text or "pasture_omission" in text
    assert "rules, not data" in text
    assert "15,654,725" in text and "12,073,602" in text

    # Per-class counts, not a single rounded headline.
    assert "99.836%" in text and "90.217%" in text
    assert "nine of the ten crop classes" in text
    assert "eight of the ten" in text and "seven of the ten" in text

    # The two exceptions share 91,020 parcels; presenting them as independent
    # double-counts the loss.
    assert "91,020" in text and "552,842" in text
    assert "95.143%" in text and "Lolium" in text
    assert "90,905" in text and "not the same measurement" in text

    # Scope of the pixel figures, which is two cohorts and exact geography.
    assert "0.182%" in text and "239,595,423" in text
    assert "14,819" in text and "8.0%" in text

    # The contested sugar_beet code, with both readings and the definition
    # that makes them differ.
    assert "49.368%" in text and "99.778%" in text and "99.552%" in text
    assert "keeps them and counts them as correct" in text
    assert "70,671" in text

    # Pasture omission at pixel level, with its own denominators.
    assert "4,776,841" in text and "2.839%" in text

    # Rasterization claims that have been checked and rejected, with scope.
    assert "all_touched=False" in text
    assert "+0.008%" in text and "2,000 CZ_2023 parcels only" in text
    assert "did not" in text

    # Boundary geometry, its four-fold spread, and the cohorts it misses.
    assert "42.20%" in text and "10.78%" in text
    assert "20-mask control sets" in text
    assert "not measured" in text and "largest cohort" in text
    assert "border_value=1" in text

    # Every figure must name the artifact it came from.
    assert "taxonomy_hcat_arbiter_performance.csv" in text
    assert "FR_2018_ryegrass_pixel_impact_summary.json" in text
    assert "pasture_omission_pixel_impact_summary.json" in text
    assert "boundary_geometry_report.md" in text


def _shipped_markdown(root: Path) -> list[tuple[str, str]]:
    """Return the (label, text) of every document a reader of the archive gets.

    This helper has to work in two places. Inside a staged archive every
    markdown file present is shipped, so the filesystem is the answer. In the
    development checkout most markdown is forensic or internal and would
    swamp any check, so the release allowlist decides -- and that is the
    stricter reading, because it also catches a shipped document referring to
    a file that exists here but will not travel.
    """
    allowlist = root / "release" / "zenodo_source_allowlist.json"
    if not allowlist.is_file():
        return [
            (path.relative_to(root).as_posix(), path.read_text(encoding="utf-8", errors="ignore"))
            for path in sorted(root.rglob("*.md"))
        ]
    import json

    manifest = json.loads(allowlist.read_text(encoding="utf-8"))
    documents: list[tuple[str, str]] = []
    for target in sorted(p for p in manifest["source_paths"] if p.endswith(".md")):
        documents.append((target, (root / target).read_text(encoding="utf-8", errors="ignore")))
    # A generated target ships under its own name but lives elsewhere here.
    for target, source in sorted(manifest["generated_files"].items()):
        if target.endswith(".md"):
            documents.append((target, (root / source).read_text(encoding="utf-8", errors="ignore")))
    assert documents, "no shipped markdown was found"
    return documents


def _shipped_paths(root: Path) -> set[str] | None:
    """The set of paths the archive will contain, or None inside an archive."""
    allowlist = root / "release" / "zenodo_source_allowlist.json"
    if not allowlist.is_file():
        return None
    import json

    manifest = json.loads(allowlist.read_text(encoding="utf-8"))
    return set(manifest["source_paths"]) | set(manifest["generated_files"])


def test_no_shipped_document_points_outside_the_archive():
    """A documented path that is absent from the archive is a dead end."""
    import re

    root = Path(__file__).resolve().parents[1]
    shipped = _shipped_paths(root)
    referenced = re.compile(
        r"(?<![\w./-])((?:examples|config|docs|tests|sentinel_crop_seg)/[\w./-]+)"
    )

    def travels(token: str) -> bool:
        if shipped is None:
            return (root / token).exists()
        # A directory reference is satisfied by anything shipped beneath it.
        return token in shipped or any(p.startswith(token.rstrip("/") + "/") for p in shipped)

    missing: dict[str, list[str]] = {}
    for label, text in _shipped_markdown(root):
        absent = sorted({
            token for token in (t.rstrip(".,);:`") for t in referenced.findall(text))
            if not travels(token)
        })
        if absent:
            missing[label] = absent
    assert not missing, f"documented paths absent from the archive: {missing}"


def test_shipped_documents_never_instruct_a_reader_into_excluded_trees():
    """`src/`, `tools/`, `scripts/` and `revision/` are not distributed.

    Naming one is allowed only to say it is absent. An instruction to run
    something from there cannot be followed by anyone holding the archive.
    """
    import re

    root = Path(__file__).resolve().parents[1]
    excluded = re.compile(
        r"(?<![\w./-])((?:src|tools|scripts|revision|local_audit)/[\w./-]+)"
    )
    # A mention is acceptable only inside a sentence that says the thing is
    # absent. Anything else is an instruction a reader cannot act on.
    disclaims = ("does not include", "not included", "not distributed",
                 "not part of", "development repository", "development checkout",
                 "historical", "historic")
    offenders: dict[str, list[str]] = {}
    for label, text in _shipped_markdown(root):
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            hits = excluded.findall(sentence)
            if hits and not any(phrase in sentence.lower() for phrase in disclaims):
                offenders.setdefault(label, []).extend(sorted(set(hits)))
    assert not offenders, f"unusable instructions into excluded trees: {offenders}"


def test_pipeline_stages_remain_independent_commands():
    """No stage may silently run another.

    Generation, aggregation, splitting, training, evaluation and plotting are
    separate entry points on purpose: a user supplies their own dataset and
    their own checkpoint, and plotting reads saved reports rather than loading
    a model. Collapsing any pair would change what a run implies.
    """
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        scripts = tomllib.load(stream)["project"]["scripts"]

    expected = {
        "sentinel-crop-generate-cohorts": "cli.generate_cohorts",
        "sentinel-crop-merge-cohorts": "cli.merge_cohorts",
        "sentinel-crop-split-spatial": "cli.split_spatial",
        "sentinel-crop-split-oof": "cli.split_oof",
        "sentinel-crop-train-unet": "cli.train_unet",
        "sentinel-crop-evaluate-unet": "cli.evaluate_unet",
        "sentinel-crop-predict-unet": "cli.predict_unet",
        "sentinel-crop-plot-training": "cli.plot_training",
        "sentinel-crop-plot-evaluation": "cli.plot_evaluation",
    }
    for command, module in expected.items():
        assert command in scripts, f"{command} is no longer a separate command"
        assert module in scripts[command], (command, scripts[command])

    # Distinct modules, so one stage cannot be an alias of another.
    targets = [scripts[command] for command in expected]
    assert len(set(targets)) == len(targets), targets

    # Plotting must not import the training or evaluation entry points, which
    # is how "read a saved report" quietly becomes "retrain" or "load a
    # checkpoint".
    for plotter in ("plot_training", "plot_evaluation"):
        source = (root / "sentinel_crop_seg" / "cli" / f"{plotter}.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("train_unet", "evaluate_unet", "predict_unet"):
            assert forbidden not in source, f"{plotter} reaches into {forbidden}"


def test_external_data_prerequisites_are_documented_before_the_generator():
    """A generator nobody can authenticate to is not usable by a researcher.

    The package reads Copernicus credentials from the environment and needs a
    EuroCrops shapefile it does not contain. Neither was stated anywhere in
    the shipped documentation, so an outside user's first real run failed on
    an undocumented prerequisite.
    """
    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "getting_started.md"
    assert guide.is_file(), "the archive ships no prerequisites guide"
    text = " ".join(guide.read_text(encoding="utf-8").split())

    # The two environment variables the code actually reads.
    auth = (root / "sentinel_crop_seg" / "core" / "auth"
            / "copernicus_auth_manager.py").read_text(encoding="utf-8")
    for variable in ("COPERNICUS_USERNAME", "COPERNICUS_PASSWORD"):
        assert variable in auth, f"{variable} is no longer read by the code"
        assert variable in text, f"{variable} is undocumented"

    # Registration is free and the account is only needed for generation.
    assert "dataspace.copernicus.eu" in text
    assert "free" in text.lower()

    # Where EuroCrops comes from, and the column that decides every label.
    assert "github.com/maja601/EuroCrops" in text
    assert "crop_column" in text and "EC_trans_n" in text

    # The honest limit: this generator does not reproduce the published
    # benchmark, which was produced by an earlier pipeline.
    assert "does not reproduce the published" in text
    assert "67,337" in text

    # A no-network verification path must exist, so the install can be
    # checked before any credential or download is involved.
    assert "--validate-only" in text
    assert "no credentials" in text or "no network" in text
