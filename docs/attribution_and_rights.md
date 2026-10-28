# Attribution, licensing and redistribution review — action required

**This is an unresolved release-rights checklist, not a license grant, legal
opinion or permission to distribute the underlying satellite/parcel data.**
Do not treat a license advertised for one upstream dataset edition as proof
about another country, year, processing stage or source record.

## Material-by-material inventory

| Item | Current evidence | Decision / action before deposition |
|---|---|---|
| Original project code | `LICENSE.md` currently declares **CC BY 4.0** and a single copyright holder. **CC BY 4.0 is the wrong instrument for source code** — see below | Choose an actual software license; do not merely confirm the current file, and do not silently replace it with MIT/Apache/GPL without an author decision |
| Pipeline documentation and original configurations | Distributed under the project's current root license | Confirm authorship and any third-party portions |
| Copernicus Sentinel-2 L2A | Official Sentinel data legal notice permits broad lawful reuse and requires attribution on distribution | Preserve original product/source/year/baseline; distinguish unmodified from modified products; include the applicable notice for redistributed material |
| EuroCrops harmonized vectors | The official EuroCrops GitHub documentation currently references **CC BY-SA 4.0**; a separate public collection description lists **CC BY 4.0** | **Resolve the edition-specific licensing discrepancy** from the exact original downloaded release and its license/metadata; honor source-country notices and any share-alike obligations that actually apply |
| Underlying national field declarations / parcel shapes | EuroCrops aggregates country-specific public administrative datasets | Check source-country and year attribution, restrictions and provenance independently; do not assume the harmonized collection description supersedes all upstream rights |
| Derived image/mask patches and complete A′ dataset | Not part of this software draft | Deferred pending supervisor agreement on archives, data licenses, privacy and redistribution |
| Existing trained checkpoint(s) | Not included or selected | Confirm paper/model selection, contributor permissions, source-data obligations and model-file license before packaging |
| Code dependencies (PyTorch, geospatial libraries, etc.) | Declared in the curated `pyproject.toml`, **not vendor-copied** | Review notices if dependency binaries, container images or source code are ever redistributed; installing dependencies is not a license grant |
| Article text, figures and author list | No final paper DOI or accepted/published record in this draft | Verify manuscript and publisher reuse rights, authors and final citation separately from software |

## Why the current code license is a defect, not just unapproved

Creative Commons itself recommends against applying CC licenses to software,
and the problem is specific rather than stylistic:

- CC licenses say nothing about **source versus object form**, so a CC BY 4.0
  archive does not state whether a recipient may redistribute compiled or
  modified builds and under what conditions.
- They contain **no patent grant and no patent peace clause**, which standard
  software licenses address explicitly.
- They include **no warranty or liability allocation** of the kind every
  software license carries, which institutional and industrial reusers check
  before adopting research code.
- CC BY's attribution requirement is written for creative works, not for
  per-file notices in a codebase, so compliance is ambiguous for someone
  vendoring a module.
- CC licenses are **not compatible** with the common copyleft and permissive
  software licenses, which blocks downstream reuse inside existing projects —
  the opposite of what depositing research software is for.

This does not make the current archive unlawful to distribute; it makes the
license unfit for its purpose and ambiguous for reusers. Resolving it means an
author decision on a software license (permissive or copyleft), recorded in
`LICENSE.md` and in the record metadata. Documentation, configuration and
figures may legitimately stay under CC BY 4.0 — a **mixed-license record** is
normal on Zenodo and is one option for an eventual licensing decision.
The final choice requires explicit rights-holder and supervisor review.

## Primary sources to review

- [Official Copernicus Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
  For publicly redistributed original Sentinel data, the notice identifies
  the source in the form **Copernicus Sentinel data [Year]**; for modified
  data, it prescribes a **Contains modified Copernicus Sentinel data [Year]**
  form. The applicable year must come from the actual acquisition/product,
  not be guessed from the manuscript.
- [Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions).
  The portal's general website-content terms and the data-specific Sentinel
  legal notice should not be conflated.
- [EuroCrops official source repository](https://github.com/maja601/EuroCrops).
  Inspect the LICENSE file and original source-version record for the
  specific datasets actually acquired by this project.
- [Euro Data Cube's EuroCrops collection description](https://collections.eurodatacube.com/eurocrops/).
  Its different license description is **not** sufficient evidence to
  silently downgrade share-alike obligations from a specific upstream release.
- [Zenodo guidance on licenses, including mixed-license records](https://help.zenodo.org/docs/deposit/describe-records/licenses/)
  and [Zenodo advice for choosing software licenses](https://support.zenodo.org/help/en-gb/1-upload-deposit/75-which-license-should-i-use/).
  Zenodo requiring a license field does **not** itself establish a
  depositor's authority to grant those rights.

## Approval record to complete with the supervising authors

| Question | Current state |
|---|---|
| Are the copyright attribution and software authors correct? | **UNAPPROVED** |
| Which **software** license replaces CC BY 4.0 for the code? (Keeping it is not a neutral option; see the defect section) | **UNAPPROVED** |
| Which material legitimately stays under CC BY 4.0 (documentation, configuration, figures) in a mixed-license record? | **UNAPPROVED** |
| Have all tracked code files and text/config examples been reviewed for outside contributions? | **UNVERIFIED** |
| Which EuroCrops edition and upstream license applies to each country/year? | **UNVERIFIED** |
| Which national declarations/third-party notices must travel with any published derived dataset? | **UNVERIFIED** |
| What is the accurate paper status, author list, DOI and any publisher-policy constraint? | **UNVERIFIED** |
| Which existing checkpoint(s) can be redistributed under what terms? | **DEFERRED** |
| Are the software and the planned combined dataset/model Zenodo records appropriately licensed and attributed? | **UNAPPROVED** |

Only mark an item approved after locating the relevant source or explicit
author/supervisor decision. This document intentionally does not assert that
any dataset, checkpoint or paper can yet be published. The software package
has passed its mechanical final-candidate staging checks; that does not
resolve the approval questions above or authorize Zenodo publication. The
combined dataset/model record is not yet prepared or published.
