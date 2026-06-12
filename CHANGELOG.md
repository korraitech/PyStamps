# Changelog

All notable changes to PyStamps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-12

### Added
- `.zenodo.json` to provide authoritative Zenodo archive metadata, listing
  KorrAI Technologies Ltd. as the lead (entity) author.

### Fixed
- Ensured the citation files (`CITATION.cff`, `CHANGELOG.md`), the ESA-PL license,
  and the restructured README are included in the archived release. The `v0.2.0`
  tag predated these additions, so this release supersedes it for citation purposes.

## [0.2.0] - 2026-06-12

First public release of PyStamps.

### Added
- Python implementation of the Stanford Method for Persistent Scatterers (StaMPS)
  for extracting ground displacement time series from SAR acquisition stacks.
- Full processing workflow: PS candidate selection, phase estimation and weeding,
  patch merging, 3-D phase unwrapping, linear APS (tropospheric) correction,
  spatially correlated look-angle (SCLA) correction, and CSV export of displacements.
- Support for SNAP-preprocessed RSLC input stacks.
- Parallel per-patch processing.
- `CITATION.cff` with KorrAI as entity author and references to the original
  StaMPS papers (Hooper et al., 2004; 2012) and the SNAP–StaMPS paper (Foumelis et al., 2018).
- "How to Cite" section in the README with paste-ready BibTeX.
- JOSS paper scaffold under `paper/`.
- Runtime citation notice printed at the end of a successful run.

### Changed
- Licensed under the European Space Agency Public License (ESA-PL) Permissive (Type 3) – v2.4.

[0.2.1]: https://github.com/korraitech/PyStamps/releases/tag/v0.2.1
[0.2.0]: https://github.com/korraitech/PyStamps/releases/tag/v0.2.0
