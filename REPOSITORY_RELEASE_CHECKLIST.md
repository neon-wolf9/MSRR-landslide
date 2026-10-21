# Repository release status

Status checked: 2026-10-21. This file describes the public GitHub repository at https://github.com/neon-wolf9/MSRR-landslide. It is a status record, not a substitute for the independent validation commands in `README.md`.

## Public release

- Repository visibility: public; default branch: `main`.
- [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1) is the published, non-prerelease Latest Release. Its tag is maintained at the reviewed documentation revision on `main`.
- `README.md`, `LICENSE`, `CITATION.cff`, the frozen configurations, scripts, reference results, and user documentation are in the tagged code release.
- The earlier incomplete Release and Tag were removed; Git history and `main` were retained.
- `CITATION.cff` identifies the code version. The separately published study-data record is [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084); it is not a code DOI.

## Scientific and reproducibility materials

- The canonical 792-to-2,446 dimensional MSRR implementation, benchmark reconstruction workflow, frozen configurations, scripts, synthetic example, tests, and reference outputs are present.
- Raw Hiroshima and Kyushu source EO/GIS products are not in Git. Eleven frozen derived study files and historical hand-offs are in the separate Zenodo data deposit; `README.md` explains extraction and SHA-256 verification. `docs/DATA_SOURCES.md` documents the remaining raw-source acquisition boundary.
- `VALIDATION_REPORT.md` records historical validation only. A full source-data-to-results rerun was not performed during release preparation.
- The principal manuscript metrics are mapped to frozen files in `docs/PAPER_RESULT_MAP.md`; later diagnostic executions remain distinct.

## Checks for subsequent documentation revisions

1. Review the complete diff for scientific-code and result changes.
2. Run the redistribution-safe commands in `README.md` and update `FILE_MANIFEST.csv` after permitted documentation or metadata edits.
3. Recheck links, placeholders, personal paths, credentials, and third-party material.
4. Verify that the existing v1.0.1 Release URL, tag target, and source-code archive reflect the intended revision. Do not change scientific code or publish a new version as part of documentation-only maintenance.
