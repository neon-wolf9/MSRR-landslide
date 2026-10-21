# Repository release status

Status checked: 2026-10-21. This file describes the public GitHub repository at https://github.com/neon-wolf9/MSRR-landslide. It is a status record, not a substitute for the independent validation commands in `README.md`.

## Public release

- Repository visibility: public; default branch: `main`.
- `v1.0.0`: published release with notes, neither draft nor prerelease. The tag points to the frozen scientific commit.
- `README.md`, `LICENSE`, and `CITATION.cff` are in the tagged release.
- `CITATION.cff` contains the public repository URL and version. No Zenodo DOI is asserted. Zenodo archival is optional and is described in `ZENODO_SETUP.md`.

## Scientific and reproducibility materials

- The canonical 792-to-2,446 dimensional MSRR implementation, benchmark reconstruction workflow, frozen configurations, scripts, synthetic example, tests, and reference outputs are present.
- Hiroshima and Kyushu source EO/GIS data and derived full-rerun tensors are excluded. `docs/DATA_SOURCES.md`, `docs/DATA_LAYOUT.md`, and `scripts/01_verify_inputs.py --full` document the acquisition and input boundary.
- `VALIDATION_REPORT.md` records the clean-environment validation of the frozen package. A full source-data-to-results rerun was not performed for the public package.
- The principal manuscript metrics are mapped to frozen files in `docs/PAPER_RESULT_MAP.md`; later diagnostic executions remain distinct.

## Checks before a later documentation release

1. Review the complete diff for scientific-code and result changes.
2. Run the redistribution-safe commands in `README.md` and update `FILE_MANIFEST.csv` after permitted documentation or metadata edits.
3. Recheck links, placeholders, personal paths, credentials, and third-party material.
4. Keep the existing `v1.0.0` tag unchanged. If documentation updates materially affect reviewer use of the frozen release, consider a separately approved documentation-only `v1.0.1` release.
