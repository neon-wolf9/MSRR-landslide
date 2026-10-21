# Repository release status

Status checked: 2026-10-21. This file describes the public GitHub repository at https://github.com/neon-wolf9/MSRR-landslide. It is a status record, not a substitute for the independent validation commands in `README.md`.

## Public release

- Repository visibility: public; default branch: `main`.
- `v1.0.0`: published release with notes, neither draft nor prerelease. The tag points to the frozen scientific commit.
- `README.md`, `LICENSE`, and `CITATION.cff` are in the tagged release.
- `v1.0.0` is incomplete relative to current `main`: the restored frozen configurations and external preprocessing script are absent from that tag. Do not cite it as the complete current package.
- The current `CITATION.cff` on `main` contains the public repository URL without claiming a new release version or Zenodo DOI. Zenodo archival is optional and is described in `ZENODO_SETUP.md`.

## Scientific and reproducibility materials

- The canonical 792-to-2,446 dimensional MSRR implementation, benchmark reconstruction workflow, frozen configurations, scripts, synthetic example, tests, and reference outputs are present.
- Hiroshima and Kyushu source EO/GIS data and derived full-rerun tensors are excluded. `docs/DATA_SOURCES.md`, `docs/DATA_LAYOUT.md`, and `scripts/01_verify_inputs.py --full` document the acquisition and input boundary.
- `VALIDATION_REPORT.md` records historical clean-environment validation of the earlier package, not an independent acceptance of the restored current `main`. A full source-data-to-results rerun was not performed for the public package.
- The principal manuscript metrics are mapped to frozen files in `docs/PAPER_RESULT_MAP.md`; later diagnostic executions remain distinct.

## Checks before any later release

1. Review the complete diff for scientific-code and result changes.
2. Run the redistribution-safe commands in `README.md` and update `FILE_MANIFEST.csv` after permitted documentation or metadata edits.
3. Recheck links, placeholders, personal paths, credentials, and third-party material.
4. Keep the existing `v1.0.0` tag unchanged. Do not create a new tag or release without independent fresh-clone acceptance and explicit author approval.
