# Code availability

The included Python files retain the experiment identifiers used during the study. The public repository is https://github.com/neon-wolf9/MSRR-landslide, with the frozen scientific release `v1.0.0`. No Zenodo DOI is claimed.

Machine-specific roots were sanitized as follows:

- scripts in `scripts/` resolve the project root from `__file__`;
- benchmark source modules under `src/benchmark/` resolve the package root from their location;
- historical absolute paths in manifests and reports are represented as `${PROJECT_ROOT}`;
- Python subprocess launchers use the active interpreter rather than a personal environment path.

The original packaging-only edits did not change a scientific calculation, model setting, seed, matching constraint, or frozen result. Current release validation is recorded in `VALIDATION_REPORT.md`.

The authoritative execution map is `docs/EXPERIMENT_MAP.md`. Development-only experiments, failed prototypes, figure-composition files, caches, and unrelated S-EHC-Net history were intentionally excluded.

