# Code availability

The included Python files retain the experiment identifiers used during the study. The complete public code package is available as [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1). The earlier `v1.0.0` snapshot predates restored frozen configurations and an external preprocessing script. The separately published research data have DOI [10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084); this is not a code DOI.

Machine-specific roots were sanitized as follows:

- scripts in `scripts/` resolve the project root from `__file__`;
- benchmark source modules under `src/benchmark/` resolve the package root from their location;
- historical absolute paths in manifests and reports are represented as `${PROJECT_ROOT}`;
- Python subprocess launchers use the active interpreter rather than a personal environment path.

The repository-preparation edits did not change a scientific calculation, model setting, seed, matching constraint, or frozen result. `VALIDATION_REPORT.md` records historical validation; release preparation did not rerun the full scientific experiments.

The authoritative execution map is `docs/EXPERIMENT_MAP.md`. Development-only experiments, failed prototypes, figure-composition files, caches, and unrelated S-EHC-Net history were intentionally excluded.

