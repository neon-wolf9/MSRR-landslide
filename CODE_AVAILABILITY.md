# Code availability

The included Python files are copied from the real local project and retain the experiment identifiers used during the study. The package does not claim a public repository or DOI.

Machine-specific roots were sanitized as follows:

- scripts in `scripts/` resolve the project root from `__file__`;
- benchmark source modules under `src/benchmark/` resolve the package root from their location;
- historical absolute paths in manifests and reports are represented as `${PROJECT_ROOT}`;
- Python subprocess launchers use the active interpreter rather than a personal environment path.

These packaging-only edits are recorded in `MSRR_code_package_audit.txt`. No scientific calculation, model setting, seed, matching constraint, or frozen result was changed.

The authoritative execution map is `docs/EXPERIMENT_MAP.md`. Development-only experiments, failed prototypes, figure-composition files, caches, and unrelated S-EHC-Net history were intentionally excluded.

