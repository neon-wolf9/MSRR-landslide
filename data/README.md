# Data directory

This public package contains metadata and a small synthetic example, not third-party scientific source data.

- `manifests/`: hashes, provenance, QA, split summaries, and frozen dataset decisions.
- `schemas/`: field order, timestamp registry, and feature definitions.
- `example/`: eight redistribution-safe synthetic matched sets for `scripts/demo_msrr.py`; these are test values, not scientific observations.
- `derived/`: reserved for locally regenerated derived inputs.

Recommended local-only directories are `raw/`, `interim/`, `processed/`, and `derived/`. They are ignored by Git because raw provider data and derived geospatial redistribution rights have not been established for public release.

For a full rerun, restore the original relative paths documented by the sanitized `${PROJECT_ROOT}` entries. See `../docs/DATA_LAYOUT.md` and `../docs/DATA_ACQUISITION.md`. Do not treat a missing restricted input as permission to create a substitute.
