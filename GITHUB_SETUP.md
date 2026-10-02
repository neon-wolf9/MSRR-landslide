# GitHub setup

## Recommended repository metadata

- Repository name: `MSRR-landslide`
- Description: `Reproducible implementation of event-matched hard-control benchmark construction and matched-set relative representation for rainfall-induced landslide discrimination.`
- Visibility: public, only after the release checklist and rights review are complete.
- Default branch: `main`

Recommended topics:

`landslide`, `geoscience`, `machine-learning`, `earth-observation`, `gis`, `reproducibility`, `positive-unlabeled-learning`, `matched-sampling`, `xgboost`, `spatial-machine-learning`

## Pre-publication sequence

1. Review `SECURITY_AUDIT.md`, `VALIDATION_REPORT.md`, and `REPOSITORY_RELEASE_CHECKLIST.md`.
2. Confirm that no third-party raw data, credentials, local paths, caches, or private files are staged.
3. Replace `REPLACE_WITH_GITHUB_URL` in `CITATION.cff` only after the repository URL exists.
4. Validate the redistribution-safe commands in `README.md` in a clean environment.
5. Initialize the public Git history and inspect the complete staged diff.
6. Push the repository without source data or secrets.
7. Create the annotated release tag `v1.0.0` and publish the release notes from `RELEASE_NOTES_v1.0.0.md`.
8. Archive that exact tag with Zenodo and update the repository with the real DOI in a later metadata commit or release.

No GitHub repository, release tag, or remote is created by this preparation task.
