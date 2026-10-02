# Public release audit

Audit date: 2026-10-02  
Source package: `MSRR_IJAEOG_Reproducibility_Package`  
Public candidate: `MSRR_landslide_CG_GitHub_release`

## Scope and inventory

The 311-file source package was scanned before the release copy was created. The scan covered 71 Python files; benchmark, matching, representation, model, evaluation, bootstrap, ablation, placebo, PairDiff, ranking, sensitivity, external-validation, domain-shift, spatial, reconstruction, and plotting workflows; configs; manifests; hashes; reference outputs; documentation; dependencies; local paths; caches; large files; data formats; and credential patterns.

The public candidate retains the existing repository structure to avoid breaking imports and frozen provenance. No original file was removed. Fourteen files were added: six requested user/reproducibility documents, four GitHub/release/Zenodo/code-availability files, a minimal-example entry point and README, the real missing eligibility-freeze helper from the project source tree, and this audit report.

## Scientific immutability

- Algorithm changed: **NO**.
- Experimental results changed: **NO**.
- Benchmark counts changed: **NO**.
- Matching rules or weights changed: **NO**.
- Fold definitions or labels changed: **NO**.
- MSRR mathematics or feature order changed: **NO**.
- Model hyperparameters or Kyushu protocol changed: **NO**.

Two non-scientific portability fixes were made: compiled dependency imports are isolated in the environment checker, and Fiona is loaded before GeoPandas in the spatial script on Windows to avoid a GDAL DLL load-order failure.

## Public-release checks

- Personal absolute project paths: **none found**. Documentation contains only sanitized placeholders; three optional system-runtime probes use standard Windows installation paths.
- Credentials, secrets, tokens, private keys, or private URLs: **none found**.
- Raw third-party geospatial formats: **none found**.
- Largest distributed file: approximately 3.12 MiB; **no unusually large raw-data asset found**.
- Non-redistributable data: third-party EO/GIS sources and source inventories are intentionally excluded and documented.
- Manuscript-claimed functionality missing: **none identified for the mapped manuscript evidence chain**.
- Retained historical provenance: some earlier non-authority baseline/neural files depend on companion modules from the larger development tree; they are explicitly documented in `docs/LEGACY_PROVENANCE.md` and are not presented as manuscript entry points.

## Validation

- Python syntax: **PASS**, 73 files.
- Dependency import/version check: **PASS** in the recorded Python 3.10.20 environment.
- Minimal synthetic example: **PASS**.
- Canonical MSRR dimension check: **PASS**, `N x 3 x 792` to `N x 3 x 2446`.
- Focused test suite: **PASS, 10 tests**.
- Reference-output and SHA-256 manifest verification: **PASS**.
- README command-path audit: **PASS**, 25 entry-point references and 0 missing files.
- Frozen benchmark replay evidence: **PASS**, Hiroshima 5,056 sets and Kyushu 1,692 sets, with zero control reuse.

Full source-data-to-results execution was not rerun because the public candidate intentionally excludes seven required scientific inputs and provider-controlled source data. The recorded clean-environment validation and the current public-candidate revalidation are documented in `VALIDATION_REPORT.md`.

## Release decision

**PUBLIC GITHUB REPOSITORY CREATED** at https://github.com/neon-wolf9/MSRR-landslide. Release `v1.0.0` is created after the final staged-file/security review. Before the final journal statement is submitted, the authors must archive that release with Zenodo and insert the real version-specific DOI.
