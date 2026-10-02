# Repository Release Checklist

Status date: 2026-10-01

This checklist applies to `MSRR_landslide_CG_GitHub_release`. It does not authorize publication or any change to the frozen scientific protocol.

## Scientific and technical content

- [x] Canonical 792-D to 2446-D MSRR implementation is included.
- [x] Hiroshima and Kyushu benchmark reconstruction workflows are documented and represented by the retained scripts/configurations.
- [x] Four learner families and the published-comparator workflow are included.
- [x] Bootstrap, ablation, placebo, PairDiff, ranking, sensitivity, domain-shift, spatial, and independent-evaluation workflows are mapped.
- [x] Frozen manuscript reference results are separated from historical diagnostic outputs.
- [x] Frozen Hiroshima and Kyushu benchmark identities/hashes are retained.
- [x] Matching rules, weights, capacity, fold definitions, labels, MSRR definition, hyperparameters, metrics, and reference values were not changed during repository preparation.
- [x] Redistribution-safe unit/reference tests are included.
- [x] A standalone synthetic-data MSRR demo is included.

## Documentation and licensing

- [x] README contains scope, installation, quick-start, workflow, data-access, testing, frozen-result, citation, and limitation guidance.
- [x] Data-source, availability, third-party, and comparator-provenance documentation is included.
- [x] Repository-authored material has an explicit MIT license scope.
- [x] Third-party EO/GIS raw data are excluded.
- [x] Public Python prose is in English; protected Japanese source-data parsing literals are documented.
- [x] A Computers & Geosciences Code Availability draft is included.
- [x] Author names, manuscript title, year, license, and release version are populated in `CITATION.cff`.
- [ ] Replace `REPLACE_WITH_GITHUB_URL` in `CITATION.cff` after the public repository exists; add a DOI only after a real archive exists.
- [ ] Replace all bracketed placeholders in `CODE_AVAILABILITY_CG_FINAL_DRAFT.md` after repository/Zenodo publication.

## Validation and security

- [x] Dependencies are pinned in `requirements.txt` and installed through `environment.yml`.
- [x] A new clean Python 3.10.20 environment was created from `environment.yml` on 2026-10-01.
- [x] Clean-environment imports, synthetic demo, unit tests, package-input check, manifest/reference verification, and frozen exact-replay evidence checks pass; see `VALIDATION_REPORT.md`.
- [x] Personal absolute project paths, credentials, tokens, private keys, and private URLs were not found.
- [x] Raw third-party EO/GIS assets were not found in the public package.
- [x] `FILE_MANIFEST.csv` records size and SHA-256 for each distributed file (regenerate immediately before release).
- [x] `.gitignore` excludes caches, locally reconstructed data, and external raw inputs.

## Publication actions — not yet performed

- [ ] Fill and validate `CITATION.cff`.
- [ ] Initialize/review the public Git history.
- [ ] Create the public GitHub repository.
- [ ] Create and push the public release tag.
- [ ] Archive the tagged release with Zenodo.
- [ ] Record the Zenodo DOI in the repository and manuscript.
- [ ] Finalize the Code Availability statement with public URLs and DOI.
- [ ] Rebuild and hash the release ZIP after final metadata replacement.

## Release gate

The scientific and reproducibility content is suitable for a publication candidate. Public release remains blocked by the intentionally unresolved citation/repository/DOI placeholders and by the external publication actions listed above. Do not publish until every unchecked mandatory item has been resolved.

## Requested release gate summary

- [x] LICENSE complete
- [ ] CITATION.cff public-location metadata complete — author and release metadata are complete, but the real repository URL and archive DOI do not yet exist
- [x] dependencies complete
- [x] English-only public code — except documented Japanese data-parsing literals that must be preserved
- [x] minimal demo passes
- [x] pytest passes
- [x] manifest check passes
- [x] Hiroshima replay evidence verified
- [x] Kyushu replay evidence verified
- [x] no secrets
- [x] no personal paths
- [x] no raw third-party EO/GIS data
- [x] data-source instructions complete
- [x] baseline provenance documented
- [x] README complete
- [x] Code Availability draft ready — placeholders remain intentionally
- [x] Git initialization ready
- [ ] GitHub publication ready — blocked by unresolved public metadata and the intentionally unperformed publication step
- [ ] release tag ready — tag/version has not been chosen or created
- [ ] Zenodo archival ready — requires a public tagged release and final metadata
