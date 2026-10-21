# Data availability and redistribution boundary

**Published-data update (2026-10-21):** The eleven reviewed frozen derived inputs and historical hand-offs described below are now distributed separately in [Zenodo record 23533084](https://zenodo.org/records/23533084) (DOI 10.5281/zenodo.23533084) as `MSRR_CG_approved_data_candidate.zip`. They remain absent from GitHub. Download, extract at the repository root without changing paths, and verify `metadata/SHA256SUMS.txt` as instructed in the README. The file-status discussion below documents the earlier GitHub-only review and is superseded for those eleven files; it must not be read as saying the published ZIP is still awaiting deposit. Raw third-party EO/GIS source products are not in that ZIP.

This repository does not bundle the third-party raw EO/GIS products or source inventories used in the study. It contains schemas, provenance manifests, hashes, exact matched-set replay memberships, a synthetic example, and frozen numerical reference results. Absence from this repository is a packaging decision, not a claim that every upstream licence prohibits redistribution.

## Included

- feature-order and timestamp schemas;
- source and processing manifests;
- data-quality, leakage, and benchmark-integrity audits;
- frozen matching protocol records and exact replay memberships;
- OOF margins, metrics, bootstrap outputs, and other numerical reference materials;
- an eight-set synthetic MSRR demo containing no scientific or personal data.

## Not included

- AJG and GSI inventory/evidence layers;
- GSI elevation products and Copernicus DEM GLO-30 tiles;
- NASA GPM IMERG granules;
- Landsat scenes and QA bands;
- SoilGrids rasters;
- Copernicus Global Land Service LC100 rasters;
- historical OpenStreetMap/Geofabrik PBF snapshots;
- J-FlwDir rasters;
- GSJ/AIST Seamless Digital Geological Map source vectors;
- credentials, API keys, cookies, tokens, passwords, or account files.

## Seven frozen *derived* inputs needed by the published full-workflow entry points

The seven paths reported by `python scripts/01_verify_inputs.py --full` are locally generated study products, **not seven raw third-party downloads**. None has been uploaded to GitHub. The four Hiroshima paths are under `data/07_FINAL_REPAIRED_DATASET_V2/`; the three Kyushu paths are under `external/kyushu_2017_asakura_toho/99_frozen_dataset/`. A separate unpublished local candidate contains `08_PAIR_TENSOR_INDEX_INT64.npy`, a pure author-generated position array, but this public code commit excludes it pending final attribution/ownership sign-off. The other six remain excluded pending rights review; no source data were changed.

| Derived product | Content and upstream rights to check | Public-package status |
|---|---|
| `04_FINAL_SAMPLE_INDEX.parquet` | Hiroshima matched-set identities, roles, labels, and fold allocation; the cited AJG fourth report is CC BY 4.0, but GSI evidence linkage and the actual frozen source edition need confirmation. | Not bundled; small candidate for lawful deposit after provenance/rights check. |
| `06_STATIC_92_RAW_BY_SAMPLE.parquet` | Hiroshima 92-feature table combining elevation, soil, satellite, cover, OSM, hydrology, and geology sources. Provider-specific attribution and ODbL/database questions need review. | Not bundled; do not assume either prohibition or permission for this mixed-source table. |
| `07_DYNAMIC_70x10_FLOAT32.npy` | Hiroshima rainfall-derived numeric tensor from GPM IMERG; source-version and NASA dataset notices must be retained. | Not bundled; potentially releasable after product-specific terms and attribution review; not a small fixture. |
| `08_PAIR_TENSOR_INDEX_INT64.npy` | Author-generated integer positions connecting the sample order to 1:2 triplets; no raw raster values. | Not bundled in the public code package; retained in a separate local candidate pending final attribution/ownership sign-off. |
| `kyushu_external_matched_triplet_index_v1.parquet` | Kyushu matched identities, roles, labels, and matching diagnostics; GSI event evidence provenance needs review. | Not bundled; small candidate after rights/provenance check. |
| `kyushu_external_matched_static_92f_v1.parquet` | Mixed-source Kyushu 92-feature table. | Not bundled; source-by-source rights and attribution review needed. |
| `kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet` | Rainfall-derived values with unit IDs and time slots; GPM terms plus linkage to event identities need review. | Not bundled; source and identity review needed. |

The actual blockers for a public end-to-end rerun include **six remaining frozen model-ready products, two additional source-derived pair files, the 121-to-128 and 124D-to-125 prediction hand-offs, and some upstream vintage inputs**—not an established blanket ban on sharing derived data. The eight runtime inputs and two row-level hand-offs, with exact paths, schemas, hashes, producer/consumer relationships and rights questions, are listed in `FINAL_REPRODUCIBILITY_GATE.md`. The local candidate has restored loader code, exact-path metadata and the byte-identical 120B output hand-off; it has not been published. `scripts/01_verify_inputs.py --full` checks only seven files, not the complete dependency closure. A viable release route is an author-approved, separately licensed data deposit with exact frozen bytes, SHA-256, source/vintage provenance, attribution, plus a tested placement/import guide; see `WORKFLOW_HANDOFF_AUDIT.md`. If that route is not lawful or feasible for a particular product, give a concrete reacquisition/reconstruction route and state the resulting verification limit. Never replace these with synthetic values to claim manuscript-result reproduction.

The public synthetic example checks the canonical 792-to-2446 MSRR transform and matched-set metrics. The test suite also checks exact replay membership invariants and a test-only invented legal-edge graph through the canonical common-support/all-or-none matching interface. These are software-behaviour checks, not an independent recreation of either event dataset or reported scores.

## Reacquisition

Use `docs/DATA_SOURCES.md` as the human-readable registry and `configs/external/`, `data/manifests/`, and `data/schemas/` as the machine-readable contracts. Restore each input at its documented `${PROJECT_ROOT}`-relative path and verify version, date/snapshot, bounds, CRS, units, field schema, and checksum before running a reconstruction script.

Several inputs require manual acquisition or authentication. A current provider product is not automatically equivalent to the event-specific or frozen asset used by the study. The current GSJ Seamless V2 download, for example, may change over time; the study asset is identified by its acquisition date and SHA-256 in `docs/DATA_SOURCES.md`.

## Redistribution

No blanket sublicence is asserted for any source product. Some source providers explicitly permit redistribution or adaptation with conditions; others have dataset-specific, registration, or share-alike questions. The exact source vintage and rights for each proposed derived file must be documented before deposit; see `docs/THIRD_PARTY_DATA.md`.
