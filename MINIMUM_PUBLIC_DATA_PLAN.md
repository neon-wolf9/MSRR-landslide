# Minimum public-data plan (proposal and local candidate)

Only the author-generated pair-position array has been copied into a **separate unpublished local candidate**; it is excluded from this public code package pending final attribution/ownership sign-off. No six other real model-ready inputs have been copied or approved for upload. The table identifies the existing frozen bytes to consider after the decisions in `DATA_RIGHTS_AUDIT.md`. Place any cleared files in a data archive with their own licence/attribution statement and stable URL, then document exact repository-relative placement. Do not put them under the MIT software licence by implication, and do not publish a ZIP-only repository.

| Priority | Candidate relative path | Bytes | Frozen SHA-256 | Why needed / remaining risk |
|---|---|---:|---|---|
| 1 | `data/07_FINAL_REPAIRED_DATASET_V2/08_PAIR_TENSOR_INDEX_INT64.npy` | 121,472 | `fa8aa5f7b8d1659199ff0aa25d7e452a363c5a226da66407d23a809d18040169` | Exact 1:2 traversal; low third-party-content risk, but meaningless without the sample order; author ownership/linkage sign-off. |
| 1 | `data/07_FINAL_REPAIRED_DATASET_V2/04_FINAL_SAMPLE_INDEX.parquet` | 275,315 | `b39a262c7f6a2741b01bbacc90d7fbcc1c3ead9ecc9cf4f166f87d4fb4839602` | Real set IDs, roles, labels and five folds; verify AJG/GSI lineage, credit and grid-ID disclosure. |
| 1 | `external/kyushu_2017_asakura_toho/99_frozen_dataset/06_matched_index/kyushu_external_matched_triplet_index_v1.parquet` | 254,602 | `c6bbc287cc6e27d82182e223d11c3e220309d2e68db92e6546f56a69f1880021` | External 1:2 membership and outcome; verify exact GSI rights and disclosure. |
| 2 | `data/07_FINAL_REPAIRED_DATASET_V2/06_STATIC_92_RAW_BY_SAMPLE.parquet` | 10,154,184 | `74710e72ed649a86fb67960f5fe1a6df4278350c4049bddbc3a9fe0719ee1922` | Hiroshima model-ready 92 static inputs; mixed-source and OSM database-law assessment. |
| 2 | `data/07_FINAL_REPAIRED_DATASET_V2/07_DYNAMIC_70x10_FLOAT32.npy` | 42,470,528 | `0fed19c0736babfae2c3ae8596786420b54243611df77ee1c49e717115b7324f` | Hiroshima 700 rainfall values; exact GPM notice/citation. |
| 2 | `external/kyushu_2017_asakura_toho/99_frozen_dataset/07_matched_static/kyushu_external_matched_static_92f_v1.parquet` | 3,098,382 | `de519fa4541c98d42da42b94ca2c6ff78753bdfd99b2142e8782da991d00a21f` | Kyushu 92 static inputs; mixed-source and OSM assessment. |
| 2 | `external/kyushu_2017_asakura_toho/99_frozen_dataset/09_matched_dynamic_70_view/kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet` | 1,327,556 | `95073f1fd8ae0f9191858b4c2c4d9154c7c524106d6f2467858936afc2f17b2a` | Kyushu 700 rainfall values with unit/time linkage; GPM and ID disclosure review. |

The three priority-1 files total **651,389 bytes** and would strengthen matched-set and fold inspection, but **cannot regenerate RAW/MSRR scores**. All seven total **57,702,039 bytes**. Even a fully cleared seven-file deposit alone would not satisfy the current executable loader.

## Additional closure before saying “main results independently evaluable”

1. Resolve the exact extra real pair files required by `20A.preflight()` (listed in `REPRODUCIBILITY_GAP_ANALYSIS.md`): `03_REPAIRED_PAIR_SETS.parquet` and `08A_PAIR_TENSOR_REGISTRY.parquet`. The time and feature schemas, dataset decision flag, two model-ready schema JSONs, and BASE-00 modules are included at their exact paths in the public code package. Review the pair files' source-derived identifiers and matching metadata before any deposit.
2. The original BASE-00 Python modules needed by the 120B import chain, frozen `sehcnm_modules.py`, and Hiroshima finalizer have been restored locally. Only the finalizer's root-path expression was made relative; its source SHA and candidate SHA are recorded separately. The finalizer still requires numerous private upstream sources and must not be run as part of lightweight acceptance.
3. The local candidate now has a lightly path/structure-tested, byte-identical original 120B selection/OOF layout for 121, 123, 124D and 125. It is historical output, not newly computed results. The distinct 124D prediction/121 row-level hand-offs remain rights-pending; see `WORKFLOW_HANDOFF_AUDIT.md`.
4. Supply a short data dictionary, ordered static/rainfall field lists, ID/row-order contract, the seven exact hashes, source vintage and attribution file, restrictions if any, and versioned deposit identifiers. Keep the data licence separate from `LICENSE` (software only).
5. Only after author permission and a fresh independent reviewer-like environment check may documentation say the primary results are reproducible from the public package. Do not run that formal experiment in this preparation task.

If a file cannot legally be shared, state the **specific verified** reason and give a real provider acquisition/reconstruction route with frozen version, inputs, scripts, checksums and expected intermediate counts. A synthetic fixture is an interface test, never a replacement for scientific evidence.
