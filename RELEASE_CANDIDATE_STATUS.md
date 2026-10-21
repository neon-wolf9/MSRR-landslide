# Unpublished v1.0.1 candidate — release-review gate (2026-10-21)

> **Historical pre-release record.** Retained for scientific and release provenance. Statements below about a candidate, missing deposit, tag, or publication status describe the audit-time snapshot, not the current release. For current code and data access, use [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1), [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), and `README.md`. Historical tests are not a new scientific rerun.

**Scope note:** This document records the earlier assembly stages; its 11-test/346-file figures are historical, not the current acceptance. `FINAL_REPRODUCIBILITY_GATE.md` is the current ten-file data/hand-off and 128 provenance decision, and `WORKFLOW_HANDOFF_AUDIT.md` records the later 14-test/369-manifest stage. See the latest candidate validation log for post-edit counts.

**Decision: NOT READY — SPECIFIC BLOCKERS.** Location: `[unpublished local candidate]`. This is a local file package, not a Git tag, GitHub commit, release, or uploaded data deposit. The original research data and `v1.0.0` were not changed. No formal training, validation, testing, ablation, bootstrap, or data rebuilding was performed. **Current hand-off update:** 12 original 120B output files and seven 128A3/128B0 authority records have now been copied unchanged to the exact executable paths; `WORKFLOW_HANDOFF_AUDIT.md` is the detailed current evidence.

## Seven frozen experiment inputs

All seven original private files were previously read successfully and matched their frozen source SHA-256 records. The table is a **distribution decision**, not an assertion that rights review has failed or that the data are legally barred.

| Input | Frozen SHA-256 | Local candidate | Rights / role |
|---|---|---|---|
| Hiroshima `04_FINAL_SAMPLE_INDEX.parquet` | `b39a262c7f6a2741b01bbacc90d7fbcc1c3ead9ecc9cf4f166f87d4fb4839602` | Excluded | NEEDS RIGHTS REVIEW: exact AJG/GSI inventory edition, grid-ID linkage, attribution. Defines 5,056 sets, labels and five folds. |
| Hiroshima `06_STATIC_92_RAW_BY_SAMPLE.parquet` | `74710e72ed649a86fb67960f5fe1a6df4278350c4049bddbc3a9fe0719ee1922` | Excluded | NEEDS RIGHTS REVIEW: mixed EO/GIS source vintages, credits and OSM ODbL/database classification. Supplies 92 static fields. |
| Hiroshima `07_DYNAMIC_70x10_FLOAT32.npy` | `0fed19c0736babfae2c3ae8596786420b54243611df77ee1c49e717115b7324f` | Excluded | NEEDS RIGHTS REVIEW: match exact IMERG collection use notice/citation and processed-tensor ownership. Supplies 70 × 10 rainfall fields. |
| Hiroshima `08_PAIR_TENSOR_INDEX_INT64.npy` | `fa8aa5f7b8d1659199ff0aa25d7e452a363c5a226da66407d23a809d18040169` | Included locally, exact bytes | Author-created pure row positions, 5,056 × 3. No third-party values identified. Final author data-licence/attribution sign-off before upload. Insufficient alone for evaluation. |
| Kyushu `kyushu_external_matched_triplet_index_v1.parquet` | `c6bbc287cc6e27d82182e223d11c3e220309d2e68db92e6546f56a69f1880021` | Excluded | NEEDS RIGHTS REVIEW: exact GSI interpretation product and third-party exceptions, grid-ID linkage, attribution. Defines 1,692 triplets. |
| Kyushu `kyushu_external_matched_static_92f_v1.parquet` | `de519fa4541c98d42da42b94ca2c6ff78753bdfd99b2142e8782da991d00a21f` | Excluded | NEEDS RIGHTS REVIEW: mixed-source vintages/credit and OSM database classification. Supplies 92 static fields. |
| Kyushu `kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet` | `95073f1fd8ae0f9191858b4c2c4d9154c7c524106d6f2467858936afc2f17b2a` | Excluded | NEEDS RIGHTS REVIEW: exact IMERG notice/citation and grid/time linkage. Supplies 5,076 × 70 dynamic rows. |

The six excluded files are not automatically forbidden: [NASA GPM policy](https://gpm.nasa.gov/data/policy), [NASA Earthdata policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy), [GSI terms](https://www.gsi.go.jp/ENGLISH/page_e30286.html), and [OSM copyright](https://www.openstreetmap.org/copyright) show lawful reuse routes with different conditions. File-level source/version matching and any exception analysis are still needed. `DATA_RIGHTS_AUDIT.md` and `DATA_LICENSE_NOTICES.md` give details.

## Restored original code and metadata

- Exact-byte original BASE-00 modules: `baseline_base.py` (`89b6d15b...`), `baseline_config.py` (`04c0f47d...`), `baseline_xgboost.py` (`80a14c4e...`), `baseline_lightgbm.py` (`15f6afd9...`). `requirements.txt` now records the original environment's LightGBM 4.7.0 because the loader eagerly imports that wrapper even though the final 120B scientific panel excludes LightGBM.
- Exact-byte original frozen encoder module: `modeling/MODEL-00/00C_model_experiment_contract/run_20260805_151152/sehcnm_modules.py`, SHA-256 `9300c0a0b424be2dcc2aef4daa8273e9dda1cf45f9410a141941d3848a516e45`, needed by `baseline_plain_fusion.py`.
- Original Hiroshima finalizer `scripts/17_FINALIZE_REPAIRED_DATASET_V2.py`: original SHA-256 `2b90e0a17c5c0f02573e93c812217046d3236b68191f90ecf49741ecc7049715`; candidate SHA-256 is recorded in `FILE_MANIFEST.csv`. Only the hard-coded root path and explanatory docstring were made portable. The builder still needs private upstream layers and **was not run**.
- Exact-byte original ancillary metadata at the loader's expected paths: `07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv`, `07B_DYNAMIC_10_FIELD_ORDER.csv`, `DATA_SECTION_CLOSED.flag`, `99_FINAL_DATASET_DECISION.json`, and both `dataset_02_*_feature_schema.json` files. The closure flag and decision JSON are original *historical attestations*, not a new candidate-data completeness certificate.
- No third-party comparator source has been copied. `third_party/PU-pullbaggingDT` is absent because the inspected upstream checkout contains no explicit redistribution licence; its source must be obtained independently at the documented commit or licensed by its rights holder.

## Static primary-chain gate

| Chain | Code/import status | Missing full-run input or hand-off | Candidate conclusion |
|---|---|---|---|
| 119 representation ablation | Entry and loader import successfully | Six real model-ready inputs plus `03_REPAIRED_PAIR_SETS.parquet` and `08A_PAIR_TENSOR_REGISTRY.parquet` | Source inspectable; formal ablation not run. |
| 120B principal RAW/MSRR evaluation | Entry, 20A/23D loader and frozen encoder import successfully; configs/reference outputs present | Same real inputs; `20A.preflight()` is stricter than the seven-file checker | Source inspectable; primary numerical results not independently regenerated. |
| 121 paired bootstrap | Entry imports; exact original 120B OOF/metric arrays staged at the consumed path | Five Hiroshima real runtime inputs | 120B hand-off complete; 121 itself not rerun. |
| 123 matched-context placebo | Entry imports; exact original 120B OOF/metrics staged | Five Hiroshima real runtime inputs | 120B hand-off complete; placebo not rerun. |
| 124D Kyushu external evaluation | Entry imports; 120B `VALIDATION_SELECTION.csv` staged at the consumed path | Five Hiroshima plus three excluded Kyushu real inputs | 120B hand-off complete; external evaluation not rerun. |
| 125 full external cross-backbone | Entry imports; 120B selection/fold outputs and 124D prediction-freeze JSON staged | Eight event inputs and 124D frozen prediction Parquet | 120B hand-off complete; 124D-to-125 hand-off incomplete. |
| 128 published comparators | 128B1 entry/adapters import; seven 128A3/128B0 authority JSONs staged | Five Hiroshima inputs, 121 row-level OOF, and independently licensed upstream PU-pullbaggingDT checkout | Author-code/config path improved; full comparator execution blocked. |

The additional real pair files exist only in the private source: `03_REPAIRED_PAIR_SETS.parquet` (145,181 bytes; SHA-256 `7999db4f3402ee760a48578bdbbadb63481a253576f1f8a58e3cc868e8f73762`) and `08A_PAIR_TENSOR_REGISTRY.parquet` (137,296 bytes; SHA-256 `2db98d8418ffc5af145d108885ddc9a822acae1b36dbabb18ab18ad0297d5735`). Their identifiers/matching fields need the same rights and linkage review as the sample index. Do not infer 120B readiness from a seven-file screen.

## Manuscript and scientific-result boundaries

Read-only review of `[author-local final manuscript]` found Hiroshima 5,056 sets/10,112 controls, Kyushu 1,692/3,384, 1:2 non-reused controls, RAW 792 and MSRR 2,446 dimensions, pooled AUROC/AUPRC and matched StrictPair/Edge definitions, principal XGBoost metrics, diagnostics, and untouched Kyushu protocol stated consistently with the frozen reference summaries. **Code Availability still calls the incomplete `v1.0.0` release the complete code** (paragraph 137); the author must correct the manuscript before submission. The Word file was not edited. This is a text/reference audit, not a rerun of the paper.

## Lightweight acceptance performed on this candidate

Interpreter: `[local Python 3.10.20 interpreter]`, Python 3.10.20, NumPy 1.24.0, pandas 2.0.0, torch 2.1.0+cpu, XGBoost 3.2.0, LightGBM 4.7.0. This is the existing project environment, **not** a freshly created independent environment. Complete command output: `CANDIDATE_LIGHT_VALIDATION.log` (ignored by the distributable manifest).

| Executed check | Exit | Observation |
|---|---:|---|
| `python -m py_compile` on restored and seven principal scripts | 0 | Syntax valid; no experiment executed. |
| Seven principal entry-point imports | 0 | 119, 120B, 121, 123, 124D, 125, 128B1 import without missing Python modules in this environment. |
| `python examples/minimal_msrr_example.py` | 0 | Synthetic RAW `(8,3,792)` → MSRR `(8,3,2446)`; StrictPair 0.500000 and Edge 0.687500 test passed. |
| `python scripts/00_check_environment.py` | 0 | Recorded core package versions and imports. |
| `python scripts/01_verify_inputs.py` | 0 | Package metadata pass; explicitly reports six unbundled real inputs. This is **not** a full-data pass. |
| `python scripts/15_verify_reference_results.py` | 0 | 346 manifest-listed files and frozen results verified. |
| `python -m pytest -q -p no:cacheprovider` | 0 | 11 lightweight tests passed. |
| Memory-mapped pair array check | 0 | `int64 (5056,3)`, indices 0–15167, all 15,168 positions unique, original SHA-256 unchanged. |

The 346-entry figure above belongs to the **preceding assembly stage**. `WORKFLOW_HANDOFF_AUDIT.md` and the final refreshed `FILE_MANIFEST.csv` give the current hand-off state and file count. The local validation log and Python caches are excluded from distribution. These tests establish software/interface and frozen-file integrity only. They do not validate the paper's event scores independently.

## Release decision and next author decisions

The package has a runnable synthetic MSRR case and exact-path archived 120B hand-off, but it cannot yet permit independent regeneration of the two-event principal scores. Per the [C&G journal code/reproducibility policy](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004), this is a substantive assessability gap, not a demand to redistribute every raw EO/GIS source. Before release review, the authors need to (1) make a file-level rights/attribution decision for the five missing Hiroshima runtime files, three Kyushu files and row-linked 121/124D intermediates; (2) publish or provide a lawful, versioned acquisition route with exact SHA-256 and separate data licence; (3) resolve the 124D-to-125 prediction hand-off after rights review; (4) specify the lawful independent checkout for the unlicensed third-party comparator source; and (5) correct the manuscript Code Availability statement. These are **specific blockers**; lack of a forbidden formal rerun during this preparation task is not itself a failure.
