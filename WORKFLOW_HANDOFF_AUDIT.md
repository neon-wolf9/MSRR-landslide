# Scientific dependency and frozen hand-off audit (local candidate, 2026-10-21)

> **Historical pre-release record.** Retained for scientific and release provenance. Statements below about a candidate, missing deposit, tag, or publication status describe the audit-time snapshot, not the current release. For current code and data access, use [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1), [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), and `README.md`. Historical tests are not a new scientific rerun.

No formal experiment was executed. Findings below come from static source inspection, safe imports, metadata reads, byte-for-byte copying of existing author results, and lightweight path tests. `reference_results/` contains curated publication evidence; it is **not** automatically the filesystem layout consumed by the formal scripts.

## 120B actual loader closure

`scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py` imports `scripts/TRAIN_SEHC_V2_TEMPLATE.py`. Its `load_dataset_loader()` loads `scripts/23D_RUN_NEURAL_BASELINES_REPAIRED_V2.py`, which loads `scripts/20A_RUN_TABULAR_BASELINES_REPAIRED_V2.py`; `20A.preflight()` is the actual dataset gate. The BASE-00 wrappers and `baseline_tuning.py`/`baseline_config.py`, `baseline_base.py`, `baseline_plain_fusion.py`, `baseline_ranknet.py`, plus frozen `sehcnm_modules.py`, are now present. All four linked Python authorities and all seven target entry points imported in Python 3.10.20 without scientific execution. LightGBM 4.7.0 is declared because 20A imports its wrapper eagerly, although the reported 120B panel excludes LightGBM.

The executable 120B method/configuration is in the script itself and the original `RUN_MANIFEST.json`, now in its real hand-off directory. `RUN_MANIFEST.json` preserves the **original** 120B script SHA-256 (`cb5a7a785b0449565542911d50a0fb038a85f358974af4d96c86411054422ac3`); the portable public script differs only in root/docstring paths. No learner choices or hyperparameters were changed in this turn.

`20A.preflight()` checks 12 data/metadata paths plus three BASE-00 module paths. All code and seven of 12 data/metadata paths are present in the candidate. The **five exact absent Hiroshima runtime inputs** are listed below; all exist in the original project and were intentionally not copied pending rights review:

| Repository-relative runtime path | Original exists | Candidate | Why not staged |
|---|---|---|---|
| `data/07_FINAL_REPAIRED_DATASET_V2/04_FINAL_SAMPLE_INDEX.parquet` | Yes, 15,168 rows | Missing | Inventory-derived labels/IDs and fold allocation; exact AJG/GSI edition and disclosure review. |
| `data/07_FINAL_REPAIRED_DATASET_V2/03_REPAIRED_PAIR_SETS.parquet` | Yes, 145,181 bytes | Missing | Source-derived set membership/matching fields; review with index. |
| `data/07_FINAL_REPAIRED_DATASET_V2/06_STATIC_92_RAW_BY_SAMPLE.parquet` | Yes, 15,168 rows | Missing | Mixed-source 92 fields; source attribution and OSM database-law review. |
| `data/07_FINAL_REPAIRED_DATASET_V2/07_DYNAMIC_70x10_FLOAT32.npy` | Yes, `(15168,70,10)` | Missing | Match exact NASA IMERG collection rights/citation and frozen source. |
| `data/07_FINAL_REPAIRED_DATASET_V2/08A_PAIR_TENSOR_REGISTRY.parquet` | Yes, 137,296 bytes | Missing | Source-derived sample IDs/positions; review with index. |

The author-created positional `08_PAIR_TENSOR_INDEX_INT64.npy` and exact-path timestamp registry, rainfall field order, closure flag, decision JSON, and two feature-schema JSONs are present. The closure flag/decision are historic frozen claims, **not** proof the current candidate contains the private data. `scripts/01_verify_inputs.py --full` tests only seven high-level files and is not a 120B preflight substitute.

## Original frozen outputs versus executable hand-off

The original project contains `experiments/MSRR_CROSS_BACKBONE_5FOLD_V1B/{OOF_RESULTS.csv,FOLD_METRICS.csv,VALIDATION_SELECTION.csv,RUN_MANIFEST.json,eight *_OOF_margin.npy}`. These 12 files are now staged at **exactly those relative paths** in the local candidate, unchanged byte-for-byte from the original executed directory. Their manifest hashes and the hand-off-specific readme distinguish them from any newly computed result. Curated OOF/metric/margin copies already existed in `reference_results/hiroshima/`, and a model-selection copy in `configs/learners/`; the latter three CSV copies have equivalent text but not the same line-ending bytes as the original experiment files. The new exact-path copies resolve the 120B-output *directory* mismatch without changing any downstream science code.

| Consumer | Actually reads from original hand-off | Hand-off state now | Other blocking inputs |
|---|---|---|---|
| `121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py` | `OOF_RESULTS.csv`, `FOLD_METRICS.csv`, eight OOF margins; its loader also rebuilds row/pair ordering | All 10 consumed 120B files staged and path-tested | The five Hiroshima inputs above. Its own `experiments/MSRR_121_PAIRED_BOOTSTRAP/120B_RECOVERED_ROW_LEVEL_OOF.parquet` is an **output** of 121, not a pre-existing input to 121. |
| `123_RUN_MATCHED_CONTEXT_PLACEBO_XGB.py` | `OOF_RESULTS.csv`, `FOLD_METRICS.csv`, selected XGBoost OOF margins | Staged | Five Hiroshima inputs; formal placebo must not be run in this task. |
| `124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py` | `VALIDATION_SELECTION.csv` for Hiroshima-only XGBoost deployment choice | Staged | Five Hiroshima plus three Kyushu model-ready files (below). It fits full-Hiroshima preprocessing in the formal run; archived preprocessing JSON is a result, not a replacement for inputs. |
| `125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py` | 120B `VALIDATION_SELECTION.csv` and `FOLD_METRICS.csv` | Staged | Eight event inputs plus original 124D frozen prediction file and freeze JSON (below). |

Other original intermediate records were located but **not** staged as public-ready files:

| Exact relative path | Original exists / SHA-256 | Candidate | Reason/status |
|---|---|---|---|
| `data/07_FINAL_REPAIRED_DATASET_V2/04_PAIR_SPATIAL_GROUP_FOLD_CROSSWALK.parquet` | Yes / `1fe4c9bc76db356743db2d6a3ca2cfc9d2c7e4752afb6a067b9c55c3269f5130` | Missing | 5,056 pair/fold rows; reconstruction evidence, not a direct 120B loader input. Pair IDs/folds need index-level rights/disclosure review. |
| `experiments/MSRR_121_PAIRED_BOOTSTRAP/120B_RECOVERED_ROW_LEVEL_OOF.parquet` | Yes / `d7687627f793fda5f28a9342720a1dbe6194b0a24d0f347e95e05f839cc9d63e` | Missing | 121 output and 128B1 input; 121,344 rows with unit/set IDs, labels and scores. Inventory-linked IDs/labels need rights review; historic 121 summary in `reference_results/hiroshima` does not substitute for it. |
| `external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_V1D/124D_EXTERNAL_PREDICTIONS_FIXED_BEFORE_LABELS.parquet` | Yes / `f55c1012ead05520ae4909e6e75dd2f5ff8e4249896c61b2b06c4497537192c8` | Missing | Required by 125; 10,152 rows contain unit/set IDs and scores. Review disclosure/rights before public deposit. |
| `external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_V1D/124D_PREDICTION_FREEZE.json` | Yes / `2b29a76b2a61cfbe743be40b79b50f214521ceb6bec18035ea2b84cd2aedb2b0` | Staged at executable path, exact bytes | Hash/count metadata already curated in `reference_results/kyushu/`; the corresponding prediction Parquet is still excluded pending review. |
| `external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_V1D/124D_HIROSHIMA_FULL_PREPROCESSOR.json` | Yes / `85bc3269d9136a8031e9a6b767523aedde711bcd083e94025e24eae65f6ebf72` | Missing | Fitted category map, dynamic/static preprocessing values; generated by 124D and not read as a 125 prerequisite. Potential source-derived values need review before release. The 125 copy has the same SHA-256. |

The **three further Kyushu runtime inputs**, all present in the private original and absent from the candidate, are `external/kyushu_2017_asakura_toho/99_frozen_dataset/06_matched_index/kyushu_external_matched_triplet_index_v1.parquet`, `.../07_matched_static/kyushu_external_matched_static_92f_v1.parquet`, and `.../09_matched_dynamic_70_view/kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet`. Exact hashes, sizes and source obligations are in `DATA_RIGHTS_AUDIT.md`. 124D/125 discover these by schema under the frozen dataset root; a reference metric CSV is not an input replacement.

## 119 and 128 closures

The seven formal entry scripts all exist and imported safely. Their fixed output targets (none was created by this audit) are:

| Entry | Script-defined output directory | Runtime-input gate in this candidate |
|---|---|---|
| 119 | `experiments/MSRR_REPRESENTATION_ABLATION_5FOLD_V1/` | Five Hiroshima loader files missing. |
| 120B | `experiments/MSRR_CROSS_BACKBONE_5FOLD_V1B/` | Five Hiroshima loader files missing; directory currently holds only copied historical output hand-off. The formal entry refuses a nonempty output directory, so a future authorized rerun must use a clean separate workspace, not this candidate path. |
| 121 | `experiments/MSRR_121_PAIRED_BOOTSTRAP/` | 120B hand-off present; five Hiroshima loader files missing. |
| 123 | `experiments/MSRR_123_MATCHED_CONTEXT_PLACEBO/` | 120B hand-off present; five Hiroshima loader files missing. |
| 124D | `external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_V1D/` | 120B selection present; eight Hiroshima/Kyushu runtime files missing. |
| 125 | `external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2/` | 120B selection/fold files and 124D freeze metadata present; eight event files and 124D prediction Parquet missing. |
| 128B1 | `experiments/MSRR_128B1_PUBLISHED_ALGORITHM_COMPARISON_5FOLD/` | Author protocol records present; five Hiroshima files, 121 row-level OOF, and unlicensed vendor checkout missing. |

119 uses the same Hiroshima loader and five real missing runtime files as 120B. Its ablation metrics are only archived references. 128B1's author-written adapters and `spy_pu_brf_formal_config.json` import and exist. Its seven original 128A3/128B0 protocol/implementation JSON authorities have now been copied byte-for-byte to their expected experiment directories; these are author-generated configuration/provenance records, not new fits. 128B1 still needs the absent 121 row-level OOF and a clean checkout at `third_party/PU-pullbaggingDT` commit `a8857dad30a454e3155644b18ddd2600df860e70`. The inspected third-party source has **no explicit redistribution licence**; it has not been bundled. `third_party/README.md` gives the independent upstream acquisition route. This is a third-party code-licence limit, not evidence that the seven derived scientific inputs are forbidden to share.

## C&G assessment and next checks

An external researcher can inspect the complete 120B Python import graph, its frozen configs, genuine MSRR implementation, synthetic dimensional/metric case, and exact archived 120B hand-off. They **cannot yet** run the two-event formal science from the candidate because eight runtime inputs and the 124D/121 downstream intermediates are not cleared/provided. No data restriction has been falsely asserted for these inputs: `NEEDS RIGHTS REVIEW` remains distinct from a proven redistribution prohibition. The documented independent source-acquisition route for raw EO/GIS products and the absence of an explicit licence for the optional upstream comparator checkout are concrete access/redistribution limitations; the six unreviewed derived files and two pair files are **not yet justified non-sharing cases**. Under [C&G's policy](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004), a justified data limitation needs a clear, source-specific explanation and a runnable dummy/synthetic test, while the main method must remain independently evaluable. Where an exact source licence ultimately prevents redistribution, document the product/version, restriction clause, attribution, lawful acquisition/rebuild steps, and the verification limit. Do not alter IDs/format to evade terms or claim synthetic metrics are paper results.

Next actions: obtain file-level author/provider review for the five Hiroshima and three Kyushu runtime files, the 121/124D row-linked intermediates, and the fold crosswalk; decide whether the exact original bytes can be deposited with a separate data licence. If approved, stage them at the paths above, run only path/schema/hash/light interface checks under this task's restrictions, and then seek a separately authorized formal independent replication. No formal rerun occurred here, and its absence alone is **not** this audit's failure reason.

## Lightweight acceptance after hand-off assembly

Using the existing `[local Python 3.10.20 interpreter]` (Python 3.10.20; this was **not** a fresh independent environment), syntax compilation of the relevant sources, ten import-smoke modules, the synthetic MSRR example, environment check, package input screen, manifest/reference verification, and `python -m pytest -q -p no:cacheprovider` were performed. The synthetic example produced RAW `(8,3,792)` and MSRR `(8,3,2446)` and passed StrictPair/Edge checks. After the 124D freeze-record test was added, an intermediate pytest run gave **13 passed, 1 failed** solely because the manifest still reflected the pre-edit `.gitattributes` bytes. The final distributable manifest contains **369 files** plus `FILE_MANIFEST.csv` (370 distributable files total), with zero missing entries; after refresh, the manifest checker passed and the complete suite reported **14 passed**. Full output for the first hand-off run is in `CANDIDATE_HANDOFF_VALIDATION.log` in the candidate directory; it is excluded from the distributable manifest. No full-data preflight or formal science command was run. The final post-staging verification supersedes the in-run manifest check.
