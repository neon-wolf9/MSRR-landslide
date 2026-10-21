# Reproducibility workflow

## 1. Scope

This document maps the manuscript evidence chain to the retained authority scripts and frozen reference outputs. It distinguishes redistribution-safe verification from a full source-data-to-results rerun. The repository preparation did not change matching rules, weights, folds, labels, the MSRR definition, model hyperparameters, Kyushu protocol, or manuscript metrics.

## 2. Environment

Create the recorded Python 3.10.20 environment:

```bash
conda env create -f environment.yml
conda activate msrr-landslide
python scripts/00_check_environment.py
```

The exact validated public package versions are in `requirements.txt`. Platform-specific numerical or geospatial differences should be recorded rather than hidden.

## 3. Data preparation

First download `MSRR_CG_approved_data_candidate.zip` from [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), extract it at the repository root without changing paths, and verify its `metadata/SHA256SUMS.txt` as shown in the README. The archive supplies eleven frozen derived inputs/hand-offs at their original `data/`, `external/`, and `experiments/` paths; raw third-party products are not included. For source-to-benchmark reconstruction, acquire those raw products using `docs/DATA_ACQUISITION.md` and validate against `docs/DATA_SOURCES.md`, `data/manifests/`, and `data/schemas/`. Then run:

```bash
python scripts/01_verify_inputs.py --full
```

This command checks only seven model-ready files; it does **not** certify every formal loader, ancillary schema or downstream result dependency. `REPRODUCIBILITY_GAP_ANALYSIS.md` records the earlier pre-deposit audit. This link update has not rerun the formal experiments and does not independently certify every full-workflow prerequisite.

The underlying data-preparation lineage includes positive-evidence freezing, hard-control eligibility construction, reconciliation, terrain/static feature processing, rainfall assembly, and event-specific dataset freezing. Exact script roles are described in `docs/BENCHMARK_PROTOCOL.md`.

## 4. Hiroshima benchmark reconstruction

The executable lineage retains:

1. `scripts/freeze_positive_evidence_grid_registry.py`
2. `scripts/build_hard_control_candidate_eligibility_pool.py`
3. `scripts/reconcile_and_freeze_hard_control_candidate_eligibility_pool_v1_1.py`
4. audited repair/allocation scripts 12, 13, 14B, and 16A
5. `scripts/127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py`
6. `scripts/130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py`

Run the final protocol and replay audits after the required inputs are restored:

```bash
python scripts/127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py
python scripts/130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py
```

The frozen target is 5,056 complete sets, 5,056 positives, 10,112 controls, 1:2 matching, and zero control reuse.

## 5. Fold reconstruction

Spatial five-fold splitting is performed at matched-set/spatial-group level. Frozen fold summaries and QA are in `data/manifests/10_FOLD_SUMMARY.csv`, `10A_FOLD_QA.json`, and `reference_results/hiroshima/FOLD_REPLAY_AUDIT.csv`. `tests/test_fold_integrity.py` verifies the redistribution-safe fold contract.

## 6. RAW representation

RAW contains 92 static predictors plus `70 x 10 = 700` rainfall values, for 792 features. Role, label, inventory flag, and fold metadata are excluded from model inputs.

## 7. MSRR representation

The canonical manuscript implementation is `dynsum7`, `raw_features`, and `msrr_features` in `scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py`. For each unlabeled three-member set it concatenates RAW state, signed deviation from the shared mean, absolute deviation, and seven rainfall-residual summaries per rainfall variable. The final dimension is `792 + 792 + 792 + 70 = 2446`.

The published synthetic fixture runs this transform and the matched-set metric interface without scientific observations. `tests/test_matching_integrity.py` also invokes the canonical common-support/all-or-none solver on invented legal edges to check two distinct controls, zero reuse, and exclusion of an unmatchable positive. These are runnable software contracts only. No public fixture currently regenerates Hiroshima or Kyushu covariate tables, source-to-benchmark construction, or manuscript scores; those require the non-bundled inputs classified in `DATA_AVAILABILITY.md`.

```bash
python examples/minimal_msrr_example.py
python -m pytest -q tests/test_msrr.py tests/test_msrr_dimensions.py -p no:cacheprovider
```

## 8. Principal learner evaluation

Experiment 120B evaluates RAW and MSRR with ElasticNet logistic regression, MLP, HistGradientBoosting, and XGBoost under the same frozen outer folds:

```bash
python scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py
```

Frozen outputs are in `reference_results/hiroshima/`. The XGBoost principal execution has RAW AUROC 0.671284 and MSRR AUROC 0.881482. The diagnostic pointwise execution near 0.8796 is separate and must not replace it.

## 9. Matched-set metrics

AUROC and AUPRC are row-level score metrics. StrictPair is the fraction of sets in which the positive score exceeds both controls. Edge is the fraction of the two positive-control comparisons that are correctly ordered. Tests in `tests/test_metrics.py` verify the definitions.

## 10. Complete-set bootstrap

Experiment 121 resamples complete matched sets and reuses identical draws within each method contrast:

```bash
python scripts/121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py
```

Frozen summaries and draws are in `reference_results/hiroshima/`.

## 11. Ablation

Experiment 119 evaluates the seven frozen representation variants:

```bash
python scripts/119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py
```

Outputs are in `reference_results/ablation/`.

## 12. Matched-context placebo

Experiment 123 compares true MSRR with RAW, global centering, and shuffled matched context:

```bash
python scripts/123_RUN_MATCHED_CONTEXT_PLACEBO_XGB.py
```

Outputs are in `reference_results/placebo/`.

## 13. PairDiff and ranking diagnostics

These analyses test task-aligned objectives and do not replace the principal pointwise execution:

```bash
python scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py
python scripts/126C_AUDIT_PAIRDIFF_CONVERGENCE_SENSITIVITY.py
python scripts/131_RUN_PAIRDIFF_XGB_FAIRNESS.py
```

Frozen results are in `reference_results/comparators/`.

The 126B entry checks the original `126A_AUDIT_FROZEN_XGB_REPLICATION.py` and its frozen `126A_DIAGNOSTIC_DECISION.json` as amendment authorities. Both are present in this local candidate. The original 126A script can perform technical fits when run; it is **not** a lightweight validation command and was not executed in release preparation.

## 14. Published comparators

Read `docs/BASELINE_PROVENANCE.md` before running the comparator panel. The upstream PU-pullbaggingDT checkout is not redistributed because the inspected source had no explicit license; only repository-authored adapters are included.

```bash
python scripts/128B0_BUILD_BASELINE_ADAPTERS_AND_SMOKE_TEST.py
python scripts/128B1_RUN_PUBLISHED_ALGORITHM_COMPARISON_5FOLD.py
python scripts/128C_RUN_PUBLISHED_METHOD_PAIRED_BOOTSTRAP.py
```

## 15. Sensitivity analyses

The conservative pre-trigger and common-DEM workflows are retained as separately frozen sensitivity analyses:

```bash
python scripts/128B_RUN_CONSERVATIVE_PRETRIGGER_SENSITIVITY.py
python scripts/128C_RUN_SCALABLE_EXACT_PRETRIGGER_REMATCHING.py
python scripts/129_RUN_COMMON_DEM_SENSITIVITY.py
python scripts/129B_RUN_FULL_COMMONDEM_EXACT_REMATCHING.py
```

Outputs are in `reference_results/robustness/`. They must not overwrite the principal benchmark.

## 16. Kyushu external evaluation

The Kyushu dataset is constructed and frozen by `run_kyushu_external_dataset_01.py`, `kyushu_match_01.py`, and `kyushu_dataset_freeze_01.py`. The final target is 1,692 complete sets, 1,692 positives, 3,384 controls, and zero control reuse.

```bash
python scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py
python scripts/125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py
```

No Kyushu outcome information is used for Hiroshima fitting, preprocessing, configuration selection, or threshold selection.

## 17. Domain-shift analysis

```bash
python scripts/126_RUN_HIROSHIMA_KYUSHU_DOMAIN_SHIFT_CHARACTERIZATION.py
```

Outputs are in `reference_results/domain_shift/`.

## 18. Spatial analysis

```bash
python scripts/127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py
```

Outputs are in `reference_results/spatial/`.

## 19. Expected reference outputs and limitations

Run the redistribution-safe verifier:

```bash
python scripts/15_verify_reference_results.py
```

`RESULT_MANIFEST.csv` and `docs/PAPER_RESULT_MAP.md` map each claim to its authority file. Frozen exact-replay evidence confirms both benchmark sizes and identities, but the public package cannot reconstruct raw provider data that are not redistributed. Full workflows are long-running and may depend on historical product snapshots, authenticated downloads, and platform-specific geospatial libraries. Missing inputs or version mismatches must be reported; they are not permission to change the protocol.
