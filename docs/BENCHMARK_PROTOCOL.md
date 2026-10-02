# Benchmark protocol

## Hiroshima 2018

The executable lineage is preserved in the original authority scripts:

1. `freeze_positive_evidence_grid_registry.py` freezes positive evidence.
2. `build_hard_control_candidate_eligibility_pool.py` constructs the label-blind candidate pool.
3. `reconcile_and_freeze_hard_control_candidate_eligibility_pool_v1_1.py` freezes the reconciled eligible domain.
4. Scripts 12, 13, 14B, and 16A perform the audited repair, exact assignment, and spatial-group recovery used by the formal benchmark.
5. `127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py` exports the frozen protocol dictionary and balance diagnostics.
6. `130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py` executes deterministic clean replay checks.

The final benchmark contains 5,056 complete 1:2 matched sets. Matching uses only pre-outcome, label-blind covariates and frozen feasibility/common-support rules. No model score enters matching.

## Kyushu 2017

`run_kyushu_external_dataset_01.py`, `kyushu_match_01.py`, and `kyushu_dataset_freeze_01.py` construct and freeze the independent external dataset. The final benchmark contains 1,692 complete 1:2 sets. Hiroshima-trained parameters are not adapted using Kyushu outcomes.

## Exact rules

The actual variable dictionary, normalization, distance components, feasibility constraints, common-support rules, tie policy, and optimization description are copied under `configs/matching/`. These files take precedence over prose summaries.

