# Experiment map

| Question | Authority script | Frozen result location |
|---|---|---|
| Matching protocol | `127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py` | `configs/matching/` |
| Exact benchmark replay | `130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py` | `reference_results/hiroshima/` |
| Representation ablation | `119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py` | `reference_results/ablation/` |
| Four-learner RAW vs MSRR | `120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py` | `reference_results/hiroshima/` |
| Paired matched-set bootstrap | `121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py` | `reference_results/hiroshima/` |
| Matched-context placebo | `123_RUN_MATCHED_CONTEXT_PLACEBO_XGB.py` | `reference_results/placebo/` |
| Primary Kyushu evaluation | `124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py` | `reference_results/kyushu/` |
| Kyushu four-learner panel | `125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py` | `reference_results/kyushu/` |
| PairDiff and rank objectives | `126B...`, `126C...`, `131...` | `reference_results/comparators/` |
| Published PU comparators | `128B1...`, `128C_RUN_PUBLISHED_METHOD_PAIRED_BOOTSTRAP.py` | `reference_results/comparators/` |
| Conservative pre-trigger | `128B...`, `128C_RUN_SCALABLE_EXACT_PRETRIGGER_REMATCHING.py` | `reference_results/robustness/` |
| Common-DEM sensitivity | `129...`, `129B...` | `reference_results/robustness/` |
| Cross-event shift | `126_RUN_HIROSHIMA_KYUSHU_DOMAIN_SHIFT_CHARACTERIZATION.py` | `reference_results/domain_shift/` |
| Spatial summaries | `127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py` | `reference_results/spatial/` |

The package does not rename these authority scripts into a cosmetic sequence because their experiment numbers are part of the frozen provenance.

