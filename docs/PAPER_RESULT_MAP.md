# Paper-result map

This table identifies the distributed authority file for each principal claim. Values are not retyped into code.

| Paper result | Authority file |
|---|---|
| Four-backbone Hiroshima RAW/MSRR metrics | `reference_results/hiroshima/OOF_RESULTS.csv` |
| Cross-backbone deltas and support decision | `reference_results/hiroshima/RAW_VS_MSRR_DELTAS.csv`, `GATE120B_DECISION.json` |
| Paired bootstrap intervals | `reference_results/hiroshima/121_BOOTSTRAP_SUMMARY.csv` |
| Seven representation variants, 792 → 2,446 dimensions | `reference_results/ablation/OOF_RESULTS.csv`, `FEATURE_DIMENSIONS.csv` |
| Global-centering and shuffled-context placebo | `reference_results/placebo/123_OOF_METRICS.csv` |
| Published PU comparators | `reference_results/comparators/128B1_PUBLISHED_METHOD_COMPARISON.csv` |
| PairDiff and rank-objective audit | `reference_results/comparators/PAIRDIFF_XGB_MAIN_RESULTS.csv` |
| Kyushu primary independent evaluation | `reference_results/kyushu/124D_EXTERNAL_METRICS.csv` |
| Kyushu cross-backbone confirmation | `reference_results/kyushu/125_EXTERNAL_METRICS_FULL_PANEL.csv` |
| Conservative pre-trigger sensitivity | `reference_results/robustness/ARM_A_VS_ARM_B_COMPARISON.csv` and companion Arm-B files |
| Common-DEM sensitivity | `reference_results/robustness/ARM_A_VS_ARM_B_COMMONDEM_COMPARISON.csv` |
| Domain-shift statistics | `reference_results/domain_shift/126_DOMAIN_SHIFT_SUMMARY.json` and associated CSVs |
| Spatial transition/block summaries | `reference_results/spatial/127_PAPER_SPATIAL_RESULT_SUMMARY.csv` |
| Deterministic matching replay | `reference_results/hiroshima/EXACT_REPLAY_COMPARISON.csv`, `DETERMINISM_AUDIT.csv` |

Some XGBoost values differ slightly between the original 120B representation panel and later task-aligned/same-backbone audits because the latter are separately frozen executions with their own selection context. They must not be averaged or substituted across tables.

