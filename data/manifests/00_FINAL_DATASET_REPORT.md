# FINAL REPAIRED DATASET V2

## Status

`PASS`

## Source version

`FINAL_REPAIRED_DATASET_V2_20260818`

## Repaired matching source

```json
{
  "source_path": "${PROJECT_ROOT}\\data\\06_hard_control\\03_hc_bug03_controlled_rebuild_v1_1\\03_hard_control_02_pair_edges_repaired_v1.csv",
  "source_mode": "DISCOVERED_COMPLETE_REPAIRED_ASSIGNMENT"
}
```

## Pair rebuild

```json
{
  "pair_sets": 5056,
  "positives": 5056,
  "hard_controls": 10112,
  "unique_hard_controls": 10112,
  "training_rows": 15168,
  "changed_control_slots": 24,
  "positive_unit_ids_unchanged": true
}
```

## Corrected HC00

```json
{
  "eligibility_table": "${PROJECT_ROOT}\\data\\02_reliable_observation_domain\\11_hard_control_candidate_eligibility_pool_frozen_v1_1\\01_frozen_full_registry\\hard_control_eligibility_full_grid_250m_frozen_v1_1.parquet",
  "core_table": "${PROJECT_ROOT}\\data\\06_hard_control\\00_candidate_pool\\hard_control_00_core_candidate_pool.csv",
  "eligible_count": 29084,
  "current_core_count": 28102,
  "corrected_core_count": 28072
}
```

## Spatial split

```json
{
  "source": "${PROJECT_ROOT}\\data\\05_final_dataset_assembly\\01_spatial_split_protocol\\dataset_01_pair_set_fold_crosswalk.parquet",
  "pair_sets": 5056,
  "spatial_groups": 500,
  "group_cross_fold_count": 0,
  "fold_pair_counts": {
    "FOLD_1": 1011,
    "FOLD_2": 1011,
    "FOLD_3": 1011,
    "FOLD_4": 1011,
    "FOLD_5": 1012
  }
}
```

## Control legality

```json
{
  "controls": 10112,
  "controls_outside_corrected_core": 0,
  "p0_edge_failures": 0,
  "control_reuse": 0
}
```

## Static

```json
{
  "rows": 15168,
  "static_fields": 92,
  "units_with_all_92_static_missing": 0,
  "fields_all_missing": [],
  "unit_id_unique_in_source": 15168
}
```

## Dynamic

```json
{
  "tensor_shape": [
    15168,
    70,
    10
  ],
  "dtype": "float32",
  "finite": true,
  "negative_count": 0,
  "start_utc": "2018-07-05 00:30:00+00:00",
  "end_utc": "2018-07-06 11:00:00+00:00",
  "timesteps": 70,
  "dynamic_fields": 10,
  "future_timesteps_read": 0
}
```

## Pair tensor

```json
{
  "shape": [
    5056,
    3
  ],
  "pair_sets": 5056,
  "all_sample_indices_in_range": true
}
```

## Fold QA

```json
{
  "fold_count": 5,
  "fold_labels": [
    "FOLD_1",
    "FOLD_2",
    "FOLD_3",
    "FOLD_4",
    "FOLD_5"
  ],
  "pair_cross_fold_count": 0,
  "spatial_group_cross_fold_count": 0,
  "fold_pair_counts": {
    "FOLD_1": 1011,
    "FOLD_2": 1011,
    "FOLD_3": 1011,
    "FOLD_4": 1011,
    "FOLD_5": 1012
  }
}
```

## Final hard checks

```json
{
  "pair_sets_5056": true,
  "positives_5056": true,
  "controls_10112": true,
  "unique_controls_10112": true,
  "training_rows_15168": true,
  "changed_slots_24": true,
  "positives_unchanged": true,
  "corrected_core_28072": true,
  "controls_inside_corrected_core": true,
  "p0_edge_failures_zero": true,
  "control_reuse_zero": true,
  "canonical_groups_500": true,
  "canonical_group_crossfold_zero": true,
  "pair_crossfold_zero": true,
  "spatial_group_crossfold_zero": true,
  "static_fields_92": true,
  "static_join_complete": true,
  "dynamic_shape_15168_70_10": true,
  "dynamic_finite": true,
  "dynamic_negative_zero": true,
  "dynamic_future_read_zero": true,
  "pair_tensor_shape_5056_3": true,
  "pair_tensor_indices_valid": true,
  "five_folds": true
}
```

## Semantics

- Formal positive anchors are unchanged.
- Hard controls are repaired under corrected HC00 eligibility.
- Old spatial group/fold is NOT used as a pre-match candidate gate.
- After matching, all three members of each pair inherit the canonical
  positive-anchor pair split.
- Static and rainfall sources remain frozen upstream assets.
- No model was trained during this data finalization.
