# MSRR definition

The authoritative implementation is `scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py`, functions `dynsum7`, `raw_features`, and `msrr_features`. Experiment 119 independently enumerates the ablation variants.

For each complete matched set containing positive `P` and controls `C1`, `C2`, a label-blind set context is computed from the three candidates. For every raw variable, MSRR appends signed deviation from that context and its absolute magnitude. Rainfall residuals additionally contribute seven fixed summaries per rainfall channel (70 values total).

For candidate `j` in set `i`, the shared mean is `mu_i = (x_i1 + x_i2 + x_i3) / 3`, the signed deviation is `d_ij = x_ij - mu_i`, and its magnitude is `abs(d_ij)`. Each candidate's 700 rainfall values are reshaped to `70 x 10` (time slots by variables). For each of the ten residual rainfall variables, `dynsum7` computes the mean, population standard deviation, maximum, minimum, sum, last residual, and maximum absolute residual across the 70 slots. The transform uses no labels or member-role indicator.

Dimension audit:

| Component | Dimensions |
|---|---:|
| Raw static | 92 |
| Signed static residual | 92 |
| Absolute static residual | 92 |
| Raw rainfall (`70 × 10`) | 700 |
| Signed rainfall residual | 700 |
| Absolute rainfall residual | 700 |
| Fixed rainfall-residual summaries (`7 × 10`) | 70 |
| **Total MSRR** | **2,446** |

RAW is `92 + 700 = 792` dimensions. Identifiers, roles, labels, inventory flags, and fold metadata never enter either representation.

