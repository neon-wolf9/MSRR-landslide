# Reproducibility protocol

## Frozen study units

- Hiroshima 2018: 5,056 complete matched sets, 15,168 rows.
- Kyushu 2017: 1,692 complete matched sets, 5,076 rows.
- Each set is ordered as one inventory-supported positive followed by two event-matched controls.
- Spatial five-fold splitting is performed at the matched-set/spatial-group level.

## Representation

RAW has 792 dimensions: 92 static predictors plus 70 half-hourly steps × 10 rainfall variables. Full MSRR has 2,446 dimensions: raw static, signed static deviation, absolute static deviation, raw rainfall sequence, signed rainfall deviation, absolute rainfall deviation, and 70 fixed rainfall-residual summaries. The executable authority is `scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py`; the seven-component audit is experiment 119.

## Training and evaluation

- Seed 7 and five spatial outer folds.
- Training-fold-only preprocessing.
- Validation-only configuration selection.
- No outer-test or Kyushu information is used for selection.
- Positive row weight 1.0; each control row weight 0.5 in the pointwise panel.
- Paired bootstrap uses the complete matched set as the resampling unit and applies identical draws to compared methods.

## Metrics

AUROC and AUPRC use row-level discrimination scores. For matched set *i*, let `m_i1` and `m_i2` be the positive-minus-control score differences and `M_i = min(m_i1, m_i2)`. Then:

- `StrictPair = mean(M_i > 0)`;
- `Edge = mean([m_i1 > 0, m_i2 > 0])` over all two-control comparisons.

## Independent event boundary

Kyushu labels, roles, and outcomes are not used for feature fitting, preprocessing, hyperparameter selection, threshold selection, or Hiroshima training. Experiment 124D freezes predictions before joining labels; experiment 125 extends the same boundary to the four-backbone panel.

## Verification

`scripts/15_verify_reference_results.py` checks dimensions, set counts, metric domains, expected method coverage, cross-file identities, and the SHA-256 manifest. The pytest suite repeats the critical checks in small isolated tests. Full numerical regeneration requires the restricted inputs described in `DATA_AVAILABILITY.md`.

