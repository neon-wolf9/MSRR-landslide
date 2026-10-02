# 126 Hiroshima–Kyushu Domain-Shift Characterization

## Status

**PASS_126_DOMAIN_SHIFT_CHARACTERIZATION_COMPLETE**

This is a post-hoc descriptive analysis. No predictive model was trained,
selected, tuned, refit, or rescued.

## Absolute performance context

Negative external-minus-internal metric cells:
**32/32**

## Static model-input shift

- features: 91
- median Wasserstein distance: 0.250759
- 90th percentile Wasserstein distance: 1.073426
- maximum Wasserstein distance: 14.275681
- categorical Jensen-Shannon distance: 0.294229

## Rainfall-process shift

- summary-field cells: 70
- median Wasserstein distance: 0.751421
- 90th percentile Wasserstein distance: 2.525585

## Matched-context geometry shift

- static MAD median Wasserstein: 0.023730
- rainfall MAD median Wasserstein: 0.029729

## Relative-signal preservation

### Static
- Pearson r: 0.061064
- Spearman rho: 0.138334
- cosine similarity: 0.065471
- sign agreement: 0.563380

### Rainfall temporal-mean
- Pearson r: nan
- Spearman rho: nan
- cosine similarity: nan
- sign agreement: nan

### Combined
- Pearson r: 0.061512
- Spearman rho: 0.144349
- cosine similarity: 0.065471
- sign agreement: 0.563380

## Interpretation guardrail

The analysis may support statements about observed covariate/process shift and
preservation or alteration of inventory-positive versus matched-control
relative contrasts. It must not be presented as proof of causal mechanisms or
as confirmation that controls are true geological negatives.
