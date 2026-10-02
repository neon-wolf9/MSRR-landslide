# 120B MSRR Cross-Backbone Validation

## Technical note

LightGBM was excluded because its native runtime crashed on the formal project
data even after a repaired-environment clone passed synthetic fitting.
No completed LightGBM scientific result was available or used.

It was replaced before the cross-backbone experiment by sklearn
HistGradientBoostingClassifier.

## Decision

**STRONG_MODEL_AGNOSTIC_SUPPORT**

Supported backbones: 4/4

ElasticNet-Logistic, MLP, HistGradientBoosting, XGBoost

## OOF results

```text
            backbone representation    AUROC    AUPRC  StrictPair     Edge
 ElasticNet-Logistic            RAW 0.637194 0.467295    0.508307 0.660107
 ElasticNet-Logistic           MSRR 0.842702 0.723678    0.693236 0.802611
                 MLP            RAW 0.637446 0.479556    0.536195 0.682555
                 MLP           MSRR 0.865090 0.770566    0.729628 0.830004
HistGradientBoosting            RAW 0.661406 0.476376    0.539953 0.691060
HistGradientBoosting           MSRR 0.875921 0.784826    0.744858 0.840585
             XGBoost            RAW 0.671284 0.489412    0.559731 0.709059
             XGBoost           MSRR 0.881482 0.791908    0.750198 0.846420
```

## MSRR - RAW deltas

```text
            backbone  delta_AUROC  delta_AUPRC  delta_StrictPair  delta_Edge  OOF_metric_wins_out_of_4 MSRR_SUPPORT  fold_wins_AUROC_out_of_5  fold_wins_AUPRC_out_of_5  fold_wins_StrictPair_out_of_5  fold_wins_Edge_out_of_5
 ElasticNet-Logistic     0.205508     0.256383          0.184929    0.142504                         4          YES                         5                         5                              5                        5
                 MLP     0.227644     0.291010          0.193434    0.147449                         4          YES                         5                         5                              5                        5
HistGradientBoosting     0.214515     0.308450          0.204905    0.149525                         4          YES                         5                         5                              5                        5
             XGBoost     0.210198     0.302496          0.190467    0.137362                         4          YES                         5                         5                              5                        5
```

## Interpretation discipline

A backbone supports MSRR only if the frozen MSRR representation beats the
frozen RAW representation on at least 3 of the 4 OOF metrics.

- 4/4: strong model-agnostic support within this benchmark.
- >=3/4 including a non-tree learner: cross-family support.
- tree-only: narrow the manuscript claim to tree ensembles.
- otherwise: do not claim cross-backbone generality.

## Next step

PAIRED_BOOTSTRAP_STATISTICAL_ANALYSIS
