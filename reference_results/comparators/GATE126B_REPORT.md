# Experiment 126B: task-aligned baselines

PASS_126B_TASK_ALIGNED_BASELINES_COMPLETE

All five methods were refitted in the same current runtime with n_jobs=1. The prospective amendment and gate were written before scientific fitting.

| method             |    AUROC |    AUPRC |   StrictPair |     Edge | Objective           | Representation   |
|:-------------------|---------:|---------:|-------------:|---------:|:--------------------|:-----------------|
| RAW_POINTWISE_XGB  | 0.676923 | 0.491093 |     0.562698 | 0.710146 | pointwise           | RAW              |
| RAW_PAIRDIFF_LOGIT | 0.571645 | 0.411887 |     0.624209 | 0.756922 | pairwise-difference | RAW difference   |
| RAW_RANKPAIR_XGB   | 0.680435 | 0.499507 |     0.617880 | 0.758505 | rank:pairwise       | RAW              |
| MSRR_POINTWISE_XGB | 0.879555 | 0.789444 |     0.749604 | 0.845036 | pointwise           | MSRR             |
| MSRR_RANKPAIR_XGB  | 0.867264 | 0.756376 |     0.753956 | 0.847409 | rank:pairwise       | MSRR             |

| contrast   | description                           |   delta_AUROC |   delta_AUPRC |   delta_StrictPair |   delta_Edge |
|:-----------|:--------------------------------------|--------------:|--------------:|-------------------:|-------------:|
| C1         | Objective effect under RAW            |      0.003512 |      0.008413 |           0.055182 |     0.048358 |
| C2         | Representation effect under pointwise |      0.202631 |      0.298351 |           0.186907 |     0.134889 |
| C3         | Representation effect under pairwise  |      0.186829 |      0.256869 |           0.136076 |     0.088904 |
| C4         | Objective effect under MSRR           |     -0.012291 |     -0.033069 |           0.004351 |     0.002373 |
| C5         | MSRR-pointwise vs RAW-rankpair        |      0.199120 |      0.289938 |           0.131725 |     0.086531 |
| C6         | MSRR-pointwise vs PairDiff            |      0.307909 |      0.377558 |           0.125396 |     0.088113 |

| contrast   | description                           | metric     |   observed_delta |   bootstrap_mean |   CI95_LOW |   CI95_HIGH |   fraction_delta_gt_0 |     B |     seed |
|:-----------|:--------------------------------------|:-----------|-----------------:|-----------------:|-----------:|------------:|----------------------:|------:|---------:|
| C1         | Objective effect under RAW            | AUROC      |         0.003512 |         0.003530 |  -0.000616 |    0.007701 |              0.951900 | 10000 | 20260829 |
| C1         | Objective effect under RAW            | AUPRC      |         0.008413 |         0.008506 |   0.002394 |    0.014576 |              0.997000 | 10000 | 20260829 |
| C1         | Objective effect under RAW            | StrictPair |         0.055182 |         0.055199 |   0.043315 |    0.067247 |              1.000000 | 10000 | 20260829 |
| C1         | Objective effect under RAW            | Edge       |         0.048358 |         0.048368 |   0.040348 |    0.056665 |              1.000000 | 10000 | 20260829 |
| C2         | Representation effect under pointwise | AUROC      |         0.202631 |         0.202592 |   0.195285 |    0.209889 |              1.000000 | 10000 | 20260829 |
| C2         | Representation effect under pointwise | AUPRC      |         0.298351 |         0.298024 |   0.286177 |    0.309633 |              1.000000 | 10000 | 20260829 |
| C2         | Representation effect under pointwise | StrictPair |         0.186907 |         0.186870 |   0.171474 |    0.202729 |              1.000000 | 10000 | 20260829 |
| C2         | Representation effect under pointwise | Edge       |         0.134889 |         0.134830 |   0.123912 |    0.145767 |              1.000000 | 10000 | 20260829 |
| C3         | Representation effect under pairwise  | AUROC      |         0.186829 |         0.186760 |   0.180270 |    0.193177 |              1.000000 | 10000 | 20260829 |
| C3         | Representation effect under pairwise  | AUPRC      |         0.256869 |         0.256466 |   0.245225 |    0.267680 |              1.000000 | 10000 | 20260829 |
| C3         | Representation effect under pairwise  | StrictPair |         0.136076 |         0.136061 |   0.121440 |    0.151108 |              1.000000 | 10000 | 20260829 |
| C3         | Representation effect under pairwise  | Edge       |         0.088904 |         0.088870 |   0.079015 |    0.098794 |              1.000000 | 10000 | 20260829 |
| C4         | Objective effect under MSRR           | AUROC      |        -0.012291 |        -0.012302 |  -0.014630 |   -0.010017 |              0.000000 | 10000 | 20260829 |
| C4         | Objective effect under MSRR           | AUPRC      |        -0.033069 |        -0.033052 |  -0.038428 |   -0.027738 |              0.000000 | 10000 | 20260829 |
| C4         | Objective effect under MSRR           | StrictPair |         0.004351 |         0.004390 |  -0.003956 |    0.012658 |              0.848400 | 10000 | 20260829 |
| C4         | Objective effect under MSRR           | Edge       |         0.002373 |         0.002408 |  -0.002472 |    0.007219 |              0.834700 | 10000 | 20260829 |
| C5         | MSRR-pointwise vs RAW-rankpair        | AUROC      |         0.199120 |         0.199063 |   0.192635 |    0.205375 |              1.000000 | 10000 | 20260829 |
| C5         | MSRR-pointwise vs RAW-rankpair        | AUPRC      |         0.289938 |         0.289518 |   0.278682 |    0.300000 |              1.000000 | 10000 | 20260829 |
| C5         | MSRR-pointwise vs RAW-rankpair        | StrictPair |         0.131725 |         0.131671 |   0.117286 |    0.146366 |              1.000000 | 10000 | 20260829 |
| C5         | MSRR-pointwise vs RAW-rankpair        | Edge       |         0.086531 |         0.086462 |   0.076938 |    0.096123 |              1.000000 | 10000 | 20260829 |
| C6         | MSRR-pointwise vs PairDiff            | AUROC      |         0.307909 |         0.307884 |   0.302716 |    0.313037 |              1.000000 | 10000 | 20260829 |
| C6         | MSRR-pointwise vs PairDiff            | AUPRC      |         0.377558 |         0.377172 |   0.367776 |    0.386286 |              1.000000 | 10000 | 20260829 |
| C6         | MSRR-pointwise vs PairDiff            | StrictPair |         0.125396 |         0.125520 |   0.111941 |    0.139241 |              1.000000 | 10000 | 20260829 |
| C6         | MSRR-pointwise vs PairDiff            | Edge       |         0.088113 |         0.088183 |   0.079015 |    0.097310 |              1.000000 | 10000 | 20260829 |

Pre-registered gate: **STRONG_REPRESENTATION_BEYOND_OBJECTIVE**. C3/C5/C6 wins: 4/4/4 of 4.

Historical reference (descriptive only; no pass/fail use):

| method             | metric     |   historical_120B_metric |   current_126B_metric |   difference | used_for_pass_fail   |
|:-------------------|:-----------|-------------------------:|----------------------:|-------------:|:---------------------|
| RAW_POINTWISE_XGB  | AUROC      |                 0.671284 |              0.676923 |     0.005639 | False                |
| RAW_POINTWISE_XGB  | AUPRC      |                 0.489412 |              0.491093 |     0.001681 | False                |
| RAW_POINTWISE_XGB  | StrictPair |                 0.559731 |              0.562698 |     0.002967 | False                |
| RAW_POINTWISE_XGB  | Edge       |                 0.709059 |              0.710146 |     0.001088 | False                |
| MSRR_POINTWISE_XGB | AUROC      |                 0.881482 |              0.879555 |    -0.001927 | False                |
| MSRR_POINTWISE_XGB | AUPRC      |                 0.791908 |              0.789444 |    -0.002464 | False                |
| MSRR_POINTWISE_XGB | StrictPair |                 0.750198 |              0.749604 |    -0.000593 | False                |
| MSRR_POINTWISE_XGB | Edge       |                 0.846420 |              0.845036 |    -0.001384 | False                |

Scores are discrimination/ranking scores, not landslide probabilities. StrictPair and Edge use strict positive margins; no threshold tuning or calibration. Bootstrap uses 10,000 identical complete-matched-set draws across methods, conditional on fitted OOF models. Folds, metrics and draws are not independent scientific replications. No unfavorable results were removed; no method or gate changed after results.
