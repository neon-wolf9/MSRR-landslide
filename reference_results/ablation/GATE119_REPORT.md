# MSRR Representation Ablation — Full 5-Fold

## Decision
**STRONG_MSRR_ABLATION_SUPPORT**

## OOF results
               variant  feature_dim    AUROC    AUPRC  StrictPair     Edge
                A0_RAW          792 0.676075 0.491203    0.561116 0.709850
    A1_RAW_PLUS_SIGNED         1584 0.882542 0.792618    0.745055 0.843058
    A2_RAW_PLUS_ABSDEV         1584 0.820745 0.703875    0.723695 0.823774
A3_FULL_REL_NO_SUMMARY         2376 0.881694 0.791370    0.744660 0.842860
    A4_STATIC_REL_ONLY          976 0.845645 0.733764    0.720926 0.826642
      A5_RAIN_REL_ONLY         2262 0.791038 0.670419    0.627769 0.759494
          A6_FULL_MSRR         2446 0.881482 0.791908    0.750198 0.846420

## Gates
CORE_MSRR_SUPPORTED=True
SIGNED_SIGNAL_SUPPORTED=True
ABSDEV_SIGNAL_SUPPORTED=True
STATIC_REL_SUPPORTED=True
RAIN_REL_SUPPORTED=True
DUAL_SOURCE_GAIN_SUPPORTED=True
TEMPORAL_SUMMARY_GAIN_SUPPORTED=True
STRONG_PAPER_LEVEL_ABLATION_SUPPORT=True

## Discipline
All seven variants are retained. Individual component claims follow their own
gates; a successful full MSRR does not automatically prove every component.
No post-test rescue of the representation definition is allowed.
