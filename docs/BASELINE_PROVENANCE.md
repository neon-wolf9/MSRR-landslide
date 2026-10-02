# Published-comparator provenance

The comparator panel uses three repository-authored, benchmark-aligned adapters under the same spatial folds and model-visible RAW features as the principal analysis. The adapters are in `scripts/msrr128_baselines/`; experiments 128B0, 128B1, and 128C are the executed authority chain.

| Comparator | Implementation status | Source basis | Distribution status |
|---|---|---|---|
| PU-BaggingDT | Paper-faithful reimplementation | Published algorithm; fixed parameters recorded by experiment 128B0 | Repository-authored adapter included under MIT |
| Spy-PU+BRF | Paper-informed benchmark adaptation | Fu et al. (2024), IEEE JSTARS; formal choices frozen in `spy_pu_brf_formal_config.json` | Repository-authored adapter included under MIT |
| PU-pullbaggingDT | Official-code-informed benchmark adaptation | `ShubingOuyangcug/PU-pullbaggingDT`, commit `a8857dad30a454e3155644b18ddd2600df860e70` | Repository-authored adapter included; upstream repository not distributed because no explicit license was present in the inspected copy |

The PU-pullbaggingDT adapter does not copy upstream source code. It retains the released network widths and method structure while adapting the input layer to the frozen 792-dimensional benchmark and enforcing fold-local preprocessing. For the original experiment's provenance guard, clone the upstream repository manually into `third_party/PU-pullbaggingDT`, checkout the exact commit above, and leave the checkout unmodified. Obtain permission or rely on rights applicable to your use; this repository grants no license to the upstream code.

The fixed comparator metrics are distributed in `reference_results/comparators/`. They are reference outputs from the frozen executions, not claims that the three adapters are drop-in reproductions of every detail of their source papers.
