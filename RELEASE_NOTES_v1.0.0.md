# v1.0.0 — Initial public research release

This release contains the code and reproducibility materials corresponding to the manuscript *Rainfall-induced landslide discrimination using multi-source Earth observation data and matched-context representation*.

Included:

- event-matched hard-control benchmark construction and exact replay workflows;
- the canonical 792-D to 2,446-D MSRR implementation;
- RAW-versus-MSRR evaluation across four learner families;
- complete-set bootstrap, ablation, placebo, PairDiff, ranking, and published-comparator workflows;
- conservative pre-trigger and common-DEM sensitivity workflows;
- independent Kyushu evaluation, domain-shift analysis, and spatial summaries;
- frozen configurations, manifests, hashes, numerical reference outputs, focused tests, and a synthetic demonstration.

Third-party EO/GIS source data and source inventories are not included. Full source-data-to-results reproduction requires users to obtain the documented products independently and comply with provider terms. The redistribution-safe validation suite verifies package integrity, reference materials, and canonical MSRR behavior; it does not claim to replace a full scientific rerun.
