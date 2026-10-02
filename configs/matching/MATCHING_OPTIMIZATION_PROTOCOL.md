# Optimization and assignment

The legal edge graph is partitioned by geology compatibility, exact matching-missingness pattern, and soil-completeness block. Common support maximizes the number of positives that can receive an all-or-none allocation of two controls under control capacity one. For the selected positives, SciPy sparse minimum-weight full bipartite matching minimizes total composite distance. Each positive is expanded to two demand rows; each candidate is a single capacity column.

Inputs and candidate IDs are sorted deterministically. Machine-epsilon candidate-index perturbations make equal-cost choices deterministic. Final control ranks are assigned after stable mergesort by `(positive_unit_id, composite_distance, candidate_unit_id)`. No replacement or automatic caliper relaxation is allowed. The current Hiroshima production membership also includes the documented HC-BUG-03 minimal augmenting-path repair of 24 control slots.
