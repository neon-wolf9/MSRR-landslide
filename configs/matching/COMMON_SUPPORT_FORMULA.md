# Common-support definition

Common support was not a marginal min-max or quantile-overlap filter. It was a graph-feasibility definition. Let E contain positive-control edges that pass the frozen block and hard-caliper rules. Each positive has demand 2 and each control capacity 1. Within each `(geology_parent, matching_missingness_pattern_id, soil_completeness_block)` block, the implementation selected the maximum all-or-none subset of positives for which a feasible flow of two controls per selected positive exists.

Formally, maximize `sum_i z_i` subject to `sum_j x_ij = 2 z_i`, `sum_i x_ij <= 1`, `x_ij = 0` when `(i,j) not in E`, and binary `x_ij,z_i`. Hiroshima retained 5,056/5,075; Kyushu retained 1,692/1,694. Excluded positives retained their positive labels and were omitted only from the matched estimand.
