# Algorithm 1. Event-matched non-inventory hard-control construction

**Input:** frozen event positives, frozen non-inventory candidate controls, pre-event/static covariates, event-anchored antecedent rainfall, group weights and hard calipers.  
**Output:** triplets `(positive, control 1, control 2)`.

1. Remove candidates failing the event-specific evidence and coverage contract; never relabel an unlabeled/non-hit grid as a confirmed negative.
2. Fit each frozen transform on the event's joint positive-candidate population using median and IQR; do not impute missing values.
3. Form geology-compatibility, matching-missingness, and soil-completeness blocks.
4. Create a legal edge only if all raw, spatial and rainfall calipers pass.
5. Compute group mean absolute robust-scaled differences and the frozen weighted composite cost; renormalize by 0.85 for the all-soil-missing block.
6. Within blocks, select the maximum all-or-none set of positives that can each receive two controls with candidate capacity one.
7. For selected positives, solve sparse minimum-weight full bipartite matching; break ties deterministically by sorted IDs and the frozen epsilon rule.
8. Rank the two controls by `(composite distance, candidate unit_id)`. Preserve unmatched positive labels and omit them only from the matched benchmark.
