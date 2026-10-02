# Experiment 128B preregistration

This preregistration was written before any Experiment 128B scientific model result was produced. The cutoffs were supplied from independently documented event evidence and will not be changed after results are observed.

- Hiroshima cutoff: 2018-07-06 18:30 JST / 09:30 UTC.
- Kyushu cutoff: 2017-07-05 16:00 JST / 07:00 UTC.
- Selection rule: last complete 30-minute timestamp strictly before the documented event-specific landslide/debris-flow occurrence time.
- Window length: 70 consecutive half-hour observations. Start is calculated as `end - 69 × 30 minutes`.
- Dynamic definitions, MSRR representation, learner family, D4/D6/D8 grid, metrics, validation selection, and bootstrap protocol remain frozen.
- Arm A retains the original matched benchmarks and changes only model-facing rainfall timing.
- Arm B runs if and only if Phase A finds that any formal matching rainfall variable depends on the shifted interval. It changes only rainfall-dependent quantities under the Experiment 127 matching protocol.
- No post-result rescue or cutoff search is allowed.

These cutoffs are conservative documented-event cutoffs. They are not asserted to be true individual failure cutoffs or the earliest occurrence times of all landslides.
