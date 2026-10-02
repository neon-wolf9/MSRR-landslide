# Reviewer reproducibility certificate

Decision: `PASS_FULL_END_TO_END_BENCHMARK_REPLAY`

Experiment 130 ran two clean, independent process replays per event. Final benchmark membership files were excluded from replay inputs and opened only after child completion for canonical comparison. Hiroshima reproduced all 5,056 repaired sets and 10,112 unique control slots; Kyushu regenerated its legal graph and reproduced all 1,692 triplets. Membership, costs, normalized hashes and Hiroshima fold mapping passed the exact gates. Protected historical inputs remained byte-identical.
