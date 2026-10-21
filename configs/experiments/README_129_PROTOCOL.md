# Experiment 129 — Common DEM Source Sensitivity

This directory is reserved for the preregistered IJAEOG Reviewer #2 Comment 7 sensitivity analysis.

- Common DEM: Copernicus DEM GLO-30, `COP-DEM_GLO-30-DGED/2024_1`.
- Arm A: frozen Hiroshima matching membership and folds; common-DEM terrain replaces model-facing terrain only.
- Arm B: common-DEM terrain is used in frozen common support, exact 1:2 rematching, fold reconstruction, and model evaluation for both events.
- The script must stop if Hiroshima tile coverage is incomplete or the Kyushu terrain replay fails.
- No alternate DEM, download, nearest-k pruning, greedy matching, control reuse, partial-positive allocation, Kyushu tuning, or result-based protocol change is allowed.
- Cross-event change is described as a source-sensitive component plus a residual cross-event difference, not as an exact causal decomposition.

Entry point: `scripts/129_RUN_COMMON_DEM_SENSITIVITY.py`.

The script writes its own `RUN.log` and contract-bound checkpoints. A checkpoint is reusable only when the source product, frozen-input hashes, temporal option, protocol contract, and script hash all agree.
