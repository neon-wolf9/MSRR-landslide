# Frozen 120B hand-off; no experiment run in this candidate

These 12 files were copied byte-for-byte from the original executed
`experiments/MSRR_CROSS_BACKBONE_5FOLD_V1B` directory on 2026-10-21.
`FILE_MANIFEST.csv` records their candidate SHA-256 values. The copies are
historical 120B outputs, **not** new scores or evidence that the candidate
re-ran 120B.

The exact relative directory is necessary because the unchanged 121, 123,
124D and 125 scripts refer to this path rather than to the curated
`reference_results/hiroshima` and `configs/learners` copies. The eight
`*_OOF_margin.npy` arrays contain author-produced scores in the frozen
15,168-row sample order; `OOF_RESULTS.csv` and `FOLD_METRICS.csv` hold
reported metrics; `VALIDATION_SELECTION.csv` holds Hiroshima-only learner
selection records. `RUN_MANIFEST.json` documents the original execution.

The portable candidate's 120B script differs from the original frozen
`source_sha256` in `RUN_MANIFEST.json` only by project-root and docstring
path adaptation already present in the public code. This hand-off does not
create or restore the underlying model-ready event inputs, fold-fitted
preprocessing data, or Kyushu predictions.

Do not run 121, 123, 124D or 125 solely because these files are present.
Their other real-data and intermediate dependencies are listed in
`WORKFLOW_HANDOFF_AUDIT.md`.
