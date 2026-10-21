# User guide

## Choose a validation level

Run commands from the repository root in the environment described in `README.md`. `examples/minimal_msrr_example.py` accepts no command-line options. It reads `data/example/synthetic_msrr_demo.npz`, calls the canonical 120B transform, and prints the input/output shapes and MSRR, StrictPair, and Edge checks; success ends with `STATUS: PASS` and exit code 0. It does not write a scientific result file.

### Redistribution-safe validation

Use this level to inspect the code package without third-party data:

```bash
python examples/minimal_msrr_example.py
python scripts/00_check_environment.py
python scripts/01_verify_inputs.py
python scripts/15_verify_reference_results.py
python -m pytest -q -p no:cacheprovider
```

`00_check_environment.py` prints the Python version and import status for the pinned dependencies. `01_verify_inputs.py` checks public metadata and lists seven derived inputs that are separately available from the Zenodo data deposit; its `--full` option makes missing items among those seven a nonzero exit, but is **not** a complete full-workflow dependency gate. `15_verify_reference_results.py` checks the frozen reference tables, arrays, and `FILE_MANIFEST.csv` hashes and prints the number of checked files. The test suite checks representation dimensions, matching and fold integrity, metrics, and reference outputs. These checks read the packaged inputs and references; they do not train models.

### Full scientific rerun

The eleven frozen derived study files and historical prediction hand-offs are publicly available through the Zenodo dataset:

https://zenodo.org/doi/10.5281/zenodo.23533084

Users should extract the archive at the repository root and verify the provided SHA-256 checksums before attempting scientific workflows.

A complete end-to-end scientific rerun has not been independently demonstrated during release preparation.

Some workflows still require separately acquired source products or additional upstream intermediates.

The `--full` input checker covers seven model-ready paths and does not certify all workflow-specific prerequisites. The intended order is recorded in `docs/REPRODUCIBILITY.md` and `QUICKSTART.md`.

The required Hiroshima and Kyushu input filenames are listed by `scripts/01_verify_inputs.py --full`; the source-product paths and versions are in `docs/DATA_SOURCES.md` and `docs/DATA_LAYOUT.md`. The matching scripts read event evidence, eligible candidates, feature tables, and frozen rules in `configs/matching/`; the principal 120B evaluation reads the reconstructed matched tensors and frozen folds. The corresponding outputs are benchmark memberships, fold records, integrity audits, score arrays, metric tables, and paired-bootstrap summaries. `RESULT_MANIFEST.csv` maps each workflow to its frozen reference output. Full workflows have script-specific input gates; where a script exposes command-line options, inspect its argument parser before execution. A missing or version-mismatched input must stop the workflow and be investigated.

## Interpret the task correctly

- Positive: inventory-supported/documented landslide occurrence.
- Control: event-matched non-inventory hard control, not a verified negative or confirmed absence.
- Prediction: matched-context landslide discrimination score, not a calibrated probability or operational risk score.
- Unit of matching and bootstrap: the complete three-member matched set.

## Read results

`RESULT_MANIFEST.csv` maps scientific questions to authority scripts and reference files. `docs/PAPER_RESULT_MAP.md` identifies the files supporting each manuscript result. Principal, diagnostic, and sensitivity executions are deliberately separated; do not substitute one category for another.

## Troubleshooting

- Missing source data: follow `docs/DATA_ACQUISITION.md`; do not fabricate a substitute.
- Geospatial installation failure: prefer the supplied Conda environment and compatible platform wheels.
- Reference verification failure: check whether any distributed file was edited after `FILE_MANIFEST.csv` was generated.
- Reconstructed count or hash mismatch: stop and audit source versions, cutoffs, CRS, feature order, and matching configuration before training.
