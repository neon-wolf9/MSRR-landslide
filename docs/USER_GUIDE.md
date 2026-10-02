# User guide

## Choose a validation level

### Redistribution-safe validation

Use this level to inspect the code package without third-party data:

```bash
python examples/minimal_msrr_example.py
python scripts/00_check_environment.py
python scripts/01_verify_inputs.py
python scripts/15_verify_reference_results.py
python -m pytest -q -p no:cacheprovider
```

### Full scientific rerun

Use this level only after acquiring and validating the excluded inputs. Run `python scripts/01_verify_inputs.py --full` before a scientific workflow. The recommended order is recorded in `docs/REPRODUCIBILITY.md` and `QUICKSTART.md`.

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
