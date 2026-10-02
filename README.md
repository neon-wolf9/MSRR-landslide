# Matched-Set Relative Representation (MSRR) for rainfall-induced landslide discrimination

Code and reproducibility materials for the manuscript *Rainfall-induced landslide discrimination using multi-source Earth observation data and matched-context representation*, prepared for submission to *Computers & Geosciences*.

## Overview

Landslide inventories provide stronger evidence for mapped occurrences than for absences. This repository therefore treats mapped positives as inventory-supported landslide occurrences and constructs event-matched non-inventory hard controls; controls are not interpreted as verified absences. Each complete matched set contains one positive and two non-reused controls.

Strict matching reduces broad environmental separability and exposes a representation mismatch: the scientific task is relational within a matched set, whereas a conventional RAW vector represents each candidate independently. Matched-Set Relative Representation (MSRR) retains the 792-dimensional RAW state and appends signed deviations, absolute deviations, and rainfall-residual summaries relative to the permutation-symmetric three-member set mean, producing 2,446 features per candidate.

The development benchmark is the July 2018 Hiroshima event (5,056 complete matched sets). Independent external evaluation uses the July 2017 northern Kyushu event (1,692 complete matched sets) without target-event tuning. Model output is a matched-context landslide discrimination score, not an occurrence probability, susceptibility probability, calibrated risk, or operational warning score.

## Repository scope

The repository contains the real manuscript code and publication reference materials for:

- Hiroshima and Kyushu benchmark construction and deterministic replay;
- eligibility, legal-edge construction, weighted matching, capacity-one allocation, and all-or-none support checks;
- canonical RAW and MSRR transformations;
- four learner families: ElasticNet logistic regression, multilayer perceptron, histogram gradient boosting, and XGBoost;
- matched-set AUROC, AUPRC, StrictPair, and Edge evaluation;
- complete-set paired bootstrap inference;
- representation ablation, matched-context placebo, PairDiff, and ranking diagnostics;
- published PU comparator adapters and provenance records;
- pre-trigger rainfall and common-DEM sensitivity analyses;
- independent Kyushu evaluation, domain-shift analysis, and spatial summaries;
- frozen numerical outputs, hashes, manifests, and result-to-paper mappings.

This is a curated release of executed project code, not a clean-room reimplementation. Historical experiment numbers are retained because they are part of the frozen provenance.

## Repository structure

- `configs/`: frozen benchmark, matching, learner, experiment, and external-event contracts.
- `src/benchmark/`: benchmark-construction authority modules retained for traceability.
- `scripts/`: benchmark, experiment, external-evaluation, statistical, and verification drivers.
- `scripts/msrr128_baselines/`: repository-authored published-comparator adapters.
- `examples/`: lightweight entry point for the synthetic MSRR demonstration.
- `data/manifests/` and `data/schemas/`: provenance, hashes, field order, timestamps, and validation contracts.
- `data/example/`: redistribution-safe synthetic matched sets; no scientific observations.
- `reference_results/`: frozen manuscript, diagnostic, sensitivity, external, comparator, domain-shift, and spatial outputs.
- `tests/`: focused MSRR, matching, fold, metric, and reference-output checks.
- `docs/`: data acquisition, benchmark protocol, MSRR definition, reproduction order, and paper-result map.

Several earlier baseline/neural provenance files are retained but are not public entry points; see `docs/LEGACY_PROVENANCE.md`.

## Installation

The publication-validation environment used Python 3.10.20. Exact package versions are recorded in `requirements.txt` and `environment.yml`.

Conda:

```bash
conda env create -f environment.yml
conda activate msrr-landslide
```

Pip in an isolated Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

The dependency list reflects imports and the clean validation environment. Full model reruns can remain sensitive to CPU/GPU libraries and platform-specific geospatial wheels; use the recorded versions where available.

## Quick start

No Hiroshima or Kyushu source data are needed for the redistribution-safe checks:

```bash
python examples/minimal_msrr_example.py
python scripts/00_check_environment.py
python scripts/01_verify_inputs.py
python scripts/15_verify_reference_results.py
python -m pytest -q -p no:cacheprovider
```

The synthetic example imports the canonical manuscript implementation and verifies an `N x 3 x 792` input to `N x 3 x 2446` output. It is not a reproduction of the landslide experiments.

## Full reproduction

Full reproduction requires independently acquired third-party EO/GIS products and the frozen derived scientific inputs at the documented relative paths.

1. Read `docs/DATA_ACQUISITION.md`, `docs/DATA_LAYOUT.md`, and `docs/THIRD_PARTY_DATA.md`.
2. Restore or reconstruct the Hiroshima and Kyushu inputs using the manifests, schemas, and frozen source versions.
3. Verify the package and full input tree with `python scripts/01_verify_inputs.py --full`.
4. Audit matching and deterministic benchmark replay:

   ```bash
   python scripts/127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py
   python scripts/130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py
   ```

5. Run the principal RAW-versus-MSRR panel:

   ```bash
   python scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py
   ```

6. Run paired bootstrap inference:

   ```bash
   python scripts/121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py
   ```

7. Follow the ordered evidence chain in `docs/REPRODUCIBILITY.md` for ablation, placebo, comparators, sensitivity analyses, Kyushu evaluation, domain shift, and spatial analysis.

Missing restricted inputs are expected to stop full-rerun scripts. Do not substitute current products for frozen event-specific sources without a documented equivalence audit.

## Expected outputs

Depending on the workflow and available inputs, authority scripts produce benchmark memberships, fold records, metric tables, paired-bootstrap intervals, integrity audits, figures, and hashes. Frozen publication references are grouped under `reference_results/`; `RESULT_MANIFEST.csv` and `docs/PAPER_RESULT_MAP.md` identify their authority scripts and scientific roles.

The principal Hiroshima XGBoost execution is the frozen 120B result. The later diagnostic pointwise result is a separate execution and must not replace the principal manuscript result. Reference values are verified by `scripts/15_verify_reference_results.py`; they are not recomputed or rewritten by repository preparation.

## Data availability

Third-party EO/GIS source data and source inventories are not redistributed. Provider-controlled or authenticated products include AJG and GSI event evidence, GSI elevation, Copernicus DEM GLO-30, SoilGrids 2.0, Landsat 8 Collection 2 Level-2, CGLS-LC100 Collection 3, historical OpenStreetMap/Geofabrik extracts, J-FlwDir v1.4, GSJ Seamless Geological Map V2, and GPM IMERG Final Run V07B.

Product identifiers, known versions or event cutoffs, official access records already present in the project, expected relative paths, and preprocessing cautions are documented in `docs/DATA_ACQUISITION.md` and `docs/DATA_SOURCES.md`. Redistribution status of derived geospatial artifacts requires a separate rights review; such artifacts are excluded from this public candidate.

## Reproducibility

Deterministic components, seeds, frozen folds, benchmark replay identities, hashes, and reference outputs are preserved. The redistribution-safe suite verifies the canonical transformation, matched-set metrics, matching integrity, fold integrity, and reference files. Exact benchmark replay evidence confirms 5,056 Hiroshima and 1,692 Kyushu complete matched sets with no control reuse.

A source-data-to-results rerun is not claimed without the excluded third-party and derived inputs. See `VALIDATION_REPORT.md` and `docs/REPRODUCIBILITY.md` for the tested boundary and known limitations.

## Citation

Citation metadata are provided in `CITATION.cff`, including the verified public repository URL: https://github.com/neon-wolf9/MSRR-landslide. Add the immutable Zenodo DOI only after release archival; no DOI is fabricated in this repository.

## License

Repository-authored software and documentation are licensed under the MIT License; see `LICENSE`. The license does not cover third-party datasets or non-distributed upstream software identified in `THIRD_PARTY_NOTICES.md`.
