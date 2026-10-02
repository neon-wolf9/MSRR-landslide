# Validation Report

## Public-candidate revalidation — 2026-10-02

After copying the validated package into `MSRR_landslide_CG_GitHub_release`, release-layer documentation and metadata were added and the redistribution-safe suite was rerun in the recorded Python 3.10.20 environment:

- all 73 Python files compiled from source: **PASS**;
- isolated dependency imports and recorded versions: **PASS**;
- synthetic minimal MSRR example, `(8, 3, 792)` to `(8, 3, 2446)`: **PASS**;
- package-input boundary check: **PASS**, with seven non-redistributed full-rerun inputs correctly reported absent;
- reference results and regenerated SHA-256 manifest: **PASS**;
- full focused test suite: **10 passed**;
- README command-path audit: **25 referenced Python entry points, 0 missing**;
- credential/private-path/raw-geodata scan: **PASS**.

The environment checker now imports compiled dependencies in isolated subprocesses. This is a release-tooling fix for a Windows GDAL DLL load-order crash caused by loading every optional geospatial backend into one diagnostic process; it does not change any scientific execution. The spatial authority script also loads Fiona before GeoPandas when Fiona is available, addressing the same Windows load-order issue without changing data or algorithm logic.

Validation date: 2026-10-01  
Package validated at the time of the recorded clean-environment run: `MSRR_IJAEOG_Reproducibility_Package`  

The current public-candidate directory is `MSRR_landslide_CG_GitHub_release`. Scientific code and frozen results were copied unchanged; release-layer documentation and metadata were added afterward and rechecked with the redistribution-safe suite.
Platform: Windows, x86-64  
Clean environment: Python 3.10.20, created in a new temporary Conda prefix from the distributed `environment.yml`

## Overall result

**PASS for redistribution-safe validation.** The clean environment was created successfully, all declared imports loaded, the synthetic MSRR demonstration passed, all ten tests passed, the file manifest and frozen reference outputs passed verification, and the distributed frozen exact-replay evidence confirms 5,056 Hiroshima and 1,692 Kyushu matched sets.

A source-data-to-results rerun was **SKIPPED BY DESIGN** because restricted/non-redistributed third-party and derived scientific inputs are not bundled. This is not represented as a successful full rerun. The missing inputs and their expected local paths are reported by `scripts/01_verify_inputs.py` and documented in the data-availability files.

## Clean-environment creation

The environment was created from `environment.yml`, which specifies Python 3.10.20 and installs the exact public package pins from `requirements.txt`. A regional Conda channel mirror was used only as the transport endpoint for creating the isolated environment because direct access to the default conda-forge endpoint was unavailable on the validation host.

Environment-creation command (the machine-specific temporary prefix is intentionally abstracted):

```powershell
$env:CONDA_CHANNEL_ALIAS = 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud'
conda env create --solver libmamba -p <temporary-clean-prefix> -f environment.yml
```

An initial attempt to express every dependency as an exact Conda package was not solvable for the required Windows versions and channels (including the recorded PyTorch/SciPy versions and geospatial constraints). The final, distributed environment specification therefore uses Conda for the Python interpreter and pip for the actual recorded dependency versions. This matches the provenance of the source environment and successfully created a clean environment.

The clean import check reported:

| Package | Observed version |
|---|---:|
| Python | 3.10.20 |
| NumPy | 1.24.0 |
| pandas | 2.0.0 |
| SciPy | 1.15.3 |
| scikit-learn | 1.5.2 |
| PyTorch | 2.1.0+cpu |
| XGBoost | 3.2.0 |
| PyArrow | 16.1.0 |
| GeoPandas | 1.1.3 |
| Matplotlib | 3.10.8 |
| PyYAML | 6.0 |
| Shapely | 2.1.2 |
| Pyogrio | 0.12.1 |
| Rasterio | 1.4.4 |
| Fiona | 1.10.1 |
| Affine | 2.4.0 |
| PyProj | 3.7.1 |
| imbalanced-learn | 0.12.4 |
| joblib | 1.6.0 |
| Pillow | 12.1.1 |
| pytest | 7.4.0 |

Result: `PASS environment imports complete`.

## Synthetic MSRR demo

Command:

```text
python scripts/demo_msrr.py
```

Observed result:

```text
Input shape: (8, 3, 792)
MSRR output shape: (8, 3, 2446)
MSRR numerical verification: PASS
StrictPair verification: PASS (0.500000)
Edge verification: PASS (0.687500)
STATUS: PASS
```

The demo imports and calls the canonical MSRR implementation. Its input is deterministic synthetic data; it is not a substitute for either scientific benchmark.

## Test suite

Command:

```text
python -m pytest -q -p no:cacheprovider
```

Final result: **10 passed in 9.33 s**.

The tests cover frozen MSRR dimensions, algebraic construction and temporal summaries, matched-set permutation symmetry, absence of label/role inputs, canonical metrics, exactly two controls per positive, no control reuse/capacity one, legal-edge matching, fold integrity, and frozen reference/manifest verification.

The permutation-symmetry assertion uses `atol=3e-6` for float32 reduction-order effects; the observed maximum difference during test development was approximately `1.9e-6`. This tolerance does not change the MSRR definition or any reference result.

## Package and frozen-result verification

Commands:

```text
python scripts/01_verify_inputs.py
python scripts/15_verify_reference_results.py
```

The package metadata check passed. The input check correctly identified seven non-redistributed full-rerun inputs as absent. The reference verifier confirmed the RAW 792-D and MSRR 2446-D definitions, seven ablation variants, four learner families, RAW/MSRR panels, placebo variants, Kyushu primary panel, valid metric ranges, expected Hiroshima out-of-fold array shapes, and every distributed file recorded in `FILE_MANIFEST.csv`.

## Exact benchmark reconstruction evidence

`reference_results/hiroshima/EXACT_REPLAY_COMPARISON.csv` was parsed in the clean environment and checked against the publication identities:

| Benchmark | Matched sets | Positives | Controls | Frozen replay status |
|---|---:|---:|---:|---|
| Hiroshima 2018 | 5,056 | 5,056 | 10,112 | PASS |
| Kyushu 2017 | 1,692 | 1,692 | 3,384 | PASS |

For both events the distributed evidence records exact positive/control identities, canonical set membership, ordered control rank, zero duplicate controls, zero partial positives, exact legal-edge counts, exact exclusions, zero cost difference, exact fold assignments, and identical independent replay runs.

Because the restricted source and derived tensors are absent from the public package, the validation performed here verifies the frozen replay evidence and its integrity; it does not rerun benchmark reconstruction from third-party raw data.

## Scientific immutability statement

No matching rule, matching weight, capacity constraint, fold definition, label, MSRR definition, model hyperparameter, manuscript reference metric, Kyushu protocol, or scientific conclusion was changed during this repository-preparation validation.

The manuscript's principal cross-backbone XGBoost result remains the frozen value (RAW AUROC approximately 0.6713; MSRR AUROC approximately 0.8815). Historical/diagnostic values near 0.8796 remain explicitly separated and must not replace the principal result.
