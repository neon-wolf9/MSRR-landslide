# Quick start

All commands below are run from the package root.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate msrr-landslide
```

The frozen local reference environment used Python 3.10.20 with the versions recorded in `environment.yml`. Exact CPU/GPU runtime equivalence is not required for the redistribution-safe checks, but a full model rerun should use the pinned versions where possible.

## 2. Verify the package

```bash
python examples/minimal_msrr_example.py
python scripts/00_check_environment.py
python scripts/01_verify_inputs.py
python scripts/15_verify_reference_results.py
pytest -q -p no:cacheprovider
```

The synthetic demo is the fastest canonical check and does not require any third-party data.

## 3. Full rerun after restoring restricted inputs

The expected root is the package root itself. Place the frozen Hiroshima inputs below `data/` and the Kyushu inputs below `external/kyushu_2017_asakura_toho/`, using `data/manifests/`, `configs/external/`, and `docs/DATA_DICTIONARY.md` as the contracts.

Recommended formal order:

```bash
python scripts/127_AUDIT_MATCHING_PROTOCOL_REPRODUCIBILITY.py
python scripts/130_AUDIT_ORIGINAL_BENCHMARK_EXACT_REPLAY.py
python scripts/119_RUN_MSRR_REPRESENTATION_ABLATION_5FOLD_ONE_SHOT.py
python scripts/120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py
python scripts/121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py
python scripts/123_RUN_MATCHED_CONTEXT_PLACEBO_XGB.py
python scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py
python scripts/125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py
python scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py
python scripts/126C_AUDIT_PAIRDIFF_CONVERGENCE_SENSITIVITY.py
python scripts/131_RUN_PAIRDIFF_XGB_FAIRNESS.py
python scripts/128B1_RUN_PUBLISHED_ALGORITHM_COMPARISON_5FOLD.py
python scripts/128C_RUN_PUBLISHED_METHOD_PAIRED_BOOTSTRAP.py
python scripts/128B_RUN_CONSERVATIVE_PRETRIGGER_SENSITIVITY.py
python scripts/128C_RUN_SCALABLE_EXACT_PRETRIGGER_REMATCHING.py
python scripts/129_RUN_COMMON_DEM_SENSITIVITY.py
python scripts/129B_RUN_FULL_COMMONDEM_EXACT_REMATCHING.py
python scripts/126_RUN_HIROSHIMA_KYUSHU_DOMAIN_SHIFT_CHARACTERIZATION.py
python scripts/127B_RUN_MSRR_SPATIAL_MATCHED_MARGIN_ANALYSIS_AUTO_GEOMETRY.py
```

Some scripts are intentionally long-running and gate on exact frozen inputs. A missing restricted asset is an expected, informative stop—not permission to substitute another dataset.
