# Retained legacy provenance files

The release retains several pre-MSRR baseline and neural-model provenance files under `modeling/`, `models/`, `losses/`, and the earlier numbered training scripts. They document the project history but are not authority entry points for the manuscript's four-learner RAW-versus-MSRR panel.

Some retained legacy modules refer to companion contract modules from the larger private development tree (for example `baseline_config`, `baseline_xgboost`, `baseline_lightgbm`, or `sehcnm_modules`). Those historical entry points are therefore not presented as standalone public workflows and are excluded from the public quick start. The manuscript workflows and their authority scripts are mapped in `docs/EXPERIMENT_MAP.md` and `RESULT_MANIFEST.csv`.

These files are retained to avoid silently erasing development provenance. Their presence does not imply that the technically excluded LightGBM learner was part of the final four-backbone manuscript panel.
