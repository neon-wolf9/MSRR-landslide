# Data layout

The repository intentionally separates public metadata and examples from non-redistributed scientific inputs.

```text
data/
|-- README.md
|-- manifests/              # provenance, hashes, QA, and frozen decisions
|-- schemas/                # field order and timestamp contracts
|-- example/                # synthetic, redistribution-safe MSRR input
|-- raw/                    # locally acquired third-party sources; ignored
|-- interim/                # local intermediate products; ignored
|-- processed/              # local processed products; ignored
`-- derived/                # locally reconstructed scientific inputs; ignored

external/
`-- kyushu_2017_asakura_toho/  # independently acquired/rebuilt external-event inputs; ignored
```

The full Hiroshima rerun expects the frozen repaired data products listed by `scripts/01_verify_inputs.py`, including the sample index, 92 static features, `70 x 10` dynamic rainfall tensor, and matched-triplet index. The full Kyushu rerun expects its matched index, static table, and dynamic tensor under the external-event frozen-dataset tree.

Paths in executable code resolve from the repository root or from script-relative project roots. Public documentation uses `${PROJECT_ROOT}` as a portable placeholder. A missing input is an informative stop and must not be replaced with fabricated or silently updated data.

Raw third-party source data remain absent from the code repository. The seven frozen derived inputs, additional pair files, and two historical prediction hand-offs are published separately in [Zenodo record 23533084](https://doi.org/10.5281/zenodo.23533084); extract its ZIP at the repository root to restore the documented relative paths. Source-specific attribution and database/share-alike terms are included in the ZIP metadata; see the README and `DATA_AVAILABILITY.md`.
