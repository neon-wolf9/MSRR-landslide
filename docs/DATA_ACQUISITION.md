# Data acquisition

## Scope

This repository does not redistribute third-party EO/GIS source data or source inventories. Acquisition is performed independently from the original providers and is subject to their terms. `docs/DATA_SOURCES.md` is the authoritative source registry assembled from existing project records.

The frozen **derived** study inputs and two historical prediction hand-offs are separately published at [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084). Download `MSRR_CG_approved_data_candidate.zip` and extract it at the code repository root, preserving `data/`, `external/`, `experiments/`, and `metadata/` paths. From that root, verify the eleven files against `metadata/SHA256SUMS.txt` using the command in the README, then run `python scripts/01_verify_inputs.py --full` for its seven-file screen. This deposit does not replace the independently acquired raw products needed for source-to-benchmark reconstruction.

## Acquisition order

1. Review `THIRD_PARTY_NOTICES.md` and `docs/THIRD_PARTY_DATA.md`.
2. Use `docs/DATA_SOURCES.md` to identify the exact product, version, event date or snapshot, official access page, and expected relative path.
3. Preserve provider metadata, scene or tile identifiers, acquisition timestamps, CRS, units, NoData definitions, and original checksums.
4. Place source files under the documented `${PROJECT_ROOT}`-relative locations. Do not edit public configs to point to personal absolute paths.
5. Run the relevant preprocessing and benchmark-construction authority scripts in `docs/BENCHMARK_PROTOCOL.md`.
6. Compare reconstructed identities, counts, costs, folds, and hashes against the frozen reference evidence.

## Product-specific constraints

- Landsat inputs must be Collection 2 Level-2 and strictly pre-event, with QA information retained.
- Historical OpenStreetMap inputs must respect the event cutoff; a current Geofabrik extract is not equivalent.
- IMERG inputs must be Final Run V07B half-hour products with continuous temporal coverage and the frozen event windows.
- Copernicus DEM inputs must retain the GLO-30 DGED product and correct tiles.
- SoilGrids requests must preserve property, depth, statistic, units, scaling, and NoData semantics.
- Event inventories and interpretation layers must preserve the source vintage used to freeze the positive evidence.

If an official access route is not fully specified in the repository, consult the manuscript and supplementary material. Do not invent or substitute a URL. Authentication material must remain outside the repository.

## Verification boundary

The public package can verify manifests, reference outputs, synthetic MSRR behavior, and frozen exact-replay evidence without source data. A complete source-data-to-results reproduction requires the excluded inputs and is not claimed by the redistribution-safe test suite alone.
