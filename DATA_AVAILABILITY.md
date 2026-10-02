# Data availability and redistribution boundary

This repository does not redistribute third-party raw EO/GIS products, source inventories, or authenticated downloads. It contains redistribution-safe schemas, provenance manifests, hashes, exact matched-set replay memberships, synthetic examples, and frozen numerical reference results.

## Included

- feature-order and timestamp schemas;
- source and processing manifests;
- data-quality, leakage, and benchmark-integrity audits;
- frozen matching protocol records and exact replay memberships;
- OOF margins, metrics, bootstrap outputs, and other numerical reference materials;
- an eight-set synthetic MSRR demo containing no scientific or personal data.

## Not included

- AJG and GSI inventory/evidence layers;
- GSI elevation products and Copernicus DEM GLO-30 tiles;
- NASA GPM IMERG granules;
- Landsat scenes and QA bands;
- SoilGrids rasters;
- Copernicus Global Land Service LC100 rasters;
- historical OpenStreetMap/Geofabrik PBF snapshots;
- J-FlwDir rasters;
- GSJ/AIST Seamless Digital Geological Map source vectors;
- credentials, API keys, cookies, tokens, passwords, or account files.

## Reacquisition

Use `docs/DATA_SOURCES.md` as the human-readable registry and `configs/external/`, `data/manifests/`, and `data/schemas/` as the machine-readable contracts. Restore each input at its documented `${PROJECT_ROOT}`-relative path and verify version, date/snapshot, bounds, CRS, units, field schema, and checksum before running a reconstruction script.

Several inputs require manual acquisition or authentication. A current provider product is not automatically equivalent to the event-specific or frozen asset used by the study. The current GSJ Seamless V2 download, for example, may change over time; the study asset is identified by its acquisition date and SHA-256 in `docs/DATA_SOURCES.md`.

## Redistribution

No blanket sublicence is asserted for any source product. Consult and comply with the provider's current terms and attribution requirements. Repository maintainers must perform a separate rights review before adding any derived geospatial artifact to a public archive.
