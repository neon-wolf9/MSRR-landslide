# Third-party data and software notices

This document records provenance and redistribution boundaries. It is not a substitute for provider licences or legal advice. The repository MIT licence covers only repository-authored software and documentation.

## Data and inventories

- **Association of Japanese Geographers (AJG):** July 2018 Hiroshima slope-failure interpretation. Source material is not bundled and no redistribution licence is asserted.
- **Geospatial Information Authority of Japan (GSI):** event evidence, interpretation products, and elevation inputs. Source material is not bundled; consult GSI terms.
- **Copernicus DEM GLO-30 (`COP-DEM_GLO-30-DGED`):** terrain source and common-DEM sensitivity input. Not bundled; consult Copernicus Data Space terms.
- **ISRIC SoilGrids 2.0:** soil covariates. Not bundled; consult ISRIC terms and attribution guidance.
- **USGS Landsat 8 Collection 2 Level-2:** pre-event NDVI and QA/water-mask inputs. Not bundled; consult USGS data policy and retain scene metadata.
- **Copernicus Global Land Service LC100 Collection 3, v3.0.1, 2015 base:** land-cover fractions. Not bundled; consult the Copernicus Land Monitoring Service data policy.
- **OpenStreetMap contributors / Geofabrik:** historical roads, waterways, and coastline/water inputs. Not bundled. OpenStreetMap data are subject to ODbL attribution and share-alike requirements; exact historical snapshots may require separate acquisition.
- **J-FlwDir v1.4:** hydrological routing, height-above-nearest-drainage, and upstream-area inputs. Not bundled; follow the official release's citation and licence instructions.
- **GSJ/AIST Seamless Digital Geological Map of Japan V2, 1:200,000:** geology input. Not bundled. The provider's English terms state Government of Japan Standard Terms of Use 2.0 with attribution; the exact frozen asset is documented in `docs/DATA_SOURCES.md`.
- **NASA GPM IMERG Final Run V07B (`GPM_3IMERGHH_07`):** half-hour rainfall. Not bundled; NASA Earthdata authentication is normally required for acquisition.

See `docs/DATA_SOURCES.md` for exact versions, supports, dates/snapshots, official pages, local paths, and study-specific acquisition cautions.

## Published comparator source

The repository-authored PU-pullbaggingDT adapter was informed by `https://github.com/ShubingOuyangcug/PU-pullbaggingDT` at commit `a8857dad30a454e3155644b18ddd2600df860e70`. The inspected upstream checkout contained no explicit `LICENSE`, `LICENCE`, `COPYING`, or `NOTICE` file. Consequently:

- no upstream PU-pullbaggingDT file is included in this package;
- the upstream repository is not relicensed under MIT;
- users who need the original provenance guard must obtain that source independently and place it at `third_party/PU-pullbaggingDT`;
- this repository grants no rights to the upstream source.

The adapter itself is repository-authored and included under MIT. `docs/BASELINE_PROVENANCE.md` distinguishes paper-faithful reimplementation, paper-informed adaptation, and official-code-informed adaptation.

## Python dependencies

The package uses the open-source libraries pinned in `environment.yml` and `requirements.txt`. Each dependency is governed by its own licence. No dependency source is vendored here.
