# Third-party data

The MIT license applies only to repository-authored software and documentation. It does not grant redistribution rights for EO/GIS products, event inventories, or provider-controlled downloads.

The study used the following third-party sources recorded in the project manifests and manuscript materials:

| Source | Product or role | Public-package status |
|---|---|---|
| Association of Japanese Geographers (AJG) | July 2018 Hiroshima slope-failure interpretation | Not bundled. The cited [fourth report, 2 August 2018](https://www.ajg.or.jp/disaster/files/201807_report004.pdf), section 6, explicitly states CC BY 4.0 and gives required credit. Confirm the frozen input really derives from this edition and preserve that credit; any additional GSI evidence has separate terms. |
| Geospatial Information Authority of Japan (GSI) | Event evidence and elevation products | Not bundled. [GSI terms](https://www.gsi.go.jp/ENGLISH/page_e30286.html) generally permit reuse with attribution but warn of third-party rights and other conditions; check the exact layers and any survey-law requirements. |
| Copernicus Data Space | Copernicus DEM GLO-30 (`COP-DEM_GLO-30-DGED`) | Not bundled. [GLO-30 licence](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf) allows reproduction, distribution, and adaptation with attribution and disclaimers. |
| ISRIC | SoilGrids 2.0 covariates | Not bundled. [ISRIC FAQ](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html) states CC BY 4.0 for SoilGrids products from 2019 onward; verify the source vintage. |
| U.S. Geological Survey | Landsat 8 Collection 2 Level-2 surface reflectance and QA | Not bundled. [USGS Landsat guidance](https://www.usgs.gov/media/images/landsat-product-citation-screenshot) says Landsat data are public domain; citation is encouraged. |
| Copernicus Land Monitoring Service | CGLS-LC100 Collection 3, v3.0.1, 2015 base | Not bundled. [CLMS data policy](https://land.copernicus.eu/en/data-policy) provides open/free access and reuse subject to acknowledgement and modification notices. |
| OpenStreetMap contributors / Geofabrik | Historical roads, waterways, coastline, and water features | Not bundled. [OSM ODbL](https://www.openstreetmap.org/copyright) attribution and possible share-alike/database obligations apply; classify the proposed derived 92-feature tables before release. |
| University of Tokyo | J-FlwDir v1.4 hydrological layers | Not bundled. [Product page](https://hydro.iis.u-tokyo.ac.jp/~yamadai/JapanDir/index.html) says CC BY 4.0, requests registration for download, and asks users not to repost the whole original-format dataset without written permission. This is not a demonstrated ban on small derived features. |
| GSJ/AIST | Seamless Digital Geological Map of Japan V2, 1:200,000 | Not bundled. [GSJ terms](https://gbank.gsj.jp/seamless/agreement_en.html) permit reuse under Government of Japan Standard Terms of Use 2.0 with attribution; preserve the exact frozen version. |
| NASA GES DISC | GPM IMERG Final Run V07B (`GPM_3IMERGHH_07`) | Not bundled; Earthdata login may be needed to download. [NASA data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) makes NASA-led data open unless specifically marked; confirm the exact IMERG product notice and citation before publishing derived arrays. |

Exact versions, event cutoffs, official pages already recorded by the project, expected paths, and acquisition cautions are in `docs/DATA_SOURCES.md`. The terms above describe provider materials reviewed for this audit, not a legal clearance of the seven mixed-source derived files. Their lineage, exact vintage, attribution text, and any third-party exceptions need file-level review. Provider terms can change; users must verify current licensing and attribution at acquisition time. A current product or snapshot is not automatically equivalent to the frozen study asset.

No passwords, tokens, cookies, authenticated downloads, browser profiles, or credential files belong in this repository.
