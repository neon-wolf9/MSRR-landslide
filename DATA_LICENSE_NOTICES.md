# Data licences and attribution for the unpublished candidate

The repository `LICENSE` is an MIT **software** licence. It does not grant rights in third-party imagery, inventories, maps, databases, or mixed-source derived tables. Nothing in this local candidate has been uploaded.

The only real-data array currently copied here is `data/07_FINAL_REPAIRED_DATASET_V2/08_PAIR_TENSOR_INDEX_INT64.npy`: 5,056 × 3 author-generated integer row positions, SHA-256 `fa8aa5f7b8d1659199ff0aa25d7e452a363c5a226da66407d23a809d18040169`. It contains no source raster values, labels, coordinates, geometry, or named-person fields. It is not meaningful without the separately reviewed sample index. Final deposit should identify the authors as the compilation creators and state a separate data licence after their sign-off.

The six other model-ready files and the additional pair table/registry are **not** in this candidate. Their precise upstream editions, rights, attribution, modification notices, and linkage/disclosure risks remain `NEEDS RIGHTS REVIEW`; do not assign them the MIT licence by default. `DATA_RIGHTS_AUDIT.md` records file-level SHA-256 and decision evidence.

Provider evidence to match against frozen acquisition records before any deposit:

- [AJG Hiroshima inventory report](https://www.ajg.or.jp/disaster/files/201807_report004.pdf): the cited fourth report says CC BY 4.0; verify that it is the exact edition used.
- [GSI terms](https://www.gsi.go.jp/ENGLISH/page_e30286.html): normally PDL 1.0 with source and modification notice; the Japanese terms are authoritative and exceptions/third-party rights must be checked for the exact event product.
- [NASA Earthdata policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) and [GPM policy](https://gpm.nasa.gov/data/policy): unmarked NASA-led data have a CC0 route; verify the exact IMERG collection record and cite product DOI/version/access date.
- [OpenStreetMap copyright](https://www.openstreetmap.org/copyright): ODbL attribution and share-alike can apply to derivative databases. Determine whether the mixed 92-feature tables meet that definition before choosing their data licence.
- Other source-specific evidence and credits are indexed in `docs/THIRD_PARTY_DATA.md` and `DATA_RIGHTS_AUDIT.md`.

This notice is a release-preparation record, not a legal opinion or a claim that all six excluded files are prohibited from distribution.
