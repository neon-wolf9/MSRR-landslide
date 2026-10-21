# Remaining release blockers (2026-10-21)

> **Historical pre-release record.** Retained for scientific and release provenance. Statements below about a candidate, missing deposit, tag, or publication status describe the audit-time snapshot, not the current release. For current code and data access, use [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1), [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), and `README.md`. Historical tests are not a new scientific rerun.

Decision for this **unpublished local remediation candidate**: **NOT READY**. No tag or GitHub release was created; `v1.0.0` was not changed. This is a focused follow-up to the earlier audit, not a claim that the full scientific experiment was rerun.

Historical local light-gate evidence from the previous focused audit: Python 3.10.20 in `[local audit environment]`; the five lightweight commands exited 0 and pytest reported **11 passed** on that earlier working tree. The log is `[local non-distributed validation log]`. These results must not be relabelled as validation of later documentation changes.

## Actual code / public-evaluability status

- No algorithm defect was established. The synthetic example exercises the canonical RAW 792 → MSRR 2446 transform and matched-set metric interface. A test-only invented edge graph exercises the canonical all-or-none/common-support matching interface. It does **not** build either event benchmark or reproduce paper metrics.
- `scripts/01_verify_inputs.py --full` still requires seven frozen **derived** inputs absent from the public repository. Four are Hiroshima sample/static/rainfall/pair-index products; three are Kyushu matched-index/static/rainfall products. Their full paths, roles, and proposed release treatment are in `DATA_AVAILABILITY.md`. Thus a reviewer cannot currently run the primary event workflows from the public package alone. This is an evaluability gap, not an automatic claim that third-party law forbids release.
- Additional **code/path blocker identified 2026-10-21**: the 120B import chain reaches `20A.preflight()`, which requires original ancillary files and BASE-00 modules not in the public package. The original Hiroshima finalizer is also absent. Scripts 121/124D expect the missing 120B experiment-output directory, not merely `reference_results/`. The seven-file checker does not cover this closure. Exact paths are in `REPRODUCIBILITY_GAP_ANALYSIS.md`.
- An author-approved deposit should prioritize the small Hiroshima pair index (~0.12 MB), Hiroshima sample index (~0.28 MB), and Kyushu matched index (~0.25 MB), with frozen SHA-256, schemas, provenance, required attribution, and a tested download/placement procedure. These would enable stronger membership and interface checks, but not by themselves restore the 92-feature/rainfall input tensors needed for score regeneration. The three feature/rainfall products need separate file-level rights review and either lawful deposit or a precise, verifiable reconstruction path.

## Data rights and scientific reproducibility limits

- The seven absent products are author-generated derivatives, not raw downloads. [AJG's cited fourth report](https://www.ajg.or.jp/disaster/files/201807_report004.pdf) explicitly uses CC BY 4.0 with credit; [Copernicus DEM](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf), [SoilGrids](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html), [Landsat](https://www.usgs.gov/media/images/landsat-product-citation-screenshot), [CLMS](https://land.copernicus.eu/en/data-policy), and [GSJ](https://gbank.gsj.jp/seamless/agreement_en.html) also have reuse routes with their stated conditions. Authentication or omission from this repository is not a licence prohibition.
- Remaining file-level questions: confirm exact AJG/GSI source vintages and any GSI third-party/Survey Act conditions; classify OSM-derived feature tables under ODbL/database rules and provide proper attribution; confirm the exact IMERG product notice and required citation; confirm J-FlwDir derivative treatment and preserve its CC BY attribution. The [J-FlwDir page](https://hydro.iis.u-tokyo.ac.jp/~yamadai/JapanDir/index.html) asks users not to repost the entire original-format dataset without permission, which does not itself ban small derived features. These are author/legal-review decisions, not established prohibitions.
- Frozen upstream historical snapshots and the seven derived products are not publicly supplied together. No source-data-to-benchmark or source-data-to-paper-score rerun was performed or claimed. Synthetic fixtures must never be described as real scientific results.

## Network validation

- An earlier independent GitHub clone succeeded; its clean local tree was verified to match public `main` commit `8534c2beebc6f57188c29f775fbd57a0a09d6ed0` by tree identity. This follow-up's two new GitHub clone attempts failed before checkout (DNS/getaddrinfo and then connection reset). Therefore a **new remote clone of the final repaired state is NETWORK BLOCKED**. Local clean-tree checks are a separate, lower-level evidence tier. A final remote clone and five lightweight commands must be repeated after any public-main update; old logs must not be relabelled as a new remote-clone pass.

## Final manuscript correction

- The author's identified final Word manuscript is `[author-local final manuscript]` (read only). Its methods, 792/2446 dimensions, Hiroshima 5,056/10,112 and Kyushu 1,692/3,384 counts, and checked principal numerical metrics agree with the frozen public references inspected in this audit. This is a targeted consistency check, not a complete independent rerun.
- Its Code Availability paragraph still describes the full public workflow as available under release **`v1.0.0`**. That tag omits files restored on `main`; the statement must be corrected in the Word manuscript by the author. The manuscript was not edited here. `CODE_AVAILABILITY.md` already describes the repository status accurately.

## Author decisions before approval

1. Correct the final Word Code Availability statement to point to the verified complete commit/future approved release, not the incomplete `v1.0.0`.
2. Approve or reject a provenance-and-rights-reviewed deposit of each small derived index; supply exact frozen bytes, attribution, and preferred archive/location if approved.
3. For the four larger/mixed-source feature and rainfall products, provide file-level source lineage and a lawful deposit or independently executable reconstruction route. Resolve ODbL, GSI, AJG edition, IMERG, and J-FlwDir conditions without blanket restrictions.
4. After public-main changes are authorized, repeat independent remote-clone validation when GitHub connectivity recovers. Do not release until the public commit, manifests, docs, and manuscript availability statement agree.
