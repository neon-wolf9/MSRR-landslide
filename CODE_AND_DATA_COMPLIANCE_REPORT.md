# Computers & Geosciences code-and-data gate — public-baseline audit, 2026-10-21

> **Historical pre-release record.** Retained for scientific and release provenance. Statements below about a candidate, missing deposit, tag, or publication status describe the audit-time snapshot, not the current release. For current code and data access, use [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1), [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), and `README.md`. Historical tests are not a new scientific rerun.

This report is the earlier **public-baseline** gate. The newer unpublished local assembly and its distinct tests are recorded in `RELEASE_CANDIDATE_STATUS.md`. Do not interpret the 331-file/11-test figures below as validation of the assembled candidate.

**Decision: NOT READY — BLOCKERS REMAIN.** This is a local, unpublished documentation candidate. No scientific method, result, frozen configuration or real-data file was changed or uploaded. No tag or release was made.

Local lightweight acceptance after these documentation edits: Python 3.10.20; the synthetic example, environment check, seven-file screen, reference/hash verifier and full lightweight pytest suite each exited 0; **331** manifest files verified and **11 tests passed**. Full command output is in `[local non-distributed validation log]`. This is **not** an independent main-result rerun or a fresh remote-clone pass.

The user-supplied [C&G Guide for Authors](https://www.elsevier.com/journals/computers-and-geosciences/0098-3004/guide-for-authors) returned HTTP 403 to this audit. The publisher's accessible [Computers & Geosciences journal page](https://shop.elsevier.com/journals/computers-and-geosciences/0098-3004) states that public documented code, a licence, English README/install/use/dependency/resource guidance, sufficient materials for main results or justified data limits with a dummy/synthetic test, tutorials, input/output user guide, non-archive-only distribution and English comments are expected. It also identifies results that cannot be independently evaluated because code/workflow/parameters/supporting data are unavailable as unsuitable. This report does not invent a rule that **all raw data** or Zenodo deposits are mandatory.

| C&G-related gate | Evidence in this candidate | Status |
|---|---|---|
| Public software repository and explicit software licence | GitHub repository URL in `README.md`; `LICENSE` covers author code, not upstream data. | PASS for repository/documentation; current local fixes are not yet on GitHub. |
| English install, basic use, dependencies, computational limits, tutorial and user guide | `README.md`, `QUICKSTART.md`, `environment.yml`, `requirements.txt`, `docs/USER_GUIDE.md`, `docs/REPRODUCIBILITY.md`. | PASS for documented light use; full-use instructions need correction below. |
| Executable synthetic case and English test/comment surface | `examples/minimal_msrr_example.py`, `data/example/synthetic_msrr_demo.npz`, `tests/`. | PASS for lightweight software behavior only; not scientific-result reproduction. |
| Main-result inputs and primary workflow evaluability | Seven private derived inputs have valid frozen hashes. `20A.preflight()` requires further absent files and BASE-00 modules; the original Hiroshima finalizer is absent; 121/124D expect the missing 120B experiment hand-off layout. | **FAIL / unresolved**. A reviewer cannot presently regenerate the primary scientific results from the public package. |
| Honest data restrictions and provenance | `DATA_AVAILABILITY.md`, `docs/THIRD_PARTY_DATA.md`, `DATA_RIGHTS_AUDIT.md` distinguish non-bundling from proven legal restriction and record provider conditions. | PARTIAL: file-level author clearance and a tested access/deposit route remain outstanding. |
| Frozen results, hashes and primary/diagnostic distinctions | `RESULT_MANIFEST.csv`, `FILE_MANIFEST.csv`, `reference_results/`, `docs/PAPER_RESULT_MAP.md`. | PASS as archived evidence; hash success is not an independent rerun. |
| Manuscript Code Availability accuracy | Final local Word manuscript still attributes the complete workflow to incomplete `v1.0.0`; repository `main` has restored files. | **FAIL until author corrects manuscript**. Word was not edited. |

## Decisions needed before a final-publication audit

1. Author grants or declines file-level sharing of each of the seven derived inputs, with exact source/version evidence, attribution, ODbL/GSI/J-FlwDir/IMERG review, ID-disclosure check and a separate data licence as appropriate. No file is cleared for immediate upload by this report.
2. Make the 120B import/data dependency closure public and reproducible, including the extra files and missing original BASE-00 modules, or document a genuinely executable, authorized alternative. The seven-file input checker must not be presented as proof of full workflow readiness.
3. Resolve 121/124D's expected original 120B output directory and selection/OOF hand-off through an audited layout or documentation solution. Do not move reference results into place and claim a rerun unless that procedure is scientifically reviewed and tested.
4. Correct the manuscript Code Availability citation of `v1.0.0`, then align it with the eventual verified public commit/release and any approved data deposit.
5. After those changes, run a fresh clone and lightweight acceptance, followed by a separately authorized independent scientific-evaluation plan. The latter was expressly outside this task.

Until then, the repository enables inspection, reuse and synthetic evaluation of the MSRR software and audit of frozen references, **not** independent verification of the paper's main event scores. Neither the existence of private complete data nor a plausible open upstream licence removes the above executable and rights-clearance gaps.
