# Security and public-release audit

> **Historical pre-release record.** Retained for scientific and release provenance. Statements below about a candidate, missing deposit, tag, or publication status describe the audit-time snapshot, not the current release. For current code and data access, use [v1.0.1](https://github.com/neon-wolf9/MSRR-landslide/releases/tag/v1.0.1), [Zenodo DOI 10.5281/zenodo.23533084](https://doi.org/10.5281/zenodo.23533084), and `README.md`. Historical tests are not a new scientific rerun.

Audit date: 2026-10-01

## Final status: PASS

The release tree was scanned recursively after the publication edits. No credential, personal-path, private-network, or raw third-party-data finding blocks publication.

| Check | Status | Evidence / disposition |
|---|---|---|
| Usernames and home-directory paths | PASS | No `[personal home path]`, `/home/...`, `/Users/...`, project-owner username, Desktop path, or Downloads path found. |
| Project-machine absolute paths | PASS | No `[local project root]` literal remains in public files. The restored comparator data contract resolves its root from `__file__`. |
| Passwords, tokens, API keys, cookies, authorization headers | PASS | No assigned credential patterns, bearer token, private key block, or credential file found. Documentation mentions these terms only to state that they are excluded. |
| Private URLs | PASS | No localhost, loopback, RFC1918, or intranet URL found. Public provider and upstream repository URLs are intentional. |
| Raw third-party EO/GIS data | PASS | No `.tif`, `.tiff`, `.gpkg`, `.shp`, `.shx`, `.dbf`, `.prj`, `.pbf`, `.hdf5`, `.h5`, `.nc`, `.nc4`, `.gdb`, or `.grd` source file is present. The `.npz` demo is synthetic and redistribution-safe. |
| Third-party comparator code | PASS WITH DOCUMENTED EXCLUSION | The unlicensed upstream PU-pullbaggingDT checkout is not included and is ignored by `.gitignore`; only repository-authored adapters are distributed. |
| Public Python prose | PASS | Chinese comments/messages/report prose were translated. Japanese schema/evidence literals required by `128_AUDIT_RAINFALL_TEMPORAL_PROTOCOL.py` remain by design and are documented in `TRANSLATION_AUDIT.md`. |

## Non-personal system paths

Three Windows installation paths remain in provenance/plotting helpers: `C:\Windows\Fonts\arial.ttf`, `C:\Windows\Fonts\arialbd.ttf`, and the standard Visual Studio 2022 Build Tools `vcvars64.bat` location. These contain no personal information or credentials. They are optional system-runtime probes, not dataset or project roots.

## Publication controls

`.gitignore` excludes independently acquired `external/` data, regenerated `data/derived/` content, the optional upstream comparator checkout, caches, and local experiment work directories. Reference results already committed under `reference_results/` are not ignored.
