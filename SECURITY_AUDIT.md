# Security and public-release audit

Audit date: 2026-10-01

## Final status: PASS

The release tree was scanned recursively after the publication edits. No credential, personal-path, private-network, or raw third-party-data finding blocks publication.

| Check | Status | Evidence / disposition |
|---|---|---|
| Usernames and home-directory paths | PASS | No `C:\Users\...`, `/home/...`, `/Users/...`, project-owner username, Desktop path, or Downloads path found. |
| Project-machine absolute paths | PASS | No `D:\landslide_s_ehc_net` literal remains in public files. The restored comparator data contract resolves its root from `__file__`. |
| Passwords, tokens, API keys, cookies, authorization headers | PASS | No assigned credential patterns, bearer token, private key block, or credential file found. Documentation mentions these terms only to state that they are excluded. |
| Private URLs | PASS | No localhost, loopback, RFC1918, or intranet URL found. Public provider and upstream repository URLs are intentional. |
| Raw third-party EO/GIS data | PASS | No `.tif`, `.tiff`, `.gpkg`, `.shp`, `.shx`, `.dbf`, `.prj`, `.pbf`, `.hdf5`, `.h5`, `.nc`, `.nc4`, `.gdb`, or `.grd` source file is present. The `.npz` demo is synthetic and redistribution-safe. |
| Third-party comparator code | PASS WITH DOCUMENTED EXCLUSION | The unlicensed upstream PU-pullbaggingDT checkout is not included and is ignored by `.gitignore`; only repository-authored adapters are distributed. |
| Public Python prose | PASS | Chinese comments/messages/report prose were translated. Japanese schema/evidence literals required by `128_AUDIT_RAINFALL_TEMPORAL_PROTOCOL.py` remain by design and are documented in `TRANSLATION_AUDIT.md`. |

## Non-personal system paths

Three Windows installation paths remain in provenance/plotting helpers: `C:\Windows\Fonts\arial.ttf`, `C:\Windows\Fonts\arialbd.ttf`, and the standard Visual Studio 2022 Build Tools `vcvars64.bat` location. These contain no personal information or credentials. They are optional system-runtime probes, not dataset or project roots.

## Publication controls

`.gitignore` excludes independently acquired `external/` data, regenerated `data/derived/` content, the optional upstream comparator checkout, caches, and local experiment work directories. Reference results already committed under `reference_results/` are not ignored.
