# Data dictionary

The authoritative machine-readable schemas are in `data/schemas/` and `configs/external/`.

## Core row identifiers

| Field | Meaning | Model input? |
|---|---|---|
| `pair_set_id` | Complete positive–control–control set identifier | No |
| `unit_id` | 250 m analytical grid identifier | No |
| `sample_role` | Positive, control 1, or control 2 | No |
| `control_rank` | Deterministic control order within a set | No |
| `y_pair` | Binary evaluation label | No |
| `spatial_group` / fold fields | Leakage-controlled split metadata | No |

## Predictor blocks

| Block | Shape per row | Notes |
|---|---:|---|
| Static EO/GIS | 92 | Terrain, land cover, pre-event NDVI, soil, road, river/hydrology, coast |
| Rainfall sequence | 70 × 10 | Half-hourly event-relative IMERG-derived variables |
| RAW | 792 | `92 + 70 × 10` |
| MSRR | 2,446 | RAW plus signed/absolute matched-context deviations and 70 summaries |

IDs, labels, inventory provenance, coverage flags, and split metadata are explicitly excluded from predictive features. Hiroshima field order is recorded by `data/schemas/06A_STATIC_92_FIELD_SCHEMA.csv`, `07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv`, and `07B_DYNAMIC_10_FIELD_ORDER.csv`. Kyushu crosswalk and whitelist files are under `configs/external/`.

## Data distribution status

Schemas, manifests, audits, matched-set replay memberships, and numerical result arrays are included. Raw third-party raster/vector products are omitted; see `DATA_AVAILABILITY.md`.

