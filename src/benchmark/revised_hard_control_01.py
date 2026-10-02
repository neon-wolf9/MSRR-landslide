from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/06_hard_control/01b_revised_matching_protocol"
HC00 = ROOT / "data/06_hard_control/00_candidate_pool"
HC01 = ROOT / "data/06_hard_control/01_matching_protocol"
HC01A = ROOT / "data/06_hard_control/01a_soil_distance_protocol"
LABEL04 = ROOT / "data/05_event_label_evidence/08_label_04_positive_grid_mapping"
STATIC03 = ROOT / "data/03_static_eogis_features/90_third_layer_assembly/05_static_03_final_audit_freeze"
STATIC_FROZEN = STATIC03 / "06_frozen"
RAIN04 = ROOT / "data/04_dynamic_rainfall_features/07_rain_04_final_audit_freeze"
RAIN_FROZEN = RAIN04 / "07_frozen"
MASTER = ROOT / "data/01_county_prediction_domain/02_grid_250m/hiroshima_grid_250m_master.gpkg"

DIRS = {
    "input": OUT / "00_input_registry",
    "protocol": OUT / "01_protocol",
    "covariates": OUT / "02_covariates",
    "edges": OUT / "03_candidate_edges",
    "feasibility": OUT / "04_feasibility",
    "audit": OUT / "05_audit",
    "frozen": OUT / "06_frozen",
    "scripts": OUT / "scripts",
}

HC00_P1 = HC00 / "hard_control_00_one_ring_guard_candidates.csv"
HC00_MARKER = HC00 / "09_frozen/HARD_CONTROL_00_CANDIDATE_POOL_FROZEN.marker.json"
LABELS = LABEL04 / "09_frozen/label_04_positive_grid_labels_250m_frozen.parquet"
LABEL_MARKER = LABEL04 / "09_frozen/LABEL_04_POSITIVE_GRID_SUBMODULE_FROZEN.marker.json"
STATIC_FULL = STATIC_FROZEN / "static_third_layer_full_frozen.parquet"
STATIC_MODEL = STATIC_FROZEN / "static_third_layer_model_candidate_frozen.parquet"
STATIC_SCHEMA = STATIC_FROZEN / "static_third_layer_field_schema_frozen.csv"
STATIC_MARKER = STATIC03 / "09_marker/STATIC_THIRD_LAYER_FROZEN.marker.json"
RAIN_DATASET = RAIN_FROZEN / "dynamic_rainfall_features_frozen.parquet_dataset"
RAIN_SCHEMA = RAIN_FROZEN / "DYNAMIC_RAINFALL_FEATURE_SCHEMA_FROZEN.json"
RAIN_TIMES = RAIN_FROZEN / "dynamic_rainfall_timestamp_registry_frozen.csv"
RAIN_MARKER = RAIN04 / "10_marker/DYNAMIC_RAINFALL_LAYER_FROZEN.marker.json"
SOIL_MARKER = HC01A / "HARD_CONTROL_01A_SOIL_DISTANCE.marker.json"
SOIL_FIELDS = HC01A / "07_frozen/hard_control_01a_soil_field_inventory.csv"
SOIL_MAPPING = HC01A / "07_frozen/hard_control_01a_soil_property_depth_mapping.csv"
SOIL_PARAMS = HC01A / "07_frozen/hard_control_01a_soil_field_transform_parameters.csv"
SOIL_CONTRACT = HC01A / "07_frozen/hard_control_01a_soil_group_distance_contract.json"
SOIL_PROPERTY_CONTRACT = HC01A / "07_frozen/hard_control_01a_property_distance_contract.json"
SOIL_WEIGHT_PATCH = HC01A / "07_frozen/hard_control_01a_group_weight_patch.json"
SOIL_MANIFEST = HC01A / "07_frozen/hard_control_01a_frozen_hashes.csv"
ORIGINAL_REGISTRY = HC01 / "hard_control_01_matching_field_registry.csv"
ORIGINAL_FORBIDDEN = HC01 / "hard_control_01_forbidden_field_registry.csv"
ORIGINAL_MARKER = HC01 / "HARD_CONTROL_01_MATCHING_PROTOCOL.marker.json"
ORIGINAL_REPORT = HC01 / "HARD_CONTROL_01_MATCHING_PROTOCOL_REPORT.md"

ANCHOR_UTC = pd.Timestamp("2018-07-06T11:00:00Z")
ANCHOR_JST = pd.Timestamp("2018-07-06T20:00:00+09:00")
EXPECTED_POSITIVE = 5075
EXPECTED_CANDIDATE = 21825
REQUIRED_ASSIGNMENTS = 15225
EPS = 1e-12

GROUP_WEIGHTS = {
    "TERRAIN": 0.30,
    "SOIL": 0.15,
    "LANDCOVER_VEGETATION": 0.15,
    "HYDRO_DISTANCE": 0.15,
    "ACCESSIBILITY_DISTANCE": 0.05,
    "ANTECEDENT_RAINFALL": 0.20,
}

GROUP_FIELDS = {
    "TERRAIN": [
        "dem_elevation_mean", "dem_slope_mean", "dem_profile_curvature_mean",
        "dem_plan_curvature_mean", "dem_twi_mean",
    ],
    "LANDCOVER_VEGETATION": [
        "ndvi_pre_event", "lc_tree_ratio", "lc_shrub_ratio", "lc_grass_ratio",
        "lc_crops_ratio", "lc_builtup_ratio", "lc_bare_ratio",
        "lc_permanent_water_ratio", "lc_seasonal_water_ratio",
    ],
    "HYDRO_DISTANCE": [
        "river_distance_to_osm_river_m", "hydro_hnd_zonal_mean_m", "hydro_upa_zonal_mean_log1p",
    ],
    "ACCESSIBILITY_DISTANCE": [
        "road_distance_to_any_road_m", "road_distance_to_major_road_m", "coast_distance_to_centroid_m",
    ],
    "ANTECEDENT_RAINFALL": [
        "rain_24h_mm", "rain_72h_mm", "rain_120h_mm", "api_k090_step30m_120h",
    ],
}

RAIN_FIELDS = GROUP_FIELDS["ANTECEDENT_RAINFALL"]
TRIGGER_FIELDS = ["rain_30m_mm", "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm"]
LOG1P_FIELDS = {
    "river_distance_to_osm_river_m", "road_distance_to_any_road_m",
    "road_distance_to_major_road_m", "coast_distance_to_centroid_m",
    *RAIN_FIELDS,
}
K_VALUES: list[int | None] = [25, 50, 100, 200, 500, None]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def semantic_hash(frame: pd.DataFrame, sort_cols: list[str] | None = None) -> str:
    data = frame.copy()
    if sort_cols:
        data = data.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    text = data.to_csv(index=False, lineterminator="\n", na_rep="<NA>", float_format="%.15g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def protected_paths() -> list[Path]:
    return [
        MASTER, HC00_P1, HC00_MARKER, LABELS, LABEL_MARKER,
        STATIC_FULL, STATIC_MODEL, STATIC_SCHEMA, STATIC_MARKER,
        RAIN_SCHEMA, RAIN_TIMES, RAIN_MARKER,
        *sorted(RAIN_DATASET.rglob("*.parquet")),
        SOIL_MARKER, SOIL_FIELDS, SOIL_MAPPING, SOIL_PARAMS, SOIL_CONTRACT,
        SOIL_PROPERTY_CONTRACT, SOIL_WEIGHT_PATCH, SOIL_MANIFEST,
        ORIGINAL_REGISTRY, ORIGINAL_FORBIDDEN, ORIGINAL_MARKER, ORIGINAL_REPORT,
    ]


def protected_hashes() -> dict[str, str]:
    return {str(p): sha256(p) if p.exists() else "MISSING" for p in protected_paths()}


def validate_soil_manifest() -> int:
    tab = pd.read_csv(SOIL_MANIFEST, encoding="utf-8-sig")
    failures = 0
    for row in tab.itertuples(index=False):
        path = Path(row.absolute_path)
        failures += int(not path.exists() or sha256(path) != row.sha256)
    failures += int(tab["filename"].eq(SOIL_MANIFEST.name).sum())
    return failures


def gate_inputs() -> dict[str, Any]:
    required = protected_paths()
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required frozen inputs: " + "; ".join(missing))
    hc00 = json.loads(HC00_MARKER.read_text(encoding="utf-8"))
    label = json.loads(LABEL_MARKER.read_text(encoding="utf-8"))
    static = json.loads(STATIC_MARKER.read_text(encoding="utf-8"))
    rain = json.loads(RAIN_MARKER.read_text(encoding="utf-8"))
    soil = json.loads(SOIL_MARKER.read_text(encoding="utf-8"))
    checks = {
        "master_grid_unique_path": MASTER.exists(),
        "hard_control_00_frozen": hc00.get("CANDIDATE_POOL_FROZEN") == "YES",
        "positive_layer_frozen": label.get("LABEL_04_POSITIVE_GRID_MAPPING_PASS") == "YES" or label.get("LABEL_04_POSITIVE_GRID_SUBMODULE_FROZEN") == "YES",
        "static_layer_frozen": static.get("STATIC_THIRD_LAYER_FROZEN") == "YES" or static.get("status") == "FROZEN",
        "rain_layer_frozen": rain.get("DYNAMIC_RAIN_LAYER_FROZEN") == "YES",
        "soil_01a_pass": soil.get("HARD_CONTROL_01A_SOIL_DISTANCE_PASS") == "YES",
        "soil_protocol_frozen": soil.get("SOIL_DISTANCE_PROTOCOL_FROZEN") == "YES",
        "soil_ready": soil.get("READY_TO_RERUN_REVISED_HARD_CONTROL_01") == "YES",
        "soil_formal_fields_18": soil.get("SOILGRIDS_FORMAL_FIELDS") == 18,
        "soil_manifest_failures": validate_soil_manifest() == 0,
        "soil_weight": float(soil.get("SOIL_GROUP_WEIGHT", -1)) == 0.15,
        "soil_previous_errors": soil.get("ERRORS") == 0,
        "failed_hc01_remains_blocked": json.loads(ORIGINAL_MARKER.read_text(encoding="utf-8")).get("FINAL_DECISION") == "BLOCKED_HARD_CONTROL_01_MAIN_CALIPER_OR_GLOBAL_ASSIGNMENT_INFEASIBLE",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Input gate failed: {checks}")
    return checks


def input_registry() -> pd.DataFrame:
    rows = []
    for path in protected_paths():
        rows.append({
            "absolute_path": str(path), "filename": path.name,
            "file_size_bytes": path.stat().st_size, "sha256": sha256(path),
            "read_only_flag": "YES", "input_status": "PASS",
        })
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(LABELS, columns=["unit_id", "label_role_main"])
    positive_ids = labels.loc[labels["label_role_main"].eq("POSITIVE"), "unit_id"].tolist()
    candidate_ids = pd.read_csv(HC00_P1, encoding="utf-8-sig")["unit_id"].tolist()
    if len(positive_ids) != EXPECTED_POSITIVE or len(set(positive_ids)) != EXPECTED_POSITIVE:
        raise RuntimeError("Positive cardinality or uniqueness gate failed")
    if len(candidate_ids) != EXPECTED_CANDIDATE or len(set(candidate_ids)) != EXPECTED_CANDIDATE:
        raise RuntimeError("P1 cardinality or uniqueness gate failed")
    if set(positive_ids) & set(candidate_ids):
        raise RuntimeError("Positive and candidate unit_id sets overlap")

    soil_inventory = pd.read_csv(SOIL_FIELDS, encoding="utf-8-sig")
    soil_fields = soil_inventory["actual_field_name"].tolist()
    if len(soil_fields) != 18:
        raise RuntimeError("Frozen Soil field count is not 18")
    static_nonrain = sorted({f for group, fs in GROUP_FIELDS.items() if group != "ANTECEDENT_RAINFALL" for f in fs})
    static_cols = ["unit_id", "geology_dominant_class"] + static_nonrain + soil_fields
    static = pd.read_parquet(STATIC_FULL, columns=static_cols)
    if len(static) != 139364 or static["unit_id"].duplicated().any() or static["unit_id"].isna().any():
        raise RuntimeError("Static layer structural gate failed")

    rain_ds = ds.dataset(str(RAIN_DATASET), format="parquet", partitioning="hive")
    anchor = rain_ds.to_table(
        filter=ds.field("timestamp_utc") == ANCHOR_UTC.to_pydatetime(),
        columns=["unit_id", "timestamp_utc", "timestamp_jst"] + RAIN_FIELDS + TRIGGER_FIELDS,
    ).to_pandas()
    if len(anchor) != 139364 or anchor["unit_id"].duplicated().any():
        raise RuntimeError("Anchor rainfall structural gate failed")
    if not pd.to_datetime(anchor["timestamp_utc"], utc=True).eq(ANCHOR_UTC).all():
        raise RuntimeError("UTC anchor mismatch")
    if not pd.to_datetime(anchor["timestamp_jst"], utc=True).eq(ANCHOR_JST.tz_convert("UTC")).all():
        raise RuntimeError("JST anchor mismatch")
    cov = static.merge(anchor.drop(columns=["timestamp_utc", "timestamp_jst"]), on="unit_id", validate="one_to_one")
    idx = cov.set_index("unit_id")
    positive = idx.loc[positive_ids].reset_index()
    candidate = idx.loc[candidate_ids].reset_index()

    grid = gpd.read_file(MASTER, layer="hiroshima_grid_250m_master")[["unit_id", "geometry"]]
    if len(grid) != 139364 or grid.crs.to_epsg() != 6671 or grid["unit_id"].duplicated().any():
        raise RuntimeError("Master grid structural gate failed")
    cent = grid.geometry.centroid
    xy = pd.DataFrame({"unit_id": grid["unit_id"], "centroid_x": cent.x, "centroid_y": cent.y}).set_index("unit_id")
    pos_xy = xy.loc[positive_ids].reset_index()
    cand_xy = xy.loc[candidate_ids].reset_index()
    return positive, candidate, cov, grid, pos_xy, cand_xy


def transform_and_blocks(positive: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    original = pd.read_csv(ORIGINAL_REGISTRY, encoding="utf-8-sig")
    source_transform = original.dropna(subset=["actual_field_name"]).drop_duplicates("actual_field_name").set_index("actual_field_name")
    nonsoil_fields = [f for fs in GROUP_FIELDS.values() for f in fs]
    joint = pd.concat([positive[["unit_id"] + nonsoil_fields], candidate[["unit_id"] + nonsoil_fields]], ignore_index=True)
    params = []
    pos = positive.copy()
    cand = candidate.copy()
    active_by_group: dict[str, list[str]] = {}
    for group, fields in GROUP_FIELDS.items():
        active_by_group[group] = []
        for field in fields:
            raw = pd.to_numeric(joint[field], errors="coerce")
            if field in LOG1P_FIELDS:
                transformed = np.log1p(raw)
                transform = "LOG1P_THEN_ROBUST_SCALE"
            elif field == "hydro_upa_zonal_mean_log1p":
                transformed = raw
                transform = "KEEP_EXISTING_LOG1P_THEN_ROBUST_SCALE"
            else:
                transformed = raw
                transform = "ROBUST_SCALE"
            med = float(transformed.median())
            q1 = float(transformed.quantile(0.25))
            q3 = float(transformed.quantile(0.75))
            iqr = q3 - q1
            status = "PASS" if np.isfinite(iqr) and iqr > 0 else "ZERO_IQR_EXCLUDED"
            params.append({
                "actual_field_name": field, "source_protocol": "REVISED_HARD_CONTROL_01",
                "group_name": group, "transform": transform, "joint_pool_rows": 26900,
                "median": med, "q25": q1, "q75": q3, "IQR": iqr,
                "non_null_count": int(raw.notna().sum()), "missing_count": int(raw.isna().sum()),
                "parameter_status": status,
            })
            for frame in (pos, cand):
                values = pd.to_numeric(frame[field], errors="coerce")
                if field in LOG1P_FIELDS:
                    values = np.log1p(values)
                frame[f"z__{field}"] = (values - med) / iqr if status == "PASS" else np.nan
            if status == "PASS":
                active_by_group[group].append(field)

    soil_inventory = pd.read_csv(SOIL_FIELDS, encoding="utf-8-sig")
    soil_params = pd.read_csv(SOIL_PARAMS, encoding="utf-8-sig")
    soil_fields = soil_inventory["actual_field_name"].tolist()
    if len(soil_params) != 18 or not soil_params["zero_IQR_flag"].eq("NO").all():
        raise RuntimeError("Frozen Soil transform gate failed")
    soil_param_idx = soil_params.set_index("actual_field_name")
    for field in soil_fields:
        med = float(soil_param_idx.loc[field, "median"])
        iqr = float(soil_param_idx.loc[field, "IQR"])
        for frame in (pos, cand):
            frame[f"z__{field}"] = (pd.to_numeric(frame[field], errors="coerce") - med) / iqr
    soil_append = soil_params.copy()
    soil_append["source_protocol"] = "HARD_CONTROL_01A_FROZEN"
    soil_append["group_name"] = "SOIL"
    soil_append["parameter_status"] = np.where(soil_append["zero_IQR_flag"].eq("YES"), "ZERO_IQR_EXCLUDED", "PASS")
    soil_append["joint_pool_rows"] = soil_append["count"]
    keep = ["actual_field_name", "source_protocol", "group_name", "transform", "joint_pool_rows", "median", "q25", "q75", "IQR", "non_null_count", "missing_count", "parameter_status"]
    param_frame = pd.concat([pd.DataFrame(params)[keep], soil_append[keep]], ignore_index=True)

    active_nonsoil = [f for fs in active_by_group.values() for f in fs]
    missing_fields = ["geology_dominant_class"] + active_nonsoil + soil_fields
    for frame in (pos, cand):
        mask = frame[missing_fields].isna().astype(np.uint8).to_numpy()
        frame["matching_missingness_pattern_id"] = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in mask]
        soil_valid = frame[soil_fields].notna().sum(axis=1)
        partial = (soil_valid != 0) & (soil_valid != 18)
        if partial.any():
            raise RuntimeError(f"Partial Soil missingness found: {int(partial.sum())}")
        frame["soil_completeness_block"] = np.where(soil_valid.eq(18), "SOIL_COMPLETE", "SOIL_ALL_MISSING")
        frame["geology_block"] = frame["geology_dominant_class"].astype("string").fillna("<NA>")
        frame["exact_block_key"] = frame["geology_block"] + "|" + frame["matching_missingness_pattern_id"] + "|" + frame["soil_completeness_block"]
        availability = []
        for group, fields in active_by_group.items():
            availability.append(frame[[f"z__{f}" for f in fields]].notna().any(axis=1))
        frame["nonsoil_groups_available_flag"] = np.logical_and.reduce(availability)
    return pos, cand, param_frame, soil_inventory, active_by_group


def revised_field_registry(param_frame: pd.DataFrame, soil_inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(ORIGINAL_REGISTRY, encoding="utf-8-sig")
    keep = original.loc[
        original["actual_field_name"].notna() & original["actual_field_name"].ne("")
        & ~original["source_module"].eq("SOILGRIDS")
    ].copy()
    soil_rows = []
    for row in soil_inventory.itertuples(index=False):
        soil_rows.append({
            "actual_field_name": row.actual_field_name,
            "semantic_name": f"{row.semantic_property}_{row.depth_top_cm}_{row.depth_bottom_cm}cm",
            "source_module": "SOILGRIDS", "temporal_role": "PRE_EVENT_STATIC",
            "matching_role": "DISTANCE_RANKING", "transform": "FIELDWISE_ROBUST_SCALING",
            "group_name": "SOIL", "weight": 0.15,
            "missingness_rule": "SOIL_COMPLETE_TO_COMPLETE_OR_ALL_MISSING_TO_ALL_MISSING; NO_IMPUTATION",
            "model_input_eligibility": "YES", "leakage_reason": "", "audit_status": "PASS",
            "distance_contribution": "YES", "PRE_EVENT_OR_CAUSALLY_AVAILABLE": "YES",
        })
    registry = pd.concat([keep, pd.DataFrame(soil_rows)], ignore_index=True)
    active = set(param_frame.loc[param_frame["parameter_status"].eq("PASS"), "actual_field_name"])
    registry["revised_parameter_status"] = registry["actual_field_name"].map(lambda x: "PASS" if x in active else ("NOT_PARAMETERIZED" if x in {"geology_dominant_class", "matching_missingness_pattern_id", "spatial_distance_m", *TRIGGER_FIELDS} else "ZERO_IQR_EXCLUDED"))
    forbidden = pd.read_csv(ORIGINAL_FORBIDDEN, encoding="utf-8-sig")
    return registry, forbidden


def add_coordinates(pos: pd.DataFrame, cand: pd.DataFrame, pos_xy: pd.DataFrame, cand_xy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pos.merge(pos_xy, on="unit_id", validate="one_to_one"),
        cand.merge(cand_xy, on="unit_id", validate="one_to_one"),
    )


def build_edges(pos: pd.DataFrame, cand: pd.DataFrame, soil_inventory: pd.DataFrame, active_by_group: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    npos, ncand = len(pos), len(cand)
    stage_names = [
        "P1_CANDIDATE_POOL", "EVIDENCE_CLEAN_P1", "GEOLOGY_EXACT_BLOCK",
        "MISSINGNESS_AND_SOIL_BLOCK", "SLOPE_CALIPER", "ELEVATION_CALIPER",
        "NDVI_CALIPER", "SPATIAL_10KM_CALIPER", "INDIVIDUAL_RAIN_CALIPERS",
        "RAINFALL_GROUP_CALIPER", "FINAL_ELIGIBLE_EDGES",
    ]
    counts = np.zeros((npos, len(stage_names)), dtype=np.int64)
    counts[:, 0] = ncand
    counts[:, 1] = ncand
    geo_groups = {key: sub.index.to_numpy(dtype=np.int32) for key, sub in cand.groupby("geology_block", sort=False)}
    exact_key = cand["exact_block_key"].to_numpy()
    candidate_arrays = {c: pd.to_numeric(cand[c], errors="coerce").to_numpy(dtype=float) for c in ["dem_slope_mean", "dem_elevation_mean", "ndvi_pre_event"]}
    cand_x = cand["centroid_x"].to_numpy(dtype=float)
    cand_y = cand["centroid_y"].to_numpy(dtype=float)
    z_arrays = {f: cand[f"z__{f}"].to_numpy(dtype=float) for fs in active_by_group.values() for f in fs}
    soil_fields = soil_inventory["actual_field_name"].tolist()
    soil_z = {f: cand[f"z__{f}"].to_numpy(dtype=float) for f in soil_fields}
    property_fields = {
        prop: soil_inventory.loc[soil_inventory["semantic_property"].eq(prop), "actual_field_name"].tolist()
        for prop in sorted(soil_inventory["semantic_property"].unique())
    }
    chunks = []
    for pi, p in pos.iterrows():
        geo_idx = geo_groups.get(p["geology_block"], np.empty(0, dtype=np.int32))
        counts[pi, 2] = len(geo_idx)
        if len(geo_idx) == 0:
            continue
        idx = geo_idx[exact_key[geo_idx] == p["exact_block_key"]]
        if not bool(p["nonsoil_groups_available_flag"]):
            idx = np.empty(0, dtype=np.int32)
        counts[pi, 3] = len(idx)
        if len(idx) == 0:
            continue
        slope_diff = np.abs(candidate_arrays["dem_slope_mean"][idx] - float(p["dem_slope_mean"]))
        mask = slope_diff <= 5.0
        idx, slope_diff = idx[mask], slope_diff[mask]
        counts[pi, 4] = len(idx)
        if len(idx) == 0:
            continue
        elev_diff = np.abs(candidate_arrays["dem_elevation_mean"][idx] - float(p["dem_elevation_mean"]))
        mask = elev_diff <= 250.0
        idx, slope_diff, elev_diff = idx[mask], slope_diff[mask], elev_diff[mask]
        counts[pi, 5] = len(idx)
        if len(idx) == 0:
            continue
        ndvi_diff = np.abs(candidate_arrays["ndvi_pre_event"][idx] - float(p["ndvi_pre_event"]))
        mask = ndvi_diff <= 0.20
        idx, slope_diff, elev_diff, ndvi_diff = idx[mask], slope_diff[mask], elev_diff[mask], ndvi_diff[mask]
        counts[pi, 6] = len(idx)
        if len(idx) == 0:
            continue
        spatial = np.hypot(cand_x[idx] - float(p["centroid_x"]), cand_y[idx] - float(p["centroid_y"]))
        mask = spatial <= 10000.0
        idx, slope_diff, elev_diff, ndvi_diff, spatial = idx[mask], slope_diff[mask], elev_diff[mask], ndvi_diff[mask], spatial[mask]
        counts[pi, 7] = len(idx)
        if len(idx) == 0:
            continue
        rain_diffs = np.column_stack([np.abs(z_arrays[f][idx] - float(p[f"z__{f}"])) for f in RAIN_FIELDS])
        mask = np.all(rain_diffs <= 0.50, axis=1)
        idx, slope_diff, elev_diff, ndvi_diff, spatial, rain_diffs = idx[mask], slope_diff[mask], elev_diff[mask], ndvi_diff[mask], spatial[mask], rain_diffs[mask]
        counts[pi, 8] = len(idx)
        if len(idx) == 0:
            continue
        rain_group = rain_diffs.mean(axis=1)
        mask = rain_group <= 0.35
        idx, slope_diff, elev_diff, ndvi_diff, spatial, rain_diffs, rain_group = idx[mask], slope_diff[mask], elev_diff[mask], ndvi_diff[mask], spatial[mask], rain_diffs[mask], rain_group[mask]
        counts[pi, 9] = len(idx)
        counts[pi, 10] = len(idx)
        if len(idx) == 0:
            continue

        group_dist: dict[str, np.ndarray] = {}
        for group, fields in active_by_group.items():
            diffs = np.column_stack([np.abs(z_arrays[f][idx] - float(p[f"z__{f}"])) for f in fields])
            group_dist[group] = np.nanmean(diffs, axis=1)
        soil_missing = p["soil_completeness_block"] == "SOIL_ALL_MISSING"
        if soil_missing:
            soil_distance = np.full(len(idx), np.nan)
            soil_unavailable = np.ones(len(idx), dtype=np.int8)
            denominator = 1.0 - GROUP_WEIGHTS["SOIL"]
        else:
            prop_dists = []
            for fields in property_fields.values():
                diffs = np.column_stack([np.abs(soil_z[f][idx] - float(p[f"z__{f}"])) for f in fields])
                prop_dists.append(diffs.mean(axis=1))
            soil_distance = np.column_stack(prop_dists).mean(axis=1)
            soil_unavailable = np.zeros(len(idx), dtype=np.int8)
            denominator = 1.0
        composite = (
            GROUP_WEIGHTS["TERRAIN"] * group_dist["TERRAIN"]
            + GROUP_WEIGHTS["LANDCOVER_VEGETATION"] * group_dist["LANDCOVER_VEGETATION"]
            + GROUP_WEIGHTS["HYDRO_DISTANCE"] * group_dist["HYDRO_DISTANCE"]
            + GROUP_WEIGHTS["ACCESSIBILITY_DISTANCE"] * group_dist["ACCESSIBILITY_DISTANCE"]
            + GROUP_WEIGHTS["ANTECEDENT_RAINFALL"] * group_dist["ANTECEDENT_RAINFALL"]
            + (0.0 if soil_missing else GROUP_WEIGHTS["SOIL"] * soil_distance)
        ) / denominator
        edge = pd.DataFrame({
            "positive_unit_id": p["unit_id"], "candidate_unit_id": cand.iloc[idx]["unit_id"].to_numpy(),
            "positive_index": pi, "candidate_index": idx,
            "geology_dominant_class": p["geology_block"],
            "matching_missingness_pattern_id": p["matching_missingness_pattern_id"],
            "soil_completeness_block": p["soil_completeness_block"],
            "centroid_distance_m": spatial, "slope_abs_diff": slope_diff,
            "elevation_abs_diff_m": elev_diff, "ndvi_abs_diff": ndvi_diff,
            "rain_24h_robust_z_diff": rain_diffs[:, 0], "rain_72h_robust_z_diff": rain_diffs[:, 1],
            "rain_120h_robust_z_diff": rain_diffs[:, 2], "api_robust_z_diff": rain_diffs[:, 3],
            "rainfall_group_distance": rain_group, "terrain_distance": group_dist["TERRAIN"],
            "soil_distance": soil_distance, "landcover_vegetation_distance": group_dist["LANDCOVER_VEGETATION"],
            "hydrology_distance": group_dist["HYDRO_DISTANCE"], "accessibility_distance": group_dist["ACCESSIBILITY_DISTANCE"],
            "composite_distance": composite, "soil_group_unavailable_flag": soil_unavailable,
            "edge_eligible_flag": 1,
        })
        edge = edge.sort_values(["composite_distance", "candidate_unit_id"], kind="mergesort").reset_index(drop=True)
        edge["rank_within_positive"] = np.arange(1, len(edge) + 1, dtype=np.int32)
        chunks.append(edge)
    edges = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    attrition_rows = []
    previous = np.zeros(npos, dtype=np.int64)
    for j, stage in enumerate(stage_names):
        current = counts[:, j]
        removed = 0 if j == 0 else int(previous.sum() - current.sum())
        affected = 0 if j == 0 else int((current < previous).sum())
        attrition_rows.append({
            "stage_order": j + 1, "stage_name": stage,
            "pairs_remaining": int(current.sum()), "pairs_removed_at_stage": removed,
            "positives_with_any_edge_after_stage": int((current > 0).sum()),
            "positives_affected_at_stage": affected,
            "positives_zero_after_stage": int((current == 0).sum()),
        })
        previous = current
    return edges, pd.DataFrame(attrition_rows)


def maxflow_for_edges(edges: pd.DataFrame, k: int | None, npos: int, ncand: int) -> dict[str, Any]:
    subset = edges if k is None else edges.loc[edges["rank_within_positive"] <= k]
    if subset.empty:
        match = np.full(npos * 3, -1, dtype=np.int64)
    else:
        ep = subset["positive_index"].to_numpy(dtype=np.int32)
        ec = subset["candidate_index"].to_numpy(dtype=np.int32)
        rows = np.repeat(ep * 3, 3) + np.tile(np.arange(3, dtype=np.int32), len(ep))
        cols = np.repeat(ec, 3)
        graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(npos * 3, ncand))
        match = maximum_bipartite_matching(graph, perm_type="column")
    assignments = int((match >= 0).sum())
    per_positive = (match.reshape(npos, 3) >= 0).sum(axis=1)
    return {
        "K": "ALL" if k is None else k, "edge_count": int(len(subset)),
        "maxflow_assignments": assignments, "assignment_shortfall": REQUIRED_ASSIGNMENTS - assignments,
        "positives_fully_assignable": int((per_positive == 3).sum()),
        "positives_not_fully_assignable": int((per_positive < 3).sum()),
        "target_reached": "YES" if assignments == REQUIRED_ASSIGNMENTS else "NO",
        "match": match, "per_positive": per_positive,
    }


def feasibility_outputs(edges: pd.DataFrame, pos: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    degree = edges.groupby("positive_index").size().reindex(range(len(pos)), fill_value=0)
    degree_audit = pos[["unit_id", "geology_block", "matching_missingness_pattern_id", "soil_completeness_block"]].copy()
    degree_audit = degree_audit.rename(columns={"unit_id": "positive_unit_id", "geology_block": "geology_dominant_class"})
    degree_audit["eligible_candidate_count"] = degree.to_numpy()
    degree_audit["candidate_degree_bin"] = pd.cut(degree_audit["eligible_candidate_count"], bins=[-1, 0, 2, np.inf], labels=["ZERO", "ONE_TO_TWO", "THREE_OR_MORE"]).astype(str)

    comp = edges.groupby("candidate_index")["positive_index"].nunique().reindex(range(len(cand)), fill_value=0)
    competition = cand[["unit_id", "geology_block", "soil_completeness_block"]].copy()
    competition = competition.rename(columns={"unit_id": "candidate_unit_id", "geology_block": "geology_dominant_class"})
    competition["competing_positive_count"] = comp.to_numpy()
    competition["competition_bin"] = pd.cut(comp, bins=[-1, 0, 1, 2, 5, 10, 50, np.inf], labels=["0", "1", "2", "3-5", "6-10", "11-50", "51+"]).astype(str).to_numpy()

    flows = [maxflow_for_edges(edges, k, len(pos), len(cand)) for k in K_VALUES]
    all_flow = flows[-1]
    match = all_flow["match"]
    per_positive = all_flow["per_positive"]

    block_base = pos[["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"]].copy()
    block_base["assigned"] = per_positive
    pos_block = block_base.groupby(["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"], dropna=False).agg(positive_count=("assigned", "size"), maxflow_assignments=("assigned", "sum")).reset_index()
    cand_block = cand.groupby(["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"], dropna=False).size().rename("candidate_supply").reset_index()
    edge_block = edges.groupby(["geology_dominant_class", "soil_completeness_block", "matching_missingness_pattern_id"]).size().rename("edge_count").reset_index().rename(columns={"geology_dominant_class": "geology_block"})
    maxflow_block = pos_block.merge(cand_block, on=["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"], how="outer").merge(edge_block, on=["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"], how="outer").fillna(0)
    maxflow_block["required_assignments"] = maxflow_block["positive_count"] * 3
    maxflow_block["assignment_shortfall"] = maxflow_block["required_assignments"] - maxflow_block["maxflow_assignments"]
    maxflow_block["block_feasible_flag"] = np.where(maxflow_block["assignment_shortfall"].eq(0), "YES", "NO")

    if edges.empty:
        component = pd.DataFrame(columns=["component_id", "positive_nodes", "candidate_nodes", "edge_count", "demand", "supply", "maxflow_assignments", "shortfall", "component_feasible_flag"])
    else:
        A = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges["positive_index"], edges["candidate_index"])), shape=(len(pos), len(cand))).tocsr()
        adjacency = bmat([[None, A], [A.T, None]], format="csr")
        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        pos_labels = labels[:len(pos)]
        cand_labels = labels[len(pos):]
        edge_components = pos_labels[edges["positive_index"].to_numpy(dtype=int)]
        edge_counts = pd.Series(edge_components).value_counts().to_dict()
        rows = []
        for cid in sorted(set(pos_labels)):
            pidx = np.where(pos_labels == cid)[0]
            cidx = np.where(cand_labels == cid)[0]
            assigned = int(per_positive[pidx].sum())
            demand = len(pidx) * 3
            rows.append({
                "component_id": int(cid), "positive_nodes": len(pidx), "candidate_nodes": len(cidx),
                "edge_count": int(edge_counts.get(cid, 0)), "demand": demand, "supply": len(cidx),
                "maxflow_assignments": assigned, "shortfall": demand - assigned,
                "component_feasible_flag": "YES" if assigned == demand else "NO",
            })
        component = pd.DataFrame(rows)

    feasibility = {
        "positive_count": len(pos), "candidate_count": len(cand),
        "required_assignments": REQUIRED_ASSIGNMENTS,
        "eligible_edge_count": len(edges),
        "positives_with_zero_candidates": int((degree == 0).sum()),
        "positives_with_lt3_candidates": int((degree < 3).sum()),
        "positives_with_at_least3_candidates": int((degree >= 3).sum()),
        "maxflow_assignments": all_flow["maxflow_assignments"],
        "assignment_shortfall": all_flow["assignment_shortfall"],
        "positives_fully_assignable": all_flow["positives_fully_assignable"],
        "control_reuse_allowed": "NO", "max_control_use_count": 1,
        "k_tests": [{k: v for k, v in f.items() if k not in {"match", "per_positive"}} for f in flows],
    }
    return degree_audit, competition, component, maxflow_block, feasibility, flows


def protocol_tables(registry: pd.DataFrame, pos: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame([{"group_name": k, "weight": v, "weight_locked": "YES"} for k, v in GROUP_WEIGHTS.items()])
    calipers = pd.DataFrame([
        {"caliper": "slope_abs_diff", "threshold": 5.0, "unit": "degree", "operator": "<="},
        {"caliper": "elevation_abs_diff_m", "threshold": 250.0, "unit": "m", "operator": "<="},
        {"caliper": "ndvi_abs_diff", "threshold": 0.20, "unit": "unitless", "operator": "<="},
        {"caliper": "centroid_distance_m", "threshold": 10000.0, "unit": "m", "operator": "<="},
        {"caliper": "each_antecedent_rain_robust_z_diff", "threshold": 0.50, "unit": "robust_z", "operator": "<="},
        {"caliper": "mean_antecedent_rain_robust_z_diff", "threshold": 0.35, "unit": "robust_z", "operator": "<="},
    ])
    blocks = pd.concat([
        pos.groupby(["geology_block", "matching_missingness_pattern_id", "soil_completeness_block"]).size().rename("positive_count"),
        cand.groupby(["geology_block", "matching_missingness_pattern_id", "soil_completeness_block"]).size().rename("candidate_count"),
    ], axis=1).fillna(0).reset_index()
    blocks["exact_matching_required"] = "YES"
    return weights, calipers, blocks


def output_paths() -> dict[str, Path]:
    return {
        "input": DIRS["input"] / "revised_hard_control_01_input_registry.csv",
        "protocol": DIRS["protocol"] / "revised_hard_control_01_protocol.json",
        "registry": DIRS["protocol"] / "revised_hard_control_01_matching_field_registry.csv",
        "forbidden": DIRS["protocol"] / "revised_hard_control_01_forbidden_fields.csv",
        "params": DIRS["protocol"] / "revised_hard_control_01_transform_parameters.csv",
        "weights": DIRS["protocol"] / "revised_hard_control_01_group_weights.csv",
        "calipers": DIRS["protocol"] / "revised_hard_control_01_calipers.csv",
        "blocks": DIRS["protocol"] / "revised_hard_control_01_missingness_blocks.csv",
        "pos_cov": DIRS["covariates"] / "revised_hard_control_01_positive_covariates.parquet",
        "cand_cov": DIRS["covariates"] / "revised_hard_control_01_candidate_covariates.parquet",
        "edges": DIRS["edges"] / "revised_hard_control_01_eligible_edges.parquet",
        "attrition": DIRS["audit"] / "revised_hard_control_01_attrition_audit.csv",
        "degree": DIRS["feasibility"] / "revised_hard_control_01_positive_degree_audit.csv",
        "competition": DIRS["feasibility"] / "revised_hard_control_01_competition_audit.csv",
        "components": DIRS["feasibility"] / "revised_hard_control_01_component_feasibility.csv",
        "blocks_flow": DIRS["feasibility"] / "revised_hard_control_01_maxflow_by_block.csv",
        "flow": DIRS["feasibility"] / "revised_hard_control_01_maxflow_feasibility.json",
        "leakage": DIRS["audit"] / "revised_hard_control_01_leakage_audit.json",
        "determinism": DIRS["audit"] / "revised_hard_control_01_determinism_audit.json",
        "protected": DIRS["audit"] / "revised_hard_control_01_upstream_hash_audit.csv",
        "hashes": OUT / "revised_hard_control_01_file_hashes.csv",
        "report": OUT / "REVISED_HARD_CONTROL_01_REPORT.md",
        "marker": OUT / "REVISED_HARD_CONTROL_01_MATCHING_PROTOCOL_FROZEN.marker.json",
        "freeze_report": OUT / "revised_hard_control_01_matching_protocol_freeze_report.json",
    }


def build(run_id: str) -> dict[str, Any]:
    ensure_dirs()
    gate = gate_inputs()
    before = protected_hashes()
    paths = output_paths()
    inp = input_registry()
    positive, candidate, _, _, pos_xy, cand_xy = load_data()
    pos, cand, params, soil_inventory, active_by_group = transform_and_blocks(positive, candidate)
    pos, cand = add_coordinates(pos, cand, pos_xy, cand_xy)
    registry, forbidden = revised_field_registry(params, soil_inventory)
    weights, calipers, blocks = protocol_tables(registry, pos, cand)
    if abs(weights["weight"].sum() - 1.0) > EPS:
        raise RuntimeError("Group weight sum is not 1")

    edges, attrition = build_edges(pos, cand, soil_inventory, active_by_group)
    degree, competition, components, maxflow_blocks, feasibility, flows = feasibility_outputs(edges, pos, cand)
    after = protected_hashes()
    changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    protected = pd.DataFrame([
        {"absolute_path": p, "sha256_run_a_or_before": before.get(p), "sha256_current_or_after": after.get(p), "hash_changed_flag": "YES" if p in changed else "NO", "protected_read_only_flag": "YES"}
        for p in sorted(set(before) | set(after))
    ])

    leakage = {
        "future_rainfall_fields_used": 0, "trigger_fields_used_in_matching": 0,
        "leakage_fields_used": 0, "label_fields_used_in_matching": 0,
        "model_output_fields_used": 0, "coordinate_model_fields_created": 0,
        "coordinates_used_only_for_10km_caliper": "YES",
        "y_zero_created": 0, "hard_control_created": "NO", "pair_id_created": 0, "pair_table_created": "NO",
    }
    protocol = {
        "stage": "REVISED_HARD_CONTROL-01", "soil_distance_protocol_source": "HARD_CONTROL_01A",
        "event_anchor_utc": "2018-07-06T11:00:00Z", "event_anchor_jst": "2018-07-06T20:00:00+09:00",
        "main_control_candidate_pool": "P1_ONE_RING_GUARD", "positive_count": len(pos), "candidate_count": len(cand),
        "control_demand_per_positive": 3, "control_capacity": 1, "control_reuse_allowed": "NO",
        "exact_blocks": ["geology_dominant_class", "matching_missingness_pattern_id", "soil_completeness_block"],
        "soil_all_missing_rule": "ALLOW_SAME_BLOCK; SOIL_DISTANCE_NA; RENORMALIZE_OTHER_GROUP_WEIGHTS",
        "automatic_caliper_relaxation": "FORBIDDEN", "p0_substitution": "FORBIDDEN",
        "formal_pair_selection_performed": "NO", "hard_control_02_performed": "NO",
    }
    pass_flag = (
        feasibility["positives_with_zero_candidates"] == 0
        and feasibility["positives_with_lt3_candidates"] == 0
        and feasibility["maxflow_assignments"] == REQUIRED_ASSIGNMENTS
        and feasibility["positives_fully_assignable"] == EXPECTED_POSITIVE
        and not changed
    )
    decision = "PASS_REVISED_HARD_CONTROL_01_MATCHING_PROTOCOL_FROZEN" if pass_flag else "BLOCKED_REVISED_HARD_CONTROL_01_FULL_GRAPH_INFEASIBLE"
    status = {
        "REVISED_HARD_CONTROL_01_MATCHING_PROTOCOL_PASS": "YES" if pass_flag else "NO",
        "SOIL_DISTANCE_PROTOCOL_SOURCE": "HARD_CONTROL_01A",
        "MAIN_CONTROL_CANDIDATE_POOL": "P1_ONE_RING_GUARD", "POSITIVE_COUNT": len(pos),
        "MAIN_CONTROL_CANDIDATE_COUNT": len(cand), "REQUIRED_ASSIGNMENTS": REQUIRED_ASSIGNMENTS,
        "ELIGIBLE_EDGE_COUNT": len(edges), "POSITIVES_WITH_ZERO_CANDIDATES": feasibility["positives_with_zero_candidates"],
        "POSITIVES_WITH_LT3_CANDIDATES": feasibility["positives_with_lt3_candidates"],
        "MAXFLOW_ASSIGNMENTS": feasibility["maxflow_assignments"], "MAXFLOW_ASSIGNMENT_SHORTFALL": feasibility["assignment_shortfall"],
        "POSITIVES_FULLY_ASSIGNABLE": feasibility["positives_fully_assignable"],
        "CONTROL_REUSE_ALLOWED": "NO", "MAX_CONTROL_USE_COUNT": 1,
        "FUTURE_RAINFALL_FIELDS_USED": 0, "LEAKAGE_FIELDS_USED": 0,
        "Y_ZERO_CREATED": 0, "HARD_CONTROL_CREATED": "NO", "PAIR_ID_CREATED": 0, "PAIR_TABLE_CREATED": "NO",
        "UPSTREAM_FROZEN_ASSET_HASH_CHANGES": len(changed),
        "MATCHING_PROTOCOL_FROZEN": "YES" if pass_flag else "NO",
        "READY_FOR_HARD_CONTROL_02": "YES" if pass_flag else "NO",
        "ERRORS": 0 if pass_flag else 1, "FINAL_DECISION": decision,
    }

    write_csv(paths["input"], inp)
    write_json(paths["protocol"], protocol)
    write_csv(paths["registry"], registry)
    write_csv(paths["forbidden"], forbidden)
    write_csv(paths["params"], params)
    write_csv(paths["weights"], weights)
    write_csv(paths["calipers"], calipers)
    write_csv(paths["blocks"], blocks)
    pos.to_parquet(paths["pos_cov"], index=False)
    cand.to_parquet(paths["cand_cov"], index=False)
    edges.to_parquet(paths["edges"], index=False)
    write_csv(paths["attrition"], attrition)
    write_csv(paths["degree"], degree)
    write_csv(paths["competition"], competition)
    write_csv(paths["components"], components)
    write_csv(paths["blocks_flow"], maxflow_blocks)
    write_json(paths["flow"], feasibility)
    write_json(paths["leakage"], leakage)
    write_csv(paths["protected"], protected)

    soil_summary = maxflow_blocks.groupby("soil_completeness_block").agg(
        positive_count=("positive_count", "sum"), candidate_supply=("candidate_supply", "sum"),
        edge_count=("edge_count", "sum"), maxflow_assignments=("maxflow_assignments", "sum"),
        required_assignments=("required_assignments", "sum"), assignment_shortfall=("assignment_shortfall", "sum"),
    ).reset_index()
    geo_summary = maxflow_blocks.groupby("geology_block").agg(
        positive_count=("positive_count", "sum"), candidate_supply=("candidate_supply", "sum"),
        edge_count=("edge_count", "sum"), maxflow_assignments=("maxflow_assignments", "sum"),
        required_assignments=("required_assignments", "sum"), assignment_shortfall=("assignment_shortfall", "sum"),
    ).reset_index()
    write_csv(DIRS["feasibility"] / "revised_hard_control_01_maxflow_by_soil_block.csv", soil_summary)
    write_csv(DIRS["feasibility"] / "revised_hard_control_01_maxflow_by_geology_block.csv", geo_summary)
    k_lines = "\n".join(f"- K={f['K']}: edges={f['edge_count']:,}; maxflow={f['maxflow_assignments']:,}; shortfall={f['assignment_shortfall']:,}" for f in feasibility["k_tests"])
    report = f"""# Revised HARD_CONTROL-01 Matching Protocol Report

## Decision

`{decision}`

The revised protocol loaded the frozen HARD_CONTROL-01A hierarchical SoilGrids distance without rerunning or modifying HARD_CONTROL-01A. The original failed HARD_CONTROL-01 record remains unchanged.

## Counts

- Positives: {len(pos):,}
- P1 candidates: {len(cand):,}
- Complete eligible edges: {len(edges):,}
- Positives with zero candidates: {feasibility['positives_with_zero_candidates']:,}
- Positives with fewer than three candidates: {feasibility['positives_with_lt3_candidates']:,}
- Required assignments: {REQUIRED_ASSIGNMENTS:,}
- ALL-graph maximum flow: {feasibility['maxflow_assignments']:,}
- Fully assignable positives: {feasibility['positives_fully_assignable']:,}

## K sensitivity

{k_lines}

No caliper was relaxed, P0 was not substituted, candidate reuse was not enabled, and no final pair assignment, hard-control label, `y=0`, `pair_id`, or pair table was created.
"""
    paths["report"].write_text(report, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "revised_hard_control_01.py")

    core_snapshot = {
        "input_registry": semantic_hash(inp, ["absolute_path"]),
        "transform_parameters": semantic_hash(params, ["actual_field_name"]),
        "eligible_edges": semantic_hash(edges, ["positive_unit_id", "rank_within_positive", "candidate_unit_id"]),
        "positive_degree": semantic_hash(degree, ["positive_unit_id"]),
        "maxflow_by_block": semantic_hash(maxflow_blocks, ["geology_block", "soil_completeness_block", "matching_missingness_pattern_id"]),
        "feasibility": hashlib.sha256(json.dumps(feasibility, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else x).encode()).hexdigest(),
        "decision": hashlib.sha256(decision.encode()).hexdigest(),
    }
    snapshot_path = DIRS["audit"] / f"_revised_hard_control_01_run_{run_id.lower()}_snapshot.json"
    write_json(snapshot_path, core_snapshot)
    baseline_path = DIRS["audit"] / "_revised_hard_control_01_run_a_snapshot.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else core_snapshot
    differences = sorted(k for k in set(baseline) | set(core_snapshot) if baseline.get(k) != core_snapshot.get(k))
    determinism = {"run_id": run_id, "baseline": "A", "semantic_differences": len(differences), "different_outputs": differences, "determinism_pass": "YES" if not differences else "NO"}
    write_json(paths["determinism"], determinism)

    if pass_flag:
        write_json(paths["marker"], status)
        freeze_report = {**status, "protocol_path": str(paths["protocol"]), "eligible_edges_path": str(paths["edges"]), "report_path": str(paths["report"])}
        write_json(paths["freeze_report"], freeze_report)
        for p in [paths["protocol"], paths["registry"], paths["params"], paths["weights"], paths["calipers"], paths["blocks"], paths["flow"], paths["report"], paths["marker"], paths["freeze_report"]]:
            shutil.copy2(p, DIRS["frozen"] / p.name)
    else:
        write_json(OUT / "REVISED_HARD_CONTROL_01_STATUS.marker.json", status)

    manifest = paths["hashes"]
    frozen_manifest = DIRS["frozen"] / "revised_hard_control_01_frozen_hashes.csv"
    if pass_flag:
        frozen_rows = [{"absolute_path": str(p), "filename": p.name, "file_size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(DIRS["frozen"].iterdir()) if p.is_file() and p != frozen_manifest]
        write_csv(frozen_manifest, pd.DataFrame(frozen_rows))
    manifest_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p not in {manifest, frozen_manifest}:
            manifest_rows.append({"relative_path": p.relative_to(OUT).as_posix(), "file_size_bytes": p.stat().st_size, "sha256": sha256(p)})
    write_csv(manifest, pd.DataFrame(manifest_rows))
    status["RERUN_SEMANTIC_DIFFERENCES"] = len(differences)
    status["MANIFEST_SELF_REFERENCES"] = 0
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=["A", "B"], default="A")
    args = parser.parse_args()
    status = build(args.run_id)
    order = [
        "REVISED_HARD_CONTROL_01_MATCHING_PROTOCOL_PASS", "SOIL_DISTANCE_PROTOCOL_SOURCE",
        "MAIN_CONTROL_CANDIDATE_POOL", "POSITIVE_COUNT", "MAIN_CONTROL_CANDIDATE_COUNT",
        "REQUIRED_ASSIGNMENTS", "ELIGIBLE_EDGE_COUNT", "POSITIVES_WITH_ZERO_CANDIDATES",
        "POSITIVES_WITH_LT3_CANDIDATES", "MAXFLOW_ASSIGNMENTS", "MAXFLOW_ASSIGNMENT_SHORTFALL",
        "POSITIVES_FULLY_ASSIGNABLE", "CONTROL_REUSE_ALLOWED", "MAX_CONTROL_USE_COUNT",
        "FUTURE_RAINFALL_FIELDS_USED", "LEAKAGE_FIELDS_USED", "Y_ZERO_CREATED",
        "HARD_CONTROL_CREATED", "PAIR_ID_CREATED", "PAIR_TABLE_CREATED", "MATCHING_PROTOCOL_FROZEN",
        "READY_FOR_HARD_CONTROL_02", "ERRORS", "FINAL_DECISION",
    ]
    for key in order:
        print(f"{key}={status[key]}")


if __name__ == "__main__":
    main()
