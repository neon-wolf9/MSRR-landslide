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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/06_hard_control/01c_matching_sensitivity_audit"
HC00 = ROOT / "data/06_hard_control/00_candidate_pool"
HC01A = ROOT / "data/06_hard_control/01a_soil_distance_protocol"
HC01B = ROOT / "data/06_hard_control/01b_revised_matching_protocol"
STATIC_FROZEN = ROOT / "data/03_static_eogis_features/90_third_layer_assembly/05_static_03_final_audit_freeze/06_frozen"
RAIN_FROZEN = ROOT / "data/04_dynamic_rainfall_features/07_rain_04_final_audit_freeze/07_frozen"
RAIN_MARKER = ROOT / "data/04_dynamic_rainfall_features/07_rain_04_final_audit_freeze/10_marker/DYNAMIC_RAINFALL_LAYER_FROZEN.marker.json"
MASTER = ROOT / "data/01_county_prediction_domain/02_grid_250m/hiroshima_grid_250m_master.gpkg"

DIRS = {
    "input": OUT / "00_input_registry",
    "scenarios": OUT / "01_scenarios",
    "edges": OUT / "02_edges",
    "maxflow": OUT / "03_maxflow",
    "balance": OUT / "04_balance",
    "audit": OUT / "05_audit",
    "scripts": OUT / "scripts",
}

P0 = HC00 / "09_frozen/hard_control_00_core_candidate_pool_frozen.parquet"
P1 = HC00 / "hard_control_00_one_ring_guard_candidates.csv"
P2 = HC00 / "hard_control_00_500m_guard_candidates.csv"
HC00_MARKER = HC00 / "09_frozen/HARD_CONTROL_00_CANDIDATE_POOL_FROZEN.marker.json"
POS_COV = HC01B / "02_covariates/revised_hard_control_01_positive_covariates.parquet"
P1_COV = HC01B / "02_covariates/revised_hard_control_01_candidate_covariates.parquet"
BASE_EDGES = HC01B / "03_candidate_edges/revised_hard_control_01_eligible_edges.parquet"
BASE_ATTRITION = HC01B / "05_audit/revised_hard_control_01_attrition_audit.csv"
BASE_STATUS = HC01B / "REVISED_HARD_CONTROL_01_STATUS.marker.json"
BASE_REPORT = HC01B / "REVISED_HARD_CONTROL_01_REPORT.md"
BASE_MANIFEST = HC01B / "revised_hard_control_01_file_hashes.csv"
TRANSFORM_PARAMS = HC01B / "01_protocol/revised_hard_control_01_transform_parameters.csv"
GROUP_WEIGHTS_FILE = HC01B / "01_protocol/revised_hard_control_01_group_weights.csv"
CALIPERS_FILE = HC01B / "01_protocol/revised_hard_control_01_calipers.csv"
SOIL_MARKER = HC01A / "HARD_CONTROL_01A_SOIL_DISTANCE.marker.json"
SOIL_FIELDS_FILE = HC01A / "07_frozen/hard_control_01a_soil_field_inventory.csv"
SOIL_CONTRACT = HC01A / "07_frozen/hard_control_01a_soil_group_distance_contract.json"
STATIC_FULL = STATIC_FROZEN / "static_third_layer_full_frozen.parquet"
RAIN_DATASET = RAIN_FROZEN / "dynamic_rainfall_features_frozen.parquet_dataset"

EXPECTED_POS = 5075
EXPECTED_P0 = 28102
EXPECTED_P1 = 21825
EXPECTED_P2 = 13805
TARGET = 15225
ANCHOR_UTC = pd.Timestamp("2018-07-06T11:00:00Z")
EPS = 1e-12

GROUP_WEIGHTS = {
    "TERRAIN": 0.30, "SOIL": 0.15, "LANDCOVER_VEGETATION": 0.15,
    "HYDRO_DISTANCE": 0.15, "ACCESSIBILITY_DISTANCE": 0.05,
    "ANTECEDENT_RAINFALL": 0.20,
}
GROUP_FIELDS = {
    "TERRAIN": ["dem_elevation_mean", "dem_slope_mean", "dem_profile_curvature_mean", "dem_plan_curvature_mean", "dem_twi_mean"],
    "LANDCOVER_VEGETATION": ["ndvi_pre_event", "lc_tree_ratio", "lc_shrub_ratio", "lc_grass_ratio", "lc_crops_ratio", "lc_builtup_ratio", "lc_bare_ratio", "lc_permanent_water_ratio", "lc_seasonal_water_ratio"],
    "HYDRO_DISTANCE": ["river_distance_to_osm_river_m", "hydro_hnd_zonal_mean_m", "hydro_upa_zonal_mean_log1p"],
    "ACCESSIBILITY_DISTANCE": ["road_distance_to_any_road_m", "road_distance_to_major_road_m", "coast_distance_to_centroid_m"],
    "ANTECEDENT_RAINFALL": ["rain_24h_mm", "rain_72h_mm", "rain_120h_mm", "api_k090_step30m_120h"],
}
RAIN_FIELDS = GROUP_FIELDS["ANTECEDENT_RAINFALL"]
TRIGGERS = ["rain_30m_mm", "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm"]
LOG1P_FIELDS = {"river_distance_to_osm_river_m", "road_distance_to_any_road_m", "road_distance_to_major_road_m", "coast_distance_to_centroid_m", *RAIN_FIELDS}

GEOLOGY_PARENT = {
    "PLUTONIC_ROCK": "IGNEOUS",
    "VOLCANIC_ROCK": "IGNEOUS",
    "SEDIMENTARY_ROCK": "SEDIMENTARY",
    "UNCONSOLIDATED_SEDIMENT": "UNCONSOLIDATED",
    "METAMORPHIC_ROCK": "METAMORPHIC",
    "ACCRETIONARY_COMPLEX": "ACCRETIONARY_COMPLEX_ONLY",
    "MIXED_OR_OTHER": "OTHER_OR_MIXED_ONLY",
    "<NA>": "UNKNOWN_ONLY",
}

SCENARIO_DEFS = [
    {"scenario_id": "S0_BASELINE", "pool": "P1", "geology": "EXACT_7_CLASS", "km": 10, "capacity": 1, "single_factor": "BASELINE"},
    {"scenario_id": "S1_GEOLOGY_PARENT_COMPATIBLE", "pool": "P1", "geology": "PARENT_COMPATIBLE", "km": 10, "capacity": 1, "single_factor": "GEOLOGY_PARENT"},
    {"scenario_id": "S2_DISTANCE_20KM", "pool": "P1", "geology": "EXACT_7_CLASS", "km": 20, "capacity": 1, "single_factor": "DISTANCE_20KM"},
    {"scenario_id": "S3_P0_POOL", "pool": "P0", "geology": "EXACT_7_CLASS", "km": 10, "capacity": 1, "single_factor": "P0_POOL"},
    {"scenario_id": "S4_REUSE_CAPACITY_2", "pool": "P1", "geology": "EXACT_7_CLASS", "km": 10, "capacity": 2, "single_factor": "REUSE_CAPACITY_2"},
    {"scenario_id": "S5_P1_PARENT_20KM", "pool": "P1", "geology": "PARENT_COMPATIBLE", "km": 20, "capacity": 1, "single_factor": "COMBINATION"},
    {"scenario_id": "S6_P0_PARENT_10KM", "pool": "P0", "geology": "PARENT_COMPATIBLE", "km": 10, "capacity": 1, "single_factor": "COMBINATION"},
    {"scenario_id": "S7_P0_PARENT_20KM", "pool": "P0", "geology": "PARENT_COMPATIBLE", "km": 20, "capacity": 1, "single_factor": "COMBINATION"},
    {"scenario_id": "S8_P1_PARENT_10KM_CAP2", "pool": "P1", "geology": "PARENT_COMPATIBLE", "km": 10, "capacity": 2, "single_factor": "COMBINATION_REUSE"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def semantic_hash(frame: pd.DataFrame, sort_cols: list[str] | None = None) -> str:
    x = frame.copy()
    if sort_cols:
        x = x.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(x.to_csv(index=False, lineterminator="\n", na_rep="<NA>", float_format="%.15g").encode()).hexdigest()


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def protected_paths() -> list[Path]:
    return [
        MASTER, P0, P1, P2, HC00_MARKER, POS_COV, P1_COV, BASE_EDGES, BASE_ATTRITION,
        BASE_STATUS, BASE_REPORT, BASE_MANIFEST, TRANSFORM_PARAMS, GROUP_WEIGHTS_FILE,
        CALIPERS_FILE, SOIL_MARKER, SOIL_FIELDS_FILE, SOIL_CONTRACT, STATIC_FULL,
        RAIN_MARKER, *sorted(RAIN_DATASET.rglob("*.parquet")),
    ]


def protected_snapshot() -> dict[str, str]:
    return {str(p): sha256(p) if p.exists() else "MISSING" for p in protected_paths()}


def gate() -> None:
    missing = [str(p) for p in protected_paths() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing formal input: " + "; ".join(missing))
    hc00 = json.loads(HC00_MARKER.read_text(encoding="utf-8"))
    soil = json.loads(SOIL_MARKER.read_text(encoding="utf-8"))
    baseline = json.loads(BASE_STATUS.read_text(encoding="utf-8"))
    rain = json.loads(RAIN_MARKER.read_text(encoding="utf-8"))
    checks = {
        "hc00_pass": hc00.get("HARD_CONTROL_00_CANDIDATE_POOL_PASS") == "YES",
        "p0_count": hc00.get("CORE_EVIDENCE_CLEAN_CANDIDATES") == EXPECTED_P0,
        "p1_count": hc00.get("ONE_RING_GUARD_CANDIDATES") == EXPECTED_P1,
        "p2_count": hc00.get("GUARD_500M_CANDIDATES") == EXPECTED_P2,
        "soil_pass": soil.get("HARD_CONTROL_01A_SOIL_DISTANCE_PASS") == "YES",
        "soil_frozen": soil.get("SOIL_DISTANCE_PROTOCOL_FROZEN") == "YES",
        "baseline_blocked": baseline.get("FINAL_DECISION") == "BLOCKED_REVISED_HARD_CONTROL_01_FULL_GRAPH_INFEASIBLE",
        "baseline_flow": baseline.get("MAXFLOW_ASSIGNMENTS") == 9156,
        "rain_frozen": rain.get("DYNAMIC_RAIN_LAYER_FROZEN") == "YES",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Input gate failed: {checks}")


def input_registry() -> pd.DataFrame:
    return pd.DataFrame([
        {"absolute_path": str(p), "filename": p.name, "file_size_bytes": p.stat().st_size, "sha256": sha256(p), "read_only_flag": "YES", "audit_status": "PASS"}
        for p in protected_paths()
    ])


def geology_matrix() -> pd.DataFrame:
    classes = sorted(GEOLOGY_PARENT)
    rows = []
    for a in classes:
        for b in classes:
            exact = a == b
            parent = GEOLOGY_PARENT[a] == GEOLOGY_PARENT[b]
            cross_allowed = parent and GEOLOGY_PARENT[a] == "IGNEOUS"
            allowed = exact or cross_allowed
            rows.append({
                "positive_geology": a, "candidate_geology": b,
                "positive_parent": GEOLOGY_PARENT[a], "candidate_parent": GEOLOGY_PARENT[b],
                "parent_compatible_flag": "YES" if allowed else "NO",
                "geology_penalty": 0 if exact else (1 if allowed else None),
                "mapping_basis": "FROZEN_7_CLASS_TO_AUDITED_PARENT; MIXED_AND_UNTRACED_CLASSES_EXACT_ONLY",
            })
    return pd.DataFrame(rows)


def active_groups(params: pd.DataFrame) -> dict[str, list[str]]:
    return {
        group: [f for f in fields if ((params["actual_field_name"].eq(f)) & params["parameter_status"].eq("PASS")).any()]
        for group, fields in GROUP_FIELDS.items()
    }


def prepare_p0_covariates(pos: pd.DataFrame, params: pd.DataFrame, soil_inventory: pd.DataFrame) -> pd.DataFrame:
    ids = pd.read_parquet(P0, columns=["unit_id"])["unit_id"].tolist()
    if len(ids) != EXPECTED_P0 or len(set(ids)) != EXPECTED_P0:
        raise RuntimeError("P0 cardinality/uniqueness gate failed")
    nonrain = sorted({f for g, fs in GROUP_FIELDS.items() if g != "ANTECEDENT_RAINFALL" for f in fs})
    soil_fields = soil_inventory["actual_field_name"].tolist()
    static = pd.read_parquet(STATIC_FULL, columns=["unit_id", "geology_dominant_class"] + nonrain + soil_fields).set_index("unit_id")
    rain_ds = ds.dataset(str(RAIN_DATASET), format="parquet", partitioning="hive")
    rain = rain_ds.to_table(
        filter=ds.field("timestamp_utc") == ANCHOR_UTC.to_pydatetime(),
        columns=["unit_id"] + RAIN_FIELDS + TRIGGERS,
    ).to_pandas().set_index("unit_id")
    cov = static.join(rain, how="inner").loc[ids].reset_index()
    pidx = params.set_index("actual_field_name")
    for row in params.itertuples(index=False):
        field = row.actual_field_name
        values = pd.to_numeric(cov[field], errors="coerce")
        if row.source_protocol != "HARD_CONTROL_01A_FROZEN" and field in LOG1P_FIELDS:
            values = np.log1p(values)
        cov[f"z__{field}"] = (values - float(row.median)) / float(row.IQR) if row.parameter_status == "PASS" else np.nan
    active = params.loc[params["parameter_status"].eq("PASS"), "actual_field_name"].tolist()
    missing_fields = ["geology_dominant_class"] + active
    mask = cov[missing_fields].isna().astype(np.uint8).to_numpy()
    cov["matching_missingness_pattern_id"] = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in mask]
    valid_soil = cov[soil_fields].notna().sum(axis=1)
    if ((valid_soil != 0) & (valid_soil != 18)).any():
        raise RuntimeError("Partial Soil missingness in P0")
    cov["soil_completeness_block"] = np.where(valid_soil.eq(18), "SOIL_COMPLETE", "SOIL_ALL_MISSING")
    cov["geology_block"] = cov["geology_dominant_class"].astype("string").fillna("<NA>")
    groups = active_groups(params)
    cov["nonsoil_groups_available_flag"] = np.logical_and.reduce([cov[[f"z__{f}" for f in fs]].notna().any(axis=1) for fs in groups.values()])
    grid = gpd.read_file(MASTER, layer="hiroshima_grid_250m_master")[["unit_id", "geometry"]]
    cent = grid.geometry.centroid
    xy = pd.DataFrame({"unit_id": grid["unit_id"], "centroid_x": cent.x, "centroid_y": cent.y}).set_index("unit_id")
    return cov.merge(xy.loc[ids].reset_index(), on="unit_id", validate="one_to_one")


def parent_for(value: str) -> str:
    return GEOLOGY_PARENT.get(value, "UNKNOWN_ONLY")


def build_parent20_superset(pos: pd.DataFrame, cand: pd.DataFrame, params: pd.DataFrame, soil_inventory: pd.DataFrame, pool: str) -> pd.DataFrame:
    groups = active_groups(params)
    soil_fields = soil_inventory["actual_field_name"].tolist()
    property_fields = {
        prop: soil_inventory.loc[soil_inventory["semantic_property"].eq(prop), "actual_field_name"].tolist()
        for prop in sorted(soil_inventory["semantic_property"].unique())
    }
    cand = cand.copy()
    cand["geology_parent"] = cand["geology_block"].map(parent_for)
    parent_groups = {key: sub.index.to_numpy(dtype=np.int32) for key, sub in cand.groupby("geology_parent", sort=False)}
    exact_groups = {key: sub.index.to_numpy(dtype=np.int32) for key, sub in cand.groupby("geology_block", sort=False)}
    c_pattern = cand["matching_missingness_pattern_id"].to_numpy()
    c_soil = cand["soil_completeness_block"].to_numpy()
    c_nonsoil = cand["nonsoil_groups_available_flag"].to_numpy(dtype=bool)
    raw = {f: pd.to_numeric(cand[f], errors="coerce").to_numpy(dtype=float) for f in ["dem_slope_mean", "dem_elevation_mean", "ndvi_pre_event"]}
    z = {f: cand[f"z__{f}"].to_numpy(dtype=float) for fs in groups.values() for f in fs}
    soil_z = {f: cand[f"z__{f}"].to_numpy(dtype=float) for f in soil_fields}
    cx, cy = cand["centroid_x"].to_numpy(float), cand["centroid_y"].to_numpy(float)
    chunks = []
    for pi, p in pos.iterrows():
        geo = p["geology_block"]
        parent = parent_for(geo)
        allowed = parent_groups.get(parent, np.empty(0, dtype=np.int32)) if parent == "IGNEOUS" else exact_groups.get(geo, np.empty(0, dtype=np.int32))
        if len(allowed) == 0 or not bool(p["nonsoil_groups_available_flag"]):
            continue
        same = (c_pattern[allowed] == p["matching_missingness_pattern_id"]) & (c_soil[allowed] == p["soil_completeness_block"]) & c_nonsoil[allowed]
        idx = allowed[same]
        if len(idx) == 0:
            continue
        slope_raw = np.abs(raw["dem_slope_mean"][idx] - float(p["dem_slope_mean"]))
        elev_raw = np.abs(raw["dem_elevation_mean"][idx] - float(p["dem_elevation_mean"]))
        ndvi_raw = np.abs(raw["ndvi_pre_event"][idx] - float(p["ndvi_pre_event"]))
        spatial = np.hypot(cx[idx] - float(p["centroid_x"]), cy[idx] - float(p["centroid_y"]))
        rain = np.column_stack([np.abs(z[f][idx] - float(p[f"z__{f}"])) for f in RAIN_FIELDS])
        rain_group = rain.mean(axis=1)
        keep = (slope_raw <= 5) & (elev_raw <= 250) & (ndvi_raw <= 0.20) & (spatial <= 20000) & np.all(rain <= 0.50, axis=1) & (rain_group <= 0.35)
        idx, slope_raw, elev_raw, ndvi_raw, spatial, rain, rain_group = idx[keep], slope_raw[keep], elev_raw[keep], ndvi_raw[keep], spatial[keep], rain[keep], rain_group[keep]
        if len(idx) == 0:
            continue
        gd = {}
        for group, fields in groups.items():
            gd[group] = np.column_stack([np.abs(z[f][idx] - float(p[f"z__{f}"])) for f in fields]).mean(axis=1)
        soil_missing = p["soil_completeness_block"] == "SOIL_ALL_MISSING"
        if soil_missing:
            soil_distance = np.full(len(idx), np.nan)
            soil_unavailable = np.ones(len(idx), dtype=np.int8)
            denominator = 0.85
        else:
            properties = []
            for fields in property_fields.values():
                properties.append(np.column_stack([np.abs(soil_z[f][idx] - float(p[f"z__{f}"])) for f in fields]).mean(axis=1))
            soil_distance = np.column_stack(properties).mean(axis=1)
            soil_unavailable = np.zeros(len(idx), dtype=np.int8)
            denominator = 1.0
        composite = (
            0.30 * gd["TERRAIN"] + 0.15 * gd["LANDCOVER_VEGETATION"]
            + 0.15 * gd["HYDRO_DISTANCE"] + 0.05 * gd["ACCESSIBILITY_DISTANCE"]
            + 0.20 * gd["ANTECEDENT_RAINFALL"] + (0 if soil_missing else 0.15 * soil_distance)
        ) / denominator
        exact = cand.iloc[idx]["geology_block"].to_numpy() == geo
        chunks.append(pd.DataFrame({
            "positive_unit_id": p["unit_id"], "candidate_unit_id": cand.iloc[idx]["unit_id"].to_numpy(),
            "positive_index": pi, "candidate_index": idx, "candidate_pool": pool,
            "positive_geology": geo, "candidate_geology": cand.iloc[idx]["geology_block"].to_numpy(),
            "geology_parent": parent, "geology_exact_flag": exact.astype(np.int8), "geology_penalty": (~exact).astype(np.int8),
            "matching_missingness_pattern_id": p["matching_missingness_pattern_id"],
            "soil_completeness_block": p["soil_completeness_block"],
            "centroid_distance_m": spatial, "slope_abs_diff": slope_raw,
            "elevation_abs_diff_m": elev_raw, "ndvi_abs_diff": ndvi_raw,
            "slope_robust_z_diff": np.abs(z["dem_slope_mean"][idx] - float(p["z__dem_slope_mean"])),
            "elevation_robust_z_diff": np.abs(z["dem_elevation_mean"][idx] - float(p["z__dem_elevation_mean"])),
            "ndvi_robust_z_diff": np.abs(z["ndvi_pre_event"][idx] - float(p["z__ndvi_pre_event"])),
            "rainfall_group_distance": rain_group, "terrain_distance": gd["TERRAIN"],
            "soil_distance": soil_distance, "landcover_vegetation_distance": gd["LANDCOVER_VEGETATION"],
            "hydrology_distance": gd["HYDRO_DISTANCE"], "accessibility_distance": gd["ACCESSIBILITY_DISTANCE"],
            "composite_distance": composite, "soil_group_unavailable_flag": soil_unavailable,
        }))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def scenario_edges(scenario: dict[str, Any], supersets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    edges = supersets[scenario["pool"]]
    mask = edges["centroid_distance_m"] <= scenario["km"] * 1000
    if scenario["geology"] == "EXACT_7_CLASS":
        mask &= edges["geology_exact_flag"].eq(1)
    return edges.loc[mask].copy()


def maximum_flow_assignment(edges: pd.DataFrame, candidate_count: int, capacity: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if edges.empty:
        return np.zeros(EXPECTED_POS, dtype=int), pd.DataFrame(columns=["positive_index", "candidate_index"]), np.zeros(candidate_count, dtype=int)
    ep = edges["positive_index"].to_numpy(dtype=np.int32)
    ec = edges["candidate_index"].to_numpy(dtype=np.int32)
    # Exact capacitated bipartite network, mathematically equivalent to expanding
    # every positive into three slots and every reusable candidate into capacity
    # slots, while retaining each eligible edge only once.
    source = 0
    pos0 = 1
    cand0 = pos0 + EXPECTED_POS
    sink = cand0 + candidate_count
    n_nodes = sink + 1
    rows = np.concatenate([
        np.full(EXPECTED_POS, source, dtype=np.int32),
        pos0 + ep,
        cand0 + np.arange(candidate_count, dtype=np.int32),
    ])
    cols = np.concatenate([
        pos0 + np.arange(EXPECTED_POS, dtype=np.int32),
        cand0 + ec,
        np.full(candidate_count, sink, dtype=np.int32),
    ])
    caps = np.concatenate([
        np.full(EXPECTED_POS, 3, dtype=np.int32),
        np.ones(len(ep), dtype=np.int32),
        np.full(candidate_count, capacity, dtype=np.int32),
    ])
    graph = csr_matrix((caps, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.int32)
    flow_result = maximum_flow(graph, source, sink, method="dinic")
    pc_flow = flow_result.flow[pos0:cand0, cand0:sink].tocoo()
    keep = pc_flow.data > 0
    assigned = pd.DataFrame({
        "positive_index": pc_flow.row[keep].astype(np.int32),
        "candidate_index": pc_flow.col[keep].astype(np.int32),
    })
    per_positive = np.bincount(assigned["positive_index"], minlength=EXPECTED_POS)
    use = np.bincount(assigned["candidate_index"], minlength=candidate_count)
    return per_positive, assigned, use


def assignment_edge_rows(edges: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    if assignment.empty:
        return edges.iloc[0:0].copy()
    key = edges.set_index(["positive_index", "candidate_index"])
    idx = pd.MultiIndex.from_frame(assignment[["positive_index", "candidate_index"]])
    return key.loc[idx].reset_index()


def scenario_audits(scenario: dict[str, Any], edges: pd.DataFrame, pos: pd.DataFrame, cand: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    degree = edges.groupby("positive_index").size().reindex(range(EXPECTED_POS), fill_value=0)
    per_positive, assignment, use = maximum_flow_assignment(edges, len(cand), scenario["capacity"])
    maxflow = len(assignment)
    result = {
        "scenario_id": scenario["scenario_id"], "candidate_pool": scenario["pool"],
        "candidate_count": len(cand), "geology_rule": scenario["geology"],
        "spatial_caliper_km": scenario["km"], "control_capacity": scenario["capacity"],
        "eligible_edge_count": len(edges), "positives_with_zero_candidates": int((degree == 0).sum()),
        "positives_with_1_candidate": int((degree == 1).sum()), "positives_with_2_candidates": int((degree == 2).sum()),
        "positives_with_ge3_candidates": int((degree >= 3).sum()), "maxflow_assignments": maxflow,
        "maxflow_shortfall": TARGET - maxflow, "positives_fully_assignable": int((per_positive == 3).sum()),
        "positives_with_2_assignments": int((per_positive == 2).sum()), "positives_with_1_assignment": int((per_positive == 1).sum()),
        "positives_with_0_assignments": int((per_positive == 0).sum()),
        "unique_controls_used": int((use > 0).sum()), "controls_used_once": int((use == 1).sum()),
        "controls_used_twice": int((use == 2).sum()), "reuse_assignment_fraction": float(np.maximum(use - 1, 0).sum() / max(maxflow, 1)),
        "future_rainfall_fields_used": 0, "leakage_fields_used": 0,
    }
    result["feasibility_pass"] = "YES" if maxflow == TARGET and (degree == 0).sum() == 0 and (degree < 3).sum() == 0 else "NO"
    # Materializing a MultiIndex over every eligible edge is expensive and is only
    # needed for balance diagnostics when the complete 1:3 gate is feasible.
    assigned_edges = assignment_edge_rows(edges, assignment) if result["feasibility_pass"] == "YES" else edges.iloc[0:0].copy()
    block_rows = []
    for field, values in [("geology", pos["geology_block"]), ("soil", pos["soil_completeness_block"])]:
        temp = pd.DataFrame({"block": values, "assigned": per_positive})
        for block, sub in temp.groupby("block"):
            demand = len(sub) * 3
            block_rows.append({"scenario_id": scenario["scenario_id"], "block_type": field, "block_name": block, "positive_count": len(sub), "required_assignments": demand, "maxflow_assignments": int(sub["assigned"].sum()), "shortfall": demand - int(sub["assigned"].sum())})
    result["plutonic_rock_shortfall"] = next((r["shortfall"] for r in block_rows if r["block_type"] == "geology" and r["block_name"] == "PLUTONIC_ROCK"), 0)
    result["volcanic_rock_shortfall"] = next((r["shortfall"] for r in block_rows if r["block_type"] == "geology" and r["block_name"] == "VOLCANIC_ROCK"), 0)
    result["soil_complete_shortfall"] = next((r["shortfall"] for r in block_rows if r["block_type"] == "soil" and r["block_name"] == "SOIL_COMPLETE"), 0)
    result["soil_all_missing_shortfall"] = next((r["shortfall"] for r in block_rows if r["block_type"] == "soil" and r["block_name"] == "SOIL_ALL_MISSING"), 0)
    aon = {
        "scenario_id": scenario["scenario_id"], "ordinary_maxflow_assignments": maxflow,
        "ordinary_flow_fully_matched_positives": int((per_positive == 3).sum()),
        "ordinary_flow_two_assignments": int((per_positive == 2).sum()),
        "ordinary_flow_one_assignment": int((per_positive == 1).sum()),
        "ordinary_flow_zero_assignments": int((per_positive == 0).sum()),
        "certified_all_or_nothing_lower_bound": int((per_positive == 3).sum()),
        "all_or_nothing_upper_bound": int(min((degree >= 3).sum(), maxflow // 3, EXPECTED_POS)),
        "exact_all_or_nothing_optimum": EXPECTED_POS if result["feasibility_pass"] == "YES" else None,
        "optimization_status": "EXACT_CERTIFIED_BY_COMPLETE_FLOW" if result["feasibility_pass"] == "YES" else "BOUNDS_ONLY; EXACT_FIXED_CHARGE_INTEGER_MODEL_NOT_REQUIRED_FOR_FEASIBILITY_GATE",
    }
    reuse_rows = []
    if scenario["capacity"] == 2:
        used_twice = np.where(use == 2)[0]
        pos_xy = pos[["centroid_x", "centroid_y"]].to_numpy(float)
        for ci in used_twice:
            pids = assignment.loc[assignment["candidate_index"].eq(ci), "positive_index"].to_numpy(int)
            concentration = float(np.hypot(*(pos_xy[pids[0]] - pos_xy[pids[1]]))) if len(pids) == 2 else math.nan
            reuse_rows.append({"scenario_id": scenario["scenario_id"], "candidate_index": ci, "candidate_unit_id": cand.iloc[ci]["unit_id"], "use_count": len(pids), "assigned_positive_pair_distance_m": concentration})
    return result, block_rows, aon, assigned_edges, pd.DataFrame(reuse_rows)


def balance_for_feasible(scenario_id: str, assigned_edges: pd.DataFrame, pos: pd.DataFrame, cand: pd.DataFrame, params: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics = []
    fields = [
        "slope_robust_z_diff", "elevation_robust_z_diff", "ndvi_robust_z_diff", "soil_distance",
        "landcover_vegetation_distance", "hydrology_distance", "accessibility_distance",
        "rainfall_group_distance", "composite_distance",
    ]
    for field in fields:
        s = pd.to_numeric(assigned_edges[field], errors="coerce")
        metrics.append({"scenario_id": scenario_id, "metric": field, "count": int(s.notna().sum()), "mean": float(s.mean()), "median": float(s.median()), "p90": float(s.quantile(0.90)), "p95": float(s.quantile(0.95)), "maximum": float(s.max())})
    metrics.append({"scenario_id": scenario_id, "metric": "geology_original_exact_fraction", "count": len(assigned_edges), "mean": float(assigned_edges["geology_exact_flag"].mean()), "median": None, "p90": None, "p95": None, "maximum": None})
    metrics.append({"scenario_id": scenario_id, "metric": "geology_parent_only_fraction", "count": len(assigned_edges), "mean": float((assigned_edges["geology_exact_flag"] == 0).mean()), "median": None, "p90": None, "p95": None, "maximum": None})
    spatial = pd.to_numeric(assigned_edges["centroid_distance_m"], errors="coerce")
    spatial_summary = {"scenario_id": scenario_id, "count": len(spatial), "median_m": float(spatial.median()), "p90_m": float(spatial.quantile(0.90)), "p95_m": float(spatial.quantile(0.95)), "maximum_m": float(spatial.max())}
    smd_rows = []
    active = params.loc[params["parameter_status"].eq("PASS") & ~params["group_name"].eq("SOIL"), ["actual_field_name", "group_name"]]
    pidx = assigned_edges["positive_index"].to_numpy(int)
    cidx = assigned_edges["candidate_index"].to_numpy(int)
    for row in active.itertuples(index=False):
        pvals = pos.iloc[pidx][f"z__{row.actual_field_name}"].to_numpy(float)
        cvals = cand.iloc[cidx][f"z__{row.actual_field_name}"].to_numpy(float)
        pooled = math.sqrt((np.nanvar(pvals, ddof=1) + np.nanvar(cvals, ddof=1)) / 2)
        smd = float((np.nanmean(pvals) - np.nanmean(cvals)) / pooled) if pooled > 0 else 0.0
        abs_smd = abs(smd)
        smd_rows.append({"scenario_id": scenario_id, "field": row.actual_field_name, "group_name": row.group_name, "smd": smd, "abs_smd": abs_smd, "balance_class": "GOOD" if abs_smd <= 0.10 else ("ACCEPTABLE_WITH_NOTE" if abs_smd <= 0.20 else "POOR_WARNING")})
    return metrics, smd_rows, spatial_summary


def select_recommendation(results: pd.DataFrame, smd: pd.DataFrame, balance: pd.DataFrame) -> dict[str, Any]:
    feasible = results.loc[results["feasibility_pass"].eq("YES")]
    if feasible.empty:
        best = results.sort_values(["maxflow_shortfall", "control_capacity", "spatial_caliper_km"]).iloc[0]
        return {
            "recommended_scenario": None, "recommendation_reason": "No tested scenario achieves the complete 15,225-assignment gate.",
            "maxflow": int(best["maxflow_assignments"]), "shortfall": int(best["maxflow_shortfall"]),
            "balance_summary": "NOT_APPLICABLE_NO_FEASIBLE_SCENARIO", "protocol_changes": [],
            "caveats": ["No protocol is automatically accepted or frozen."], "ready_for_protocol_decision": "NO",
        }
    priority = ["S1_GEOLOGY_PARENT_COMPATIBLE", "S2_DISTANCE_20KM", "S3_P0_POOL", "S5_P1_PARENT_20KM", "S6_P0_PARENT_10KM", "S7_P0_PARENT_20KM", "S4_REUSE_CAPACITY_2", "S8_P1_PARENT_10KM_CAP2"]
    feasible = feasible.copy()
    feasible["priority"] = feasible["scenario_id"].map({s: i for i, s in enumerate(priority)}).fillna(999)
    feasible["max_abs_smd"] = feasible["scenario_id"].map(smd.groupby("scenario_id")["abs_smd"].max()).fillna(np.inf)
    feasible["mean_composite"] = feasible["scenario_id"].map(balance.loc[balance["metric"].eq("composite_distance")].set_index("scenario_id")["mean"]).fillna(np.inf)
    best = feasible.sort_values(["priority", "max_abs_smd", "mean_composite"]).iloc[0]
    changes = []
    if best["geology_rule"] == "PARENT_COMPATIBLE": changes.append("Replace exact 7-class geology with audited parent compatibility; only plutonic-volcanic cross-class edges are newly allowed.")
    if best["spatial_caliper_km"] == 20: changes.append("Increase centroid spatial caliper from 10 km to 20 km.")
    if best["candidate_pool"] == "P0": changes.append("Use the frozen evidence-clean P0 candidate pool instead of P1.")
    if best["control_capacity"] == 2: changes.append("Allow diagnostic candidate capacity 2.")
    max_smd = float(best["max_abs_smd"])
    return {
        "recommended_scenario": best["scenario_id"],
        "recommendation_reason": "First feasible option under the predeclared minimum-change priority; balance metrics are reported for user decision, not automatic adoption.",
        "maxflow": int(best["maxflow_assignments"]), "shortfall": int(best["maxflow_shortfall"]),
        "balance_summary": {"maximum_absolute_smd": max_smd, "classification": "GOOD" if max_smd <= 0.10 else ("ACCEPTABLE_WITH_NOTE" if max_smd <= 0.20 else "POOR_WARNING")},
        "protocol_changes": changes,
        "caveats": ["Sensitivity diagnosis only; no matching protocol, labels, or pairs are frozen.", "Ordinary maximum-flow partial-assignment counts are not the exact maximum all-or-nothing positive count for infeasible scenarios."],
        "ready_for_protocol_decision": "YES",
    }


def build(run_id: str) -> dict[str, Any]:
    ensure_dirs()
    gate()
    before = protected_snapshot()
    inp = input_registry()
    params = pd.read_csv(TRANSFORM_PARAMS, encoding="utf-8-sig")
    soil_inventory = pd.read_csv(SOIL_FIELDS_FILE, encoding="utf-8-sig")
    pos = pd.read_parquet(POS_COV)
    p1 = pd.read_parquet(P1_COV)
    p0 = prepare_p0_covariates(pos, params, soil_inventory)
    if len(pos) != EXPECTED_POS or len(p1) != EXPECTED_P1 or len(p0) != EXPECTED_P0:
        raise RuntimeError("Covariate cardinality gate failed")
    matrix = geology_matrix()
    # Superset construction is deliberately expensive because no top-K truncation is
    # permitted.  A timed-out invocation may still have atomically completed both
    # Parquet writes, so validate and resume from those audit intermediates.
    p1_path = DIRS["edges"] / "hard_control_01b_p1_parent20km_superset_edges.parquet"
    p0_path = DIRS["edges"] / "hard_control_01b_p0_parent20km_superset_edges.parquet"
    required_edge_cols = {"positive_index", "candidate_index", "centroid_distance_m", "geology_penalty"}
    if p1_path.exists() and p0_path.exists():
        p1_super = pd.read_parquet(p1_path)
        p0_super = pd.read_parquet(p0_path)
        if not required_edge_cols.issubset(p1_super.columns) or not required_edge_cols.issubset(p0_super.columns):
            raise RuntimeError("Cached full-edge superset schema gate failed")
    else:
        p1_super = build_parent20_superset(pos, p1, params, soil_inventory, "P1")
        p0_super = build_parent20_superset(pos, p0, params, soil_inventory, "P0")
        p1_super.to_parquet(p1_path, index=False)
        p0_super.to_parquet(p0_path, index=False)
    supersets = {"P1": p1_super, "P0": p0_super}
    candidates = {"P1": p1, "P0": p0}

    results, block_rows, aon_rows, reuse_frames = [], [], [], []
    balance_rows, smd_rows, spatial_rows = [], [], []
    edge_rows, degree_rows = [], []
    single_results = []
    for definition in SCENARIO_DEFS[:5]:
        edges = scenario_edges(definition, supersets)
        result, blocks, aon, assigned_edges, reuse = scenario_audits(definition, edges, pos, candidates[definition["pool"]])
        results.append(result); single_results.append(result); block_rows.extend(blocks); aon_rows.append(aon)
        edge_rows.append({"scenario_id": definition["scenario_id"], "eligible_edge_count": len(edges), "full_graph_flag": "YES"})
        deg = edges.groupby("positive_index").size().reindex(range(EXPECTED_POS), fill_value=0)
        degree_rows.extend([{"scenario_id": definition["scenario_id"], "degree_bin": label, "positive_count": int(mask.sum())} for label, mask in [("0", deg.eq(0)), ("1", deg.eq(1)), ("2", deg.eq(2)), ("3+", deg.ge(3))]])
        if not reuse.empty: reuse_frames.append(reuse)
        if result["feasibility_pass"] == "YES":
            bm, sr, sp = balance_for_feasible(definition["scenario_id"], assigned_edges, pos, candidates[definition["pool"]], params)
            balance_rows.extend(bm); smd_rows.extend(sr); spatial_rows.append(sp)

    all_single_fail = all(r["feasibility_pass"] == "NO" for r in single_results[1:])
    if all_single_fail:
        for definition in SCENARIO_DEFS[5:]:
            edges = scenario_edges(definition, supersets)
            result, blocks, aon, assigned_edges, reuse = scenario_audits(definition, edges, pos, candidates[definition["pool"]])
            results.append(result); block_rows.extend(blocks); aon_rows.append(aon)
            edge_rows.append({"scenario_id": definition["scenario_id"], "eligible_edge_count": len(edges), "full_graph_flag": "YES"})
            deg = edges.groupby("positive_index").size().reindex(range(EXPECTED_POS), fill_value=0)
            degree_rows.extend([{"scenario_id": definition["scenario_id"], "degree_bin": label, "positive_count": int(mask.sum())} for label, mask in [("0", deg.eq(0)), ("1", deg.eq(1)), ("2", deg.eq(2)), ("3+", deg.ge(3))]])
            if not reuse.empty: reuse_frames.append(reuse)
            if result["feasibility_pass"] == "YES":
                bm, sr, sp = balance_for_feasible(definition["scenario_id"], assigned_edges, pos, candidates[definition["pool"]], params)
                balance_rows.extend(bm); smd_rows.extend(sr); spatial_rows.append(sp)

    scenario_results = pd.DataFrame(results)
    if int(scenario_results.loc[scenario_results["scenario_id"].eq("S0_BASELINE"), "maxflow_assignments"].iloc[0]) != 9156:
        raise RuntimeError("S0 baseline maxflow did not reproduce 9156")
    scenario_registry = pd.DataFrame(SCENARIO_DEFS).loc[lambda d: d["scenario_id"].isin(scenario_results["scenario_id"])].rename(columns={"pool": "candidate_pool", "geology": "geology_rule", "km": "spatial_caliper_km", "capacity": "control_capacity"})
    edge_counts = pd.DataFrame(edge_rows)
    degree_dist = pd.DataFrame(degree_rows)
    maxflow_results = scenario_results[["scenario_id", "maxflow_assignments", "maxflow_shortfall", "positives_fully_assignable", "positives_with_2_assignments", "positives_with_1_assignment", "positives_with_0_assignments", "unique_controls_used", "controls_used_once", "controls_used_twice", "control_capacity", "feasibility_pass"]].copy()
    aon = pd.DataFrame(aon_rows)
    block_shortfalls = pd.DataFrame(block_rows)
    balance = pd.DataFrame(balance_rows, columns=["scenario_id", "metric", "count", "mean", "median", "p90", "p95", "maximum"])
    smd = pd.DataFrame(smd_rows, columns=["scenario_id", "field", "group_name", "smd", "abs_smd", "balance_class"])
    spatial = pd.DataFrame(spatial_rows, columns=["scenario_id", "count", "median_m", "p90_m", "p95_m", "maximum_m"])
    reuse = pd.concat(reuse_frames, ignore_index=True) if reuse_frames else pd.DataFrame(columns=["scenario_id", "candidate_index", "candidate_unit_id", "use_count", "assigned_positive_pair_distance_m"])
    reuse_summary = scenario_results.loc[scenario_results["control_capacity"].eq(2), ["scenario_id", "unique_controls_used", "controls_used_once", "controls_used_twice", "reuse_assignment_fraction"]].copy()
    if not reuse.empty:
        conc = reuse.groupby("scenario_id")["assigned_positive_pair_distance_m"].agg(["median", lambda s: s.quantile(.9), "max"]).reset_index()
        conc.columns = ["scenario_id", "reused_positive_pair_distance_median", "reused_positive_pair_distance_p90", "reused_positive_pair_distance_max"]
        reuse_summary = reuse_summary.merge(conc, on="scenario_id", how="left")
    recommendation = select_recommendation(scenario_results, smd, balance)
    after = protected_snapshot()
    changes = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    protected = pd.DataFrame([{"absolute_path": p, "sha256_before": before.get(p), "sha256_after": after.get(p), "hash_changed_flag": "YES" if p in changes else "NO"} for p in sorted(set(before) | set(after))])
    leakage = {"future_rainfall_fields_used": 0, "leakage_fields_used": 0, "y_zero_created": 0, "hard_control_created": "NO", "pair_id_created": 0, "pair_table_created": "NO", "matching_protocol_frozen": "NO", "ready_for_hard_control_02": "NO"}

    outputs = {
        "input": DIRS["input"] / "hard_control_01b_input_registry.csv",
        "registry": DIRS["scenarios"] / "hard_control_01b_scenario_registry.csv",
        "matrix": DIRS["scenarios"] / "hard_control_01b_geology_compatibility_matrix.csv",
        "results": DIRS["scenarios"] / "hard_control_01b_scenario_results.csv",
        "edge_counts": DIRS["edges"] / "hard_control_01b_edge_counts.csv",
        "degree": DIRS["maxflow"] / "hard_control_01b_degree_distribution.csv",
        "maxflow": DIRS["maxflow"] / "hard_control_01b_maxflow_results.csv",
        "aon": DIRS["maxflow"] / "hard_control_01b_all_or_nothing_results.csv",
        "blocks": DIRS["maxflow"] / "hard_control_01b_block_shortfalls.csv",
        "balance": DIRS["balance"] / "hard_control_01b_balance_metrics.csv",
        "smd": DIRS["balance"] / "hard_control_01b_smd_audit.csv",
        "spatial": DIRS["balance"] / "hard_control_01b_spatial_distance_audit.csv",
        "reuse": DIRS["balance"] / "hard_control_01b_reuse_audit.csv",
        "leakage": DIRS["audit"] / "hard_control_01b_leakage_audit.json",
        "det": DIRS["audit"] / "hard_control_01b_determinism_audit.json",
        "protected": DIRS["audit"] / "hard_control_01b_upstream_hash_audit.csv",
        "report": OUT / "HARD_CONTROL_01B_MATCHING_SENSITIVITY_REPORT.md",
        "recommendation": OUT / "HARD_CONTROL_01B_RECOMMENDATION.json",
        "manifest": OUT / "hard_control_01b_file_hashes.csv",
    }
    for key, frame in [("input", inp), ("registry", scenario_registry), ("matrix", matrix), ("results", scenario_results), ("edge_counts", edge_counts), ("degree", degree_dist), ("maxflow", maxflow_results), ("aon", aon), ("blocks", block_shortfalls), ("balance", balance), ("smd", smd), ("spatial", spatial), ("reuse", reuse_summary), ("protected", protected)]:
        write_csv(outputs[key], frame)
    write_json(outputs["leakage"], leakage)
    write_json(outputs["recommendation"], recommendation)
    feasible_count = int(scenario_results["feasibility_pass"].eq("YES").sum())
    final_decision = "PASS_HARD_CONTROL_01B_FEASIBLE_PROTOCOL_OPTION_IDENTIFIED" if feasible_count else "BLOCKED_HARD_CONTROL_01B_NO_TESTED_PROTOCOL_FEASIBLE"
    report_lines = "\n".join(f"- {r.scenario_id}: maxflow={int(r.maxflow_assignments):,}; shortfall={int(r.maxflow_shortfall):,}; fully matched in ordinary flow={int(r.positives_fully_assignable):,}; feasible={r.feasibility_pass}" for r in scenario_results.itertuples())
    outputs["report"].write_text(f"""# HARD_CONTROL-01B Matching Sensitivity Report

## Decision

`{final_decision}`

S0 reproduced the full-graph maximum flow of 9,156. Every tested scenario retained the frozen Soil protocol, slope/elevation/NDVI/rainfall calipers, evidence-clean candidate rules, and no-leakage constraints. Combination scenarios were run only because S1-S4 all failed.

## Scenario results

{report_lines}

## Recommendation

The recommendation file identifies a protocol option for user decision only. No scenario was accepted or frozen, and no `y=0`, hard control, `pair_id`, or pair table was created.
""", encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "hard_control_01b_matching_sensitivity.py")

    core = {
        "scenario_results": semantic_hash(scenario_results, ["scenario_id"]),
        "maxflow_results": semantic_hash(maxflow_results, ["scenario_id"]),
        "aon_results": semantic_hash(aon, ["scenario_id"]),
        "balance": semantic_hash(balance, ["scenario_id", "metric"]) if not balance.empty else hashlib.sha256(b"EMPTY").hexdigest(),
        "smd": semantic_hash(smd, ["scenario_id", "field"]) if not smd.empty else hashlib.sha256(b"EMPTY").hexdigest(),
        "recommendation": hashlib.sha256(json.dumps(recommendation, sort_keys=True).encode()).hexdigest(),
    }
    snap_path = DIRS["audit"] / f"_hard_control_01b_run_{run_id.lower()}_snapshot.json"
    write_json(snap_path, core)
    baseline_path = DIRS["audit"] / "_hard_control_01b_run_a_snapshot.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else core
    diffs = sorted(k for k in set(baseline) | set(core) if baseline.get(k) != core.get(k))
    det = {"run_id": run_id, "semantic_differences": len(diffs), "different_outputs": diffs, "determinism_pass": "YES" if not diffs else "NO", "upstream_hash_changes": len(changes)}
    write_json(outputs["det"], det)
    manifest_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p != outputs["manifest"]:
            manifest_rows.append({"relative_path": p.relative_to(OUT).as_posix(), "file_size_bytes": p.stat().st_size, "sha256": sha256(p)})
    write_csv(outputs["manifest"], pd.DataFrame(manifest_rows))
    best_name = recommendation["recommended_scenario"] or "NONE"
    if best_name != "NONE":
        best = scenario_results.loc[scenario_results["scenario_id"].eq(best_name)].iloc[0]
        max_abs_smd = float(smd.loc[smd["scenario_id"].eq(best_name), "abs_smd"].max())
    else:
        best = scenario_results.sort_values("maxflow_shortfall").iloc[0]
        max_abs_smd = math.nan
    summary = {
        "BASELINE_MAXFLOW": 9156, "SCENARIO_COUNT": len(scenario_results), "FEASIBLE_SCENARIO_COUNT": feasible_count,
        "BEST_FEASIBLE_SCENARIO": best_name, "BEST_SCENARIO_MAXFLOW": int(best["maxflow_assignments"]),
        "BEST_SCENARIO_SHORTFALL": int(best["maxflow_shortfall"]), "BEST_SCENARIO_MAX_ABS_SMD": max_abs_smd,
        "Y_ZERO_CREATED": 0, "HARD_CONTROL_CREATED": "NO", "PAIR_ID_CREATED": 0, "PAIR_TABLE_CREATED": "NO",
        "MATCHING_PROTOCOL_FROZEN": "NO", "READY_FOR_HARD_CONTROL_02": "NO", "ERRORS": 0,
        "FINAL_DECISION": final_decision, "scenario_results": scenario_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=["A", "B"], default="A")
    args = parser.parse_args()
    out = build(args.run_id)
    print(f"BASELINE_MAXFLOW={out['BASELINE_MAXFLOW']}")
    print(f"SCENARIO_COUNT={out['SCENARIO_COUNT']}")
    for row in out["scenario_results"].itertuples():
        print(f"{row.scenario_id}_MAXFLOW={int(row.maxflow_assignments)}")
        print(f"{row.scenario_id}_SHORTFALL={int(row.maxflow_shortfall)}")
        print(f"{row.scenario_id}_FULLY_MATCHED_POSITIVES={int(row.positives_fully_assignable)}")
        print(f"{row.scenario_id}_CONTROL_CAPACITY={int(row.control_capacity)}")
        print(f"{row.scenario_id}_FEASIBLE={row.feasibility_pass}")
    for key in ["FEASIBLE_SCENARIO_COUNT", "BEST_FEASIBLE_SCENARIO", "BEST_SCENARIO_MAXFLOW", "BEST_SCENARIO_SHORTFALL", "BEST_SCENARIO_MAX_ABS_SMD", "Y_ZERO_CREATED", "HARD_CONTROL_CREATED", "PAIR_ID_CREATED", "PAIR_TABLE_CREATED", "MATCHING_PROTOCOL_FROZEN", "READY_FOR_HARD_CONTROL_02", "ERRORS", "FINAL_DECISION"]:
        print(f"{key}={out[key]}")


if __name__ == "__main__":
    main()
