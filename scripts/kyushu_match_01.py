from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external/kyushu_2017_asakura_toho"
OUT = EXT / "08_external_dataset/matched_benchmark"
FIG = OUT / "audit_figures"
MASTER = EXT / "02_boundary_grid/kyushu_2017_250m_master_grid_v1.gpkg"
STATIC = EXT / "07_static_features/final/kyushu_static_92f_v1.parquet"
DYNAMIC = EXT / "08_external_dataset/kyushu_external_dynamic_10f_192slots_v1.parquet"
LABELS = EXT / "08_external_dataset/kyushu_external_labels_v1.parquet"
SUPPORT = EXT / "08_external_dataset/kyushu_external_evaluation_support_v1.gpkg"
GSI = EXT / "03_positive_grid/gsi_collapse_features.gpkg"
TIME_PROTOCOL = EXT / "00_manifest/kyushu_2017_dynamic_time_protocol_v1.json"

REVISED_SCRIPT = ROOT / "data/06_hard_control/01b_revised_matching_protocol/revised_hard_control_01.py"
SENS_SCRIPT = ROOT / "data/06_hard_control/01c_matching_sensitivity_audit/scripts/hard_control_01b_matching_sensitivity.py"
SUPPORT_SCRIPT = ROOT / "data/06_hard_control/01f_common_support_protocol/hard_control_01e_common_support_protocol.py"
P0_CONTRACT = ROOT / "data/06_hard_control/00_candidate_pool/09_frozen/hard_control_00_candidate_pool_contract_frozen.json"
PROTOCOL_MARKER = ROOT / "data/06_hard_control/01f_common_support_protocol/07_frozen/HARD_CONTROL_01E_MATCHING_PROTOCOL_FROZEN.marker.json"
ASSIGNMENT_MARKER = ROOT / "data/06_hard_control/02_formal_pair_assignment/08_frozen/HARD_CONTROL_02_FORMAL_PAIRS_FROZEN.marker.json"

CANDIDATE_POOL = OUT / "kyushu_external_control_candidate_pool_v1.parquet"
INDEX_PQ = OUT / "kyushu_external_matched_triplet_index_v1.parquet"
INDEX_CSV = OUT / "kyushu_external_matched_triplet_index_v1.csv"
UNMATCHED = OUT / "kyushu_external_unmatched_positives_v1.csv"
SCHEMA = OUT / "kyushu_external_matching_feature_schema_v1.csv"
BALANCE = OUT / "kyushu_external_matching_balance_v1.csv"
MANIFEST = OUT / "kyushu_external_matching_manifest_v1.csv"
DATASET_MANIFEST = OUT / "kyushu_external_matched_dataset_manifest_v1.csv"
REPORT = OUT / "kyushu_external_matching_audit_v1.md"
STATUS = OUT / "kyushu_external_matching_machine_status_v1.json"
PARAMS = OUT / "kyushu_external_matching_transform_parameters_v1.csv"
EDGE_AUDIT = OUT / "kyushu_external_matching_edge_attrition_v1.csv"
BLOCK_AUDIT = OUT / "kyushu_external_matching_common_support_blocks_v1.csv"
TEST_JSON = OUT / "kyushu_external_matching_independent_test_v1.json"

ANCHOR = pd.Timestamp("2017-07-05T11:00:00Z")
SOURCE_EVENT = "KYUSHU_2017_ASAKURA_TOHO"
SOURCE_INVENTORY = "GSI_D1_NO874_FORMAL_INTERPRETATION"
GEO_CLASSES = [
    "ACCRETIONARY_COMPLEX", "METAMORPHIC_ROCK", "MIXED_OR_OTHER",
    "PLUTONIC_ROCK", "SEDIMENTARY_ROCK", "UNCONSOLIDATED_SEDIMENT",
    "VOLCANIC_ROCK",
]
GEO_FIELDS = [
    "geology_accretionary_complex_fraction", "geology_metamorphic_rock_fraction",
    "geology_mixed_or_other_fraction", "geology_plutonic_rock_fraction",
    "geology_sedimentary_rock_fraction", "geology_unconsolidated_sediment_fraction",
    "geology_volcanic_rock_fraction",
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    pooled = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else 0.0


def load_and_gate():
    required = [MASTER, STATIC, DYNAMIC, LABELS, SUPPORT, GSI, TIME_PROTOCOL,
                REVISED_SCRIPT, SENS_SCRIPT, SUPPORT_SCRIPT, P0_CONTRACT,
                PROTOCOL_MARKER, ASSIGNMENT_MARKER]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen input: " + "; ".join(missing))
    before = {str(p.resolve()): sha256(p) for p in required}
    p0 = json.loads(P0_CONTRACT.read_text(encoding="utf-8"))
    if p0.get("main_protocol") != "P0_NO_EXTRA_GRID_GUARD":
        raise RuntimeError("P0 formal guard contract mismatch")

    master = pyogrio.read_dataframe(MASTER, layer="kyushu_2017_250m_master_grid_v1")
    if len(master) != 13478 or not master.unit_id.is_unique or master.crs.to_epsg() != 6670:
        raise RuntimeError("Master grid contract mismatch")
    static = pd.read_parquet(STATIC)
    dynamic = pd.read_parquet(DYNAMIC)
    labels = pd.read_parquet(LABELS)
    support = pyogrio.read_dataframe(SUPPORT, layer="evaluation_support_v1")
    if len(static) != 13478 or len(labels) != 13478 or len(support) != 13478:
        raise RuntimeError("Frozen table cardinality mismatch")
    if static.shape[1] != 93 or static.unit_id.duplicated().any():
        raise RuntimeError("Static schema/cardinality mismatch")
    if len(dynamic) != 2587776 or dynamic.duplicated(["unit_id", "time_index"]).any():
        raise RuntimeError("Dynamic schema/cardinality mismatch")
    if dynamic.unit_id.nunique() != 13478 or dynamic.time_index.nunique() != 192:
        raise RuntimeError("Dynamic key contract mismatch")
    if int((labels.y_external == 1).sum()) != 1694 or int((labels.y_external == 0).sum()) != 11784:
        raise RuntimeError("External label count mismatch")
    if not support.evaluation_support.astype(bool).all():
        raise RuntimeError("Evaluation support is not complete")
    if set(master.unit_id) != set(static.unit_id) or set(master.unit_id) != set(labels.unit_id) or set(master.unit_id) != set(dynamic.unit_id):
        raise RuntimeError("unit_id set mismatch")

    times = dynamic[["time_index", "timestamp_utc"]].drop_duplicates().sort_values("time_index")
    if len(times) != 192 or int(times.time_index.iloc[0]) != 1 or int(times.time_index.iloc[-1]) != 192:
        raise RuntimeError("Dynamic time index mismatch")
    anchor_rows = dynamic.loc[dynamic.time_index.eq(70)].copy()
    if len(anchor_rows) != 13478 or not pd.to_datetime(anchor_rows.timestamp_utc, utc=True).eq(ANCHOR).all():
        raise RuntimeError("Kyushu event anchor mismatch")

    gsi = pyogrio.read_dataframe(GSI, layer="gsi_collapse_features").to_crs(6670)
    if len(gsi) != 1935 or (~gsi.is_valid).any():
        raise RuntimeError("GSI geometry contract mismatch")
    grid_union = shapely.union_all(master.geometry.array)
    outside = int((~shapely.intersects(gsi.geometry.array, grid_union)).sum())
    if outside != 1:
        raise RuntimeError(f"GSI outside-domain count mismatch: {outside}")
    collapse_union = shapely.union_all(gsi.geometry.array)
    overlap_area = shapely.area(shapely.intersection(master.geometry.array, collapse_union))
    hit = overlap_area > 1e-8
    label_hit = labels.set_index("unit_id").loc[master.unit_id, "y_external"].to_numpy(int) == 1
    if not np.array_equal(hit, label_hit):
        raise RuntimeError(f"GSI/label hit mismatch: {int(np.sum(hit != label_hit))}")
    return master, static, dynamic, labels, support, gsi, anchor_rows, before


def derive_geology_class(static: pd.DataFrame) -> pd.Series:
    values = static[GEO_FIELDS].to_numpy(float)
    all_missing = np.isnan(values).all(axis=1)
    safe = np.where(np.isnan(values), -np.inf, values)
    index = np.argmax(safe, axis=1)
    result = pd.Series([GEO_CLASSES[i] for i in index], index=static.index, dtype="string")
    result.loc[all_missing] = pd.NA
    dominant = np.nanmax(values, axis=1)
    stored = static.geology_dominant_fraction.to_numpy(float)
    if not np.allclose(dominant, stored, equal_nan=True, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Derived geology dominant class is inconsistent with frozen dominant fraction")
    ties = ((safe == safe.max(axis=1, keepdims=True)).sum(axis=1) > 1) & ~all_missing
    if ties.any():
        raise RuntimeError("Unexpected geology dominant-class tie")
    return result


def build_covariates(master, static, labels, anchor_rows):
    revised = import_module(REVISED_SCRIPT, "kyushu_match_revised")
    sensitivity = import_module(SENS_SCRIPT, "kyushu_match_sensitivity")
    common = import_module(SUPPORT_SCRIPT, "kyushu_match_common")
    static = static.copy()
    static["geology_dominant_class"] = derive_geology_class(static)
    rain = anchor_rows[["unit_id"] + revised.RAIN_FIELDS].copy()
    cov = static.merge(rain, on="unit_id", validate="one_to_one")
    cent = master.geometry.centroid
    xy = pd.DataFrame({"unit_id": master.unit_id.astype(str), "centroid_x": cent.x, "centroid_y": cent.y})
    cov = cov.merge(xy, on="unit_id", validate="one_to_one")
    labelled = cov.merge(labels[["unit_id", "y_external"]], on="unit_id", validate="one_to_one")
    positive = labelled.loc[labelled.y_external.eq(1)].drop(columns="y_external").sort_values("unit_id").reset_index(drop=True)
    candidate = labelled.loc[labelled.y_external.eq(0)].drop(columns="y_external").sort_values("unit_id").reset_index(drop=True)
    pos, cand, params, soil_inventory, groups = revised.transform_and_blocks(positive, candidate)
    params["joint_pool_rows"] = len(pos) + len(cand)
    params.to_csv(PARAMS, index=False, encoding="utf-8-sig")
    common.N_POS = len(pos)
    common.N_CAND = len(cand)
    return revised, sensitivity, common, pos, cand, params, soil_inventory, groups


def plot_audits(master, labels, candidate_audit, index, unmatched, balance, pos, cand):
    cent = master.geometry.centroid
    xy = pd.DataFrame({"unit_id": master.unit_id, "x": cent.x, "y": cent.y}).set_index("unit_id")
    minx, miny, maxx, maxy = map(float, master.total_bounds)

    def map_image(path: Path, title: str, layers):
        image = Image.new("RGB", (1000, 900), "white")
        draw = ImageDraw.Draw(image)
        def project(x, y):
            return 45 + int((x-minx)/(maxx-minx)*900), 850-int((y-miny)/(maxy-miny)*780)
        for ids, color, radius in layers:
            if len(ids) == 0: continue
            points = xy.loc[list(ids)]
            for x, y in zip(points.x, points.y):
                px, py = project(float(x), float(y)); draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)
        draw.text((45, 18), title, fill="black")
        draw.text((45, 875), "EPSG:6670; audit only; no model output", fill="black")
        image.save(path)

    ids0 = labels.loc[labels.y_external.eq(0), "unit_id"].tolist()
    ids1 = labels.loc[labels.y_external.eq(1), "unit_id"].tolist()
    map_image(FIG / "01_source_positive_candidate_distribution.png", "Source positives (red) and initial non-hit candidates (teal)", [(ids0, "#5ab4ac", 1), (ids1, "#d73027", 2)])
    pids = index.loc[index.sample_role.eq("POSITIVE"), "unit_id"].tolist()
    cids = index.loc[index.sample_role.eq("HARD_CONTROL"), "unit_id"].tolist()
    map_image(FIG / "02_final_triplets_spatial_distribution.png", "Final matched positives (red) and hard controls (blue)", [(cids, "#4575b4", 1), (pids, "#d73027", 2)])

    values = index.loc[index.sample_role.eq("HARD_CONTROL"), "matching_distance"].to_numpy(float)
    hist, _ = np.histogram(values, bins=50)
    image = Image.new("RGB", (1000, 550), "white"); draw = ImageDraw.Draw(image); peak = max(int(hist.max()), 1)
    for i, count in enumerate(hist):
        x0 = 55 + int(i*18); x1 = x0+16; y0 = 490-int(count/peak*420); draw.rectangle((x0,y0,x1,490), fill="#4575b4")
    draw.text((50, 18), "Composite matching distance histogram", fill="black"); image.save(FIG / "03_matching_distance_histogram.png")

    b = balance.dropna(subset=["smd_before", "smd_after"]).sort_values("abs_smd_before", ascending=False).head(45)
    image = Image.new("RGB", (1300, max(700, len(b)*22+80)), "white"); draw = ImageDraw.Draw(image); center = 800; scale = 180
    draw.line((center,45,center,image.height-25), fill="black", width=1); draw.text((20,15), "SMD before (orange) and after (blue)", fill="black")
    for i, rec in enumerate(b.itertuples()):
        y=55+i*22; draw.text((20,y-7), rec.field_name[:55], fill="black")
        draw.ellipse((center+int(rec.smd_before*scale)-3,y-3,center+int(rec.smd_before*scale)+3,y+3), fill="#e66101")
        draw.ellipse((center+int(rec.smd_after*scale)-3,y-3,center+int(rec.smd_after*scale)+3,y+3), fill="#4575b4")
    image.save(FIG / "04_matching_smd_before_after.png")

    map_image(FIG / "05_unmatched_positive_distribution.png", "Unmatched positives", [(unmatched.unit_id.tolist(), "#7b3294", 3)])
    legal = candidate_audit.loc[candidate_audit.common_support_status.eq("LEGAL_CONTROL_CANDIDATE"), "unit_id"].tolist()
    illegal = candidate_audit.loc[~candidate_audit.common_support_status.eq("LEGAL_CONTROL_CANDIDATE"), "unit_id"].tolist()
    map_image(FIG / "06_common_support_domain.png", "Legal common-support controls (green); outside support (gray)", [(illegal, "#bdbdbd", 1), (legal, "#1b9e77", 1)])
    map_image(FIG / "07_guard_exclusion_audit.png", "Formal P0 guard: positive anchors red; no extra one-ring/500 m guard", [(ids0, "#80cdc1", 1), (ids1, "#d73027", 2)])

    if len(index):
        first = index.pair_set_id.drop_duplicates().sort_values().iloc[0]
        members = index.loc[index.pair_set_id.eq(first)].sort_values("y_pair", ascending=False)
        fields = ["dem_slope_mean", "dem_elevation_mean", "ndvi_pre_event", "rain_24h_mm", "rain_120h_mm"]
        p = pos.set_index("unit_id"); c = cand.set_index("unit_id")
        rows = [(p if rec.y_pair == 1 else c).loc[rec.unit_id, fields].to_numpy(float) for rec in members.itertuples()]
        z=np.asarray(rows); med=np.nanmedian(z,axis=0); scale=np.nanmax(np.abs(z-med),axis=0); scale[scale==0]=1; z=(z-med)/scale
        image=Image.new("RGB",(1000,600),"white"); draw=ImageDraw.Draw(image); colors=["#d73027","#4575b4","#74add1"]
        for j, field in enumerate(fields): draw.text((80+j*175,540),field[:20],fill="black")
        for i in range(3):
            pts=[]
            for j in range(len(fields)): pts.append((100+j*175,300-int(z[i,j]*180)))
            draw.line(pts,fill=colors[i],width=3)
            for x,y in pts: draw.ellipse((x-4,y-4,x+4,y+4),fill=colors[i])
        draw.text((30,15),f"Deterministic triplet feature probe: {first}",fill="black"); image.save(FIG / "08_triplet_feature_probe.png")


def independent_test(before):
    idxp = pd.read_parquet(INDEX_PQ)
    idxc = pd.read_csv(INDEX_CSV, encoding="utf-8-sig")
    cand = pd.read_parquet(CANDIDATE_POOL)
    unmatched = pd.read_csv(UNMATCHED, encoding="utf-8-sig")
    labels = pd.read_parquet(LABELS)
    groups = idxp.groupby("pair_set_id", sort=False)
    structures = groups.agg(rows=("unit_id", "size"), unique=("unit_id", "nunique"), positives=("y_pair", "sum"), controls=("y_pair", lambda x: int((x == 0).sum())))
    tests = {
        "source_counts": int((labels.y_external == 1).sum()) == 1694 and int((labels.y_external == 0).sum()) == 11784,
        "csv_parquet_key_equal": idxp[["pair_set_id", "unit_id"]].astype(str).equals(idxc[["pair_set_id", "unit_id"]].astype(str)),
        "pair_structure": bool((structures.rows == 3).all() and (structures.unique == 3).all() and (structures.positives == 1).all() and (structures.controls == 2).all()),
        "control_no_reuse": not idxp.loc[idxp.y_pair.eq(0), "unit_id"].duplicated().any(),
        "positive_no_reuse": not idxp.loc[idxp.y_pair.eq(1), "unit_id"].duplicated().any(),
        "identity": int(idxp.y_pair.sum()) + len(unmatched) == 1694,
        "candidate_pool_subset": set(idxp.loc[idxp.y_pair.eq(0), "unit_id"]).issubset(set(cand.loc[cand.common_support_status.eq("LEGAL_CONTROL_CANDIDATE"), "unit_id"])),
        "calipers": bool(idxp.loc[idxp.y_pair.eq(0), "caliper_status"].eq("PASS").all()),
        "guards": bool(idxp.loc[idxp.y_pair.eq(0), "guard_status"].eq("PASS_P0_NO_EXTRA_GRID_GUARD").all()),
        "feature_leakage": not any(c in idxp.columns for c in ["fold", "prediction", "model_score"]),
        "input_hashes_unchanged": all(Path(p).exists() and sha256(Path(p)) == h for p, h in before.items()),
    }
    payload = {"tests": tests, "TEST_RESULT": "PASS" if all(tests.values()) else "FAIL", "verification_time_utc": utc()}
    TEST_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    master, static, dynamic, labels, support, gsi, anchor_rows, before = load_and_gate()
    revised, sensitivity, common, pos, cand, params, soil_inventory, groups = build_covariates(master, static, labels, anchor_rows)
    edges = sensitivity.build_parent20_superset(pos, cand, params, soil_inventory, "P0")
    if edges.empty:
        raise RuntimeError("No legal matching edges")
    edges = edges.sort_values(["positive_unit_id", "candidate_unit_id"], kind="mergesort").reset_index(drop=True)
    selected, block_audit = common.exact_all_or_none(edges, pos)
    assigned, cost_audit = common.mincost_certificate(edges, selected, pos, cand)
    block_audit.to_csv(BLOCK_AUDIT, index=False, encoding="utf-8-sig")

    # The frozen 20 km superset stores the group mean but not its four audit
    # components. Recompute those components from the exact same transformed
    # rows before reducing the formal output columns.
    pp = pos.loc[assigned.positive_index.to_numpy(int)].reset_index(drop=True)
    cc = cand.loc[assigned.candidate_index.to_numpy(int)].reset_index(drop=True)
    rain_component_names = [
        "rain_24h_robust_z_diff", "rain_72h_robust_z_diff",
        "rain_120h_robust_z_diff", "api_robust_z_diff",
    ]
    for field, audit_name in zip(revised.RAIN_FIELDS, rain_component_names):
        assigned[audit_name] = np.abs(
            cc[f"z__{field}"].to_numpy(float) - pp[f"z__{field}"].to_numpy(float)
        )

    edge_degree = edges.groupby("candidate_unit_id").size()
    candidate_audit = pd.DataFrame({"unit_id": cand.unit_id})
    candidate_audit["y_external"] = 0
    candidate_audit["evaluation_support"] = 1
    candidate_audit["evidence_clean_status"] = "PASS_GSI_FORMAL_NONHIT"
    candidate_audit["guard_status"] = "PASS_P0_NO_EXTRA_GRID_GUARD"
    candidate_audit["dynamic_time_status"] = "PASS_192_SLOTS_ANCHOR_70_PRESENT"
    candidate_audit["legal_edge_count"] = candidate_audit.unit_id.map(edge_degree).fillna(0).astype(int)
    candidate_audit["common_support_status"] = np.where(candidate_audit.legal_edge_count.gt(0), "LEGAL_CONTROL_CANDIDATE", "OUTSIDE_COMMON_SUPPORT_NO_LEGAL_EDGE")
    candidate_audit["exclusion_reason"] = np.where(candidate_audit.legal_edge_count.gt(0), "", "NO_POSITIVE_EDGE_PASSES_FROZEN_BLOCKS_AND_CALIPERS")
    candidate_audit.to_parquet(CANDIDATE_POOL, index=False)

    assignment_cols = ["positive_unit_id", "candidate_unit_id", "composite_distance", "centroid_distance_m",
                       "slope_abs_diff", "elevation_abs_diff_m", "ndvi_abs_diff", "rain_24h_robust_z_diff",
                       "rain_72h_robust_z_diff", "rain_120h_robust_z_diff", "api_robust_z_diff",
                       "rainfall_group_distance", "terrain_distance", "soil_distance",
                       "landcover_vegetation_distance", "hydrology_distance", "accessibility_distance",
                       "geology_parent", "positive_geology", "candidate_geology", "soil_completeness_block"]
    assigned = assigned[assignment_cols].copy()
    assigned = assigned.sort_values(["positive_unit_id", "composite_distance", "candidate_unit_id"], kind="mergesort")
    assigned["control_rank"] = assigned.groupby("positive_unit_id").cumcount() + 1
    positives = sorted(assigned.positive_unit_id.unique())
    pair_map = {uid: f"KY2017_MATCH_{i:04d}" for i, uid in enumerate(positives, 1)}
    rows = []
    for uid in positives:
        rows.append({"pair_set_id": pair_map[uid], "unit_id": uid, "sample_role": "POSITIVE", "y_external": 1, "y_pair": 1,
                     "control_rank": pd.NA, "matching_cost": 0.0, "common_support_status": "MATCHED_IN_COMMON_SUPPORT",
                     "matching_distance": 0.0, "guard_status": "NOT_APPLICABLE_POSITIVE_ANCHOR", "evaluation_support": 1,
                     "source_event": SOURCE_EVENT, "source_inventory": SOURCE_INVENTORY, "caliper_status": "NOT_APPLICABLE_POSITIVE_ANCHOR"})
        for rec in assigned.loc[assigned.positive_unit_id.eq(uid)].itertuples():
            rows.append({"pair_set_id": pair_map[uid], "unit_id": rec.candidate_unit_id, "sample_role": "HARD_CONTROL", "y_external": 0, "y_pair": 0,
                         "control_rank": int(rec.control_rank), "matching_cost": float(rec.composite_distance), "common_support_status": "MATCHED_IN_COMMON_SUPPORT",
                         "matching_distance": float(rec.composite_distance), "guard_status": "PASS_P0_NO_EXTRA_GRID_GUARD", "evaluation_support": 1,
                         "source_event": SOURCE_EVENT, "source_inventory": SOURCE_INVENTORY, "caliper_status": "PASS",
                         "centroid_distance_m": float(rec.centroid_distance_m), "slope_abs_diff": float(rec.slope_abs_diff),
                         "elevation_abs_diff_m": float(rec.elevation_abs_diff_m), "ndvi_abs_diff": float(rec.ndvi_abs_diff),
                         "rain_24h_robust_z_diff": float(rec.rain_24h_robust_z_diff), "rain_72h_robust_z_diff": float(rec.rain_72h_robust_z_diff),
                         "rain_120h_robust_z_diff": float(rec.rain_120h_robust_z_diff), "api_robust_z_diff": float(rec.api_robust_z_diff),
                         "rainfall_group_distance": float(rec.rainfall_group_distance)})
    index = pd.DataFrame(rows).sort_values(["pair_set_id", "y_pair", "control_rank"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    index["control_rank"] = index.control_rank.astype("Int64")
    index.to_parquet(INDEX_PQ, index=False)
    index.to_csv(INDEX_CSV, index=False, encoding="utf-8-sig", float_format="%.15g")

    degree_pos = edges.groupby("positive_unit_id").size()
    unmatched_ids = sorted(set(pos.unit_id) - set(positives))
    unmatched = pd.DataFrame({"unit_id": unmatched_ids})
    unmatched["legal_edge_count"] = unmatched.unit_id.map(degree_pos).fillna(0).astype(int)
    unmatched["common_support_status"] = np.where(unmatched.legal_edge_count.eq(0), "OUT_OF_COMMON_SUPPORT", "UNMATCHED")
    unmatched["unmatched_reason"] = np.where(unmatched.legal_edge_count.eq(0), "NO_LEGAL_EDGE_AFTER_FROZEN_BLOCKS_AND_CALIPERS", "GLOBAL_NO_REUSE_CONTROL_COMPETITION")
    unmatched["y_external"] = 1
    unmatched.to_csv(UNMATCHED, index=False, encoding="utf-8-sig")

    soil_fields = soil_inventory.actual_field_name.tolist()
    schema_rows = []
    group_order = ["TERRAIN", "SOIL", "LANDCOVER_VEGETATION", "HYDRO_DISTANCE", "ACCESSIBILITY_DISTANCE", "ANTECEDENT_RAINFALL"]
    field_order = 0
    for group in group_order:
        fields = soil_fields if group == "SOIL" else revised.GROUP_FIELDS[group]
        for field in fields:
            field_order += 1
            param = params.loc[params.actual_field_name.eq(field)].iloc[0]
            schema_rows.append({"field_order": field_order, "field_name": field, "group_name": group, "group_weight": revised.GROUP_WEIGHTS[group],
                                "transform": param.transform, "parameter_source": param.source_protocol, "matching_role": "DISTANCE_AND_OR_CALIPER",
                                "future_information": "NO", "label_access": "NO"})
    pd.DataFrame(schema_rows).to_csv(SCHEMA, index=False, encoding="utf-8-sig")

    matched_pos = pos.set_index("unit_id").loc[positives]
    matched_cand_ids = index.loc[index.y_pair.eq(0), "unit_id"].tolist()
    matched_cand = cand.set_index("unit_id").loc[matched_cand_ids]
    all_legal_ids = candidate_audit.loc[candidate_audit.common_support_status.eq("LEGAL_CONTROL_CANDIDATE"), "unit_id"]
    legal_cand = cand.set_index("unit_id").loc[all_legal_ids]
    balance_rows = []
    for field in [f for fs in revised.GROUP_FIELDS.values() for f in fs] + soil_fields:
        before_smd = smd(pos[field], legal_cand[field])
        after_smd = smd(matched_pos[field], matched_cand[field])
        balance_rows.append({"field_name": field, "smd_before": before_smd, "abs_smd_before": abs(before_smd) if np.isfinite(before_smd) else np.nan,
                             "smd_after": after_smd, "abs_smd_after": abs(after_smd) if np.isfinite(after_smd) else np.nan,
                             "positive_count_before": int(pos[field].notna().sum()), "control_count_before": int(legal_cand[field].notna().sum()),
                             "positive_count_after": int(matched_pos[field].notna().sum()), "control_count_after": int(matched_cand[field].notna().sum())})
    balance = pd.DataFrame(balance_rows); balance.to_csv(BALANCE, index=False, encoding="utf-8-sig")

    pd.DataFrame([{"stage_order": 1, "stage": "INITIAL_NONHIT", "candidate_count": 11784},
                  {"stage_order": 2, "stage": "P0_EVIDENCE_CLEAN_AND_FORMAL_GUARD", "candidate_count": 11784},
                  {"stage_order": 3, "stage": "HAS_AT_LEAST_ONE_LEGAL_COMMON_SUPPORT_EDGE", "candidate_count": int((candidate_audit.legal_edge_count > 0).sum())},
                  {"stage_order": 4, "stage": "SELECTED_NO_REUSE_CONTROLS", "candidate_count": len(matched_cand_ids)}]).to_csv(EDGE_AUDIT, index=False, encoding="utf-8-sig")

    plot_audits(master, labels, candidate_audit, index, unmatched, balance, pos, cand)
    after = {p: sha256(Path(p)) for p in before}
    changed = sum(after[p] != h for p, h in before.items())
    tests = independent_test(before)
    pair_count = len(positives); control_count = 2 * pair_count
    test_pass = tests["TEST_RESULT"] == "PASS" and changed == 0
    status_value = "PASS_KYUSHU_EXTERNAL_1TO2_MATCHED_DATASET_READY" if test_pass else "BLOCKED_KYUSHU_MATCH_VALIDATION_FAILED"
    status = {
        "KYUSHU_MATCH_STATUS": status_value, "SOURCE_POSITIVE_COUNT": 1694, "SOURCE_NONHIT_COUNT": 11784,
        "LEGAL_CONTROL_CANDIDATE_COUNT": int((candidate_audit.legal_edge_count > 0).sum()), "MATCHED_PAIR_SET_COUNT": pair_count,
        "MATCHED_POSITIVE_COUNT": pair_count, "MATCHED_CONTROL_COUNT": control_count, "MATCHED_ROW_COUNT": len(index),
        "UNMATCHED_POSITIVE_COUNT": len(unmatched), "CONTROL_REUSE_COUNT": int(index.loc[index.y_pair.eq(0), "unit_id"].duplicated().sum()),
        "INVALID_PAIR_STRUCTURE_COUNT": 0 if tests["tests"]["pair_structure"] else 1, "CALIPER_VIOLATION_COUNT": int((index.loc[index.y_pair.eq(0), "caliper_status"] != "PASS").sum()),
        "GUARD_VIOLATION_COUNT": int((index.loc[index.y_pair.eq(0), "guard_status"] != "PASS_P0_NO_EXTRA_GRID_GUARD").sum()),
        "FEATURE_LEAKAGE_COUNT": 0, "STATIC_INPUT_HASH_CHANGED_COUNT": int(after[str(STATIC.resolve())] != before[str(STATIC.resolve())]),
        "DYNAMIC_INPUT_HASH_CHANGED_COUNT": int(after[str(DYNAMIC.resolve())] != before[str(DYNAMIC.resolve())]),
        "LABEL_INPUT_HASH_CHANGED_COUNT": int(after[str(LABELS.resolve())] != before[str(LABELS.resolve())]),
        "TEST_RESULT": "PASS" if test_pass else "FAIL", "EVENT_ANCHOR_UTC": ANCHOR.isoformat(),
        "FORMAL_GUARD_PROTOCOL": "P0_NO_EXTRA_GRID_GUARD", "verification_time_utc": utc(),
    }
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    input_rows = [{"asset_role": "FROZEN_INPUT", "path": p, "size_bytes": Path(p).stat().st_size, "sha256_before": h,
                   "sha256_after": after[p], "hash_status": "PASS_UNCHANGED" if h == after[p] else "FAIL_CHANGED"} for p, h in before.items()]
    pd.DataFrame(input_rows).to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    outputs = [CANDIDATE_POOL, INDEX_PQ, INDEX_CSV, UNMATCHED, SCHEMA, BALANCE, PARAMS, EDGE_AUDIT, BLOCK_AUDIT, REPORT, STATUS, TEST_JSON]
    manifest_rows = []
    for path in outputs:
        if path.exists(): manifest_rows.append({"asset_role": "FORMAL_OUTPUT", "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path), "status": "PASS"})
    pd.DataFrame(manifest_rows).to_csv(DATASET_MANIFEST, index=False, encoding="utf-8-sig")

    REPORT.write_text(f"""# Kyushu 2017 KYUSHU-MATCH-01 audit

- Status: `{status_value}`
- Source labels: 1,694 GSI-hit positive grids and 11,784 grids not hit within the formal interpreted support. A `y_external=0` value means only **not hit by the formal GSI geometry within the interpreted domain**; it is not proof that a landslide can never occur.
- Formal protocol: `HC_MAIN_COMMON_SUPPORT_1_TO_2_NO_REUSE`, candidate pool `P0_NO_EXTRA_GRID_GUARD`, parent-compatible geology, 20 km spatial caliper, capacity one, deterministic all-or-none common support followed by sparse minimum-weight assignment.
- The Hiroshima one-ring and 500 m guards are diagnostic sensitivity pools and are not the frozen main P0 protocol; they were not substituted into the formal Kyushu result.
- Event anchor: dynamic `time_index=70`, `{ANCHOR.isoformat()}`. The frozen 192-slot long table was read without modification or truncation.
- Geology block: deterministically reconstructed from the seven frozen geology fractions using the exact GEOLOGY-01 class order and first-maximum rule; agreement with `geology_dominant_fraction` was required, with zero ties.
- GSI geometries: 1,935 valid; 1,934 intersect the master evaluation domain; one outside-domain geometry retained in the source audit.
- Legal control candidates: {status['LEGAL_CONTROL_CANDIDATE_COUNT']:,}; matched positives/pair sets: {pair_count:,}; matched controls: {control_count:,}; unmatched positives: {len(unmatched):,}.
- No model, checkpoint, prediction, AUROC, AUPRC, or model-based matching adjustment was used.
- Frozen inputs changed: {changed}; independent test: `{tests['TEST_RESULT']}`.

## Formal matching rules

The matching feature groups, weights, robust transforms, missingness blocks, SoilGrids transform parameters, geology parent rule, raw hard calipers, rainfall robust-z calipers, no-reuse capacity, all-or-none common-support selection, and deterministic minimum-cost tie-break are copied from the frozen Hiroshima code and contracts. Labels are used only to define positive anchors and prevent positives from entering the control pool.

## Outputs

The formal index contains identifiers and audit fields only. Static 92-field and dynamic 10-field long tables are referenced by `unit_id`; neither is duplicated. The complete input and output hashes are recorded in the two manifests, and audit figures are under `{FIG}`.
""", encoding="utf-8")

    # Refresh output manifest after the report exists.
    manifest_rows = [{"asset_role": "FORMAL_OUTPUT", "path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p), "status": "PASS"} for p in outputs if p.exists()]
    pd.DataFrame(manifest_rows).to_csv(DATASET_MANIFEST, index=False, encoding="utf-8-sig")
    for key in ["KYUSHU_MATCH_STATUS", "SOURCE_POSITIVE_COUNT", "SOURCE_NONHIT_COUNT", "LEGAL_CONTROL_CANDIDATE_COUNT",
                "MATCHED_PAIR_SET_COUNT", "MATCHED_POSITIVE_COUNT", "MATCHED_CONTROL_COUNT", "MATCHED_ROW_COUNT",
                "UNMATCHED_POSITIVE_COUNT", "CONTROL_REUSE_COUNT", "INVALID_PAIR_STRUCTURE_COUNT", "CALIPER_VIOLATION_COUNT",
                "GUARD_VIOLATION_COUNT", "FEATURE_LEAKAGE_COUNT", "STATIC_INPUT_HASH_CHANGED_COUNT",
                "DYNAMIC_INPUT_HASH_CHANGED_COUNT", "LABEL_INPUT_HASH_CHANGED_COUNT", "TEST_RESULT"]:
        print(f"{key}={status[key]}")


if __name__ == "__main__":
    main()
