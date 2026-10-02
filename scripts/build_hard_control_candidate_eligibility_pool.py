#!/usr/bin/env python3
"""Build the audited 250 m hard-control *eligibility* pool.

This stage creates no negative samples, hard controls, labels, pairs, feature
matrices, rainfall matches, or static-environment screening decisions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUCCESS = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_COMPLETE"
FAILURE = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_REJECTED"
NEXT_STEP = "Await user confirmation of the candidate eligibility-pool results; do not freeze the pool or create hard controls."
CRS_EPSG = 6671
FORBIDDEN_FIELDS = {
    "label", "y", "negative_flag", "negative_sample", "hard_control_flag",
    "control_type", "pair_id", "positive_sample_id", "control_sample_id",
    "split", "fold_id",
}
STATUS_PRIORITY = [
    "INELIGIBLE_INVENTORY_EVIDENCE",
    "INELIGIBLE_COVERAGE_EXCLUSION",
    "INELIGIBLE_MASTER_GRID",
    "ELIGIBLE_BASE_FULL_RELIABLE",
    "REVIEW_PARTIAL_RELIABLE",
    "INELIGIBLE_OUTSIDE_INTERPRETED",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def yn(value: bool) -> str:
    return "YES" if value else "NO"


def scalar(value: Any) -> str:
    if value is None or value is pd.NA:
        return "<NULL>"
    try:
        if pd.isna(value):
            return "<NULL>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "<NULL>"
        return format(float(value), ".9g")
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    text = str(value).strip()
    return "<NULL>" if text == "" else text


def normalized_hash(frame: pd.DataFrame, include_geometry: bool = False) -> str:
    geometry_name = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else None
    attrs = [c for c in frame.columns if c != geometry_name]
    h = hashlib.sha256()
    h.update(("|".join(attrs) + "\n").encode("utf-8"))
    for row in frame[attrs].itertuples(index=False, name=None):
        h.update(("\x1f".join(scalar(v) for v in row) + "\n").encode("utf-8"))
    if include_geometry:
        if geometry_name is None:
            raise ValueError("Geometry hash requested for nonspatial table")
        h.update(b"<GEOMETRY>\n")
        for geom in frame[geometry_name]:
            h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
            h.update(b"\n")
    return h.hexdigest()


def geometry_hash(frame: gpd.GeoDataFrame) -> str:
    h = hashlib.sha256()
    for uid, geom in zip(frame["unit_id"].astype(str), frame.geometry):
        h.update((uid + "\x1f").encode("utf-8"))
        h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def verify_hash_rows(
    table: Path, base: Path, errors: list[str], only_paths: set[str] | None = None
) -> list[dict[str, Any]]:
    rows = read_csv(table).fillna("")
    details: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        raw = str(row.get("absolute_path", row.get("path", row.get("relative_path", ""))))
        path = Path(raw) if Path(raw).is_absolute() else base / raw.replace("\\", "/")
        if only_paths is not None and str(path) not in only_paths:
            continue
        if "a26" in str(path).lower():
            errors.append(f"Prohibited external A26 input listed; not read: {path}")
            continue
        expected_size_raw = row.get("size_bytes", row.get("file_size_bytes", ""))
        expected_size = int(expected_size_raw)
        expected_hash = str(row["sha256"]).lower()
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        passed = bool(exists and actual_size == expected_size and actual_hash == expected_hash)
        if not passed:
            errors.append(f"Registered input hash mismatch: {path}")
        details.append({
            "source_manifest": str(table), "input_role": row.get("role", row.get("file_role", "registered_input")),
            "absolute_path": str(path), "size_bytes": actual_size, "sha256": actual_hash,
            "hash_verified": yn(passed), "read_only": "YES",
        })
    return details


def add_explicit_inputs(paths: dict[str, Path], details: list[dict[str, Any]]) -> None:
    known = {item["absolute_path"] for item in details}
    for role, path in paths.items():
        if str(path) not in known:
            details.append({
                "source_manifest": "explicit_formal_input", "input_role": role,
                "absolute_path": str(path), "size_bytes": path.stat().st_size,
                "sha256": sha256(path), "hash_verified": "YES", "read_only": "YES",
            })


def unique_candidate(search_root: Path, expected: Path, label: str, errors: list[str]) -> None:
    candidates = [p.resolve() for p in search_root.rglob(expected.name) if p.is_file()]
    if len(candidates) != 1 or candidates[0] != expected.resolve():
        errors.append(f"Ambiguous {label} candidates: {[str(p) for p in candidates]}")


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, geometry: bool = False) -> bool:
    if geometry:
        a_geom = a.geometry.name if isinstance(a, gpd.GeoDataFrame) else "geometry"
        b_geom = b.geometry.name if isinstance(b, gpd.GeoDataFrame) else "geometry"
        a_attrs = [c for c in a.columns if c != a_geom]
        b_attrs = [c for c in b.columns if c != b_geom]
        columns_equal = a_attrs == b_attrs
    else:
        columns_equal = list(a.columns) == list(b.columns)
    return len(a) == len(b) and columns_equal and normalized_hash(a, geometry) == normalized_hash(b, geometry)


def write_attribute_pair(frame: pd.DataFrame, parquet: Path, csv_path: Path) -> dict[str, Any]:
    frame.to_parquet(parquet, index=False)
    write_csv(frame, csv_path)
    pq = pd.read_parquet(parquet)
    cs = read_csv(csv_path)
    return {
        "paths": [parquet, csv_path], "rows": len(frame),
        "normalized_content_sha256": normalized_hash(frame),
        "reload_pass": frames_equal(frame, pq) and frames_equal(frame, cs),
    }


def write_full_formats(
    frame: gpd.GeoDataFrame, attribute_parquet: Path, csv_path: Path,
    geoparquet: Path, gpkg: Path, layer: str,
) -> dict[str, Any]:
    attrs = pd.DataFrame(frame.drop(columns=frame.geometry.name))
    attrs.to_parquet(attribute_parquet, index=False)
    write_csv(attrs, csv_path)
    frame.to_parquet(geoparquet, index=False)
    frame.to_file(gpkg, layer=layer, driver="GPKG", index=False)
    pq = pd.read_parquet(attribute_parquet)
    cs = read_csv(csv_path)
    geo = gpd.read_parquet(geoparquet)
    gp = gpd.read_file(gpkg, layer=layer)
    passed = (
        frames_equal(attrs, pq) and frames_equal(attrs, cs)
        and geo.crs is not None and gp.crs is not None
        and geo.crs.to_epsg() == CRS_EPSG and gp.crs.to_epsg() == CRS_EPSG and geo.crs == gp.crs
        and frames_equal(frame, geo, True) and frames_equal(frame, gp, True)
    )
    return {
        "paths": [attribute_parquet, csv_path, geoparquet, gpkg], "rows": len(frame),
        "normalized_content_sha256": normalized_hash(frame, True),
        "geometry_sha256": geometry_hash(frame), "reload_pass": passed,
    }


def print_summary(report: dict[str, Any]) -> None:
    keys = [
        "PROJECT_ROOT", "PREVIOUS_FIRST_LAYER_FREEZE_PASS",
        "PREVIOUS_GSI_COVERAGE_MAPPING_PASS", "PREVIOUS_POSITIVE_REGISTRY_FREEZE_PASS",
        "MASTER_GRID_COUNT", "PRELIMINARY_ELIGIBLE_GRID_COUNT", "FULL_RELIABLE_GRID_COUNT",
        "PARTIAL_RELIABLE_GRID_COUNT", "COVERAGE_EXCLUSION_GRID_COUNT",
        "OUTSIDE_INTERPRETED_GRID_COUNT", "CONTROL_EXCLUSION_GRID_COUNT",
        "BASE_CANDIDATE_COUNT", "PARTIAL_REVIEW_COUNT",
        "INELIGIBLE_INVENTORY_EVIDENCE_COUNT", "INELIGIBLE_COVERAGE_EXCLUSION_COUNT",
        "INELIGIBLE_MASTER_GRID_COUNT", "INELIGIBLE_OUTSIDE_INTERPRETED_COUNT",
        "STATUS_PARTITION_PASS", "BASE_CANDIDATE_SET_IDENTITY_PASS",
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT",
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT",
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT",
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT",
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT", "UNIT_ID_INTEGRITY_PASS",
        "MASTER_GEOMETRY_UNCHANGED", "CROSS_FORMAT_RELOAD_PASS", "LABEL_FIELD_CREATED",
        "HARD_CONTROLS_CREATED", "PAIR_TABLE_CREATED", "RAINFALL_MATCHING_PERFORMED",
        "STATIC_EOGIS_SCREENING_PERFORMED", "EXTERNAL_A26_CONTENT_READ",
        "WARNINGS", "ERRORS", "FINAL_DECISION", "NEXT_STEP",
    ]
    for key in keys:
        value = report[key]
        if key == "ERRORS" and not value:
            value = "NONE"
        print(f"{key}: {value}")


def validate_existing(target: Path, explicit_paths: dict[str, Path]) -> dict[str, Any] | None:
    report_path = target / "06_audit/hard_control_candidate_build_report_v1.json"
    manifest_path = target / "00_manifest/hard_control_candidate_input_manifest_v1.csv"
    if not report_path.is_file() or not manifest_path.is_file():
        return None
    report = load_json(report_path)
    if report.get("FINAL_DECISION") != SUCCESS:
        return None
    manifest = read_csv(manifest_path)
    for row in manifest.to_dict("records"):
        path = Path(row["absolute_path"])
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            return None
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    first = root / "data/01_county_prediction_domain"
    reliable = root / "data/02_reliable_observation_domain"
    coverage_frozen = reliable / "03_gsi_coverage_frozen"
    mapping = reliable / "04_gsi_coverage_grid_mapping"
    positive = reliable / "09_positive_evidence_grid_registry_frozen"
    target = reliable / "10_hard_control_candidate_eligibility_pool"
    staging = reliable / "10_hard_control_candidate_eligibility_pool_staging"
    errors: list[str] = []
    warnings: list[str] = []
    differences: list[dict[str, Any]] = []

    paths = {
        "first_marker": first / "06_audit/FIRST_LAYER_FROZEN.marker.json",
        "first_report": first / "06_audit/first_layer_freeze_report.json",
        "first_manifest": first / "06_audit/first_layer_asset_manifest.csv",
        "master_grid": first / "02_grid_250m/hiroshima_grid_250m_master.gpkg",
        "coverage_marker": coverage_frozen / "04_audit/GSI_COVERAGE_FROZEN.marker.json",
        "coverage_report": coverage_frozen / "04_audit/gsi_coverage_freeze_report.json",
        "coverage_hashes": coverage_frozen / "04_audit/gsi_coverage_freeze_file_hashes.csv",
        "mapping_audit": mapping / "03_audit/gsi_coverage_grid_mapping_audit_v1.json",
        "mapping_hashes": mapping / "03_audit/gsi_coverage_grid_mapping_file_hashes_v1.csv",
        "mapping_attributes": mapping / "01_mapped_grid/gsi_coverage_mapping_250m_attributes_v1.csv",
        "mapping_spatial": mapping / "01_mapped_grid/gsi_coverage_mapping_250m_v1.gpkg",
        "positive_marker": positive / "03_audit/POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN.marker.json",
        "positive_report": positive / "03_audit/positive_evidence_grid_freeze_report_v1.json",
        "positive_hashes": positive / "00_manifest/positive_evidence_freeze_file_hashes_v1.csv",
        "primary_grids": positive / "01_frozen_registry/primary_positive_grid_registry_250m_frozen_v1.parquet",
        "affected_grids": positive / "01_frozen_registry/ajg_affected_grid_registry_250m_frozen_v1.parquet",
        "control_grids": positive / "01_frozen_registry/control_exclusion_grid_registry_250m_frozen_v1.parquet",
        "inventory": positive / "01_frozen_registry/inventory_evidence_registry_frozen_v1.parquet",
    }
    candidate_roots = {
        "first_marker": first / "06_audit", "coverage_marker": coverage_frozen / "04_audit",
        "mapping_audit": mapping / "03_audit", "positive_marker": positive / "03_audit",
    }
    for name, search in candidate_roots.items():
        unique_candidate(search, paths[name], name, errors)
    for role, path in paths.items():
        if not path.is_file():
            errors.append(f"Missing formal input {role}: {path}")
    if errors:
        print(f"ERRORS: {errors}")
        print(f"FINAL_DECISION: {FAILURE}")
        return 2

    first_marker = load_json(paths["first_marker"])
    first_report = load_json(paths["first_report"])
    coverage_marker = load_json(paths["coverage_marker"])
    coverage_report = load_json(paths["coverage_report"])
    mapping_audit = load_json(paths["mapping_audit"])
    positive_marker = load_json(paths["positive_marker"])
    positive_report = load_json(paths["positive_report"])
    previous = {
        "first": first_marker.get("status") == "FIRST_LAYER_FROZEN" and first_report.get("final_decision") == "FIRST_LAYER_FROZEN",
        "coverage": coverage_marker.get("final_decision") == "GSI_RELIABLE_COVERAGE_FROZEN" and coverage_report.get("final_decision") == "GSI_RELIABLE_COVERAGE_FROZEN",
        "mapping": mapping_audit.get("final_decision") == "GSI_COVERAGE_MAPPED_TO_GRID_250M" and not mapping_audit.get("errors"),
        "positive": positive_marker.get("current_FINAL_DECISION") == "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN" and positive_report.get("FINAL_DECISION") == "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN",
    }
    for name, passed in previous.items():
        if not passed:
            errors.append(f"Previous formal decision failed: {name}")

    input_details: list[dict[str, Any]] = []
    first_required = {str(paths["master_grid"]), str(paths["first_marker"]), str(paths["first_report"])}
    input_details.extend(verify_hash_rows(paths["first_manifest"], first, errors, first_required))
    input_details.extend(verify_hash_rows(paths["coverage_hashes"], coverage_frozen, errors))
    input_details.extend(verify_hash_rows(paths["mapping_hashes"], mapping, errors))
    input_details.extend(verify_hash_rows(paths["positive_hashes"], positive, errors))
    add_explicit_inputs(paths, input_details)
    input_df = pd.DataFrame(input_details).drop_duplicates("absolute_path").sort_values("absolute_path")

    if target.exists():
        existing = validate_existing(target, paths)
        if existing is None:
            print(f"ERRORS: Existing output is not a valid idempotent result: {target}")
            print(f"FINAL_DECISION: {FAILURE}")
            return 2
        print_summary(existing)
        return 0
    if staging.exists():
        print(f"ERRORS: Staging directory exists and will not be overwritten: {staging}")
        print(f"FINAL_DECISION: {FAILURE}")
        return 2

    master = gpd.read_file(paths["master_grid"], layer="hiroshima_grid_250m_master")
    map_attrs = read_csv(paths["mapping_attributes"])
    primary = pd.read_parquet(paths["primary_grids"])
    affected = pd.read_parquet(paths["affected_grids"])
    control = pd.read_parquet(paths["control_grids"])
    inventory = pd.read_parquet(paths["inventory"])

    master_order = master["unit_id"].astype(str).tolist()
    master_ids = set(master_order)
    if master_order != list(master.sort_values(["grid_row", "grid_col"], kind="mergesort")["unit_id"].astype(str)):
        errors.append("Master grid is not in canonical natural unit_id (grid_row, grid_col) order")
    mapping_ids = set(map_attrs["unit_id"].astype(str))
    primary_ids = set(primary["unit_id"].astype(str))
    affected_ids = set(affected["unit_id"].astype(str))
    control_ids = set(control["unit_id"].astype(str))
    if mapping_ids != master_ids:
        errors.append(f"Mapping/master unit ID difference: missing={len(master_ids-mapping_ids)}, extra={len(mapping_ids-master_ids)}")

    # Use the formal binary coverage semantics. Six sub-tolerance boundary
    # fragments were historically labeled FULL_RELIABLE despite zero
    # interpreted area and any_interpreted_flag=0; retain their IDs in audit.
    interpreted = map_attrs["any_interpreted_flag"].astype(int).eq(1)
    exclusion = map_attrs["any_exclusion_flag"].astype(int).eq(1)
    full = interpreted & ~exclusion & map_attrs["full_reliable_flag"].astype(int).eq(1)
    partial = interpreted & ~exclusion & ~full
    outside = ~interpreted
    map_attrs["gsi_coverage_class"] = np.select(
        [exclusion, outside, full, partial],
        ["COVERAGE_EXCLUSION", "OUTSIDE_INTERPRETED", "FULL_RELIABLE", "PARTIAL_RELIABLE"],
        default="UNCLASSIFIED",
    )
    tolerance_anomalies = map_attrs.loc[
        (map_attrs["coverage_status"] == "FULL_RELIABLE") & ~interpreted,
        ["unit_id", "analysis_area_m2", "interpreted_area_m2", "any_interpreted_flag", "coverage_status"],
    ].copy()
    if len(tolerance_anomalies):
        warnings.append(
            f"{len(tolerance_anomalies)} sub-tolerance boundary fragments have prior coverage_status=FULL_RELIABLE but formal any_interpreted_flag=0; classified OUTSIDE_INTERPRETED without modifying input"
        )

    coverage_counts = map_attrs["gsi_coverage_class"].value_counts().to_dict()
    expected_counts = {
        "MASTER_GRID_COUNT": (len(master), 139364),
        "PRELIMINARY_ELIGIBLE_GRID_COUNT": (int(pd.Series(master["valid_prediction_unit_preliminary"]).astype(int).eq(1).sum()), 135474),
        "FULL_RELIABLE_GRID_COUNT": (int(coverage_counts.get("FULL_RELIABLE", 0)), 38605),
        "PARTIAL_RELIABLE_GRID_COUNT": (int(coverage_counts.get("PARTIAL_RELIABLE", 0)), 849),
        "COVERAGE_EXCLUSION_GRID_COUNT": (int(coverage_counts.get("COVERAGE_EXCLUSION", 0)), 2041),
        "OUTSIDE_INTERPRETED_GRID_COUNT": (int(coverage_counts.get("OUTSIDE_INTERPRETED", 0)), 97869),
        "PRIMARY_POSITIVE_GRID_COUNT": (len(primary_ids), 5337),
        "AJG_AFFECTED_GRID_COUNT": (len(affected_ids), 8478),
        "CONTROL_EXCLUSION_GRID_COUNT": (len(control_ids), 8704),
        "GSI_EVIDENCE_COUNT": (int((inventory["source"].astype(str).str.upper() == "GSI").sum()), 11595),
        "INVENTORY_EVIDENCE_COUNT": (len(inventory), 19116),
    }
    for name, (actual, expected) in expected_counts.items():
        if actual != expected:
            errors.append(f"{name}={actual}, expected {expected}")
            differences.append({"check": name, "actual": actual, "expected": expected})
    if sum(coverage_counts.values()) != len(master) or set(coverage_counts) != {
        "FULL_RELIABLE", "PARTIAL_RELIABLE", "COVERAGE_EXCLUSION", "OUTSIDE_INTERPRETED"
    }:
        errors.append(f"Coverage classification partition failed: {coverage_counts}")

    selected_master = [
        "unit_id", "grid_row", "grid_col", "grid_size_m", "centroid_lon", "centroid_lat",
        "boundary_fraction", "valid_prediction_unit_preliminary", "water_source", "geometry",
    ]
    selected_mapping = [
        "unit_id", "interpreted_area_m2", "interpreted_fraction", "exclusion_area_m2",
        "exclusion_fraction", "reliable_area_m2", "reliable_fraction", "gsi_coverage_class",
    ]
    full_grid = master[selected_master].merge(
        map_attrs[selected_mapping], on="unit_id", how="left", validate="one_to_one", sort=False
    )
    full_grid = gpd.GeoDataFrame(full_grid, geometry="geometry", crs=master.crs)
    if full_grid["unit_id"].astype(str).tolist() != master_order:
        errors.append("Master input order was not preserved")

    control_flags = control[[
        "unit_id", "ajg_primary_positive_flag", "ajg_affected_flag", "gsi_evidence_flag",
    ]].copy()
    control_flags["control_exclusion_flag"] = "YES"
    full_grid = full_grid.merge(control_flags, on="unit_id", how="left", validate="one_to_one", sort=False)
    for col in ["ajg_primary_positive_flag", "ajg_affected_flag", "gsi_evidence_flag", "control_exclusion_flag"]:
        full_grid[col] = full_grid[col].fillna("NO").astype(str).str.upper()
    full_grid = full_grid.rename(columns={
        "valid_prediction_unit_preliminary": "preliminary_eligible_flag",
        "ajg_primary_positive_flag": "primary_positive_grid_flag",
    })
    full_grid["preliminary_eligible_flag"] = np.where(
        full_grid["preliminary_eligible_flag"].astype(int).eq(1), "YES", "NO"
    )
    full_grid["has_ajg_evidence"] = full_grid["ajg_affected_flag"]
    full_grid["has_gsi_evidence"] = full_grid["gsi_evidence_flag"]
    full_grid["inventory_evidence_exclusion_flag"] = full_grid["control_exclusion_flag"]

    evidence_mask = full_grid["control_exclusion_flag"].eq("YES") | full_grid["has_ajg_evidence"].eq("YES") | full_grid["has_gsi_evidence"].eq("YES")
    coverage_exclusion_mask = full_grid["gsi_coverage_class"].eq("COVERAGE_EXCLUSION")
    master_ineligible_mask = full_grid["preliminary_eligible_flag"].eq("NO")
    full_mask = full_grid["gsi_coverage_class"].eq("FULL_RELIABLE")
    partial_mask = full_grid["gsi_coverage_class"].eq("PARTIAL_RELIABLE")
    outside_mask = full_grid["gsi_coverage_class"].eq("OUTSIDE_INTERPRETED")
    base_mask = full_mask & ~master_ineligible_mask & ~evidence_mask
    review_mask = partial_mask & ~master_ineligible_mask & ~evidence_mask
    full_grid["primary_eligibility_status"] = np.select(
        [evidence_mask, coverage_exclusion_mask, master_ineligible_mask, base_mask, review_mask, outside_mask],
        STATUS_PRIORITY,
        default="UNCLASSIFIED",
    )

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if row["control_exclusion_flag"] == "YES": values.append("CONTROL_EXCLUSION")
        if row["primary_positive_grid_flag"] == "YES": values.append("PRIMARY_POSITIVE_GRID")
        if row["ajg_affected_flag"] == "YES": values.append("AJG_AFFECTED_GRID")
        if row["gsi_evidence_flag"] == "YES": values.append("GSI_EVIDENCE_GRID")
        if row["gsi_coverage_class"] == "COVERAGE_EXCLUSION": values.append("COVERAGE_EXCLUSION")
        if row["gsi_coverage_class"] == "OUTSIDE_INTERPRETED": values.append("OUTSIDE_INTERPRETED")
        if row["preliminary_eligible_flag"] == "NO": values.append("PRELIMINARY_INELIGIBLE")
        if row["gsi_coverage_class"] == "FULL_RELIABLE": values.append("FULL_RELIABLE")
        if row["gsi_coverage_class"] == "PARTIAL_RELIABLE": values.append("PARTIAL_RELIABLE")
        return ";".join(values)

    full_grid["eligibility_reason_codes"] = full_grid.apply(reasons, axis=1)
    full_grid["eligible_for_further_screening"] = np.where(base_mask, "YES", "NO")
    full_grid["partial_review_flag"] = np.where(review_mask, "YES", "NO")
    full_grid["water_audit_status"] = "PENDING_STATIC_EOGIS"
    full_grid["static_eogis_audit_status"] = "PENDING"
    full_grid["rainfall_matching_status"] = "NOT_PERFORMED"
    full_grid["final_control_status"] = "NOT_CREATED"
    full_grid = full_grid.drop(columns=["water_source"])

    candidate = full_grid[base_mask].copy()
    review = full_grid[review_mask].drop(columns="geometry").copy()
    ineligible = full_grid[full_grid["primary_eligibility_status"].str.startswith("INELIGIBLE_")].drop(columns="geometry").copy()
    status_counts = full_grid["primary_eligibility_status"].value_counts().to_dict()
    status_partition_pass = (
        len(status_counts) == 6 and set(status_counts) == set(STATUS_PRIORITY)
        and sum(status_counts.values()) == len(master)
        and not full_grid["primary_eligibility_status"].eq("UNCLASSIFIED").any()
    )
    expected_base_ids = (
        set(full_grid.loc[full_mask, "unit_id"].astype(str))
        & set(full_grid.loc[~master_ineligible_mask, "unit_id"].astype(str))
    ) - control_ids
    actual_base_ids = set(candidate["unit_id"].astype(str))
    base_identity_pass = actual_base_ids == expected_base_ids
    intersections = {
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(actual_base_ids & control_ids),
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT": len(actual_base_ids & primary_ids),
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT": len(actual_base_ids & affected_ids),
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT": len(actual_base_ids & set(control.loc[control["gsi_evidence_flag"] == "YES", "unit_id"].astype(str))),
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(set(review["unit_id"].astype(str)) & control_ids),
    }
    unit_integrity_pass = (
        len(full_grid) == 139364 and full_grid["unit_id"].notna().all()
        and not full_grid["unit_id"].duplicated().any()
        and set(full_grid["unit_id"].astype(str)) == master_ids
        and full_grid["unit_id"].astype(str).tolist() == master_order
    )
    if not status_partition_pass: errors.append(f"Primary eligibility status partition failed: {status_counts}")
    if not base_identity_pass: errors.append("Base candidate set identity failed")
    if any(intersections.values()): errors.append(f"Forbidden candidate/review intersections: {intersections}")
    if not unit_integrity_pass: errors.append("Full registry unit ID/order integrity failed")
    forbidden_created = sorted({c.lower() for c in full_grid.columns} & FORBIDDEN_FIELDS)
    if forbidden_created: errors.append(f"Forbidden fields created: {forbidden_created}")

    master_file_hash_before = sha256(paths["master_grid"])
    master_geometry_before = geometry_hash(master)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        dirs = [staging / name for name in (
            "00_manifest", "01_full_grid_registry", "02_candidate_pool", "03_review_pool",
            "04_ineligible_registry", "05_schema_and_rules", "06_audit", "07_preview",
        )]
        for d in dirs: d.mkdir(parents=True, exist_ok=True)
        manifest_dir, full_dir, candidate_dir, review_dir, ineligible_dir, schema_dir, audit_dir, preview_dir = dirs

        full_result = write_full_formats(
            full_grid,
            full_dir / "hard_control_eligibility_full_grid_250m_v1.parquet",
            full_dir / "hard_control_eligibility_full_grid_250m_v1.csv",
            full_dir / "hard_control_eligibility_full_grid_250m_v1.geoparquet",
            full_dir / "hard_control_eligibility_full_grid_250m_v1.gpkg",
            "hard_control_eligibility_full_grid_250m_v1",
        )
        candidate_result = write_full_formats(
            candidate,
            candidate_dir / "hard_control_base_candidate_pool_250m_v1.parquet",
            candidate_dir / "hard_control_base_candidate_pool_250m_v1.csv",
            candidate_dir / "hard_control_base_candidate_pool_250m_v1.geoparquet",
            candidate_dir / "hard_control_base_candidate_pool_250m_v1.gpkg",
            "hard_control_base_candidate_pool_250m_v1",
        )
        review_result = write_attribute_pair(
            review, review_dir / "partial_reliable_review_pool_250m_v1.parquet",
            review_dir / "partial_reliable_review_pool_250m_v1.csv",
        )
        ineligible_result = write_attribute_pair(
            ineligible, ineligible_dir / "hard_control_ineligible_grid_registry_250m_v1.parquet",
            ineligible_dir / "hard_control_ineligible_grid_registry_250m_v1.csv",
        )
        format_results = {
            "full_grid_registry": full_result, "base_candidate_pool": candidate_result,
            "partial_review_pool": review_result, "ineligible_registry": ineligible_result,
        }
        cross_format_pass = all(v["reload_pass"] for v in format_results.values())
        master_geometry_unchanged = (
            master_file_hash_before == sha256(paths["master_grid"])
            and master_geometry_before == geometry_hash(gpd.read_file(paths["master_grid"], layer="hiroshima_grid_250m_master"))
            and master_geometry_before == geometry_hash(full_grid)
        )
        if not cross_format_pass: errors.append("Cross-format reload/content consistency failed")
        if not master_geometry_unchanged: errors.append("Master grid file or geometry changed")

        rules = pd.DataFrame([
            {"priority": 1, "primary_eligibility_status": "INELIGIBLE_INVENTORY_EVIDENCE", "rule": "control_exclusion_flag=YES OR AJG/GSI evidence exists", "meaning": "Evidence exclusion; not a negative label"},
            {"priority": 2, "primary_eligibility_status": "INELIGIBLE_COVERAGE_EXCLUSION", "rule": "gsi_coverage_class=COVERAGE_EXCLUSION and no higher-priority evidence exclusion", "meaning": "Cloud/uninterpreted/quality exclusion"},
            {"priority": 3, "primary_eligibility_status": "INELIGIBLE_MASTER_GRID", "rule": "preliminary_eligible_flag=NO and no higher-priority exclusion", "meaning": "Fails first-layer preliminary eligibility"},
            {"priority": 4, "primary_eligibility_status": "ELIGIBLE_BASE_FULL_RELIABLE", "rule": "FULL_RELIABLE AND preliminary eligible AND no control exclusion/evidence", "meaning": "May enter later screening only; not a negative or hard control"},
            {"priority": 5, "primary_eligibility_status": "REVIEW_PARTIAL_RELIABLE", "rule": "PARTIAL_RELIABLE AND preliminary eligible AND no control exclusion/evidence", "meaning": "Sensitivity review only"},
            {"priority": 6, "primary_eligibility_status": "INELIGIBLE_OUTSIDE_INTERPRETED", "rule": "OUTSIDE_INTERPRETED and no higher-priority exclusion", "meaning": "Outside interpreted GSI coverage"},
        ])
        write_csv(rules, schema_dir / "hard_control_eligibility_rule_dictionary_v1.csv")
        reason_rows = [
            ("CONTROL_EXCLUSION", "Frozen AJG/GSI control exclusion grid"),
            ("PRIMARY_POSITIVE_GRID", "Frozen primary positive grid"),
            ("AJG_AFFECTED_GRID", "Frozen AJG affected grid"),
            ("GSI_EVIDENCE_GRID", "Frozen GSI evidence-linked grid"),
            ("COVERAGE_EXCLUSION", "GSI exclusion coverage"),
            ("OUTSIDE_INTERPRETED", "Outside interpreted coverage"),
            ("PRELIMINARY_INELIGIBLE", "First-layer preliminary eligibility is NO"),
            ("FULL_RELIABLE", "Full reliable GSI coverage"),
            ("PARTIAL_RELIABLE", "Partial reliable GSI coverage"),
        ]
        write_csv(pd.DataFrame(reason_rows, columns=["reason_code", "definition"]), schema_dir / "hard_control_exclusion_reason_dictionary_v1.csv")
        schema_rows = []
        leakage_terms = ("evidence", "eligibility", "candidate", "coverage", "exclusion", "status", "reason")
        for col in full_grid.columns:
            schema_rows.append({
                "field_name": col, "data_type": "geometry" if col == "geometry" else str(full_grid[col].dtype),
                "nullable": yn(bool(full_grid[col].isna().any())),
                "field_source": "master_grid" if col in selected_master else ("gsi_mapping_or_frozen_evidence" if any(x in col for x in ("coverage", "reliable", "interpreted", "exclusion", "evidence", "positive", "affected")) else "derived_rule"),
                "field_role": "PRIMARY_KEY" if col == "unit_id" else ("GEOMETRY" if col == "geometry" else "ELIGIBILITY_AUDIT"),
                "model_input_allowed": "NO" if col == "geometry" or col == "unit_id" or any(t in col.lower() for t in leakage_terms) else "REVIEW_REQUIRED",
                "audit_only": yn(col == "unit_id" or any(t in col.lower() for t in leakage_terms)),
                "leakage_risk": yn(col == "unit_id" or any(t in col.lower() for t in leakage_terms)),
            })
        write_csv(pd.DataFrame(schema_rows), schema_dir / "hard_control_candidate_schema_v1.csv")

        if len(tolerance_anomalies):
            write_csv(tolerance_anomalies, audit_dir / "coverage_status_tolerance_anomalies_v1.csv")
        else:
            write_csv(pd.DataFrame(columns=["unit_id", "analysis_area_m2", "interpreted_area_m2", "any_interpreted_flag", "coverage_status"]), audit_dir / "coverage_status_tolerance_anomalies_v1.csv")
        write_csv(pd.DataFrame(differences, columns=["check", "actual", "expected"]), audit_dir / "hard_control_candidate_differences_v1.csv")

        # A lightweight audit preview based on centroids only; no geometry mutation.
        colors = {
            "ELIGIBLE_BASE_FULL_RELIABLE": "#1a9850", "REVIEW_PARTIAL_RELIABLE": "#fee08b",
            "INELIGIBLE_INVENTORY_EVIDENCE": "#d73027", "INELIGIBLE_COVERAGE_EXCLUSION": "#f46d43",
            "INELIGIBLE_MASTER_GRID": "#8073ac", "INELIGIBLE_OUTSIDE_INTERPRETED": "#d9d9d9",
        }
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        for status in STATUS_PRIORITY:
            x = full_grid[full_grid["primary_eligibility_status"] == status]
            ax.scatter(x["centroid_lon"], x["centroid_lat"], s=0.12, c=colors[status], label=status, rasterized=True)
        ax.set_title("250 m hard-control base eligibility audit (not controls or labels)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(markerscale=12, fontsize=6, loc="best")
        fig.tight_layout(); fig.savefig(preview_dir / "hard_control_candidate_eligibility_preview_v1.png"); plt.close(fig)

        status_counts_complete = {status: int(status_counts.get(status, 0)) for status in STATUS_PRIORITY}
        report = {
            "PROJECT_ROOT": str(root),
            "PREVIOUS_FIRST_LAYER_FREEZE_PASS": yn(previous["first"]),
            "PREVIOUS_GSI_COVERAGE_FREEZE_PASS": yn(previous["coverage"]),
            "PREVIOUS_GSI_COVERAGE_MAPPING_PASS": yn(previous["mapping"]),
            "PREVIOUS_POSITIVE_REGISTRY_FREEZE_PASS": yn(previous["positive"]),
            "MASTER_GRID_COUNT": len(master),
            "PRELIMINARY_ELIGIBLE_GRID_COUNT": expected_counts["PRELIMINARY_ELIGIBLE_GRID_COUNT"][0],
            "FULL_RELIABLE_GRID_COUNT": coverage_counts.get("FULL_RELIABLE", 0),
            "PARTIAL_RELIABLE_GRID_COUNT": coverage_counts.get("PARTIAL_RELIABLE", 0),
            "COVERAGE_EXCLUSION_GRID_COUNT": coverage_counts.get("COVERAGE_EXCLUSION", 0),
            "OUTSIDE_INTERPRETED_GRID_COUNT": coverage_counts.get("OUTSIDE_INTERPRETED", 0),
            "CONTROL_EXCLUSION_GRID_COUNT": len(control_ids),
            "BASE_CANDIDATE_COUNT": len(candidate), "PARTIAL_REVIEW_COUNT": len(review),
            "INELIGIBLE_INVENTORY_EVIDENCE_COUNT": status_counts_complete["INELIGIBLE_INVENTORY_EVIDENCE"],
            "INELIGIBLE_COVERAGE_EXCLUSION_COUNT": status_counts_complete["INELIGIBLE_COVERAGE_EXCLUSION"],
            "INELIGIBLE_MASTER_GRID_COUNT": status_counts_complete["INELIGIBLE_MASTER_GRID"],
            "INELIGIBLE_OUTSIDE_INTERPRETED_COUNT": status_counts_complete["INELIGIBLE_OUTSIDE_INTERPRETED"],
            "STATUS_COUNTS": status_counts_complete,
            "STATUS_PARTITION_PASS": yn(status_partition_pass),
            "BASE_CANDIDATE_SET_IDENTITY_PASS": yn(base_identity_pass), **intersections,
            "UNIT_ID_INTEGRITY_PASS": yn(unit_integrity_pass),
            "MASTER_INPUT_ORDER_PRESERVED": yn(full_grid["unit_id"].astype(str).tolist() == master_order),
            "MASTER_GEOMETRY_UNCHANGED": yn(master_geometry_unchanged),
            "CROSS_FORMAT_RELOAD_PASS": yn(cross_format_pass),
            "LABEL_FIELD_CREATED": "NO", "HARD_CONTROLS_CREATED": "NO", "PAIR_TABLE_CREATED": "NO",
            "RAINFALL_MATCHING_PERFORMED": "NO", "STATIC_EOGIS_SCREENING_PERFORMED": "NO",
            "EXTERNAL_A26_CONTENT_READ": "NO", "RANDOM_NEGATIVE_SAMPLING_PERFORMED": "NO",
            "WARNINGS": warnings, "ERRORS": errors,
            "FINAL_DECISION": SUCCESS if not errors else FAILURE, "NEXT_STEP": NEXT_STEP,
        }
        consistency = {
            "coverage_class_counts": coverage_counts,
            "prior_mapping_status_counts": map_attrs["coverage_status"].value_counts().to_dict(),
            "tolerance_anomaly_entity_ids": tolerance_anomalies["unit_id"].astype(str).tolist(),
            "status_priority": STATUS_PRIORITY, "status_counts": status_counts_complete,
            "expected_base_set_sha256": hashlib.sha256("\n".join(sorted(expected_base_ids)).encode()).hexdigest(),
            "actual_base_set_sha256": hashlib.sha256("\n".join(sorted(actual_base_ids)).encode()).hexdigest(),
            "intersections": intersections, "unit_integrity_pass": unit_integrity_pass,
            "master_geometry_unchanged": master_geometry_unchanged,
            "cross_format_results": format_results,
            "forbidden_created_fields": forbidden_created,
            "errors": errors,
        }
        (audit_dir / "hard_control_candidate_consistency_audit_v1.json").write_text(
            json.dumps(consistency, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        build_report = {
            **report, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "classification_definition": "FULL_RELIABLE ∩ PRELIMINARY_ELIGIBLE − CONTROL_EXCLUSION by exact unit_id",
            "normalized_content_hashes": {name: value["normalized_content_sha256"] for name, value in format_results.items()},
        }
        (audit_dir / "hard_control_candidate_build_report_v1.json").write_text(
            json.dumps(build_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        md = [
            "# 250 m Hard-Control Base Candidate Eligibility Pool", "",
            f"- Final decision: `{report['FINAL_DECISION']}`",
            f"- Base candidate count: {len(candidate):,}", f"- Partial review count: {len(review):,}",
            f"- Full grid registry: {len(full_grid):,}", "",
            "Eligibility means permission to enter later screening only. It is not a negative label, a hard control, or a paired sample.",
            "No rainfall matching, static EOGIS screening, random sampling, labels, hard controls, pairs, or external A26 reads were performed.",
            "", f"Warnings: `{warnings if warnings else 'NONE'}`", f"Errors: `{errors if errors else 'NONE'}`",
            f"Next step: {NEXT_STEP}",
        ]
        (audit_dir / "HARD_CONTROL_CANDIDATE_BUILD_REPORT_v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        write_csv(input_df, manifest_dir / "hard_control_candidate_input_manifest_v1.csv")
        normalized_by_path: dict[str, str] = {}
        for result in format_results.values():
            for p in result["paths"]:
                normalized_by_path[str(p.relative_to(staging)).replace("\\", "/")] = result["normalized_content_sha256"]
        output_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            output_rows.append({
                "relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p),
                "normalized_content_sha256": normalized_by_path.get(rel, "NOT_APPLICABLE"),
            })
        write_csv(pd.DataFrame(output_rows), manifest_dir / "hard_control_candidate_output_manifest_v1.csv")
        hash_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            hash_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p)})
        write_csv(pd.DataFrame(hash_rows), manifest_dir / "hard_control_candidate_file_hashes_v1.csv")

        os.replace(staging, target)
        staging = None
        print_summary(report)
        return 0 if report["FINAL_DECISION"] == SUCCESS else 2
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
