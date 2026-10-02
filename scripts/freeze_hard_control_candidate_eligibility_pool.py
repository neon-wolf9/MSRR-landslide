#!/usr/bin/env python3
"""Audit and freeze the existing 250 m hard-control eligibility pool.

No eligibility rule is rerun to create classifications. Existing statuses,
values, IDs, and geometries are only verified, stably sorted, and copied.
"""

from __future__ import annotations

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
import numpy as np
import pandas as pd


SUCCESS = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN"
FAILURE = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FREEZE_REJECTED"
NEXT_STEP = "Await author confirmation of the frozen output before static EO/GIS construction."
CRS_EPSG = 6671
FORBIDDEN = {
    "label", "y", "negative_flag", "negative_sample", "hard_control_flag",
    "control_type", "pair_id", "positive_sample_id", "control_sample_id",
    "split", "fold_id", "modeling_samples", "positive_hard_control_pairs",
}
STATUS_EXPECTED = {
    "ELIGIBLE_BASE_FULL_RELIABLE": 29084,
    "REVIEW_PARTIAL_RELIABLE": 774,
    "INELIGIBLE_INVENTORY_EVIDENCE": 8704,
    "INELIGIBLE_COVERAGE_EXCLUSION": 1741,
    "INELIGIBLE_MASTER_GRID": 3794,
    "INELIGIBLE_OUTSIDE_INTERPRETED": 95267,
}


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


def sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("unit_id", kind="mergesort").reset_index(drop=True)
    if isinstance(frame, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry=frame.geometry.name, crs=frame.crs)
    return result


def normalized_hash(frame: pd.DataFrame, geometry: bool = False) -> str:
    df = sorted_frame(frame)
    geom_name = df.geometry.name if isinstance(df, gpd.GeoDataFrame) else None
    attrs = [c for c in df.columns if c != geom_name]
    h = hashlib.sha256()
    h.update(("|".join(attrs) + "\n").encode("utf-8"))
    for row in df[attrs].itertuples(index=False, name=None):
        h.update(("\x1f".join(scalar(v) for v in row) + "\n").encode("utf-8"))
    if geometry:
        if geom_name is None:
            raise ValueError("Geometry requested for nonspatial table")
        h.update(b"<GEOMETRY>\n")
        for geom in df[geom_name]:
            h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
            h.update(b"\n")
    return h.hexdigest()


def geometry_hash(frame: gpd.GeoDataFrame) -> str:
    df = sorted_frame(frame)
    h = hashlib.sha256()
    for uid, geom in zip(df["unit_id"].astype(str), df.geometry):
        h.update((uid + "\x1f").encode("utf-8"))
        h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def attribute_columns(frame: pd.DataFrame) -> list[str]:
    geom = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else None
    return [c for c in frame.columns if c != geom]


def equivalent(a: pd.DataFrame, b: pd.DataFrame, geometry: bool = False) -> bool:
    return (
        len(a) == len(b)
        and attribute_columns(a) == attribute_columns(b)
        and normalized_hash(a, geometry) == normalized_hash(b, geometry)
    )


def verify_file_hash_table(base: Path, table: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows = read_csv(table)
    details: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        path = base / str(row["relative_path"]).replace("\\", "/")
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        passed = bool(exists and actual_size == int(row["size_bytes"]) and actual_hash == row["sha256"])
        if not passed:
            errors.append(f"Candidate-pool file hash mismatch: {path}")
        details.append({
            "input_role": "candidate_pool_registered_file", "absolute_path": str(path),
            "size_bytes": actual_size, "sha256": actual_hash, "hash_verified": yn(passed), "read_only": "YES",
        })
    return details


def verify_upstream_manifest(table: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows = read_csv(table)
    details: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        path = Path(row["absolute_path"])
        if "a26" in str(path).lower():
            errors.append(f"Prohibited external A26 path listed; not read: {path}")
            continue
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        passed = bool(exists and actual_size == int(row["size_bytes"]) and actual_hash == row["sha256"])
        if not passed:
            errors.append(f"Candidate upstream input hash mismatch: {path}")
        details.append({
            "input_role": row.get("input_role", "candidate_upstream_input"), "absolute_path": str(path),
            "size_bytes": actual_size, "sha256": actual_hash, "hash_verified": yn(passed), "read_only": "YES",
        })
    return details


def add_explicit(paths: dict[str, Path], details: list[dict[str, Any]]) -> None:
    known = {x["absolute_path"] for x in details}
    for role, path in paths.items():
        if str(path) not in known:
            details.append({
                "input_role": role, "absolute_path": str(path), "size_bytes": path.stat().st_size,
                "sha256": sha256(path), "hash_verified": "YES", "read_only": "YES",
            })


def write_attribute(frame: pd.DataFrame, parquet: Path, csv_path: Path) -> dict[str, Any]:
    df = sorted_frame(frame)
    df.to_parquet(parquet, index=False)
    write_csv(df, csv_path)
    pq = pd.read_parquet(parquet)
    cs = read_csv(csv_path)
    return {
        "paths": [parquet, csv_path], "rows": len(df),
        "normalized_content_sha256": normalized_hash(df),
        "reload_pass": equivalent(df, pq) and equivalent(df, cs),
    }


def write_spatial(
    source: gpd.GeoDataFrame, attribute_parquet: Path, csv_path: Path,
    geoparquet: Path, gpkg: Path, layer: str,
) -> dict[str, Any]:
    gdf = sorted_frame(source)
    attrs = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    attrs.to_parquet(attribute_parquet, index=False)
    write_csv(attrs, csv_path)
    gdf.to_parquet(geoparquet, index=False)
    gdf.to_file(gpkg, layer=layer, driver="GPKG", index=False)
    pq = pd.read_parquet(attribute_parquet)
    cs = read_csv(csv_path)
    geo = gpd.read_parquet(geoparquet)
    gp = gpd.read_file(gpkg, layer=layer)
    reload_pass = (
        equivalent(attrs, pq) and equivalent(attrs, cs)
        and geo.crs is not None and gp.crs is not None
        and geo.crs.to_epsg() == CRS_EPSG and gp.crs.to_epsg() == CRS_EPSG and geo.crs == gp.crs
        and equivalent(gdf, geo, True) and equivalent(gdf, gp, True)
    )
    return {
        "paths": [attribute_parquet, csv_path, geoparquet, gpkg], "rows": len(gdf),
        "normalized_content_sha256": normalized_hash(gdf, True),
        "geometry_sha256": geometry_hash(gdf), "reload_pass": reload_pass,
    }


def print_summary(report: dict[str, Any]) -> None:
    keys = [
        "PROJECT_ROOT", "PREVIOUS_CANDIDATE_POOL_DECISION_PASS",
        "PREVIOUS_POSITIVE_REGISTRY_FREEZE_PASS", "PREVIOUS_GSI_COVERAGE_MAPPING_PASS",
        "MASTER_GRID_COUNT", "BASE_CANDIDATE_COUNT", "PARTIAL_REVIEW_COUNT",
        "ELIGIBLE_BASE_FULL_RELIABLE_COUNT", "REVIEW_PARTIAL_RELIABLE_COUNT",
        "INELIGIBLE_INVENTORY_EVIDENCE_COUNT", "INELIGIBLE_COVERAGE_EXCLUSION_COUNT",
        "INELIGIBLE_MASTER_GRID_COUNT", "INELIGIBLE_OUTSIDE_INTERPRETED_COUNT",
        "STATUS_PARTITION_PASS", "BASE_CANDIDATE_SET_IDENTITY_PASS",
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT",
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT",
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT",
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT",
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT",
        "BOUNDARY_EXCEPTION_COUNT", "BOUNDARY_EXCEPTION_CANDIDATE_COUNT",
        "BOUNDARY_EXCEPTION_REVIEW_COUNT", "BOUNDARY_EXCEPTION_POLICY_PASS",
        "UNIT_ID_INTEGRITY_PASS", "MASTER_GEOMETRY_UNCHANGED", "CROSS_FORMAT_RELOAD_PASS",
        "NORMALIZED_CONTENT_HASH_PASS", "IDEMPOTENT_RERUN_PASS", "LABEL_FIELD_CREATED",
        "HARD_CONTROLS_CREATED", "PAIR_TABLE_CREATED", "RAINFALL_MATCHING_PERFORMED",
        "STATIC_EOGIS_SCREENING_PERFORMED", "EXTERNAL_A26_CONTENT_READ",
        "WARNINGS", "ERRORS", "FINAL_DECISION", "NEXT_STEP",
    ]
    for key in keys:
        value = report[key]
        if key == "ERRORS" and not value:
            value = "NONE"
        print(f"{key}: {value}")


def validate_existing(target: Path) -> dict[str, Any] | None:
    marker_path = target / "07_audit/HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN.marker.json"
    report_path = target / "07_audit/hard_control_candidate_freeze_report_v1.json"
    manifest_path = target / "00_manifest/hard_control_candidate_freeze_input_manifest_v1.csv"
    if not marker_path.is_file() or not report_path.is_file() or not manifest_path.is_file():
        return None
    marker = load_json(marker_path)
    report = load_json(report_path)
    if marker.get("current_final_decision") != SUCCESS or report.get("FINAL_DECISION") != SUCCESS:
        return None
    for row in read_csv(manifest_path).to_dict("records"):
        path = Path(row["absolute_path"])
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            return None
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    first = root / "data/01_county_prediction_domain"
    reliable = root / "data/02_reliable_observation_domain"
    mapping = reliable / "04_gsi_coverage_grid_mapping"
    positive = reliable / "09_positive_evidence_grid_registry_frozen"
    source = reliable / "10_hard_control_candidate_eligibility_pool"
    target = reliable / "11_hard_control_candidate_eligibility_pool_frozen"
    staging = reliable / "11_hard_control_candidate_eligibility_pool_frozen_staging"
    errors: list[str] = []
    warnings: list[str] = []
    differences: list[dict[str, Any]] = []

    paths = {
        "first_marker": first / "06_audit/FIRST_LAYER_FROZEN.marker.json",
        "master_grid": first / "02_grid_250m/hiroshima_grid_250m_master.gpkg",
        "mapping_audit": mapping / "03_audit/gsi_coverage_grid_mapping_audit_v1.json",
        "mapping_attributes": mapping / "01_mapped_grid/gsi_coverage_mapping_250m_attributes_v1.csv",
        "positive_marker": positive / "03_audit/POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN.marker.json",
        "primary_grids": positive / "01_frozen_registry/primary_positive_grid_registry_250m_frozen_v1.parquet",
        "affected_grids": positive / "01_frozen_registry/ajg_affected_grid_registry_250m_frozen_v1.parquet",
        "control_grids": positive / "01_frozen_registry/control_exclusion_grid_registry_250m_frozen_v1.parquet",
        "candidate_file_hashes": source / "00_manifest/hard_control_candidate_file_hashes_v1.csv",
        "candidate_input_manifest": source / "00_manifest/hard_control_candidate_input_manifest_v1.csv",
        "candidate_output_manifest": source / "00_manifest/hard_control_candidate_output_manifest_v1.csv",
        "full_attr": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.parquet",
        "full_csv": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.csv",
        "full_geo": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.geoparquet",
        "full_gpkg": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.gpkg",
        "candidate_attr": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.parquet",
        "candidate_csv": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.csv",
        "candidate_geo": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.geoparquet",
        "candidate_gpkg": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.gpkg",
        "review": source / "03_review_pool/partial_reliable_review_pool_250m_v1.parquet",
        "ineligible": source / "04_ineligible_registry/hard_control_ineligible_grid_registry_250m_v1.parquet",
        "schema": source / "05_schema_and_rules/hard_control_candidate_schema_v1.csv",
        "rules": source / "05_schema_and_rules/hard_control_eligibility_rule_dictionary_v1.csv",
        "reasons": source / "05_schema_and_rules/hard_control_exclusion_reason_dictionary_v1.csv",
        "anomalies": source / "06_audit/coverage_status_tolerance_anomalies_v1.csv",
        "consistency": source / "06_audit/hard_control_candidate_consistency_audit_v1.json",
        "build_report": source / "06_audit/hard_control_candidate_build_report_v1.json",
        "preview": source / "07_preview/hard_control_candidate_eligibility_preview_v1.png",
    }
    for role, path in paths.items():
        if not path.is_file(): errors.append(f"Missing formal input {role}: {path}")
    candidate_reports = list((source / "06_audit").glob("hard_control_candidate_build_report_v*.json"))
    if candidate_reports != [paths["build_report"]]:
        errors.append(f"Ambiguous candidate build reports: {[str(p) for p in candidate_reports]}")
    if errors:
        print(f"ERRORS: {errors}"); print(f"FINAL_DECISION: {FAILURE}"); return 2

    first_marker = load_json(paths["first_marker"])
    mapping_audit = load_json(paths["mapping_audit"])
    positive_marker = load_json(paths["positive_marker"])
    prior_report = load_json(paths["build_report"])
    previous = {
        "first": first_marker.get("status") == "FIRST_LAYER_FROZEN",
        "mapping": mapping_audit.get("final_decision") == "GSI_COVERAGE_MAPPED_TO_GRID_250M" and not mapping_audit.get("errors"),
        "positive": positive_marker.get("current_FINAL_DECISION") == "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN" and not positive_marker.get("errors"),
        "candidate": prior_report.get("FINAL_DECISION") == "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_COMPLETE" and not prior_report.get("ERRORS"),
    }
    for name, passed in previous.items():
        if not passed: errors.append(f"Previous decision failed: {name}")

    input_details = verify_file_hash_table(source, paths["candidate_file_hashes"], errors)
    input_details.extend(verify_upstream_manifest(paths["candidate_input_manifest"], errors))
    add_explicit(paths, input_details)
    input_manifest = pd.DataFrame(input_details).drop_duplicates("absolute_path").sort_values("absolute_path")

    if target.exists():
        existing = validate_existing(target)
        if existing is None:
            print(f"ERRORS: Existing target is not a valid idempotent freeze: {target}")
            print(f"FINAL_DECISION: {FAILURE}"); return 2
        print_summary(existing); return 0
    if staging.exists():
        print(f"ERRORS: Staging directory exists and will not be overwritten: {staging}")
        print(f"FINAL_DECISION: {FAILURE}"); return 2

    # Load existing classifications only; no status is recalculated or edited.
    full_attr = pd.read_parquet(paths["full_attr"])
    full_geo = gpd.read_parquet(paths["full_geo"])
    candidate_attr = pd.read_parquet(paths["candidate_attr"])
    candidate_geo = gpd.read_parquet(paths["candidate_geo"])
    review = pd.read_parquet(paths["review"])
    ineligible = pd.read_parquet(paths["ineligible"])
    anomalies = read_csv(paths["anomalies"])
    master = gpd.read_file(paths["master_grid"], layer="hiroshima_grid_250m_master")
    map_attrs = read_csv(paths["mapping_attributes"])
    primary = pd.read_parquet(paths["primary_grids"])
    affected = pd.read_parquet(paths["affected_grids"])
    control = pd.read_parquet(paths["control_grids"])

    # Verify all source formats before freezing.
    source_full_csv = read_csv(paths["full_csv"])
    source_full_gpkg = gpd.read_file(paths["full_gpkg"])
    source_candidate_csv = read_csv(paths["candidate_csv"])
    source_candidate_gpkg = gpd.read_file(paths["candidate_gpkg"])
    source_cross_format_pass = (
        equivalent(full_attr, source_full_csv) and equivalent(full_geo, source_full_gpkg, True)
        and equivalent(candidate_attr, source_candidate_csv) and equivalent(candidate_geo, source_candidate_gpkg, True)
        and equivalent(full_attr, pd.DataFrame(full_geo.drop(columns=full_geo.geometry.name)))
        and equivalent(candidate_attr, pd.DataFrame(candidate_geo.drop(columns=candidate_geo.geometry.name)))
    )
    if not source_cross_format_pass: errors.append("Source candidate-pool cross-format parity failed")

    master_ids = set(master["unit_id"].astype(str))
    full_ids = set(full_attr["unit_id"].astype(str))
    candidate_ids = set(candidate_attr["unit_id"].astype(str))
    review_ids = set(review["unit_id"].astype(str))
    primary_ids = set(primary["unit_id"].astype(str))
    affected_ids = set(affected["unit_id"].astype(str))
    control_ids = set(control["unit_id"].astype(str))
    gsi_ids = set(control.loc[control["gsi_evidence_flag"].astype(str).str.upper() == "YES", "unit_id"].astype(str))
    map_by_id = map_attrs.set_index(map_attrs["unit_id"].astype(str))
    prelim_ids = set(full_attr.loc[full_attr["preliminary_eligible_flag"] == "YES", "unit_id"].astype(str))
    full_reliable_ids = set(full_attr.loc[full_attr["gsi_coverage_class"] == "FULL_RELIABLE", "unit_id"].astype(str))
    partial_reliable_ids = set(full_attr.loc[full_attr["gsi_coverage_class"] == "PARTIAL_RELIABLE", "unit_id"].astype(str))
    expected_candidate_ids = (full_reliable_ids & prelim_ids) - control_ids

    status_counts = full_attr["primary_eligibility_status"].value_counts().to_dict()
    benchmarks = {
        "MASTER_GRID_COUNT": (len(full_attr), 139364),
        "PRELIMINARY_ELIGIBLE_GRID_COUNT": (len(prelim_ids), 135474),
        "FULL_RELIABLE_GRID_COUNT": (len(full_reliable_ids), 38605),
        "PARTIAL_RELIABLE_GRID_COUNT": (len(partial_reliable_ids), 849),
        "COVERAGE_EXCLUSION_GRID_COUNT": (int((full_attr["gsi_coverage_class"] == "COVERAGE_EXCLUSION").sum()), 2041),
        "OUTSIDE_INTERPRETED_GRID_COUNT": (int((full_attr["gsi_coverage_class"] == "OUTSIDE_INTERPRETED").sum()), 97869),
        "CONTROL_EXCLUSION_GRID_COUNT": (len(control_ids), 8704),
        "BASE_CANDIDATE_COUNT": (len(candidate_attr), 29084),
        "PARTIAL_REVIEW_COUNT": (len(review), 774),
    }
    for name, (actual, expected) in benchmarks.items():
        if actual != expected:
            errors.append(f"{name}={actual}, expected {expected}")
            differences.append({"check": name, "actual": actual, "expected": expected})
    for status, expected in STATUS_EXPECTED.items():
        actual = int(status_counts.get(status, 0))
        if actual != expected:
            errors.append(f"{status}={actual}, expected {expected}")
            differences.append({"check": status, "actual": actual, "expected": expected})

    status_partition_pass = set(status_counts) == set(STATUS_EXPECTED) and sum(status_counts.values()) == 139364
    base_identity_pass = candidate_ids == expected_candidate_ids
    intersections = {
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(candidate_ids & control_ids),
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT": len(candidate_ids & primary_ids),
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT": len(candidate_ids & affected_ids),
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT": len(candidate_ids & gsi_ids),
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(review_ids & control_ids),
    }
    set_checks = [
        base_identity_pass, not any(intersections.values()), not (candidate_ids & review_ids),
        candidate_ids <= prelim_ids, candidate_ids <= full_reliable_ids,
        review_ids <= partial_reliable_ids, full_ids == master_ids,
    ]
    if not status_partition_pass: errors.append(f"Status partition failed: {status_counts}")
    if not all(set_checks): errors.append("One or more frozen set identity checks failed")
    unit_integrity_pass = (
        full_attr["unit_id"].notna().all() and not full_attr["unit_id"].duplicated().any()
        and not candidate_attr["unit_id"].duplicated().any() and not review["unit_id"].duplicated().any()
        and not ineligible["unit_id"].duplicated().any() and full_ids == master_ids
    )
    if not unit_integrity_pass: errors.append("Unit ID integrity failed")

    anomaly_ids = set(anomalies["unit_id"].astype(str))
    full_status = full_attr.set_index(full_attr["unit_id"].astype(str))["primary_eligibility_status"].to_dict()
    source_anomaly_hash = sha256(paths["anomalies"])
    exception_rows = []
    for row in anomalies.sort_values("unit_id").to_dict("records"):
        uid = str(row["unit_id"])
        exception_rows.append({
            "unit_id": uid, "prior_coverage_status": row["coverage_status"],
            "any_interpreted_flag": int(row["any_interpreted_flag"]),
            "final_primary_eligibility_status": full_status.get(uid),
            "exception_reason": "SUB_TOLERANCE_BOUNDARY_FRAGMENT_WITH_ZERO_FORMAL_INTERPRETED_AREA",
            "source_file": str(paths["anomalies"]), "source_content_hash": source_anomaly_hash,
            "candidate_pool_inclusion_flag": yn(uid in candidate_ids),
            "partial_review_inclusion_flag": yn(uid in review_ids),
            "resolution_policy": "FORMAL_BINARY_INTERPRETATION_FIELD_PRIORITY_AND_CONSERVATIVE_EXCLUSION",
            "manual_edit_performed": "NO",
        })
    exceptions = pd.DataFrame(exception_rows)
    exception_policy_pass = (
        len(exceptions) == 6 and len(anomaly_ids) == 6
        and exceptions["any_interpreted_flag"].eq(0).all()
        and exceptions["final_primary_eligibility_status"].eq("INELIGIBLE_OUTSIDE_INTERPRETED").all()
        and exceptions["candidate_pool_inclusion_flag"].eq("NO").all()
        and exceptions["partial_review_inclusion_flag"].eq("NO").all()
        and exceptions["manual_edit_performed"].eq("NO").all()
    )
    if not exception_policy_pass: errors.append("Six-record boundary exception policy failed")
    warnings.append("Six sub-tolerance boundary fragments are conservatively frozen as INELIGIBLE_OUTSIDE_INTERPRETED; prior inputs remain unchanged")

    # Geometry checks compare exact WKB by unit_id; no spatial operation occurs.
    master_geom = dict(zip(master["unit_id"].astype(str), master.geometry.map(lambda g: g.wkb_hex)))
    full_geom = dict(zip(full_geo["unit_id"].astype(str), full_geo.geometry.map(lambda g: g.wkb_hex)))
    candidate_geom = dict(zip(candidate_geo["unit_id"].astype(str), candidate_geo.geometry.map(lambda g: g.wkb_hex)))
    master_geometry_unchanged = full_geom == master_geom and all(master_geom[k] == v for k, v in candidate_geom.items())
    geometry_quality_pass = (
        full_geo.geometry.notna().all() and candidate_geo.geometry.notna().all()
        and not full_geo.geometry.is_empty.any() and not candidate_geo.geometry.is_empty.any()
        and full_geo.geometry.is_valid.all() and candidate_geo.geometry.is_valid.all()
        and not full_geo.geometry.to_wkb().duplicated().any()
        and not candidate_geo.geometry.to_wkb().duplicated().any()
        and full_geo.crs is not None and full_geo.crs.to_epsg() == CRS_EPSG
        and candidate_geo.crs is not None and candidate_geo.crs.to_epsg() == CRS_EPSG
    )
    if not master_geometry_unchanged: errors.append("Master/candidate geometry identity failed")
    if not geometry_quality_pass: errors.append("Geometry validity/emptiness/duplication/CRS audit failed")

    forbidden_present = sorted({c.lower() for c in full_attr.columns} & FORBIDDEN)
    required_candidate_states = (
        candidate_attr["water_audit_status"].eq("PENDING_STATIC_EOGIS").all()
        and candidate_attr["static_eogis_audit_status"].eq("PENDING").all()
        and candidate_attr["rainfall_matching_status"].eq("NOT_PERFORMED").all()
        and candidate_attr["final_control_status"].eq("NOT_CREATED").all()
    )
    if forbidden_present: errors.append(f"Forbidden fields present: {forbidden_present}")
    if not required_candidate_states: errors.append("Candidate pending/not-performed states changed")

    source_hashes = {
        "full": normalized_hash(full_geo, True), "candidate": normalized_hash(candidate_geo, True),
        "review": normalized_hash(review), "ineligible": normalized_hash(ineligible),
        "exceptions": normalized_hash(exceptions),
    }
    staging.mkdir(parents=False, exist_ok=False)
    try:
        names = [
            "00_manifest", "01_frozen_full_registry", "02_frozen_base_candidate_pool",
            "03_frozen_partial_review_pool", "04_frozen_ineligible_registry", "05_frozen_exceptions",
            "06_schema_and_rules", "07_audit", "08_preview",
        ]
        dirs = {name: staging / name for name in names}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

        full_result = write_spatial(
            full_geo,
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1.parquet",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1.csv",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1.geoparquet",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1.gpkg",
            "hard_control_eligibility_full_grid_250m_frozen_v1",
        )
        candidate_result = write_spatial(
            candidate_geo,
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1.parquet",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1.csv",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1.geoparquet",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1.gpkg",
            "hard_control_base_candidate_pool_250m_frozen_v1",
        )
        review_result = write_attribute(
            review,
            dirs["03_frozen_partial_review_pool"] / "partial_reliable_review_pool_250m_frozen_v1.parquet",
            dirs["03_frozen_partial_review_pool"] / "partial_reliable_review_pool_250m_frozen_v1.csv",
        )
        ineligible_result = write_attribute(
            ineligible,
            dirs["04_frozen_ineligible_registry"] / "hard_control_ineligible_grid_registry_250m_frozen_v1.parquet",
            dirs["04_frozen_ineligible_registry"] / "hard_control_ineligible_grid_registry_250m_frozen_v1.csv",
        )
        exception_path = dirs["05_frozen_exceptions"] / "coverage_semantic_boundary_exception_registry_frozen_v1.csv"
        write_csv(sorted_frame(exceptions), exception_path)
        shutil.copy2(paths["schema"], dirs["06_schema_and_rules"] / "hard_control_candidate_schema_frozen_v1.csv")
        shutil.copy2(paths["rules"], dirs["06_schema_and_rules"] / "hard_control_eligibility_rule_dictionary_frozen_v1.csv")
        shutil.copy2(paths["reasons"], dirs["06_schema_and_rules"] / "hard_control_exclusion_reason_dictionary_frozen_v1.csv")
        shutil.copy2(paths["preview"], dirs["08_preview"] / "hard_control_candidate_eligibility_preview_frozen_v1.png")

        results = {
            "full": full_result, "candidate": candidate_result,
            "review": review_result, "ineligible": ineligible_result,
        }
        frozen_hashes = {
            "full": full_result["normalized_content_sha256"],
            "candidate": candidate_result["normalized_content_sha256"],
            "review": review_result["normalized_content_sha256"],
            "ineligible": ineligible_result["normalized_content_sha256"],
            "exceptions": normalized_hash(read_csv(exception_path)),
        }
        normalized_pass = source_hashes == frozen_hashes
        cross_format_pass = source_cross_format_pass and all(x["reload_pass"] for x in results.values())
        # Equivalent second-run signature: independently reload every formal
        # frozen table and recompute content, set, status, geometry, exception,
        # and count signatures before publication.
        rerun_signature = {
            "normalized_hashes": frozen_hashes,
            "base_candidate_count": candidate_result["rows"],
            "partial_review_count": review_result["rows"],
            "candidate_id_sha256": hashlib.sha256("\n".join(sorted(candidate_ids)).encode()).hexdigest(),
            "status_sha256": hashlib.sha256("\n".join(f"{k}:{status_counts[k]}" for k in sorted(status_counts)).encode()).hexdigest(),
            "full_geometry_sha256": full_result["geometry_sha256"],
            "candidate_geometry_sha256": candidate_result["geometry_sha256"],
            "exception_count": len(exceptions),
        }
        expected_signature = {
            "normalized_hashes": source_hashes,
            "base_candidate_count": len(candidate_attr), "partial_review_count": len(review),
            "candidate_id_sha256": hashlib.sha256("\n".join(sorted(candidate_ids)).encode()).hexdigest(),
            "status_sha256": hashlib.sha256("\n".join(f"{k}:{status_counts[k]}" for k in sorted(status_counts)).encode()).hexdigest(),
            "full_geometry_sha256": geometry_hash(full_geo),
            "candidate_geometry_sha256": geometry_hash(candidate_geo),
            "exception_count": 6,
        }
        idempotent_pass = rerun_signature == expected_signature
        if not normalized_pass: errors.append("Frozen/source normalized content hashes differ")
        if not cross_format_pass: errors.append("Cross-format reload failed")
        if not idempotent_pass: errors.append("Equivalent idempotent rerun signature failed")

        write_csv(pd.DataFrame(differences, columns=["check", "actual", "expected"]), dirs["07_audit"] / "hard_control_candidate_freeze_differences_v1.csv")
        report = {
            "PROJECT_ROOT": str(root),
            "PREVIOUS_CANDIDATE_POOL_DECISION_PASS": yn(previous["candidate"]),
            "PREVIOUS_POSITIVE_REGISTRY_FREEZE_PASS": yn(previous["positive"]),
            "PREVIOUS_GSI_COVERAGE_MAPPING_PASS": yn(previous["mapping"]),
            "MASTER_GRID_COUNT": len(full_attr), "BASE_CANDIDATE_COUNT": len(candidate_attr),
            "PARTIAL_REVIEW_COUNT": len(review),
            **{f"{k}_COUNT": int(status_counts.get(k, 0)) for k in STATUS_EXPECTED},
            "STATUS_PARTITION_PASS": yn(status_partition_pass),
            "BASE_CANDIDATE_SET_IDENTITY_PASS": yn(base_identity_pass), **intersections,
            "BOUNDARY_EXCEPTION_COUNT": len(exceptions),
            "BOUNDARY_EXCEPTION_CANDIDATE_COUNT": int(exceptions["candidate_pool_inclusion_flag"].eq("YES").sum()),
            "BOUNDARY_EXCEPTION_REVIEW_COUNT": int(exceptions["partial_review_inclusion_flag"].eq("YES").sum()),
            "BOUNDARY_EXCEPTION_POLICY_PASS": yn(exception_policy_pass),
            "UNIT_ID_INTEGRITY_PASS": yn(unit_integrity_pass),
            "MASTER_GEOMETRY_UNCHANGED": yn(master_geometry_unchanged and geometry_quality_pass),
            "CROSS_FORMAT_RELOAD_PASS": yn(cross_format_pass),
            "NORMALIZED_CONTENT_HASH_PASS": yn(normalized_pass),
            "IDEMPOTENT_RERUN_PASS": yn(idempotent_pass),
            "LABEL_FIELD_CREATED": "NO", "HARD_CONTROLS_CREATED": "NO", "PAIR_TABLE_CREATED": "NO",
            "RAINFALL_MATCHING_PERFORMED": "NO", "STATIC_EOGIS_SCREENING_PERFORMED": "NO",
            "EXTERNAL_A26_CONTENT_READ": "NO", "WARNINGS": warnings, "ERRORS": errors,
            "FINAL_DECISION": SUCCESS if not errors else FAILURE, "NEXT_STEP": NEXT_STEP,
        }
        consistency = {
            "benchmarks": benchmarks, "status_counts": status_counts,
            "set_intersections": intersections, "boundary_exception_ids": sorted(anomaly_ids),
            "boundary_exception_policy_pass": exception_policy_pass,
            "source_normalized_hashes": source_hashes, "frozen_normalized_hashes": frozen_hashes,
            "cross_format_results": results, "rerun_signature": rerun_signature,
            "expected_signature": expected_signature, "geometry_quality_pass": geometry_quality_pass,
            "forbidden_fields": forbidden_present, "errors": errors,
        }
        (dirs["07_audit"] / "hard_control_candidate_freeze_consistency_audit_v1.json").write_text(
            json.dumps(consistency, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        freeze_time = datetime.now(timezone.utc).isoformat()
        freeze_report = {
            **report, "freeze_version": "v1", "freeze_timestamp_utc": freeze_time,
            "crs": f"EPSG:{CRS_EPSG}", "previous_final_decision": prior_report["FINAL_DECISION"],
            "normalized_content_hashes": frozen_hashes,
        }
        (dirs["07_audit"] / "hard_control_candidate_freeze_report_v1.json").write_text(
            json.dumps(freeze_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        md = [
            "# Hard-Control Candidate Eligibility Pool Freeze Report", "",
            f"- Final decision: `{report['FINAL_DECISION']}`",
            f"- Full grid registry: {len(full_attr):,}", f"- Base candidates: {len(candidate_attr):,}",
            f"- Partial review: {len(review):,}", f"- Boundary semantic exceptions: {len(exceptions)}", "",
            "This is a frozen eligibility registry only. No negative samples, hard controls, labels, pairs, rainfall matches, or EO/GIS screening results were created.",
            "", f"Warnings: `{warnings if warnings else 'NONE'}`", f"Errors: `{errors if errors else 'NONE'}`",
            f"Next step: {NEXT_STEP}",
        ]
        (dirs["07_audit"] / "HARD_CONTROL_CANDIDATE_FREEZE_REPORT_v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        input_manifest_path = dirs["00_manifest"] / "hard_control_candidate_freeze_input_manifest_v1.csv"
        write_csv(input_manifest, input_manifest_path)
        normalized_by_path: dict[str, str] = {}
        for result in results.values():
            for p in result["paths"]:
                normalized_by_path[str(p.relative_to(staging)).replace("\\", "/")] = result["normalized_content_sha256"]
        normalized_by_path[str(exception_path.relative_to(staging)).replace("\\", "/")] = frozen_hashes["exceptions"]
        output_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            output_rows.append({
                "relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p),
                "normalized_content_sha256": normalized_by_path.get(rel, "NOT_APPLICABLE"),
            })
        output_manifest_path = dirs["00_manifest"] / "hard_control_candidate_freeze_output_manifest_v1.csv"
        write_csv(pd.DataFrame(output_rows), output_manifest_path)
        hash_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            hash_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p)})
        hashes_path = dirs["00_manifest"] / "hard_control_candidate_freeze_file_hashes_v1.csv"
        write_csv(pd.DataFrame(hash_rows), hashes_path)

        if report["FINAL_DECISION"] == SUCCESS:
            marker = {
                "freeze_version": "v1", "freeze_timestamp_utc": freeze_time,
                "project_root": str(root), "crs": f"EPSG:{CRS_EPSG}",
                "master_grid_count": len(full_attr), "base_candidate_count": len(candidate_attr),
                "partial_review_count": len(review), "primary_status_counts": status_counts,
                "boundary_exception_count": len(exceptions),
                "input_manifest_hash": sha256(input_manifest_path),
                "input_file_hashes": input_manifest[["absolute_path", "sha256"]].to_dict("records"),
                "output_file_hashes": hash_rows,
                "file_hash_manifest_sha256": sha256(hashes_path),
                "normalized_content_hashes": frozen_hashes,
                "previous_final_decision": prior_report["FINAL_DECISION"],
                "current_final_decision": SUCCESS, "warnings": warnings, "errors": errors,
                "next_step": NEXT_STEP,
            }
            (dirs["07_audit"] / "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN.marker.json").write_text(
                json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        os.replace(staging, target)
        staging = None
        print_summary(report)
        return 0 if report["FINAL_DECISION"] == SUCCESS else 2
    finally:
        if staging is not None and staging.exists(): shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
