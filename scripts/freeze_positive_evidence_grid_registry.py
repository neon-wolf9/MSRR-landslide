#!/usr/bin/env python3
"""Audit and freeze the second-layer positive-evidence/grid registries.

The operation is deliberately read-only with respect to every upstream v1
asset.  New outputs are written to a sibling temporary directory, fully
audited, then atomically renamed to the formal frozen directory.
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
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd


SUCCESS = "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN"
FAILURE = "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FREEZE_REJECTED"
NEXT_STEP = "Await user confirmation of the frozen result; do not build the hard-control candidate eligibility pool."
CRS = "EPSG:6671"
FORBIDDEN_CREATED_FIELDS = {
    "label", "y", "negative_flag", "hard_control_flag", "pair_id",
    "control_sample_id", "positive_sample_id",
}
LEAKAGE_TERMS = (
    "source", "confidence", "inventory_role", "gsi", "ajg", "match",
    "candidate", "use_for_training", "coverage", "exclusion", "label",
    "training_decision", "control",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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
    if isinstance(value, (bool,)):
        return "1" if value else "0"
    if isinstance(value, (float,)):
        if math.isnan(value):
            return "<NULL>"
        # CSV text round-trips can move the final binary floating-point bit;
        # 12 significant digits preserve registry semantics while making the
        # normalized content hash container-independent.
        return format(value, ".12g")
    text = str(value).strip()
    return "<NULL>" if text == "" else text


def normalized_hash(frame: pd.DataFrame, keys: list[str], include_geometry: bool = False) -> str:
    """Stable semantic hash independent of file container metadata."""
    df = frame.copy()
    geometry_name = df.geometry.name if isinstance(df, gpd.GeoDataFrame) else None
    if geometry_name and geometry_name in df.columns:
        attrs = [c for c in df.columns if c != geometry_name]
    else:
        attrs = list(df.columns)
    ordered_cols = list(keys) + sorted(c for c in attrs if c not in keys)
    df = df.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)
    h = hashlib.sha256()
    h.update(("|".join(ordered_cols) + "\n").encode("utf-8"))
    for row in df[ordered_cols].itertuples(index=False, name=None):
        h.update(("\x1f".join(scalar(v) for v in row) + "\n").encode("utf-8"))
    if include_geometry:
        if not geometry_name:
            raise ValueError("Geometry requested for a non-spatial table")
        h.update(b"<GEOMETRY>\n")
        for geom in df[geometry_name]:
            h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
            h.update(b"\n")
    return h.hexdigest()


def geometry_hash(frame: gpd.GeoDataFrame, key: str) -> str:
    df = frame.sort_values(key, kind="mergesort")
    h = hashlib.sha256()
    for ident, geom in zip(df[key], df.geometry):
        h.update((scalar(ident) + "\x1f").encode("utf-8"))
        h.update(("<NULL>" if geom is None else geom.wkb_hex).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def unique_candidate(root: Path, expected: Path, label: str, errors: list[str]) -> Path:
    candidates = [p.resolve() for p in root.rglob(expected.name) if p.is_file()]
    expected_resolved = expected.resolve()
    if len(candidates) != 1 or candidates[0] != expected_resolved:
        errors.append(f"{label} candidate ambiguity: {[str(p) for p in candidates]}")
    return expected


def verify_registered_hashes(base: Path, table: Path, errors: list[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    rows = read_csv(table).fillna("")
    for row in rows.to_dict("records"):
        raw_path = str(row.get("relative_path", row.get("path", "")))
        path = Path(raw_path) if Path(raw_path).is_absolute() else base / raw_path.replace("\\", "/")
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        ok = bool(
            exists
            and actual_size == int(row["size_bytes"])
            and actual_hash == str(row["sha256"]).lower()
        )
        if not ok:
            errors.append(f"Registered upstream hash mismatch: {path}")
        details.append({
            "registry": str(table), "role": row.get("role", "registered_upstream_asset"), "path": str(path),
            "size_bytes": actual_size, "sha256": actual_hash, "verified": yn(ok),
        })
    return details


def verify_positive_input_manifest(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in read_csv(path).fillna("").to_dict("records"):
        raw = str(row["absolute_path"])
        if "a26" in raw.lower():
            errors.append(f"Prohibited external A26 path in manifest; not read: {raw}")
            continue
        p = Path(raw)
        exists = p.is_file()
        actual_size = p.stat().st_size if exists else None
        actual_hash = sha256(p) if exists else None
        expected = str(row["registered_sha256"]).lower()
        ok = bool(
            exists and actual_size == int(row["size_bytes"])
            and actual_hash == expected
            and str(row["actual_sha256"]).lower() == expected
            and str(row["sha256_verified"]).upper() == "YES"
            and str(row["read_only"]).upper() == "YES"
        )
        if not ok:
            errors.append(f"Positive registry input manifest mismatch: {raw}")
        details.append({
            "registry": str(path), "role": row["input_role"], "path": raw,
            "size_bytes": actual_size, "sha256": actual_hash, "verified": yn(ok),
        })
    return details


def frame_equal_normalized(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], geometry: bool = False) -> bool:
    return normalized_hash(a, keys, geometry) == normalized_hash(b, keys, geometry)


def write_attribute_formats(frame: pd.DataFrame, stem: Path, keys: list[str]) -> dict[str, Any]:
    df = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    parquet = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    df.to_parquet(parquet, index=False)
    write_csv(df, csv_path)
    pq = pd.read_parquet(parquet)
    cs = read_csv(csv_path)
    semantic = normalized_hash(df, keys)
    return {
        "paths": [parquet, csv_path], "rows": len(df), "normalized_hash": semantic,
        "reload_pass": frame_equal_normalized(df, pq, keys) and frame_equal_normalized(df, cs, keys),
    }


def write_spatial_formats(
    frame: gpd.GeoDataFrame, stem: Path, keys: list[str], layer: str
) -> dict[str, Any]:
    gdf = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    parquet = stem.with_suffix(".parquet")
    gpkg = stem.with_suffix(".gpkg")
    csv_path = stem.with_suffix(".csv")
    gdf.to_parquet(parquet, index=False)
    gdf.to_file(gpkg, layer=layer, driver="GPKG", index=False)
    write_csv(pd.DataFrame(gdf.drop(columns=gdf.geometry.name)), csv_path)
    pq = gpd.read_parquet(parquet)
    gp = gpd.read_file(gpkg, layer=layer)
    cs = read_csv(csv_path)
    attrs = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    semantic = normalized_hash(gdf, keys, include_geometry=True)
    reload_pass = (
        pq.crs is not None and gp.crs is not None
        and pq.crs.to_epsg() == 6671 and gp.crs.to_epsg() == 6671 and pq.crs == gp.crs
        and frame_equal_normalized(gdf, pq, keys, True)
        and frame_equal_normalized(gdf, gp, keys, True)
        and frame_equal_normalized(attrs, cs, keys)
    )
    return {
        "paths": [parquet, gpkg, csv_path], "rows": len(gdf),
        "normalized_hash": semantic, "geometry_hash": geometry_hash(gdf, keys[0]),
        "reload_pass": reload_pass,
    }


def duplicate_count(frame: pd.DataFrame, keys: list[str]) -> int:
    return int(frame.duplicated(keys, keep=False).sum())


def record_check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> bool:
    passed = actual == expected
    checks.append({"check": name, "actual": actual, "expected": expected, "pass": yn(passed)})
    return passed


def build_field_dictionary(
    tables: dict[str, tuple[pd.DataFrame, list[str], str]], registry_dir: Path
) -> pd.DataFrame:
    foreign = {
        "unit_id": "master_grid.unit_id",
        "primary_unit_id": "master_grid.unit_id",
        "ajg_entity_id": "ajg_entity_registry.landslide_entity_id",
        "landslide_entity_id": "",
        "evidence_id": "",
        "gsi_evidence_id": "inventory_evidence.evidence_id",
    }
    rows: list[dict[str, Any]] = []
    for table, (frame, keys, source) in tables.items():
        geom_name = frame.geometry.name if isinstance(frame, gpd.GeoDataFrame) else None
        for col in frame.columns:
            lower = col.lower()
            leakage = any(term in lower for term in LEAKAGE_TERMS)
            is_key = col in keys
            rows.append({
                "table_name": table,
                "field_name": col,
                "data_type": "geometry" if col == geom_name else str(frame[col].dtype),
                "primary_key": yn(is_key),
                "foreign_key": foreign.get(col, ""),
                "nullable": yn(bool(frame[col].isna().any())),
                "field_source": source,
                "field_role": "GEOMETRY" if col == geom_name else ("PRIMARY_KEY" if is_key else "REGISTRY_ATTRIBUTE"),
                "model_input_allowed": "NO" if leakage or col == geom_name or is_key else "REVIEW_REQUIRED",
                "audit_only": yn(leakage or is_key),
                "leakage_risk": yn(leakage or "id" in lower),
            })
    return pd.DataFrame(rows).sort_values(["table_name", "field_name"])


def print_summary(audit: dict[str, Any]) -> None:
    keys = [
        "PREVIOUS_POSITIVE_REGISTRY_PASS", "OUTSIDE_ASSIGNMENT_VERIFICATION_PASS",
        "MASTER_GRID_COUNT", "AJG_ENTITY_COUNT", "AJG_IN_DOMAIN_ENTITY_COUNT",
        "AJG_OUTSIDE_ENTITY_COUNT", "PRIMARY_POSITIVE_GRID_COUNT",
        "AJG_AFFECTED_GRID_COUNT", "GSI_EVIDENCE_COUNT", "INVENTORY_EVIDENCE_COUNT",
        "CONTROL_EXCLUSION_GRID_COUNT", "PRIMARY_POSITIVE_SUBSET_AFFECTED_PASS",
        "AJG_AFFECTED_SUBSET_CONTROL_EXCLUSION_PASS", "FOREIGN_KEY_INTEGRITY_PASS",
        "OUTSIDE_ENTITY_ASSIGNMENT_PASS", "GSI_AUXILIARY_ROLE_PASS",
        "GEOMETRY_UNCHANGED_PASS", "CROSS_FORMAT_RELOAD_PASS", "LABEL_FIELD_CREATED",
        "HARD_CONTROLS_CREATED", "PAIR_TABLE_CREATED", "WARNINGS", "ERRORS",
        "FINAL_DECISION", "NEXT_STEP",
    ]
    for key in keys:
        value = audit[key]
        if key == "ERRORS" and not value:
            value = "NONE"
        print(f"{key}: {value}")


def validate_existing(target: Path, current_inputs: dict[str, str]) -> dict[str, Any] | None:
    marker = target / "03_audit/POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN.marker.json"
    report = target / "03_audit/positive_evidence_grid_freeze_report_v1.json"
    if not marker.is_file() or not report.is_file():
        return None
    m = json_load(marker)
    audit = json_load(report)
    if m.get("current_FINAL_DECISION") != SUCCESS or audit.get("FINAL_DECISION") != SUCCESS:
        return None
    recorded = {item["path"]: item["sha256"] for item in m.get("input_hashes", [])}
    if recorded != current_inputs:
        return None
    return audit


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reliable = root / "data/02_reliable_observation_domain"
    first = root / "data/01_county_prediction_domain"
    coverage = reliable / "04_gsi_coverage_grid_mapping"
    normalization = reliable / "06_source_internal_entity_normalization"
    matching = reliable / "07_ajg_gsi_cross_source_matching"
    registry = reliable / "08_positive_evidence_grid_registry"
    target = reliable / "09_positive_evidence_grid_registry_frozen"
    errors: list[str] = []
    warnings: list[str] = []
    unresolved: list[str] = []
    checks: list[dict[str, Any]] = []

    paths = {
        "first_marker": first / "06_audit/FIRST_LAYER_FROZEN.marker.json",
        "first_report": first / "06_audit/first_layer_freeze_report.json",
        "master_grid": first / "02_grid_250m/hiroshima_grid_250m_master.gpkg",
        "coverage_grid": coverage / "01_mapped_grid/gsi_coverage_mapping_250m_v1.gpkg",
        "coverage_audit": coverage / "03_audit/gsi_coverage_grid_mapping_audit_v1.json",
        "normalization_audit": normalization / "07_audit/source_internal_entity_normalization_audit_v1.json",
        "ajg_canonical": normalization / "02_ajg_entities/ajg_canonical_landslide_entities_v1.gpkg",
        "gsi_canonical": normalization / "03_gsi_entities/gsi_canonical_landslide_entities_v1.gpkg",
        "matching_audit": matching / "07_audit/ajg_gsi_cross_source_matching_audit_v1.json",
        "support_summary": matching / "04_ajg_summary/ajg_entity_gsi_evidence_summary_v1.csv",
        "positive_manifest": registry / "00_input_registry/positive_evidence_grid_registry_input_manifest_v1.csv",
        "inventory": registry / "01_inventory_evidence/inventory_evidence_registry_v1.csv",
        "ajg_registry": registry / "02_positive_entities/primary_positive_entity_registry_v1.gpkg",
        "ajg_registry_csv": registry / "02_positive_entities/primary_positive_entity_registry_attributes_v1.csv",
        "ajg_links": registry / "03_ajg_grid_links/ajg_entity_to_250m_grid_links_v1.csv",
        "gsi_links": registry / "04_gsi_grid_links/gsi_evidence_to_250m_grid_links_v1.csv",
        "positive_grids": registry / "05_positive_grid_registry/primary_positive_grid_registry_v1.gpkg",
        "control_grids": registry / "06_control_exclusion_grid/positive_and_evidence_control_exclusion_grid_v1.gpkg",
        "positive_schema": registry / "07_schema/positive_evidence_grid_registry_schema_v1.csv",
        "positive_audit": registry / "08_audit/positive_evidence_grid_registry_audit_v1.json",
        "outside_registry": registry / "10_outside_hiroshima_verification/outside_hiroshima_entity_verification_v1.csv",
        "outside_audit": registry / "10_outside_hiroshima_verification/outside_hiroshima_entity_verification_audit_v1.json",
    }
    search_roots = {
        "first_marker": first, "master_grid": first, "coverage_audit": coverage,
        "normalization_audit": normalization, "matching_audit": matching,
        "positive_audit": registry / "08_audit", "outside_audit": registry / "10_outside_hiroshima_verification",
    }
    for name, search_root in search_roots.items():
        unique_candidate(search_root, paths[name], name, errors)
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"Missing formal input {name}: {path}")
    if errors:
        print("ERRORS:", errors)
        print("FINAL_DECISION:", FAILURE)
        return 2

    first_marker = json_load(paths["first_marker"])
    first_report = json_load(paths["first_report"])
    coverage_audit = json_load(paths["coverage_audit"])
    norm_audit = json_load(paths["normalization_audit"])
    match_audit = json_load(paths["matching_audit"])
    positive_audit = json_load(paths["positive_audit"])
    outside_audit = json_load(paths["outside_audit"])

    decision_checks = {
        "FIRST_LAYER_FROZEN": first_marker.get("status") == "FIRST_LAYER_FROZEN" and first_report.get("final_decision") == "FIRST_LAYER_FROZEN",
        "GSI_COVERAGE": coverage_audit.get("final_decision") == "GSI_COVERAGE_MAPPED_TO_GRID_250M" and not coverage_audit.get("errors"),
        "SOURCE_NORMALIZATION": norm_audit.get("FINAL_DECISION") == "SOURCE_INTERNAL_ENTITY_NORMALIZATION_COMPLETE" and not norm_audit.get("ERRORS"),
        "CROSS_SOURCE_MATCHING": match_audit.get("FINAL_DECISION") == "AJG_GSI_CROSS_SOURCE_EVIDENCE_MATCHING_COMPLETE" and not match_audit.get("ERRORS"),
        "POSITIVE_REGISTRY": positive_audit.get("FINAL_DECISION") == "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_COMPLETE" and not positive_audit.get("ERRORS"),
        "OUTSIDE_ASSIGNMENT": outside_audit.get("FINAL_DECISION") == "OUTSIDE_HIROSHIMA_ENTITY_ASSIGNMENT_VERIFIED" and not outside_audit.get("ERRORS"),
    }
    for name, passed in decision_checks.items():
        if not passed:
            errors.append(f"Prior decision check failed: {name}")

    input_details: list[dict[str, Any]] = []
    input_details.extend(verify_positive_input_manifest(paths["positive_manifest"], errors))
    input_details.extend(verify_registered_hashes(
        coverage, coverage / "03_audit/gsi_coverage_grid_mapping_file_hashes_v1.csv", errors
    ))
    input_details.extend(verify_registered_hashes(
        normalization, normalization / "07_audit/source_internal_entity_normalization_file_hashes_v1.csv", errors
    ))
    input_details.extend(verify_registered_hashes(
        matching, matching / "07_audit/ajg_gsi_cross_source_matching_file_hashes_v1.csv", errors
    ))
    input_details.extend(verify_registered_hashes(
        registry, registry / "08_audit/positive_evidence_grid_registry_file_hashes_v1.csv", errors
    ))
    # Add every explicit formal input, including audits not covered by a prior
    # table, exactly once to the freeze manifest.
    detail_paths = {item["path"] for item in input_details}
    for role, path in paths.items():
        if str(path) not in detail_paths:
            input_details.append({
                "registry": "explicit_formal_input", "role": role, "path": str(path),
                "size_bytes": path.stat().st_size, "sha256": sha256(path), "verified": "YES",
            })
    current_inputs = {item["path"]: item["sha256"] for item in input_details}

    if target.exists():
        existing = validate_existing(target, current_inputs)
        if existing is None:
            print(f"ERRORS: Existing target is not a valid idempotent freeze: {target}")
            print(f"FINAL_DECISION: {FAILURE}")
            return 2
        print_summary(existing)
        return 0

    # Load only the approved formal assets. No model features, rainfall, or A26.
    master = gpd.read_file(paths["master_grid"], layer="hiroshima_grid_250m_master")
    coverage_grid = gpd.read_file(paths["coverage_grid"])
    ajg_canonical = gpd.read_file(paths["ajg_canonical"], layer="ajg_canonical_entities")
    gsi_canonical = gpd.read_file(paths["gsi_canonical"], layer="gsi_canonical_entities")
    ajg = gpd.read_file(paths["ajg_registry"], layer="primary_positive_entities")
    inventory = read_csv(paths["inventory"])
    ajg_links = read_csv(paths["ajg_links"])
    gsi_links = read_csv(paths["gsi_links"])
    positive = gpd.read_file(paths["positive_grids"], layer="primary_positive_grids")
    control = gpd.read_file(paths["control_grids"], layer="control_exclusion_grids")
    affected = control[control["ajg_affected_flag"].astype(str).str.upper() == "YES"].copy()
    support = read_csv(paths["support_summary"])
    outside = read_csv(paths["outside_registry"])

    master_ids = set(master["unit_id"].astype(str))
    ajg_ids = set(ajg["landslide_entity_id"].astype(str))
    outside_ids = set(ajg.loc[ajg["county_relation_status"] == "OUTSIDE_HIROSHIMA", "landslide_entity_id"].astype(str))
    in_domain = ajg[ajg["county_relation_status"].isin(["INSIDE_HIROSHIMA", "PARTIAL_BOUNDARY"])]
    in_domain_ids = set(in_domain["landslide_entity_id"].astype(str))
    positive_ids = set(positive["unit_id"].astype(str))
    affected_ids = set(affected["unit_id"].astype(str))
    control_ids = set(control["unit_id"].astype(str))
    gsi_inventory = inventory[inventory["source"].astype(str).str.upper() == "GSI"]
    ajg_inventory = inventory[inventory["source"].astype(str).str.upper() == "AJG"]

    # Benchmark quantities.
    benchmarks = {
        "MASTER_GRID_COUNT": (len(master), 139364),
        "AJG_ENTITY_COUNT": (len(ajg), 7521),
        "AJG_INSIDE_HIROSHIMA_COUNT": (int((ajg["county_relation_status"] == "INSIDE_HIROSHIMA").sum()), 7402),
        "AJG_PARTIAL_BOUNDARY_COUNT": (int((ajg["county_relation_status"] == "PARTIAL_BOUNDARY").sum()), 39),
        "AJG_IN_DOMAIN_ENTITY_COUNT": (len(in_domain), 7441),
        "AJG_OUTSIDE_HIROSHIMA_COUNT": (len(outside_ids), 80),
        "UNIQUE_PRIMARY_POSITIVE_GRID_COUNT": (len(positive_ids), 5337),
        "AJG_AFFECTED_UNIQUE_GRID_COUNT": (len(affected_ids), 8478),
        "GRID_WITH_MULTIPLE_AJG_ENTITIES_COUNT": (int((positive["positive_entity_count"] > 1).sum()), 1455),
        "MAX_AJG_ENTITY_COUNT_PER_GRID": (int(positive["positive_entity_count"].max()), 8),
        "GSI_EVIDENCE_COUNT": (len(gsi_canonical), 11595),
        "GSI_EVIDENCE_WITH_GRID_LINK_COUNT": (gsi_links["gsi_evidence_id"].nunique(), 11501),
        "GSI_OUTSIDE_HIROSHIMA_COUNT": (int((gsi_inventory["county_relation_status"] == "OUTSIDE_HIROSHIMA").sum()), 94),
        "GSI_EVIDENCE_UNIQUE_GRID_COUNT": (gsi_links["unit_id"].nunique(), 4846),
        "INVENTORY_EVIDENCE_REGISTRY_COUNT": (len(inventory), 19116),
        "CONTROL_EXCLUSION_UNIQUE_GRID_COUNT": (len(control_ids), 8704),
        "MAKE_VALID_ENTITY_COUNT": (int((ajg["make_valid_flag"] == "YES").sum()), 66),
        "MAKE_VALID_PRIMARY_GRID_COUNT": (ajg.loc[ajg["make_valid_flag"] == "YES", "primary_unit_id"].nunique(), 66),
        "MAKE_VALID_ENTITY_OUTSIDE_HIROSHIMA_COUNT": (int(((ajg["make_valid_flag"] == "YES") & (ajg["county_relation_status"] == "OUTSIDE_HIROSHIMA")).sum()), 0),
    }
    for name, (actual, expected) in benchmarks.items():
        if not record_check(checks, name, int(actual), expected):
            errors.append(f"Benchmark failed: {name}={actual}, expected {expected}")

    # Outside verification must agree exactly with its formal prior audit.
    outside_benchmarks = {
        "OUTSIDE_HIROSHIMA_ENTITY_COUNT": 80,
        "OUTSIDE_WITH_NON_NULL_PRIMARY_UNIT_COUNT": 0,
        "OUTSIDE_PRIMARY_CANDIDATE_YES_COUNT": 0,
        "OUTSIDE_USE_FOR_TRAINING_TRUE_COUNT": 0,
        "OUTSIDE_ENTITY_PRESENT_IN_AJG_GRID_LINK_COUNT": 0,
        "OUTSIDE_ENTITY_PRESENT_IN_POSITIVE_GRID_COUNT": 0,
        "OUTSIDE_ENTITY_PRESENT_IN_INVENTORY_EVIDENCE_COUNT": 80,
    }
    for name, expected in outside_benchmarks.items():
        actual = outside_audit.get(name)
        if not record_check(checks, name, actual, expected):
            errors.append(f"Outside audit benchmark failed: {name}={actual}, expected {expected}")

    # Set and key integrity.
    primary_subset = positive_ids <= affected_ids
    affected_subset = affected_ids <= control_ids
    all_unit_ids = (
        set(ajg_links["unit_id"].astype(str)) | set(gsi_links["unit_id"].astype(str)) |
        positive_ids | affected_ids | control_ids |
        set(in_domain["primary_unit_id"].dropna().astype(str))
    )
    foreign_key_pass = all_unit_ids <= master_ids
    primary_link_rows = ajg_links[ajg_links["is_primary_unit"].astype(str).str.upper() == "YES"]
    primary_link_counts = primary_link_rows.groupby("ajg_entity_id").size()
    in_domain_one_primary = (
        in_domain["primary_unit_id"].notna().all()
        and in_domain["primary_unit_id"].astype(str).str.strip().ne("").all()
        and set(primary_link_counts.index.astype(str)) == in_domain_ids
        and bool((primary_link_counts == 1).all())
    )
    registry_primary = dict(zip(in_domain["landslide_entity_id"].astype(str), in_domain["primary_unit_id"].astype(str)))
    link_primary = dict(zip(primary_link_rows["ajg_entity_id"].astype(str), primary_link_rows["unit_id"].astype(str)))
    in_domain_one_primary = in_domain_one_primary and registry_primary == link_primary
    outside_assignment_pass = (
        ajg.loc[ajg["landslide_entity_id"].astype(str).isin(outside_ids), "primary_unit_id"]
        .fillna("").astype(str).str.strip().eq("").all()
        and not bool(ajg_links["ajg_entity_id"].astype(str).isin(outside_ids).any())
        and not bool(outside_ids & set(";".join(positive["positive_entity_ids"].fillna("").astype(str)).split(";")))
        and outside_ids <= set(ajg_inventory["evidence_id"].astype(str))
    )
    inventory_composition_pass = (
        len(inventory) == len(ajg) + len(gsi_canonical)
        and set(ajg_inventory["evidence_id"].astype(str)) == ajg_ids
        and set(gsi_inventory["evidence_id"].astype(str)) == set(gsi_canonical["source_entity_id"].astype(str))
    )
    gsi_auxiliary_pass = (
        (gsi_inventory["primary_positive_candidate_flag"].astype(str).str.upper() == "NO").all()
        and (gsi_inventory["use_for_primary_training"].astype(str).str.upper() == "NO").all()
        and (gsi_inventory["source_role"].astype(str) != "PRIMARY_AJG_EVIDENCE").all()
        and (gsi_inventory["exclude_from_future_control_pool"].astype(str).str.upper() == "YES").all()
    )
    duplicate_checks = {
        "ajg_entity": duplicate_count(ajg, ["landslide_entity_id"]),
        "inventory": duplicate_count(inventory, ["evidence_id"]),
        "ajg_links": duplicate_count(ajg_links, ["ajg_entity_id", "unit_id"]),
        "positive_grid": duplicate_count(positive, ["unit_id"]),
        "affected_grid": duplicate_count(affected, ["unit_id"]),
        "gsi_links": duplicate_count(gsi_links, ["gsi_evidence_id", "unit_id"]),
        "control_grid": duplicate_count(control, ["unit_id"]),
        "support_summary": duplicate_count(support, ["ajg_entity_id"]),
        "outside_registry": duplicate_count(outside, ["landslide_entity_id"]),
    }
    duplicate_pass = all(value == 0 for value in duplicate_checks.values())
    if not primary_subset: errors.append("Primary positive grids are not a subset of AJG affected grids")
    if not affected_subset: errors.append("AJG affected grids are not a subset of control exclusion grids")
    if not foreign_key_pass: errors.append(f"Unknown unit IDs: {sorted(all_unit_ids - master_ids)[:100]}")
    if not in_domain_one_primary: errors.append("In-domain AJG entities do not have exactly one consistent primary unit")
    if not outside_assignment_pass: errors.append("Outside-Hiroshima entity assignment relation failed")
    if not inventory_composition_pass: errors.append("Inventory is not exactly AJG 7,521 plus GSI 11,595")
    if not gsi_auxiliary_pass: errors.append("GSI auxiliary-only evidence role failed")
    if not duplicate_pass: errors.append(f"Duplicate primary-key rows: {duplicate_checks}")

    # Source and registered geometry identity, without modifying either.
    source_hash_before = {name: sha256(paths[name]) for name in ("master_grid", "ajg_canonical", "gsi_canonical")}
    crs_pass = all(str(g.crs).upper() == CRS for g in (master, coverage_grid, ajg_canonical, gsi_canonical, ajg, positive, control))
    canonical_ajg_geom = geometry_hash(ajg_canonical.rename(columns={"source_entity_id": "landslide_entity_id"}), "landslide_entity_id")
    registered_ajg_geom = geometry_hash(ajg, "landslide_entity_id")
    geometry_identity_pass = canonical_ajg_geom == registered_ajg_geom and crs_pass

    tmp = target.parent / "09_positive_evidence_grid_registry_frozen_staging"
    if tmp.exists():
        print(f"ERRORS: Staging directory already exists and will not be overwritten: {tmp}")
        print(f"FINAL_DECISION: {FAILURE}")
        return 2
    tmp.mkdir(parents=False, exist_ok=False)
    try:
        manifest_dir = tmp / "00_manifest"
        frozen_dir = tmp / "01_frozen_registry"
        schema_dir = tmp / "02_schema_and_roles"
        audit_dir = tmp / "03_audit"
        preview_dir = tmp / "04_preview"
        for directory in (manifest_dir, frozen_dir, schema_dir, audit_dir, preview_dir):
            directory.mkdir(parents=True, exist_ok=True)

        tables: dict[str, tuple[pd.DataFrame, list[str], str]] = {
            "ajg_entity_registry": (ajg, ["landslide_entity_id"], str(paths["ajg_registry"])),
            "inventory_evidence_registry": (inventory, ["evidence_id"], str(paths["inventory"])),
            "ajg_entity_grid_links_250m": (ajg_links, ["ajg_entity_id", "unit_id"], str(paths["ajg_links"])),
            "primary_positive_grid_registry_250m": (positive, ["unit_id"], str(paths["positive_grids"])),
            "ajg_affected_grid_registry_250m": (affected, ["unit_id"], str(paths["control_grids"])),
            "gsi_evidence_grid_links_250m": (gsi_links, ["gsi_evidence_id", "unit_id"], str(paths["gsi_links"])),
            "control_exclusion_grid_registry_250m": (control, ["unit_id"], str(paths["control_grids"])),
            "ajg_gsi_evidence_support_summary": (support, ["ajg_entity_id"], str(paths["support_summary"])),
            "outside_hiroshima_entity_registry": (outside, ["landslide_entity_id"], str(paths["outside_registry"])),
        }
        output_specs = {
            "ajg_entity_registry": ("ajg_entity_registry_frozen_v1", True, "ajg_entity_registry_frozen"),
            "inventory_evidence_registry": ("inventory_evidence_registry_frozen_v1", False, ""),
            "ajg_entity_grid_links_250m": ("ajg_entity_grid_links_250m_frozen_v1", False, ""),
            "primary_positive_grid_registry_250m": ("primary_positive_grid_registry_250m_frozen_v1", True, "primary_positive_grid_registry_250m_frozen"),
            "ajg_affected_grid_registry_250m": ("ajg_affected_grid_registry_250m_frozen_v1", True, "ajg_affected_grid_registry_250m_frozen"),
            "gsi_evidence_grid_links_250m": ("gsi_evidence_grid_links_250m_frozen_v1", False, ""),
            "control_exclusion_grid_registry_250m": ("control_exclusion_grid_registry_250m_frozen_v1", True, "control_exclusion_grid_registry_250m_frozen"),
            "ajg_gsi_evidence_support_summary": ("ajg_gsi_evidence_support_summary_frozen_v1", False, ""),
            "outside_hiroshima_entity_registry": ("outside_hiroshima_entity_registry_frozen_v1", False, ""),
        }
        results: dict[str, dict[str, Any]] = {}
        for name, (frame, keys, _) in tables.items():
            stem_name, spatial, layer = output_specs[name]
            stem = frozen_dir / stem_name
            results[name] = (
                write_spatial_formats(frame, stem, keys, layer)
                if spatial else write_attribute_formats(frame, stem, keys)
            )
        cross_format_pass = all(item["reload_pass"] for item in results.values())
        if not cross_format_pass:
            errors.append("One or more cross-format reload/content checks failed")

        # Required schema and role documentation.
        shutil.copy2(paths["positive_schema"], schema_dir / "positive_evidence_grid_registry_schema_v1.csv")
        roles = pd.DataFrame([
            {"evidence_role": "AJG_PRIMARY_POSITIVE_EVIDENCE", "source": "AJG", "meaning": "In-domain AJG entity may define a primary positive grid", "may_be_primary_positive": "YES", "may_be_negative": "NO", "model_input_allowed": "NO"},
            {"evidence_role": "OUT_OF_DOMAIN_EVIDENCE", "source": "AJG", "meaning": "Outside-Hiroshima AJG entity retained as evidence only", "may_be_primary_positive": "NO", "may_be_negative": "NO", "model_input_allowed": "NO"},
            {"evidence_role": "AJG_SUPPORT_EVIDENCE", "source": "GSI", "meaning": "GSI evidence supporting an AJG entity", "may_be_primary_positive": "NO", "may_be_negative": "NO", "model_input_allowed": "NO"},
            {"evidence_role": "AUXILIARY_EVIDENCE", "source": "GSI", "meaning": "Supplementary or unmatched GSI evidence", "may_be_primary_positive": "NO", "may_be_negative": "NO", "model_input_allowed": "NO"},
            {"evidence_role": "CONTROL_EXCLUSION", "source": "AJG/GSI", "meaning": "Grid prohibited from future negative-control eligibility; neither positive nor negative label", "may_be_primary_positive": "NO", "may_be_negative": "NO", "model_input_allowed": "NO"},
        ])
        write_csv(roles, schema_dir / "evidence_role_dictionary_v1.csv")
        field_dict = build_field_dictionary(tables, frozen_dir)
        write_csv(field_dict, schema_dir / "frozen_field_dictionary_v1.csv")

        output_columns = {str(c).lower() for frame, _, _ in tables.values() for c in frame.columns}
        forbidden_created = sorted(output_columns & FORBIDDEN_CREATED_FIELDS)
        if forbidden_created:
            errors.append(f"Forbidden label/control/pair fields present: {forbidden_created}")

        # Copy only the already generated audit previews; no new decisions are inferred.
        for source in sorted((registry / "09_preview").glob("*.png")):
            shutil.copy2(source, preview_dir / source.name)

        source_hash_after = {name: sha256(paths[name]) for name in source_hash_before}
        geometry_unchanged_pass = geometry_identity_pass and source_hash_before == source_hash_after
        if not geometry_unchanged_pass:
            errors.append("AJG/GSI/master source geometry identity or immutable file hash check failed")

        # Input manifest contains both file and semantic hashes where applicable.
        semantic_inputs = {
            str(paths["master_grid"]): normalized_hash(master, ["unit_id"], True),
            str(paths["ajg_canonical"]): normalized_hash(ajg_canonical, ["source_entity_id"], True),
            str(paths["gsi_canonical"]): normalized_hash(gsi_canonical, ["source_entity_id"], True),
            str(paths["ajg_registry"]): normalized_hash(ajg, ["landslide_entity_id"], True),
            str(paths["inventory"]): normalized_hash(inventory, ["evidence_id"]),
            str(paths["ajg_links"]): normalized_hash(ajg_links, ["ajg_entity_id", "unit_id"]),
            str(paths["gsi_links"]): normalized_hash(gsi_links, ["gsi_evidence_id", "unit_id"]),
            str(paths["positive_grids"]): normalized_hash(positive, ["unit_id"], True),
            str(paths["control_grids"]): normalized_hash(control, ["unit_id"], True),
            str(paths["support_summary"]): normalized_hash(support, ["ajg_entity_id"]),
            str(paths["outside_registry"]): normalized_hash(outside, ["landslide_entity_id"]),
        }
        input_manifest_rows = []
        for item in sorted(input_details, key=lambda x: x["path"]):
            input_manifest_rows.append({
                "input_role": item.get("role", "registered_upstream_asset"),
                "absolute_path": item["path"], "size_bytes": item["size_bytes"],
                "sha256": item["sha256"], "normalized_content_sha256": semantic_inputs.get(item["path"], "NOT_APPLICABLE"),
                "hash_verified": item["verified"], "read_only": "YES",
            })
        input_manifest = pd.DataFrame(input_manifest_rows).drop_duplicates("absolute_path").sort_values("absolute_path")
        write_csv(input_manifest, manifest_dir / "positive_evidence_freeze_input_manifest_v1.csv")

        relation_audit = {
            "PRIMARY_POSITIVE_SUBSET_AFFECTED_PASS": yn(primary_subset),
            "AJG_AFFECTED_SUBSET_CONTROL_EXCLUSION_PASS": yn(affected_subset),
            "FOREIGN_KEY_INTEGRITY_PASS": yn(foreign_key_pass),
            "IN_DOMAIN_EXACTLY_ONE_PRIMARY_UNIT_PASS": yn(in_domain_one_primary),
            "OUTSIDE_ENTITY_ASSIGNMENT_PASS": yn(outside_assignment_pass),
            "INVENTORY_COMPOSITION_PASS": yn(inventory_composition_pass),
            "GSI_AUXILIARY_ROLE_PASS": yn(gsi_auxiliary_pass),
            "DUPLICATE_PRIMARY_KEY_PASS": yn(duplicate_pass),
            "DUPLICATE_COUNTS": duplicate_checks,
            "UNKNOWN_UNIT_IDS": sorted(all_unit_ids - master_ids),
            "BENCHMARK_CHECKS": checks,
        }
        (audit_dir / "positive_evidence_grid_consistency_audit_v1.json").write_text(
            json.dumps(relation_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        final = SUCCESS if not errors else FAILURE
        audit = {
            "PROJECT_ROOT": str(root),
            "PREVIOUS_POSITIVE_REGISTRY_PASS": yn(decision_checks["POSITIVE_REGISTRY"]),
            "OUTSIDE_ASSIGNMENT_VERIFICATION_PASS": yn(decision_checks["OUTSIDE_ASSIGNMENT"]),
            "MASTER_GRID_COUNT": len(master), "AJG_ENTITY_COUNT": len(ajg),
            "AJG_IN_DOMAIN_ENTITY_COUNT": len(in_domain), "AJG_OUTSIDE_ENTITY_COUNT": len(outside_ids),
            "PRIMARY_POSITIVE_GRID_COUNT": len(positive_ids), "AJG_AFFECTED_GRID_COUNT": len(affected_ids),
            "GSI_EVIDENCE_COUNT": len(gsi_canonical), "INVENTORY_EVIDENCE_COUNT": len(inventory),
            "CONTROL_EXCLUSION_GRID_COUNT": len(control_ids),
            "PRIMARY_POSITIVE_SUBSET_AFFECTED_PASS": yn(primary_subset),
            "AJG_AFFECTED_SUBSET_CONTROL_EXCLUSION_PASS": yn(affected_subset),
            "FOREIGN_KEY_INTEGRITY_PASS": yn(foreign_key_pass and duplicate_pass and in_domain_one_primary),
            "OUTSIDE_ENTITY_ASSIGNMENT_PASS": yn(outside_assignment_pass),
            "GSI_AUXILIARY_ROLE_PASS": yn(gsi_auxiliary_pass),
            "GEOMETRY_UNCHANGED_PASS": yn(geometry_unchanged_pass),
            "CROSS_FORMAT_RELOAD_PASS": yn(cross_format_pass),
            "LABEL_FIELD_CREATED": "NO", "HARD_CONTROLS_CREATED": "NO", "PAIR_TABLE_CREATED": "NO",
            "MODEL_FEATURES_READ": "NO", "RAINFALL_DATA_READ": "NO", "EXTERNAL_A26_CONTENT_READ": "NO",
            "WARNINGS": warnings, "ERRORS": errors, "FINAL_DECISION": final, "NEXT_STEP": NEXT_STEP,
        }
        freeze_report = {
            **audit,
            "freeze_version": "v1",
            "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "CRS": CRS,
            "previous_FINAL_DECISION": {
                "first_layer": first_marker.get("status"),
                "positive_registry": positive_audit.get("FINAL_DECISION"),
                "outside_assignment": outside_audit.get("FINAL_DECISION"),
            },
            "core_registry_rows": {name: result["rows"] for name, result in results.items()},
            "normalized_output_hashes": {name: result["normalized_hash"] for name, result in results.items()},
            "source_geometry_file_hash_before": source_hash_before,
            "source_geometry_file_hash_after": source_hash_after,
            "unresolved_items": unresolved,
        }
        (audit_dir / "positive_evidence_grid_freeze_report_v1.json").write_text(
            json.dumps(freeze_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report_lines = [
            "# Positive Evidence and 250 m Grid Registry Freeze Report", "",
            f"- Final decision: `{final}`", f"- CRS: `{CRS}`",
            f"- Master grid: {len(master):,}", f"- AJG entities: {len(ajg):,}",
            f"- Primary positive grids: {len(positive_ids):,}", f"- AJG affected grids: {len(affected_ids):,}",
            f"- GSI evidence: {len(gsi_canonical):,}", f"- Inventory evidence: {len(inventory):,}",
            f"- Control exclusion grids: {len(control_ids):,}", "",
            "All tables were stably sorted and cross-format reloaded. Control exclusion denotes future negative-sample prohibition only; it is neither a positive nor a negative label.",
            "No labels, negative samples, hard controls, pairs, model features, rainfall data, or external A26 content were created or read.",
            "", f"Errors: `{errors if errors else 'NONE'}`", f"Warnings: `{warnings if warnings else 'NONE'}`",
            f"Next step: {NEXT_STEP}",
        ]
        (audit_dir / "POSITIVE_EVIDENCE_GRID_FREEZE_REPORT_v1.md").write_text(
            "\n".join(report_lines) + "\n", encoding="utf-8"
        )

        # Output manifest: core files plus schema/audit/preview created so far.
        output_rows: list[dict[str, Any]] = []
        normalized_by_path: dict[str, str] = {}
        for name, result in results.items():
            for p in result["paths"]:
                normalized_by_path[str(p.relative_to(tmp)).replace("\\", "/")] = result["normalized_hash"]
        for p in sorted(x for x in tmp.rglob("*") if x.is_file()):
            rel = str(p.relative_to(tmp)).replace("\\", "/")
            output_rows.append({
                "relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p),
                "normalized_content_sha256": normalized_by_path.get(rel, "NOT_APPLICABLE"),
            })
        output_manifest_path = manifest_dir / "positive_evidence_freeze_output_manifest_v1.csv"
        write_csv(pd.DataFrame(output_rows), output_manifest_path)

        file_hash_rows = []
        for p in sorted(x for x in tmp.rglob("*") if x.is_file()):
            rel = str(p.relative_to(tmp)).replace("\\", "/")
            file_hash_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": sha256(p)})
        file_hash_path = manifest_dir / "positive_evidence_freeze_file_hashes_v1.csv"
        write_csv(pd.DataFrame(file_hash_rows), file_hash_path)

        if final == SUCCESS:
            marker = {
                "freeze_version": "v1", "freeze_timestamp_utc": freeze_report["freeze_timestamp_utc"],
                "project_root": str(root), "CRS": CRS, "master_grid_count": len(master),
                "core_registry_rows": freeze_report["core_registry_rows"],
                "input_hashes": [{"path": k, "sha256": v} for k, v in sorted(current_inputs.items())],
                "output_hashes": file_hash_rows,
                "file_hash_manifest_sha256": sha256(file_hash_path),
                "normalized_output_hashes": freeze_report["normalized_output_hashes"],
                "previous_FINAL_DECISION": freeze_report["previous_FINAL_DECISION"],
                "current_FINAL_DECISION": final, "unresolved_items": unresolved,
                "warnings": warnings, "errors": errors,
            }
            (audit_dir / "POSITIVE_EVIDENCE_AND_GRID_REGISTRY_FROZEN.marker.json").write_text(
                json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        # Atomic publication only after every success check and marker write.
        os.replace(tmp, target)
        tmp = None
        print_summary(audit)
        return 0 if final == SUCCESS else 2
    finally:
        if tmp is not None and tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
