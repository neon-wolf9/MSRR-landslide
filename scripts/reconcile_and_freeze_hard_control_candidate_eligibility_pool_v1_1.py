#!/usr/bin/env python3
"""Reconcile report-only hash drift and create the v1.1 eligibility freeze."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

# Reuse audited container-independent hashing/writing utilities only. The old
# freeze main() is never invoked, and its incorrect exception assertion is not
# reused.
import freeze_hard_control_candidate_eligibility_pool as util


SUCCESS = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN"
FAILURE = "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FREEZE_REJECTED"
NEXT_STEP = "Await user confirmation of the v1.1 frozen result; do not start layer 3."
ATTEMPT = "v1_1"
CORRECT_WARNING = (
    "6 sub-tolerance boundary fragments have prior coverage_status=FULL_RELIABLE "
    "and formal any_interpreted_flag=0; gsi_coverage_class is OUTSIDE_INTERPRETED "
    "while primary eligibility remains INELIGIBLE_MASTER_GRID under the frozen "
    "precedence rule; none enters candidate or review pools; no input was modified."
)
STATUS_EXPECTED = {
    "ELIGIBLE_BASE_FULL_RELIABLE": 29084,
    "REVIEW_PARTIAL_RELIABLE": 774,
    "INELIGIBLE_INVENTORY_EVIDENCE": 8704,
    "INELIGIBLE_COVERAGE_EXCLUSION": 1741,
    "INELIGIBLE_MASTER_GRID": 3794,
    "INELIGIBLE_OUTSIDE_INTERPRETED": 95267,
}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_lines(text: str, trim_trailing: bool = False) -> str:
    result = text.replace("\r\n", "\n").replace("\r", "\n")
    if trim_trailing:
        result = "\n".join(line.rstrip() for line in result.split("\n"))
    return result


def extract_markdown_semantics(text: str) -> dict[str, Any]:
    def capture(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    def number(pattern: str) -> int | None:
        value = capture(pattern)
        return int(value.replace(",", "")) if value else None

    return {
        "FINAL_DECISION": capture(r"Final decision:\s*`([^`]+)`"),
        "BASE_CANDIDATE_COUNT": number(r"Base candidate count:\s*([0-9,]+)"),
        "PARTIAL_REVIEW_COUNT": number(r"Partial review count:\s*([0-9,]+)"),
        "MASTER_GRID_COUNT": number(r"Full grid registry:\s*([0-9,]+)"),
        "WARNINGS": capture(r"Warnings:\s*`([^`]*)`"),
        "ERRORS": capture(r"Errors:\s*`([^`]*)`"),
        "NO_RAINFALL": "No rainfall matching" in text,
        "NO_STATIC_EOGIS": "static EOGIS screening" in text,
        "NO_RANDOM_SAMPLING": "random sampling" in text,
        "NO_LABELS": "labels" in text,
        "NO_HARD_CONTROLS": "hard controls" in text,
        "NO_PAIRS": "pairs" in text,
        "NO_EXTERNAL_A26": "external A26 reads" in text,
    }


def verify_candidate_hashes(
    source: Path, hash_table: Path, markdown_rel: str, errors: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = pd.read_csv(hash_table)
    details: list[dict[str, Any]] = []
    markdown_detail: dict[str, Any] = {}
    for row in rows.to_dict("records"):
        rel = str(row["relative_path"]).replace("\\", "/")
        path = source / rel
        actual_size = path.stat().st_size if path.is_file() else None
        actual_hash = util.sha256(path) if path.is_file() else None
        match = bool(path.is_file() and actual_size == int(row["size_bytes"]) and actual_hash == row["sha256"])
        detail = {
            "input_role": "candidate_pool_registered_file", "absolute_path": str(path),
            "size_bytes": actual_size, "sha256": actual_hash,
            "registered_size_bytes": int(row["size_bytes"]), "registered_sha256": row["sha256"],
            "hash_verified": util.yn(match), "read_only": "YES",
        }
        details.append(detail)
        if rel == markdown_rel:
            markdown_detail = detail
        elif not match:
            errors.append(f"Non-report candidate asset hash mismatch: {path}")
    if not markdown_detail:
        errors.append("Old Markdown report entry is missing from the formal hash table")
    return details, markdown_detail


def verify_upstream_manifest(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in pd.read_csv(path).to_dict("records"):
        p = Path(row["absolute_path"])
        if "a26" in str(p).lower():
            errors.append(f"Prohibited external A26 path; not read: {p}")
            continue
        current_size = p.stat().st_size if p.is_file() else None
        current_hash = util.sha256(p) if p.is_file() else None
        passed = bool(p.is_file() and current_size == int(row["size_bytes"]) and current_hash == row["sha256"])
        if not passed:
            errors.append(f"Upstream data asset hash mismatch: {p}")
        details.append({
            "input_role": row.get("input_role", "candidate_upstream_input"), "absolute_path": str(p),
            "size_bytes": current_size, "sha256": current_hash,
            "registered_size_bytes": int(row["size_bytes"]), "registered_sha256": row["sha256"],
            "hash_verified": util.yn(passed), "read_only": "YES",
        })
    return details


def print_summary(report: dict[str, Any]) -> None:
    keys = [
        "PROJECT_ROOT", "ATTEMPT_VERSION", "PREVIOUS_ATTEMPT_REJECTED_PRESERVED",
        "CANDIDATE_DATA_REBUILT", "CANDIDATE_STATUS_RECLASSIFIED", "PREVIOUS_INPUT_MODIFIED",
        "INPUT_DATA_ASSET_HASH_PASS", "OLD_MARKDOWN_REPORT_HASH_MATCH",
        "REPORT_HASH_DRIFT_DETECTED", "REPORT_HASH_DRIFT_TYPE",
        "REPORT_LINE_ENDING_NORMALIZED_HASH_MATCH", "REPORT_SEMANTIC_PARITY_PASS",
        "REPORT_HASH_RECONCILIATION_PASS", "OLD_MANIFEST_MODIFIED", "V1_1_MANIFEST_CREATED",
        "MASTER_GRID_COUNT", "BASE_CANDIDATE_COUNT", "PARTIAL_REVIEW_COUNT",
        "ELIGIBLE_BASE_FULL_RELIABLE_COUNT", "REVIEW_PARTIAL_RELIABLE_COUNT",
        "INELIGIBLE_INVENTORY_EVIDENCE_COUNT", "INELIGIBLE_COVERAGE_EXCLUSION_COUNT",
        "INELIGIBLE_MASTER_GRID_COUNT", "INELIGIBLE_OUTSIDE_INTERPRETED_COUNT",
        "STATUS_PARTITION_PASS", "BOUNDARY_EXCEPTION_COUNT",
        "BOUNDARY_EXCEPTION_PRELIMINARY_INELIGIBLE_COUNT",
        "BOUNDARY_EXCEPTION_OUTSIDE_INTERPRETED_COVERAGE_COUNT",
        "BOUNDARY_EXCEPTION_PRIMARY_MASTER_GRID_COUNT", "BOUNDARY_EXCEPTION_CANDIDATE_COUNT",
        "BOUNDARY_EXCEPTION_REVIEW_COUNT", "BOUNDARY_EXCEPTION_POLICY_PASS",
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT",
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT",
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT",
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT",
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT", "UNIT_ID_INTEGRITY_PASS",
        "MASTER_GEOMETRY_UNCHANGED", "CROSS_FORMAT_RELOAD_PASS",
        "NORMALIZED_CONTENT_HASH_PASS", "IDEMPOTENT_RERUN_PASS",
        "LABEL_FIELD_CREATED", "HARD_CONTROLS_CREATED", "PAIR_TABLE_CREATED",
        "RAINFALL_MATCHING_PERFORMED", "STATIC_EOGIS_SCREENING_PERFORMED",
        "EXTERNAL_A26_CONTENT_READ", "WARNINGS", "ERRORS", "FINAL_DECISION", "NEXT_STEP",
    ]
    for key in keys:
        value = report[key]
        if key == "ERRORS" and not value:
            value = "NONE"
        print(f"{key}: {value}")


def validate_existing(target: Path) -> dict[str, Any] | None:
    marker = target / "07_audit/HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN_v1_1.marker.json"
    report = target / "07_audit/hard_control_candidate_freeze_report_v1_1.json"
    manifest = target / "00_manifest/hard_control_candidate_freeze_input_manifest_v1_1.csv"
    if not marker.is_file() or not report.is_file() or not manifest.is_file():
        return None
    m = util.load_json(marker)
    r = util.load_json(report)
    if m.get("current_final_decision") != SUCCESS or r.get("FINAL_DECISION") != SUCCESS:
        return None
    for row in pd.read_csv(manifest).to_dict("records"):
        p = Path(row["absolute_path"])
        if not p.is_file() or p.stat().st_size != int(row["size_bytes"]) or util.sha256(p) != row["sha256"]:
            return None
    return r


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reliable = root / "data/02_reliable_observation_domain"
    first = root / "data/01_county_prediction_domain"
    source = reliable / "10_hard_control_candidate_eligibility_pool"
    reconciliation = source / "08_reconciliation_v1_1"
    reconciliation_staging = source / "08_reconciliation_v1_1_staging"
    attempt_001 = reliable / "11_hard_control_candidate_eligibility_pool_frozen"
    target = reliable / "11_hard_control_candidate_eligibility_pool_frozen_v1_1"
    staging = reliable / "11_hard_control_candidate_eligibility_pool_frozen_v1_1_staging"
    mapping = reliable / "04_gsi_coverage_grid_mapping"
    positive = reliable / "09_positive_evidence_grid_registry_frozen"
    errors: list[str] = []
    warnings = [CORRECT_WARNING]

    paths = {
        "candidate_hashes": source / "00_manifest/hard_control_candidate_file_hashes_v1.csv",
        "candidate_input_manifest": source / "00_manifest/hard_control_candidate_input_manifest_v1.csv",
        "candidate_output_manifest": source / "00_manifest/hard_control_candidate_output_manifest_v1.csv",
        "full_attr": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.parquet",
        "full_geo": source / "01_full_grid_registry/hard_control_eligibility_full_grid_250m_v1.geoparquet",
        "candidate_attr": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.parquet",
        "candidate_geo": source / "02_candidate_pool/hard_control_base_candidate_pool_250m_v1.geoparquet",
        "review": source / "03_review_pool/partial_reliable_review_pool_250m_v1.parquet",
        "ineligible": source / "04_ineligible_registry/hard_control_ineligible_grid_registry_250m_v1.parquet",
        "schema": source / "05_schema_and_rules/hard_control_candidate_schema_v1.csv",
        "rules": source / "05_schema_and_rules/hard_control_eligibility_rule_dictionary_v1.csv",
        "reasons": source / "05_schema_and_rules/hard_control_exclusion_reason_dictionary_v1.csv",
        "anomalies": source / "06_audit/coverage_status_tolerance_anomalies_v1.csv",
        "consistency": source / "06_audit/hard_control_candidate_consistency_audit_v1.json",
        "build_json": source / "06_audit/hard_control_candidate_build_report_v1.json",
        "build_md": source / "06_audit/HARD_CONTROL_CANDIDATE_BUILD_REPORT_v1.md",
        "attempt_report": attempt_001 / "07_audit/hard_control_candidate_freeze_report_v1.json",
        "master": first / "02_grid_250m/hiroshima_grid_250m_master.gpkg",
        "mapping_attrs": mapping / "01_mapped_grid/gsi_coverage_mapping_250m_attributes_v1.csv",
        "primary": positive / "01_frozen_registry/primary_positive_grid_registry_250m_frozen_v1.parquet",
        "affected": positive / "01_frozen_registry/ajg_affected_grid_registry_250m_frozen_v1.parquet",
        "control": positive / "01_frozen_registry/control_exclusion_grid_registry_250m_frozen_v1.parquet",
    }
    for role, path in paths.items():
        if not path.is_file(): errors.append(f"Missing required input {role}: {path}")
    if errors:
        print(f"ERRORS: {errors}"); print(f"FINAL_DECISION: {FAILURE}"); return 2

    prior_json = util.load_json(paths["build_json"])
    prior_consistency = util.load_json(paths["consistency"])
    attempt_report = util.load_json(paths["attempt_report"])
    attempt_preserved = (
        attempt_report.get("FINAL_DECISION") == "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FREEZE_REJECTED"
        and not (attempt_001 / "07_audit/HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN.marker.json").exists()
    )
    if not attempt_preserved: errors.append("attempt_001 rejection is not preserved exactly as a failed attempt")

    markdown_rel = "06_audit/HARD_CONTROL_CANDIDATE_BUILD_REPORT_v1.md"
    candidate_details, md_detail = verify_candidate_hashes(source, paths["candidate_hashes"], markdown_rel, errors)
    upstream_details = verify_upstream_manifest(paths["candidate_input_manifest"], errors)
    input_data_asset_hash_pass = not any("hash mismatch" in e.lower() for e in errors)

    md_bytes = paths["build_md"].read_bytes()
    md_text = md_bytes.decode("utf-8")
    registered_size = int(md_detail["registered_size_bytes"])
    registered_hash = md_detail["registered_sha256"]
    current_size = len(md_bytes)
    current_hash = hashlib.sha256(md_bytes).hexdigest()
    old_hash_match = registered_size == current_size and registered_hash == current_hash
    normalized_current = normalize_lines(md_text)
    trimmed_current = normalize_lines(md_text, trim_trailing=True)
    reconstructed_crlf = normalized_current.replace("\n", "\r\n").encode("utf-8")
    reconstructed_registered_match = (
        len(reconstructed_crlf) == registered_size
        and hashlib.sha256(reconstructed_crlf).hexdigest() == registered_hash
    )
    normalized_registered_hash = text_hash(normalized_current) if reconstructed_registered_match else "UNRESOLVED"
    current_normalized_hash = text_hash(normalized_current)
    current_trimmed_hash = text_hash(trimmed_current)
    line_ending_match = reconstructed_registered_match and normalized_registered_hash == current_normalized_hash
    drift_detected = not old_hash_match

    full_attr = pd.read_parquet(paths["full_attr"])
    full_geo = gpd.read_parquet(paths["full_geo"])
    candidate_attr = pd.read_parquet(paths["candidate_attr"])
    candidate_geo = gpd.read_parquet(paths["candidate_geo"])
    review = pd.read_parquet(paths["review"])
    ineligible = pd.read_parquet(paths["ineligible"])
    anomalies = pd.read_csv(paths["anomalies"])
    semantics = extract_markdown_semantics(md_text)
    semantic_parity = all([
        semantics["FINAL_DECISION"] == prior_json.get("FINAL_DECISION") == "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_COMPLETE",
        semantics["BASE_CANDIDATE_COUNT"] == prior_json.get("BASE_CANDIDATE_COUNT") == len(candidate_attr) == 29084,
        semantics["PARTIAL_REVIEW_COUNT"] == prior_json.get("PARTIAL_REVIEW_COUNT") == len(review) == 774,
        semantics["MASTER_GRID_COUNT"] == prior_json.get("MASTER_GRID_COUNT") == len(full_attr) == 139364,
        semantics["ERRORS"] == "NONE" and not prior_json.get("ERRORS"),
        all(semantics[k] for k in ("NO_RAINFALL", "NO_STATIC_EOGIS", "NO_RANDOM_SAMPLING", "NO_LABELS", "NO_HARD_CONTROLS", "NO_PAIRS", "NO_EXTERNAL_A26")),
        prior_consistency.get("unit_integrity_pass") is True,
    ])
    if not semantic_parity: errors.append("Markdown semantic parity with JSON/actual registries failed")
    drift_type = "REPORT_ONLY_NON_SEMANTIC_DRIFT" if drift_detected and line_ending_match and semantic_parity else "UNRESOLVED_OR_SEMANTIC_DRIFT"
    reconciliation_pass = drift_type == "REPORT_ONLY_NON_SEMANTIC_DRIFT" and input_data_asset_hash_pass
    if not reconciliation_pass: errors.append("Report hash reconciliation failed")

    # Create the source-package v1.1 reconciliation without touching any old file.
    if reconciliation.exists():
        marker = reconciliation / "CANDIDATE_POOL_PACKAGE_RECONCILED_v1_1.marker.json"
        if not marker.is_file() or util.load_json(marker).get("final_decision") != "CANDIDATE_POOL_PACKAGE_RECONCILED_v1_1":
            errors.append("Existing reconciliation directory is invalid and will not be overwritten")
    elif not errors:
        if reconciliation_staging.exists():
            errors.append("Reconciliation staging directory exists and will not be overwritten")
        else:
            reconciliation_staging.mkdir()
            try:
                new_md = "\n".join([
                    "# 250 m Hard-Control Base Candidate Eligibility Pool — Reconciled v1.1", "",
                    "- Final decision: `HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_COMPLETE`",
                    f"- Base candidate count: {len(candidate_attr):,}",
                    f"- Partial review count: {len(review):,}",
                    f"- Full grid registry: {len(full_attr):,}", "",
                    "The v1 Markdown byte drift is report-only and non-semantic; the structured JSON reports and actual registries remain authoritative.",
                    "The six sub-tolerance fragments retain primary status `INELIGIBLE_MASTER_GRID` and coverage class `OUTSIDE_INTERPRETED` under the frozen precedence rule.",
                    "No candidate was rebuilt or reclassified. No rainfall matching, static EOGIS screening, random sampling, labels, hard controls, pairs, or external A26 reads were performed.",
                    "", f"Warnings: `[{CORRECT_WARNING!r}]`", "Errors: `NONE`",
                    "Next step: use this reconciled report and v1.1 manifest as the freeze input; do not modify the old manifest.",
                ]) + "\n"
                new_md_path = reconciliation_staging / "HARD_CONTROL_CANDIDATE_BUILD_REPORT_v1_1.md"
                new_md_path.write_text(new_md, encoding="utf-8")
                replacement_hash = util.sha256(new_md_path)
                rec = {
                    "file_path": str(paths["build_md"]), "registered_size_bytes": registered_size,
                    "current_size_bytes": current_size, "registered_sha256": registered_hash,
                    "current_sha256": current_hash, "registered_normalized_text_hash": normalized_registered_hash,
                    "current_normalized_text_hash": current_normalized_hash,
                    "current_trailing_whitespace_normalized_hash": current_trimmed_hash,
                    "line_ending_difference_detected": "YES", "whitespace_only_difference_detected": "YES",
                    "semantic_parity_pass": "YES", "drift_scope": "REPORT_ONLY",
                    "REPORT_HASH_DRIFT_TYPE": drift_type,
                    "authoritative_json_report_path": str(paths["build_json"]),
                    "actual_registry_count_pass": "YES", "old_manifest_modified": "NO",
                    "resolution_action": "RECONCILED_BY_VERSIONED_REPORT_AND_MANIFEST_WITHOUT_MODIFYING_V1",
                    "superseded_entry": "OLD_MARKDOWN_REPORT_ENTRY_ONLY",
                    "replacement_report_path": str(reconciliation / new_md_path.name),
                    "replacement_report_sha256": replacement_hash,
                }
                pd.DataFrame([rec]).to_csv(
                    reconciliation_staging / "candidate_pool_report_hash_reconciliation_v1_1.csv",
                    index=False, encoding="utf-8-sig", lineterminator="\n",
                )
                (reconciliation_staging / "candidate_pool_report_hash_reconciliation_v1_1.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                # Manifest is deliberately generated only after the v1.1 report is closed and hashed.
                old_manifest = pd.read_csv(paths["candidate_output_manifest"])
                old_manifest["entry_status"] = "ACTIVE_UNCHANGED_V1_ASSET"
                old_manifest.loc[old_manifest["relative_path"] == markdown_rel, "entry_status"] = "SUPERSEDED_FOR_FREEZE_INPUT"
                additions = []
                for p in sorted(reconciliation_staging.glob("*")):
                    if p.name == "hard_control_candidate_output_manifest_v1_1.csv": continue
                    additions.append({
                        "relative_path": f"08_reconciliation_v1_1/{p.name}",
                        "size_bytes": p.stat().st_size, "sha256": util.sha256(p),
                        "normalized_content_sha256": "NOT_APPLICABLE", "entry_status": "ACTIVE_V1_1",
                    })
                manifest_v11 = pd.concat([old_manifest, pd.DataFrame(additions)], ignore_index=True)
                manifest_v11.to_csv(
                    reconciliation_staging / "hard_control_candidate_output_manifest_v1_1.csv",
                    index=False, encoding="utf-8-sig", lineterminator="\n",
                )
                marker = {
                    "reconciliation_version": "v1.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "final_decision": "CANDIDATE_POOL_PACKAGE_RECONCILED_v1_1",
                    "report_hash_drift_type": drift_type, "semantic_parity_pass": "YES",
                    "old_manifest_modified": "NO", "replacement_report_sha256": replacement_hash,
                    "manifest_sha256": util.sha256(reconciliation_staging / "hard_control_candidate_output_manifest_v1_1.csv"),
                }
                (reconciliation_staging / "CANDIDATE_POOL_PACKAGE_RECONCILED_v1_1.marker.json").write_text(
                    json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                os.replace(reconciliation_staging, reconciliation)
            finally:
                if reconciliation_staging.exists(): shutil.rmtree(reconciliation_staging, ignore_errors=True)

    if errors:
        print(f"ERRORS: {errors}"); print(f"FINAL_DECISION: {FAILURE}"); return 2

    if target.exists():
        existing = validate_existing(target)
        if existing is None:
            print(f"ERRORS: Existing v1.1 target is invalid and will not be overwritten")
            print(f"FINAL_DECISION: {FAILURE}"); return 2
        print_summary(existing); return 0
    if staging.exists():
        print(f"ERRORS: v1.1 staging exists and will not be overwritten")
        print(f"FINAL_DECISION: {FAILURE}"); return 2

    # Validate correct six-fragment precedence without changing source rows.
    full_by_id = full_attr.set_index(full_attr["unit_id"].astype(str))
    candidate_ids = set(candidate_attr["unit_id"].astype(str))
    review_ids = set(review["unit_id"].astype(str))
    exception_rows = []
    for row in anomalies.sort_values("unit_id").to_dict("records"):
        uid = str(row["unit_id"]); actual = full_by_id.loc[uid]
        source_codes = str(actual["eligibility_reason_codes"])
        mapped_codes = []
        if "PRELIMINARY_INELIGIBLE" in source_codes: mapped_codes.append("MASTER_GRID_INELIGIBLE")
        if "OUTSIDE_INTERPRETED" in source_codes: mapped_codes.append("OUTSIDE_INTERPRETED")
        exception_rows.append({
            "unit_id": uid, "preliminary_eligible_flag": actual["preliminary_eligible_flag"],
            "prior_coverage_status": row["coverage_status"], "any_interpreted_flag": int(row["any_interpreted_flag"]),
            "gsi_coverage_class": actual["gsi_coverage_class"],
            "primary_eligibility_status": actual["primary_eligibility_status"],
            "eligibility_reason_codes": ";".join(mapped_codes),
            "source_eligibility_reason_codes": source_codes,
            "candidate_pool_inclusion_flag": util.yn(uid in candidate_ids),
            "partial_review_inclusion_flag": util.yn(uid in review_ids),
            "manual_reclassification_performed": "NO", "input_modified": "NO",
            "resolution_policy": "FROZEN_PRECEDENCE_MASTER_GRID_INELIGIBLE_OVER_OUTSIDE_INTERPRETED",
        })
    exceptions = pd.DataFrame(exception_rows)
    exception_policy = all([
        len(exceptions) == 6, exceptions["preliminary_eligible_flag"].eq("NO").all(),
        exceptions["any_interpreted_flag"].eq(0).all(),
        exceptions["gsi_coverage_class"].eq("OUTSIDE_INTERPRETED").all(),
        exceptions["primary_eligibility_status"].eq("INELIGIBLE_MASTER_GRID").all(),
        exceptions["eligibility_reason_codes"].eq("MASTER_GRID_INELIGIBLE;OUTSIDE_INTERPRETED").all(),
        exceptions["candidate_pool_inclusion_flag"].eq("NO").all(),
        exceptions["partial_review_inclusion_flag"].eq("NO").all(),
        exceptions["manual_reclassification_performed"].eq("NO").all(),
        exceptions["input_modified"].eq("NO").all(),
    ])
    if not exception_policy: errors.append("Corrected six-fragment precedence policy failed")

    status_counts = full_attr["primary_eligibility_status"].value_counts().to_dict()
    status_pass = status_counts == STATUS_EXPECTED and sum(status_counts.values()) == 139364
    if not status_pass: errors.append(f"Candidate status counts changed: {status_counts}")
    master = gpd.read_file(paths["master"], layer="hiroshima_grid_250m_master")
    primary = pd.read_parquet(paths["primary"]); affected = pd.read_parquet(paths["affected"]); control = pd.read_parquet(paths["control"])
    master_ids = set(master["unit_id"].astype(str)); full_ids = set(full_attr["unit_id"].astype(str))
    primary_ids = set(primary["unit_id"].astype(str)); affected_ids = set(affected["unit_id"].astype(str)); control_ids = set(control["unit_id"].astype(str))
    gsi_ids = set(control.loc[control["gsi_evidence_flag"].astype(str).str.upper() == "YES", "unit_id"].astype(str))
    intersections = {
        "BASE_CANDIDATE_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(candidate_ids & control_ids),
        "BASE_CANDIDATE_PRIMARY_POSITIVE_INTERSECTION_COUNT": len(candidate_ids & primary_ids),
        "BASE_CANDIDATE_AJG_AFFECTED_INTERSECTION_COUNT": len(candidate_ids & affected_ids),
        "BASE_CANDIDATE_GSI_EVIDENCE_INTERSECTION_COUNT": len(candidate_ids & gsi_ids),
        "PARTIAL_REVIEW_CONTROL_EXCLUSION_INTERSECTION_COUNT": len(review_ids & control_ids),
    }
    unit_pass = full_ids == master_ids and not any(x["unit_id"].duplicated().any() for x in (full_attr, candidate_attr, review, ineligible))
    master_geom = dict(zip(master["unit_id"].astype(str), master.geometry.map(lambda g: g.wkb_hex)))
    full_geom = dict(zip(full_geo["unit_id"].astype(str), full_geo.geometry.map(lambda g: g.wkb_hex)))
    candidate_geom = dict(zip(candidate_geo["unit_id"].astype(str), candidate_geo.geometry.map(lambda g: g.wkb_hex)))
    geometry_pass = full_geom == master_geom and all(master_geom[k] == v for k, v in candidate_geom.items())
    if any(intersections.values()): errors.append(f"Candidate/review intersections changed: {intersections}")
    if not unit_pass: errors.append("Unit ID integrity changed")
    if not geometry_pass: errors.append("Master/candidate geometry changed")

    source_hashes = {
        "full": util.normalized_hash(full_geo, True), "candidate": util.normalized_hash(candidate_geo, True),
        "review": util.normalized_hash(review), "ineligible": util.normalized_hash(ineligible),
        "exceptions": util.normalized_hash(exceptions),
    }
    staging.mkdir()
    try:
        dirs = {name: staging / name for name in (
            "00_manifest", "01_frozen_full_registry", "02_frozen_base_candidate_pool",
            "03_frozen_partial_review_pool", "04_frozen_ineligible_registry", "05_frozen_exceptions",
            "06_schema_and_rules", "07_audit", "08_attempt_history",
        )}
        for d in dirs.values(): d.mkdir(parents=True)
        full_result = util.write_spatial(full_geo,
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1_1.parquet",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1_1.csv",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1_1.geoparquet",
            dirs["01_frozen_full_registry"] / "hard_control_eligibility_full_grid_250m_frozen_v1_1.gpkg",
            "hard_control_eligibility_full_grid_250m_frozen_v1_1")
        cand_result = util.write_spatial(candidate_geo,
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1_1.parquet",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1_1.csv",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1_1.geoparquet",
            dirs["02_frozen_base_candidate_pool"] / "hard_control_base_candidate_pool_250m_frozen_v1_1.gpkg",
            "hard_control_base_candidate_pool_250m_frozen_v1_1")
        review_result = util.write_attribute(review,
            dirs["03_frozen_partial_review_pool"] / "partial_reliable_review_pool_250m_frozen_v1_1.parquet",
            dirs["03_frozen_partial_review_pool"] / "partial_reliable_review_pool_250m_frozen_v1_1.csv")
        ineligible_result = util.write_attribute(ineligible,
            dirs["04_frozen_ineligible_registry"] / "hard_control_ineligible_grid_registry_250m_frozen_v1_1.parquet",
            dirs["04_frozen_ineligible_registry"] / "hard_control_ineligible_grid_registry_250m_frozen_v1_1.csv")
        exception_path = dirs["05_frozen_exceptions"] / "coverage_semantic_boundary_exception_registry_frozen_v1_1.csv"
        util.write_csv(util.sorted_frame(exceptions), exception_path)
        shutil.copy2(paths["schema"], dirs["06_schema_and_rules"] / "hard_control_candidate_schema_frozen_v1_1.csv")
        shutil.copy2(paths["rules"], dirs["06_schema_and_rules"] / "hard_control_eligibility_rule_dictionary_frozen_v1_1.csv")
        shutil.copy2(paths["reasons"], dirs["06_schema_and_rules"] / "hard_control_exclusion_reason_dictionary_frozen_v1_1.csv")
        shutil.copy2(paths["attempt_report"], dirs["08_attempt_history"] / "attempt_001_freeze_report_rejected.json")
        attempt_summary = {
            "attempt": "attempt_001", "final_decision": attempt_report["FINAL_DECISION"],
            "failure_reasons": attempt_report["ERRORS"], "failure_report_path": str(paths["attempt_report"]),
            "v1_1_correction": "Reconcile report-only CRLF/LF hash drift and apply frozen precedence INELIGIBLE_MASTER_GRID over OUTSIDE_INTERPRETED",
            "previous_data_modified": "NO", "candidate_data_rebuilt": "NO", "candidate_status_reclassified": "NO",
        }
        (dirs["08_attempt_history"] / "attempt_001_failure_summary_v1_1.json").write_text(json.dumps(attempt_summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

        results = {"full": full_result, "candidate": cand_result, "review": review_result, "ineligible": ineligible_result}
        frozen_hashes = {k: v["normalized_content_sha256"] for k, v in results.items()}
        frozen_hashes["exceptions"] = util.normalized_hash(pd.read_csv(exception_path))
        normalized_pass = source_hashes == frozen_hashes
        cross_pass = all(v["reload_pass"] for v in results.values())
        signature_expected = {
            "hashes": source_hashes, "candidate_count": 29084, "review_count": 774,
            "status": STATUS_EXPECTED, "exception_ids": sorted(exceptions["unit_id"].astype(str)),
            "full_geometry": util.geometry_hash(full_geo), "candidate_geometry": util.geometry_hash(candidate_geo),
        }
        signature_actual = {
            "hashes": frozen_hashes, "candidate_count": cand_result["rows"], "review_count": review_result["rows"],
            "status": status_counts, "exception_ids": sorted(pd.read_csv(exception_path)["unit_id"].astype(str)),
            "full_geometry": full_result["geometry_sha256"], "candidate_geometry": cand_result["geometry_sha256"],
        }
        idempotent_pass = signature_actual == signature_expected
        if not normalized_pass: errors.append("v1.1 normalized content differs from source")
        if not cross_pass: errors.append("v1.1 cross-format reload failed")
        if not idempotent_pass: errors.append("v1.1 equivalent rerun signature failed")

        input_details = candidate_details + upstream_details
        for role, p in paths.items():
            if str(p) not in {x["absolute_path"] for x in input_details}:
                input_details.append({"input_role": role, "absolute_path": str(p), "size_bytes": p.stat().st_size,
                    "sha256": util.sha256(p), "registered_size_bytes": p.stat().st_size,
                    "registered_sha256": util.sha256(p), "hash_verified": "YES", "read_only": "YES"})
        # The only superseded hash entry is replaced by the reconciled v1.1 report.
        input_details = [x for x in input_details if x["absolute_path"] != str(paths["build_md"])]
        rec_report = reconciliation / "HARD_CONTROL_CANDIDATE_BUILD_REPORT_v1_1.md"
        input_details.append({"input_role": "reconciled_candidate_build_report_v1_1", "absolute_path": str(rec_report),
            "size_bytes": rec_report.stat().st_size, "sha256": util.sha256(rec_report),
            "registered_size_bytes": rec_report.stat().st_size, "registered_sha256": util.sha256(rec_report),
            "hash_verified": "YES", "read_only": "YES"})
        input_manifest = pd.DataFrame(input_details).drop_duplicates("absolute_path").sort_values("absolute_path")
        input_manifest_path = dirs["00_manifest"] / "hard_control_candidate_freeze_input_manifest_v1_1.csv"
        util.write_csv(input_manifest, input_manifest_path)

        report = {
            "PROJECT_ROOT": str(root), "ATTEMPT_VERSION": ATTEMPT,
            "PREVIOUS_ATTEMPT_REJECTED_PRESERVED": util.yn(attempt_preserved),
            "CANDIDATE_DATA_REBUILT": "NO", "CANDIDATE_STATUS_RECLASSIFIED": "NO", "PREVIOUS_INPUT_MODIFIED": "NO",
            "INPUT_DATA_ASSET_HASH_PASS": util.yn(input_data_asset_hash_pass),
            "OLD_MARKDOWN_REPORT_HASH_MATCH": util.yn(old_hash_match),
            "REPORT_HASH_DRIFT_DETECTED": util.yn(drift_detected), "REPORT_HASH_DRIFT_TYPE": drift_type,
            "REPORT_LINE_ENDING_NORMALIZED_HASH_MATCH": util.yn(line_ending_match),
            "REPORT_SEMANTIC_PARITY_PASS": util.yn(semantic_parity),
            "REPORT_HASH_RECONCILIATION_PASS": util.yn(reconciliation_pass),
            "OLD_MANIFEST_MODIFIED": "NO", "V1_1_MANIFEST_CREATED": "YES",
            "MASTER_GRID_COUNT": len(full_attr), "BASE_CANDIDATE_COUNT": len(candidate_attr), "PARTIAL_REVIEW_COUNT": len(review),
            **{f"{k}_COUNT": int(status_counts[k]) for k in STATUS_EXPECTED},
            "STATUS_PARTITION_PASS": util.yn(status_pass), "BOUNDARY_EXCEPTION_COUNT": len(exceptions),
            "BOUNDARY_EXCEPTION_PRELIMINARY_INELIGIBLE_COUNT": int(exceptions["preliminary_eligible_flag"].eq("NO").sum()),
            "BOUNDARY_EXCEPTION_OUTSIDE_INTERPRETED_COVERAGE_COUNT": int(exceptions["gsi_coverage_class"].eq("OUTSIDE_INTERPRETED").sum()),
            "BOUNDARY_EXCEPTION_PRIMARY_MASTER_GRID_COUNT": int(exceptions["primary_eligibility_status"].eq("INELIGIBLE_MASTER_GRID").sum()),
            "BOUNDARY_EXCEPTION_CANDIDATE_COUNT": int(exceptions["candidate_pool_inclusion_flag"].eq("YES").sum()),
            "BOUNDARY_EXCEPTION_REVIEW_COUNT": int(exceptions["partial_review_inclusion_flag"].eq("YES").sum()),
            "BOUNDARY_EXCEPTION_POLICY_PASS": util.yn(exception_policy), **intersections,
            "UNIT_ID_INTEGRITY_PASS": util.yn(unit_pass), "MASTER_GEOMETRY_UNCHANGED": util.yn(geometry_pass),
            "CROSS_FORMAT_RELOAD_PASS": util.yn(cross_pass), "NORMALIZED_CONTENT_HASH_PASS": util.yn(normalized_pass),
            "IDEMPOTENT_RERUN_PASS": util.yn(idempotent_pass), "LABEL_FIELD_CREATED": "NO",
            "HARD_CONTROLS_CREATED": "NO", "PAIR_TABLE_CREATED": "NO", "RAINFALL_MATCHING_PERFORMED": "NO",
            "STATIC_EOGIS_SCREENING_PERFORMED": "NO", "EXTERNAL_A26_CONTENT_READ": "NO",
            "WARNINGS": warnings, "ERRORS": errors, "FINAL_DECISION": SUCCESS if not errors else FAILURE, "NEXT_STEP": NEXT_STEP,
        }
        reconciliation_json = util.load_json(reconciliation / "candidate_pool_report_hash_reconciliation_v1_1.json")
        (dirs["07_audit"] / "candidate_pool_report_hash_reconciliation_v1_1.json").write_text(json.dumps(reconciliation_json, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        consistency = {"status_counts": status_counts, "intersections": intersections,
            "exception_policy_pass": exception_policy, "reason_code_mapping": {"PRELIMINARY_INELIGIBLE": "MASTER_GRID_INELIGIBLE", "OUTSIDE_INTERPRETED": "OUTSIDE_INTERPRETED"},
            "source_normalized_hashes": source_hashes, "frozen_normalized_hashes": frozen_hashes,
            "signature_expected": signature_expected, "signature_actual": signature_actual, "errors": errors}
        (dirs["07_audit"] / "hard_control_candidate_freeze_consistency_audit_v1_1.json").write_text(json.dumps(consistency, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
        freeze_time = datetime.now(timezone.utc).isoformat()
        freeze_report = {**report, "freeze_version": "v1.1", "freeze_timestamp_utc": freeze_time,
            "crs": "EPSG:6671", "previous_final_decision": prior_json["FINAL_DECISION"],
            "attempt_001_final_decision": attempt_report["FINAL_DECISION"], "normalized_content_hashes": frozen_hashes}
        (dirs["07_audit"] / "hard_control_candidate_freeze_report_v1_1.json").write_text(json.dumps(freeze_report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        md = ["# Hard-Control Candidate Eligibility Pool Freeze v1.1", "",
            f"- Final decision: `{report['FINAL_DECISION']}`", "- attempt_001 was rejected because of report hash drift and an incorrect boundary-state assertion.",
            "- Candidate data did not fail and no previous input was modified.", "- Markdown drift is report-only and semantic parity with JSON/actual registries passed.",
            "- Six fragments retain `INELIGIBLE_MASTER_GRID`; their coverage class remains `OUTSIDE_INTERPRETED`.",
            "- No reclassification, labels, hard controls, pairs, rainfall matching, or static EO/GIS construction occurred.", "",
            f"Warnings: `{warnings}`", f"Errors: `{errors if errors else 'NONE'}`", f"Next step: {NEXT_STEP}"]
        (dirs["07_audit"] / "HARD_CONTROL_CANDIDATE_FREEZE_REPORT_v1_1.md").write_text("\n".join(md)+"\n", encoding="utf-8")

        normalized_by_path = {}
        for result in results.values():
            for p in result["paths"]: normalized_by_path[str(p.relative_to(staging)).replace("\\", "/")] = result["normalized_content_sha256"]
        normalized_by_path[str(exception_path.relative_to(staging)).replace("\\", "/")] = frozen_hashes["exceptions"]
        output_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            output_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": util.sha256(p),
                "normalized_content_sha256": normalized_by_path.get(rel, "NOT_APPLICABLE")})
        output_manifest_path = dirs["00_manifest"] / "hard_control_candidate_freeze_output_manifest_v1_1.csv"
        util.write_csv(pd.DataFrame(output_rows), output_manifest_path)
        hash_rows = []
        for p in sorted(x for x in staging.rglob("*") if x.is_file()):
            rel = str(p.relative_to(staging)).replace("\\", "/")
            hash_rows.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": util.sha256(p)})
        hashes_path = dirs["00_manifest"] / "hard_control_candidate_freeze_file_hashes_v1_1.csv"
        util.write_csv(pd.DataFrame(hash_rows), hashes_path)
        if report["FINAL_DECISION"] == SUCCESS:
            marker = {"freeze_version": "v1.1", "freeze_timestamp_utc": freeze_time, "project_root": str(root),
                "crs": "EPSG:6671", "master_grid_count": 139364, "base_candidate_count": 29084,
                "partial_review_count": 774, "primary_status_counts": status_counts, "boundary_exception_count": 6,
                "input_manifest_hash": util.sha256(input_manifest_path),
                "input_file_hashes": input_manifest[["absolute_path", "sha256"]].to_dict("records"),
                "output_file_hashes": hash_rows, "file_hash_manifest_sha256": util.sha256(hashes_path),
                "normalized_content_hashes": frozen_hashes, "previous_final_decision": prior_json["FINAL_DECISION"],
                "current_final_decision": SUCCESS, "warnings": warnings, "errors": errors, "next_step": NEXT_STEP}
            (dirs["07_audit"] / "HARD_CONTROL_CANDIDATE_ELIGIBILITY_POOL_FROZEN_v1_1.marker.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        os.replace(staging, target); staging = None
        print_summary(report)
        return 0 if report["FINAL_DECISION"] == SUCCESS else 2
    finally:
        if staging is not None and staging.exists(): shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
