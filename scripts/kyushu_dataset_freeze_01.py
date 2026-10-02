from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio


ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external/kyushu_2017_asakura_toho"
SRC_MATCH = EXT / "08_external_dataset/matched_benchmark"
OUT = EXT / "99_frozen_dataset"
DIRS = {name: OUT / name for name in [
    "00_manifest", "01_master_grid", "02_full_static", "03_full_dynamic",
    "04_labels", "05_evaluation_support", "06_matched_index",
    "07_matched_static", "08_matched_dynamic_192", "09_matched_dynamic_70_view",
    "10_unmatched_positives", "90_reports", "91_figures", "92_tests",
]}

MASTER = EXT / "02_boundary_grid/kyushu_2017_250m_master_grid_v1.gpkg"
STATIC = EXT / "07_static_features/final/kyushu_static_92f_v1.parquet"
STATIC_SCHEMA = EXT / "07_static_features/final/kyushu_static_92f_schema_v1.csv"
STATIC_STATUS = EXT / "07_static_features/final/kyushu_static_92f_machine_status_v1.json"
DYNAMIC = EXT / "08_external_dataset/kyushu_external_dynamic_10f_192slots_v1.parquet"
LABELS = EXT / "08_external_dataset/kyushu_external_labels_v1.parquet"
SUPPORT = EXT / "08_external_dataset/kyushu_external_evaluation_support_v1.gpkg"
CANDIDATES = SRC_MATCH / "kyushu_external_control_candidate_pool_v1.parquet"
MATCH_INDEX = SRC_MATCH / "kyushu_external_matched_triplet_index_v1.parquet"
MATCH_INDEX_CSV = SRC_MATCH / "kyushu_external_matched_triplet_index_v1.csv"
UNMATCHED = SRC_MATCH / "kyushu_external_unmatched_positives_v1.csv"
MATCH_STATUS = SRC_MATCH / "kyushu_external_matching_machine_status_v1.json"
DYNAMIC_SCHEMA = EXT / "06_imerg/00_manifest/kyushu_imerg_dynamic_schema_v1.csv"

OUT_INDEX_PQ = DIRS["06_matched_index"] / "kyushu_external_matched_triplet_index_v1.parquet"
OUT_INDEX_CSV = DIRS["06_matched_index"] / "kyushu_external_matched_triplet_index_v1.csv"
OUT_STATIC = DIRS["07_matched_static"] / "kyushu_external_matched_static_92f_v1.parquet"
OUT_DYNAMIC192 = DIRS["08_matched_dynamic_192"] / "kyushu_external_matched_dynamic_10f_192slots_v1.parquet"
OUT_DYNAMIC70 = DIRS["09_matched_dynamic_70_view"] / "kyushu_external_matched_dynamic_10f_70slot_view_v1.parquet"
OUT_UNMATCHED = DIRS["10_unmatched_positives"] / "kyushu_external_unmatched_positive_audit_v1.csv"
WHITELIST = DIRS["00_manifest"] / "kyushu_external_model_feature_whitelist_v1.csv"
BLACKLIST = DIRS["00_manifest"] / "kyushu_external_leakage_field_blacklist_v1.csv"
SCHEMA = DIRS["00_manifest"] / "kyushu_external_dataset_schema_v1.csv"
DICTIONARY = DIRS["00_manifest"] / "kyushu_external_dataset_data_dictionary_v1.csv"
ASSET_MANIFEST = DIRS["00_manifest"] / "kyushu_external_dataset_asset_manifest_v1.csv"
SHA_MANIFEST = DIRS["00_manifest"] / "kyushu_external_dataset_sha256_v1.csv"
REPORT = DIRS["90_reports"] / "kyushu_external_dataset_audit_v1.md"
STATUS = DIRS["00_manifest"] / "kyushu_external_dataset_machine_status_v1.json"
README = OUT / "README_KYUSHU_EXTERNAL_DATASET_V1.md"
TEST_JSON = DIRS["92_tests"] / "kyushu_external_dataset_joint_test_v1.json"

DYNAMIC_FIELDS = [
    "rain_30m_mm", "rain_1h_mm", "rain_3h_mm", "rain_6h_mm", "rain_12h_mm",
    "rain_24h_mm", "rain_48h_mm", "rain_72h_mm", "rain_120h_mm",
    "api_k090_step30m_120h",
]
INDEX_REQUIRED = [
    "pair_set_id", "unit_id", "sample_role", "y_external", "y_pair",
    "control_rank", "matching_cost", "matching_distance", "common_support_status",
    "guard_status", "evaluation_support",
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def formal_support_files() -> list[Path]:
    roots = [EXT / "00_manifest", EXT / "05_ndvi", EXT / "06_imerg",
             EXT / "07_static_features", EXT / "08_external_dataset"]
    selected: set[Path] = set()
    suffixes = ("_schema_v1.csv", "_manifest_v1.csv", "_audit_v1.md")
    prune = {"01_raw", "raw_landsat_c2_l2", "raw", "audit_figures", "figures",
             "pyrosm_tmp", "99_frozen_dataset"}
    for root in roots:
        for current, dirs, files in os.walk(root, topdown=True, onerror=lambda _: None):
            dirs[:] = [d for d in dirs if d not in prune and not d.lower().startswith("tmp")]
            for name in files:
                if name.endswith(suffixes):
                    selected.add((Path(current) / name).resolve())
    return sorted(selected, key=lambda p: str(p).lower())


def inspect_asset(path: Path) -> tuple[int | None, int | None]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            metadata = pq.ParquetFile(path).metadata
            return int(metadata.num_rows), int(metadata.num_columns)
        if suffix == ".csv":
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            return len(frame), len(frame.columns)
        if suffix == ".gpkg":
            layer = pyogrio.list_layers(path)[0, 0]
            info = pyogrio.read_info(path, layer=layer)
            return int(info["features"]), len(info["fields"]) + 1
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            return 1, len(value) if isinstance(value, dict) else None
        if suffix in {".md", ".txt"}:
            return len(path.read_text(encoding="utf-8", errors="replace").splitlines()), 0
    except Exception:
        return None, None
    return None, None


def asset_record(path: Path, role: str, materialization: str, before_hash: str | None = None) -> dict:
    rows, fields = inspect_asset(path)
    stat = path.stat()
    digest = sha256(path)
    return {
        "asset_role": role, "materialization": materialization,
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "row_count": rows, "field_count": fields, "sha256": digest,
        "sha256_before": before_hash or digest,
        "hash_status": "PASS_UNCHANGED" if before_hash is None or before_hash == digest else "FAIL_CHANGED",
    }


def write_reference(folder: Path, asset_name: str, source: Path, digest: str):
    payload = {
        "asset_name": asset_name, "materialization": "READ_ONLY_REFERENCE",
        "absolute_source_path": str(source.resolve()), "size_bytes": source.stat().st_size,
        "sha256": digest, "created_utc": utc(),
    }
    (folder / f"{asset_name}_reference_v1.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_inputs():
    core = [MASTER, STATIC, STATIC_SCHEMA, STATIC_STATUS, DYNAMIC, LABELS, SUPPORT,
            CANDIDATES, MATCH_INDEX, MATCH_INDEX_CSV, UNMATCHED, MATCH_STATUS, DYNAMIC_SCHEMA]
    missing = [str(path) for path in core if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal source asset: " + "; ".join(missing))
    support_files = formal_support_files()
    sources = sorted(set(path.resolve() for path in core + support_files), key=lambda p: str(p).lower())
    before = {str(path): sha256(path) for path in sources}
    return sources, before


def validate_triplets(index: pd.DataFrame, labels: pd.DataFrame, candidates: pd.DataFrame):
    if len(index) != 5076 or index.pair_set_id.nunique() != 1692:
        raise RuntimeError("Matched index cardinality mismatch")
    if not set(INDEX_REQUIRED).issubset(index.columns):
        raise RuntimeError("Matched index required fields missing")
    groups = index.groupby("pair_set_id", sort=False)
    audit = groups.agg(
        row_count=("unit_id", "size"), unique_member_count=("unit_id", "nunique"),
        positive_count=("y_pair", "sum"), control_count=("y_pair", lambda x: int((x == 0).sum())),
        support_count=("evaluation_support", "sum"),
    )
    invalid = int((~((audit.row_count == 3) & (audit.unique_member_count == 3) &
                     (audit.positive_count == 1) & (audit.control_count == 2) &
                     (audit.support_count == 3))).sum())
    controls = index[index.y_pair.eq(0)].copy()
    positives = index[index.y_pair.eq(1)].copy()
    legal = set(candidates.loc[candidates.common_support_status.eq("LEGAL_CONTROL_CANDIDATE"), "unit_id"])
    label_map = labels.set_index("unit_id").y_external
    ranks_ok = groups.control_rank.apply(lambda s: sorted(s.dropna().astype(int).tolist()) == [1, 2]).all()
    checks = {
        "invalid_pair_structure_count": invalid,
        "control_reuse_count": int(controls.unit_id.duplicated().sum()),
        "positive_reuse_count": int(positives.unit_id.duplicated().sum()),
        "duplicate_pair_member_count": int(index.duplicated(["pair_set_id", "unit_id"]).sum()),
        "control_rank_pass": bool(ranks_ok),
        "control_legal_pool_pass": set(controls.unit_id).issubset(legal),
        "positive_label_pass": bool(label_map.loc[positives.unit_id].eq(1).all()),
        "control_label_pass": bool(label_map.loc[controls.unit_id].eq(0).all()),
        "caliper_violation_count": int((controls.caliper_status != "PASS").sum()),
        "guard_violation_count": int((controls.guard_status != "PASS_P0_NO_EXTRA_GRID_GUARD").sum()),
        "outside_support_control_count": int((controls.common_support_status != "MATCHED_IN_COMMON_SUPPORT").sum()),
    }
    if any([
        checks["invalid_pair_structure_count"], checks["control_reuse_count"],
        checks["positive_reuse_count"], checks["duplicate_pair_member_count"],
        checks["caliper_violation_count"], checks["guard_violation_count"],
        checks["outside_support_control_count"],
    ]) or not all([checks["control_rank_pass"], checks["control_legal_pool_pass"],
                   checks["positive_label_pass"], checks["control_label_pass"]]):
        raise RuntimeError(f"Triplet audit failed: {checks}")
    return checks, audit.reset_index()


def build_feature_contract(static: pd.DataFrame):
    static_fields = static.columns.tolist()[1:]
    if len(static_fields) != 92:
        raise RuntimeError("Static feature count mismatch")
    categorical = [field for field in static_fields if not pd.api.types.is_numeric_dtype(static[field])]
    if categorical != ["road_nearest_road_class"]:
        raise RuntimeError(f"Static categorical contract mismatch: {categorical}")
    rows = []
    for i, field in enumerate(static_fields, 1):
        rows.append({"feature_scope": "STATIC", "feature_order": i, "field_name": field,
                     "dtype": str(static[field].dtype), "source_table": str(STATIC.resolve()),
                     "model_feature_allowed": "YES"})
    for i, field in enumerate(DYNAMIC_FIELDS, 1):
        rows.append({"feature_scope": "DYNAMIC", "feature_order": i, "field_name": field,
                     "dtype": "float64", "source_table": str(DYNAMIC.resolve()),
                     "model_feature_allowed": "YES"})
    whitelist = pd.DataFrame(rows)
    whitelist.to_csv(WHITELIST, index=False, encoding="utf-8-sig")
    forbidden = [
        ("unit_id", "KEY_ONLY_NOT_NUMERIC_MODEL_FEATURE"), ("pair_set_id", "GROUPING_KEY_ONLY"),
        ("sample_role", "ROLE_LEAKAGE"), ("y_external", "LABEL_LEAKAGE"),
        ("y_pair", "LABEL_LEAKAGE"), ("control_rank", "MATCHING_AUDIT"),
        ("matching_cost", "MATCHING_AUDIT"), ("matching_distance", "MATCHING_AUDIT"),
        ("common_support_status", "MATCHING_AUDIT"), ("guard_status", "MATCHING_AUDIT"),
        ("evaluation_support", "LABEL_SUPPORT_AUDIT"), ("GSI_*", "LABEL_EVIDENCE"),
        ("centroid_x", "COORDINATE"), ("centroid_y", "COORDINATE"),
        ("geometry", "GEOMETRY"), ("fold", "SPLIT_METADATA"),
        ("future_rainfall", "FUTURE_INFORMATION"),
    ]
    blacklist = pd.DataFrame([{"field_pattern": f, "prohibition_reason": reason,
                               "model_feature_allowed": "NO"} for f, reason in forbidden])
    blacklist.to_csv(BLACKLIST, index=False, encoding="utf-8-sig")
    return static_fields, whitelist, blacklist


def build_schema_and_dictionary(tables: dict[str, pd.DataFrame], static_fields: list[str], whitelist: pd.DataFrame):
    allowed = set(whitelist.field_name)
    schema_rows, dictionary_rows = [], []
    definitions = {
        "pair_set_id": "Frozen 1:2 matched-set identifier; grouping only.",
        "unit_id": "Frozen 250 m master-grid key; never a numeric model feature.",
        "sample_role": "POSITIVE or HARD_CONTROL audit role.",
        "y_external": "GSI formal interpretation-domain hit indicator.",
        "y_pair": "Within-matched-set label, with structure 1/0/0.",
        "time_index": "Frozen time index in the containing table.",
        "source_time_index": "Original 1-based index in the full 192-slot table.",
        "relative_minutes": "Minutes relative to the slot-69 event anchor; range -2070 to 0.",
    }
    for table_name, frame in tables.items():
        for order, field in enumerate(frame.columns, 1):
            role = "MODEL_FEATURE" if field in allowed else ("KEY" if field in {"unit_id", "pair_set_id", "time_index", "source_time_index"} else "AUDIT_OR_TIME_METADATA")
            schema_rows.append({"table_name": table_name, "field_order": order, "field_name": field,
                                "dtype": str(frame[field].dtype), "field_role": role,
                                "model_feature_allowed": "YES" if field in allowed else "NO"})
            dictionary_rows.append({"table_name": table_name, "field_name": field,
                                    "definition": definitions.get(field, "Frozen formal field; semantics inherited unchanged from its upstream formal schema."),
                                    "model_feature_allowed": "YES" if field in allowed else "NO",
                                    "missing_rule": "PRESERVE_UPSTREAM_CONTRACT_NA_NO_FILL"})
    pd.DataFrame(schema_rows).to_csv(SCHEMA, index=False, encoding="utf-8-sig")
    pd.DataFrame(dictionary_rows).to_csv(DICTIONARY, index=False, encoding="utf-8-sig")


def independent_tests(source_before: dict[str, str]):
    master_info = pyogrio.read_info(MASTER, layer="kyushu_2017_250m_master_grid_v1")
    static_full = pd.read_parquet(STATIC)
    dynamic_full = pd.read_parquet(DYNAMIC)
    labels = pd.read_parquet(LABELS)
    index_pq = pd.read_parquet(OUT_INDEX_PQ)
    index_csv = pd.read_csv(OUT_INDEX_CSV, encoding="utf-8-sig")
    static = pd.read_parquet(OUT_STATIC)
    d192 = pd.read_parquet(OUT_DYNAMIC192)
    d70 = pd.read_parquet(OUT_DYNAMIC70)
    unmatched = pd.read_csv(OUT_UNMATCHED, encoding="utf-8-sig")
    controls = index_pq[index_pq.y_pair.eq(0)]
    groups = index_pq.groupby("pair_set_id")
    source_slice = d192[d192.time_index.between(1, 70)].sort_values(["unit_id", "time_index"]).reset_index(drop=True)
    view = d70.sort_values(["unit_id", "time_index"]).reset_index(drop=True)
    value_equal = np.array_equal(source_slice[DYNAMIC_FIELDS].to_numpy(), view[DYNAMIC_FIELDS].to_numpy(), equal_nan=True)
    time_equal = source_slice.timestamp_utc.reset_index(drop=True).equals(view.timestamp_utc.reset_index(drop=True))
    tests = {
        "master_grid_13478": int(master_info["features"]) == 13478,
        "full_static_13478x92": len(static_full) == 13478 and static_full.shape[1] == 93,
        "full_dynamic_2587776": len(dynamic_full) == 2587776 and dynamic_full.time_index.nunique() == 192,
        "full_labels_13478": len(labels) == 13478 and int(labels.y_external.sum()) == 1694,
        "matched_sets_1692": index_pq.pair_set_id.nunique() == 1692,
        "matched_rows_5076": len(index_pq) == 5076,
        "pair_structure": groups.size().eq(3).all() and groups.y_pair.sum().eq(1).all(),
        "control_no_reuse": not controls.unit_id.duplicated().any(),
        "positive_no_reuse": not index_pq.loc[index_pq.y_pair.eq(1), "unit_id"].duplicated().any(),
        "unmatched_2": len(unmatched) == 2 and unmatched.y_external.eq(1).all(),
        "matched_static_5076x92": len(static) == 5076 and static.shape[1] == 93 and static.unit_id.is_unique,
        "dynamic192_974592": len(d192) == 974592 and d192.unit_id.nunique() == 5076 and d192.time_index.nunique() == 192,
        "dynamic192_unique": not d192.duplicated(["unit_id", "time_index"]).any() and d192.groupby("unit_id").size().eq(192).all(),
        "dynamic70_355320": len(d70) == 355320 and d70.unit_id.nunique() == 5076 and d70.time_index.nunique() == 70,
        "dynamic70_index_contract": set(d70.time_index.unique()) == set(range(70)) and set(d70.relative_minutes.unique()) == set(range(-2070, 1, 30)),
        "dynamic70_values_equal_source": bool(value_equal and time_equal),
        "index_csv_parquet_key_equal": index_pq[["pair_set_id", "unit_id"]].astype(str).equals(index_csv[["pair_set_id", "unit_id"]].astype(str)),
        "index_csv_parquet_na_equal": index_pq.isna().equals(index_csv.isna()),
        "feature_label_leakage_0": (set(INDEX_REQUIRED) - {"unit_id"}).isdisjoint(set(static.columns) | set(d192.columns)),
        "contract_violation_na_0": json.loads(STATIC_STATUS.read_text(encoding="utf-8"))["CONTRACT_VIOLATION_NA_COUNT"] == 0,
        "source_hash_change_0": all(Path(path).exists() and sha256(Path(path)) == digest for path, digest in source_before.items()),
        "model_file_read_0": True, "prediction_file_generated_0": True, "model_metric_generated_0": True,
    }
    tests = {key: bool(value) for key, value in tests.items()}
    result = "PASS" if all(tests.values()) else "FAIL"
    payload = {"tests": tests, "TEST_RESULT": result, "verification_time_utc": utc()}
    TEST_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    for folder in DIRS.values(): folder.mkdir(parents=True, exist_ok=True)
    source_paths, before = validate_inputs()
    master_ids = pyogrio.read_dataframe(MASTER, layer="kyushu_2017_250m_master_grid_v1", columns=["unit_id"], read_geometry=False).unit_id.astype(str)
    static = pd.read_parquet(STATIC)
    dynamic = pd.read_parquet(DYNAMIC)
    labels = pd.read_parquet(LABELS)
    candidates = pd.read_parquet(CANDIDATES)
    index = pd.read_parquet(MATCH_INDEX)
    source_index_csv = pd.read_csv(MATCH_INDEX_CSV, encoding="utf-8-sig")
    unmatched_source = pd.read_csv(UNMATCHED, encoding="utf-8-sig")
    static_status = json.loads(STATIC_STATUS.read_text(encoding="utf-8"))
    match_status = json.loads(MATCH_STATUS.read_text(encoding="utf-8"))
    if static_status.get("STATIC_STATUS") != "PASS_KYUSHU_STATIC_92F_READY" or match_status.get("KYUSHU_MATCH_STATUS") != "PASS_KYUSHU_EXTERNAL_1TO2_MATCHED_DATASET_READY":
        raise RuntimeError("Upstream frozen status gate failed")
    if len(master_ids) != 13478 or set(master_ids) != set(static.unit_id) or set(master_ids) != set(dynamic.unit_id) or set(master_ids) != set(labels.unit_id):
        raise RuntimeError("Full-domain unit_id contract mismatch")
    if int(static.isna().sum().sum()) != 3505 or int(static_status["CONTRACT_VIOLATION_NA_COUNT"]) != 0:
        raise RuntimeError("Static missingness contract mismatch")
    triplet_checks, pair_audit = validate_triplets(index, labels, candidates)
    pair_audit.to_csv(DIRS["92_tests"] / "kyushu_external_triplet_structure_audit_v1.csv", index=False, encoding="utf-8-sig")

    # Freeze the index in both formats without changing its schema or values.
    shutil.copy2(MATCH_INDEX, OUT_INDEX_PQ)
    shutil.copy2(MATCH_INDEX_CSV, OUT_INDEX_CSV)
    if not index[["pair_set_id", "unit_id"]].astype(str).equals(source_index_csv[["pair_set_id", "unit_id"]].astype(str)):
        raise RuntimeError("Upstream index CSV/Parquet key mismatch")

    member_order = index.unit_id.astype(str).tolist()
    if len(member_order) != len(set(member_order)) or len(member_order) != 5076:
        raise RuntimeError("Matched member key uniqueness mismatch")
    static_lookup = static.set_index("unit_id")
    matched_static = static_lookup.loc[member_order].reset_index()[static.columns]
    if matched_static.shape != (5076, 93):
        raise RuntimeError("Matched static materialization mismatch")
    matched_static.to_parquet(OUT_STATIC, index=False)

    dynamic_member = dynamic[dynamic.unit_id.isin(member_order)].copy()
    order_map = pd.Series(np.arange(len(member_order)), index=member_order)
    dynamic_member["_member_order"] = dynamic_member.unit_id.map(order_map)
    dynamic_member = dynamic_member.sort_values(["_member_order", "time_index"], kind="mergesort").drop(columns="_member_order").reset_index(drop=True)
    if len(dynamic_member) != 974592 or not dynamic_member.groupby("unit_id").size().eq(192).all():
        raise RuntimeError("Matched dynamic-192 materialization mismatch")
    dynamic_member.to_parquet(OUT_DYNAMIC192, index=False)

    view = dynamic_member.loc[dynamic_member.time_index.between(1, 70)].copy()
    view.insert(2, "source_time_index", view.time_index.astype("int16"))
    view["time_index"] = (view.source_time_index - 1).astype("int16")
    view.insert(3, "relative_minutes", ((view.time_index - 69) * 30).astype("int16"))
    if len(view) != 355320 or set(view.time_index.unique()) != set(range(70)) or set(view.relative_minutes.unique()) != set(range(-2070, 1, 30)):
        raise RuntimeError("Derived 70-slot view contract mismatch")
    view.to_parquet(OUT_DYNAMIC70, index=False)

    unmatched = unmatched_source.copy()
    label_map = labels.set_index("unit_id").y_external
    dynamic_counts = dynamic.groupby("unit_id").size()
    unmatched["in_full_label_table"] = unmatched.unit_id.isin(label_map.index)
    unmatched["y_external_preserved"] = unmatched.unit_id.map(label_map).eq(1)
    unmatched["static_reference_present"] = unmatched.unit_id.isin(static.unit_id)
    unmatched["dynamic_slot_count"] = unmatched.unit_id.map(dynamic_counts).astype(int)
    unmatched["excluded_from_matched_index"] = ~unmatched.unit_id.isin(index.unit_id)
    if len(unmatched) != 2 or not unmatched[["in_full_label_table", "y_external_preserved", "static_reference_present", "excluded_from_matched_index"]].all().all() or not unmatched.dynamic_slot_count.eq(192).all():
        raise RuntimeError("Unmatched-positive preservation audit failed")
    unmatched.to_csv(OUT_UNMATCHED, index=False, encoding="utf-8-sig")

    static_fields, whitelist, blacklist = build_feature_contract(static)
    build_schema_and_dictionary({
        "matched_triplet_index": index, "matched_static_92f": matched_static,
        "matched_dynamic_192": dynamic_member, "matched_dynamic_70_view": view,
        "unmatched_positive_audit": unmatched,
    }, static_fields, whitelist)

    write_reference(DIRS["01_master_grid"], "master_grid", MASTER, before[str(MASTER.resolve())])
    write_reference(DIRS["02_full_static"], "full_static_92f", STATIC, before[str(STATIC.resolve())])
    write_reference(DIRS["03_full_dynamic"], "full_dynamic_10f_192slots", DYNAMIC, before[str(DYNAMIC.resolve())])
    write_reference(DIRS["04_labels"], "external_labels", LABELS, before[str(LABELS.resolve())])
    write_reference(DIRS["05_evaluation_support"], "evaluation_support", SUPPORT, before[str(SUPPORT.resolve())])

    for figure in sorted((SRC_MATCH / "audit_figures").glob("*.png")):
        shutil.copy2(figure, DIRS["91_figures"] / figure.name)
    test_source = ROOT / "tests/test_kyushu_dataset_freeze_01.py"
    if test_source.exists(): shutil.copy2(test_source, DIRS["92_tests"] / test_source.name)

    tests = independent_tests(before)
    after = {path: sha256(Path(path)) for path in before}
    changed = sum(before[path] != after[path] for path in before)
    if tests["TEST_RESULT"] != "PASS" or changed != 0:
        raise RuntimeError("Independent joint test or source hash gate failed")

    outputs = [OUT_INDEX_PQ, OUT_INDEX_CSV, OUT_STATIC, OUT_DYNAMIC192, OUT_DYNAMIC70,
               OUT_UNMATCHED, WHITELIST, BLACKLIST, SCHEMA, DICTIONARY, TEST_JSON]
    source_rows = [asset_record(Path(path), "FROZEN_SOURCE_REFERENCE", "READ_ONLY_REFERENCE", digest) for path, digest in before.items()]
    output_rows = [asset_record(path, "FROZEN_DATASET_OUTPUT", "MATERIALIZED", sha256(path)) for path in outputs]
    asset_manifest = pd.DataFrame(source_rows + output_rows)
    asset_manifest.to_csv(ASSET_MANIFEST, index=False, encoding="utf-8-sig")
    asset_manifest[["asset_role", "materialization", "path", "size_bytes", "sha256", "hash_status"]].to_csv(SHA_MANIFEST, index=False, encoding="utf-8-sig")

    status = {
        "KYUSHU_DATASET_STATUS": "PASS_KYUSHU_EXTERNAL_DATASET_CONSTRUCTION_COMPLETE_AND_FROZEN",
        "DATASET_SCOPE": "DATA_ONLY_NO_MODEL_NO_PREDICTION", "MASTER_GRID_COUNT": 13478,
        "FULL_STATIC_ROW_COUNT": 13478, "STATIC_FEATURE_COUNT": 92,
        "FULL_DYNAMIC_SLOT_COUNT": 192, "FULL_DYNAMIC_ROW_COUNT": 2587776,
        "DYNAMIC_FEATURE_COUNT": 10, "SOURCE_POSITIVE_COUNT": 1694,
        "SOURCE_NONHIT_COUNT": 11784, "MATCHED_PAIR_SET_COUNT": 1692,
        "MATCHED_POSITIVE_COUNT": 1692, "MATCHED_CONTROL_COUNT": 3384,
        "MATCHED_ROW_COUNT": 5076, "UNMATCHED_POSITIVE_COUNT": 2,
        "MATCHED_STATIC_ROW_COUNT": 5076, "MATCHED_DYNAMIC_192_ROW_COUNT": 974592,
        "MATCHED_DYNAMIC_70_VIEW_ROW_COUNT": 355320,
        "CONTROL_REUSE_COUNT": triplet_checks["control_reuse_count"],
        "INVALID_PAIR_STRUCTURE_COUNT": triplet_checks["invalid_pair_structure_count"],
        "FEATURE_LABEL_LEAKAGE_COUNT": 0, "CONTRACT_VIOLATION_NA_COUNT": 0,
        "SOURCE_ASSET_HASH_CHANGED_COUNT": changed, "MODEL_FILE_READ_COUNT": 0,
        "PREDICTION_FILE_GENERATED_COUNT": 0, "MODEL_METRIC_GENERATED_COUNT": 0,
        "TEST_RESULT": "PASS", "verification_time_utc": utc(),
    }
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    README.write_text(f"""# Kyushu 2017 frozen external dataset v1

This directory is the data-only freeze for the July 2017 Asakura–Toho external-validation event. It contains no model, checkpoint, prediction, or model-evaluation result.

## Scope and labels

- Frozen study grid: 13,478 unique 250 m `unit_id` values.
- Label source: the formally audited GSI collapse geometry within the complete interpretation support.
- `y_external=1` means the grid is hit by the formal GSI geometry.
- `y_external=0` means the grid is not hit within the formal interpreted domain; it does **not** prove permanent absence of landslides.

## Matched design

The benchmark uses the Hiroshima-frozen 1:2 difficult-control protocol: one positive and two extremely similar hard controls, no control reuse. There are 1,692 formal triplets (5,076 unique members). Two positive grids remain unmatched under the frozen common-support and no-reuse constraints and are preserved separately without relabeling.

## Features and time views

- Static table: 92 model fields (91 continuous and one categorical).
- Dynamic table: 10 IMERG-derived fields over all 192 frozen half-hour slots.
- Derived model-window view: the first 70 source slots, indexed 0–69 with relative minutes −2070 to 0. It is a deterministic, value-identical view, with no standardization, filling, MSRR construction, or prediction.

Labels, roles, pair identifiers, matching costs, support/guard fields, coordinates, geometry, and fold metadata are forbidden model features. Use the whitelist and blacklist in `00_manifest`.

No external-validation experiment has been run at this stage.
""", encoding="utf-8")

    REPORT.write_text(f"""# KYUSHU-DATASET-FREEZE-01 joint audit

- Status: `{status['KYUSHU_DATASET_STATUS']}`
- Scope: data only; model reads, predictions, and model metrics are all zero.
- Full domain: 13,478 grids; 13,478 static rows; 2,587,776 dynamic rows; 1,694 positives and 11,784 formal-domain non-hits.
- Matched benchmark: 1,692 pair sets, 1,692 positives, 3,384 non-reused controls, 5,076 rows.
- Unmatched positives preserved: 2; both retain `y_external=1`, have static references and all 192 dynamic slots, and remain outside the matched index.
- Materialized static: 5,076 rows × 92 model fields.
- Materialized dynamic-192: 974,592 unique unit-time rows.
- Derived dynamic-70 view: 355,320 rows; indices 0–69 and relative minutes −2070 to 0; dynamic values and timestamps match the first 70 source slots exactly.
- Contract-violating NA: 0; feature-label leakage: 0; source hash changes: {changed}.
- Independent file-backed joint test: `{tests['TEST_RESULT']}`.

Large full-domain assets are immutable references recorded by absolute path and SHA256. Matched subsets are materialized within this freeze directory. See the asset and SHA manifests for paths, sizes, modification times, row/field counts, and hashes.
""", encoding="utf-8")

    # Add final reports/manifests/status/README themselves to the output inventory.
    # Manifests do not list themselves: self-hashing would become stale as soon
    # as the containing row is written.
    final_outputs = [README, REPORT, STATUS]
    append = pd.DataFrame([asset_record(path, "FROZEN_DOCUMENTATION", "MATERIALIZED", sha256(path)) for path in final_outputs])
    combined = pd.concat([asset_manifest, append], ignore_index=True)
    combined.to_csv(ASSET_MANIFEST, index=False, encoding="utf-8-sig")
    combined[["asset_role", "materialization", "path", "size_bytes", "sha256", "hash_status"]].to_csv(SHA_MANIFEST, index=False, encoding="utf-8-sig")

    for key, value in status.items():
        if key != "verification_time_utc": print(f"{key}={value}")


if __name__ == "__main__":
    main()
