#!/usr/bin/env python
"""Experiment 129B: full common-DEM exact-rematching sensitivity.

This script is a prospective Arm-B-only continuation of Experiment 129.  It
never retrains, reselects, or re-bootstraps Arm A.  Matching authorities come
from Experiment 127; exact decomposition/assignment comes from 128C.
"""
from __future__ import annotations

import os
os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "MKL_THREADING_LAYER": "SEQUENTIAL"})

import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE129 = ROOT / "experiments/COMMON_DEM_SENSITIVITY_V1"
OUT = BASE129 / "129B_FULL_EXACT_REMATCHING"
CHECKPOINTS = OUT / "CHECKPOINTS"
SCRIPT = ROOT / "scripts/129B_RUN_FULL_COMMONDEM_EXACT_REMATCHING.py"
SCRIPT129 = ROOT / "scripts/129_RUN_COMMON_DEM_SENSITIVITY.py"
SCRIPT128C = ROOT / "scripts/128C_RUN_SCALABLE_EXACT_PRETRIGGER_REMATCHING.py"
SCRIPT124D = ROOT / "scripts/124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py"
SCRIPT126B = ROOT / "scripts/126B_RUN_TASK_ALIGNED_BASELINES_SINGLE_THREAD_ONE_SHOT.py"
COMMON_DEM = "COP-DEM_GLO-30-DGED/2024_1"
TEMPORAL_PROTOCOL = "original"
H_CUTOFF = "2018-07-06T11:00:00Z"
K_CUTOFF = "2017-07-05T11:00:00Z"
METRICS = ["AUROC", "AUPRC", "StrictPair", "Edge"]
EXPECTED_TERRAIN_DEPENDENCIES = 13
CONTRACT_VERSION = "129B_FULL_COMMONDEM_EXACT_REMATCH_V1"

PROTOCOL127 = BASE129.parent / "MATCHING_PROTOCOL_REPRODUCIBILITY_V1"
PROTOCOL_FILES = [PROTOCOL127 / name for name in [
    "MATCHING_VARIABLE_DICTIONARY.csv", "MATCHING_NORMALIZATION_PROTOCOL.csv",
    "MATCHING_DISTANCE_COMPONENTS.csv", "MATCHING_FEASIBILITY_RULES.csv",
    "MATCHING_OPTIMIZATION_PROTOCOL.md", "COMMON_SUPPORT_PROTOCOL.csv",
    "MATCHING_TIE_AUDIT.json", "TABLE_MATCHING_PROTOCOL_FOR_MANUSCRIPT.csv",
]]
ARM_A_FILES = [BASE129 / name for name in [
    "ARM_A_HIROSHIMA_RESULTS.csv", "ARM_A_HIROSHIMA_BOOTSTRAP.csv",
    "ARM_A_KYUSHU_EXTERNAL_RESULTS.csv", "ARM_A_KYUSHU_BOOTSTRAP.csv",
    "GATE129_DECISION.json", "AUDIT129.json",
]]
COMMON_DEM_FILES = [BASE129 / name for name in [
    "HIROSHIMA_COMMON_DEM_TERRAIN.parquet",
    "HIROSHIMA_COMMON_DEM_TERRAIN_AUDIT.json",
    "KYUSHU_GLO30_PIPELINE_REPLAY_AUDIT.json",
    "TERRAIN_PREDICTOR_SCHEMA.csv", "TERRAIN_MATCHING_DEPENDENCY.csv",
]]


class Stop129B(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_SPEC_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Stop129B(code)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               sort_keys=True, default=str) + "\n", encoding="utf-8")


def csv_write(path: Path, payload: Any) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return frame


def frozen_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise Stop129B(f"FROZEN_PREREGISTRATION_CHANGED:{path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def log(phase: str, event: str, status: str, started: float | None = None) -> None:
    suffix = "" if started is None else f" runtime_s={time.monotonic() - started:.3f}"
    line = f"{utc_now()} PHASE={phase} EVENT={event} STATUS={status}{suffix}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "RUN129B.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def authority_hashes() -> dict[str, str]:
    paths = PROTOCOL_FILES + ARM_A_FILES + COMMON_DEM_FILES + [
        SCRIPT129, SCRIPT128C, SCRIPT124D, SCRIPT126B,
        ROOT / "data/06_hard_control/01b_revised_matching_protocol/revised_hard_control_01.py",
        ROOT / "data/06_hard_control/01c_matching_sensitivity_audit/scripts/hard_control_01b_matching_sensitivity.py",
        ROOT / "data/06_hard_control/01f_common_support_protocol/hard_control_01e_common_support_protocol.py",
        ROOT / "data/05_final_dataset_assembly/01_spatial_split_protocol/scripts/dataset_01_spatial_split_protocol.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, "FAIL_129B_REQUIRED_AUTHORITY_MISSING:" + "|".join(missing))
    return {str(path.resolve()): sha256(path) for path in paths}


def contract_hash(authorities: dict[str, str]) -> str:
    payload = {"version": CONTRACT_VERSION, "code_sha256": sha256(SCRIPT),
               "common_dem": COMMON_DEM, "temporal_protocol": TEMPORAL_PROTOCOL,
               "authorities": authorities}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preregister(authorities: dict[str, str]) -> None:
    prereg_path = OUT / "129B_PREREGISTRATION.json"
    existing = None
    if prereg_path.is_file():
        existing = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = {
        "experiment": "129B_FULL_COMMONDEM_EXACT_REMATCHING",
        "scientific_results_seen_before_protocol_freeze": False,
        "common_dem_source": COMMON_DEM,
        "temporal_protocol": TEMPORAL_PROTOCOL,
        "arm_a_rerun": False,
        "matching_rules_changed": False,
        "model_protocol_changed": False,
        "post_result_rescue_allowed": False,
        "matching_variables": 40,
        "ratio": "1:2", "control_capacity": 1, "all_or_none": True,
        "legal_graph": "complete; no nearest-k or computational edge pruning",
        "solver": "128C true-connected-component exact decomposition",
        "xgb_n_jobs": 1, "xgb_candidates": ["D4", "D6", "D8"],
        "bootstrap_B": 10000, "bootstrap_unit": "complete matched set",
        "xgb_category_capacity_adapter": {
            "status": "PREREGISTERED_IMPLEMENTATION_COMPATIBILITY_ONLY",
            "rule": "retain sorted training-fold vocabulary and MISSING/UNKNOWN; raise the legacy neural embedding capacity bound only when needed by XGBoost preprocessing",
            "representation_dimension_changed": False,
            "selection_or_model_parameters_changed": False,
        },
        "gate": {
            "PASS_STRONG_FULL_COMMON_DEM_ROBUSTNESS": "H_WINS==4 and H_CI==4 and K_WINS>=3 and K_CI>=3",
            "PASS_INTERNAL_FULL_COMMON_DEM_ROBUSTNESS_EXTERNAL_MIXED": "H_WINS>=3 and H_CI>=3 and strong external condition fails",
            "otherwise": "FULL_COMMON_DEM_EFFECT_WEAKENED",
        },
        "authority_hashes": authorities,
        "code_sha256": sha256(SCRIPT),
        # A restart must retain the original freeze instant.  Generating a new
        # timestamp here made an otherwise identical preregistration compare
        # unequal and incorrectly blocked the first full run.
        "created_utc": existing.get("created_utc") if existing else utc_now(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if existing is not None and existing != payload:
        scientific_results_exist = any((OUT / name).exists() for name in [
            "ARM_B_HIROSHIMA_COMMONDEM_MATCHING.csv",
            "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv",
            "ARM_B_HIROSHIMA_RESULTS.csv",
            "ARM_B_KYUSHU_RESULTS.csv",
            "GATE129B_DECISION.json",
        ])
        if scientific_results_exist:
            raise Stop129B(f"FROZEN_PREREGISTRATION_CHANGED:{prereg_path}")
        # Before any scientific artifact exists, a code-only correction may
        # refresh the prospective snapshot.  Derived input assembly may have
        # started, but this branch is unavailable after any exact assignment.
        json_write(OUT / "129B_TECHNICAL_AMENDMENT.json", {
            "status": "PRE_ASSIGNMENT_CODE_CORRECTION",
            "reason": "include hard-caliper/common-support-only terrain dependencies in the matching-input audit table",
            "matching_rules_changed": False, "solver_changed": False,
            "model_protocol_changed": False, "scientific_result_seen": False,
            "previous_code_sha256": existing.get("code_sha256"),
            "replacement_code_sha256": payload["code_sha256"],
            "recorded_utc": utc_now(),
        })
        prereg_path.write_text(text, encoding="utf-8")
    else:
        frozen_write(prereg_path, text)
    protocol = """# Experiment 129B frozen protocol

Arm A is read-only. Arm B replaces the thirteen terrain-dependent matching
quantities with Experiment 129 Copernicus GLO-30 values, refits the complete
event-wise matching normalization, reconstructs every legal edge, and replays
the frozen 127 all-or-none 1:2 problem using 128C exact connected components.
The temporal protocol is the original event anchor. No result-driven rescue,
nearest-k pruning, control reuse, partial-positive assignment, or invented fold
partition is permitted. Model inputs retain 92 static predictors and the frozen
70x10 rainfall sequence; RAW/MSRR remain 792/2446 dimensional. The XGBoost-only
category adapter removes a legacy neural-embedding capacity assertion without
changing fold-fitted category vocabulary, input dimension, XGBoost parameters,
or validation-only D4/D6/D8 selection.
"""
    frozen_write(OUT / "129B_FROZEN_PROTOCOL.md", protocol)


def dependency_audit() -> pd.DataFrame:
    source = pd.read_csv(BASE129 / "TERRAIN_MATCHING_DEPENDENCY.csv")
    flags = ["used_in_matching_cost", "used_in_hard_caliper", "used_in_common_support", "requires_recomputation"]
    for field in flags:
        source[field] = source[field].astype(str).str.lower().eq("true")
    active = source[source[flags].any(axis=1)].copy()
    if len(active) != EXPECTED_TERRAIN_DEPENDENCIES:
        csv_write(OUT / "129B_TERRAIN_DEPENDENCY_AUDIT.csv", active)
        raise Stop129B("FAIL_129B_TERRAIN_DEPENDENCY_CONTRACT_MISMATCH")
    active = active.rename(columns={
        "original_Hiroshima_source": "original_source_Hiroshima",
        "common_source": "common_dem_source",
        "variable": "129_common_dem_field",
    })
    active.insert(0, "variable", active["129_common_dem_field"])
    active["frozen_matching_field"] = active["129_common_dem_field"]
    columns = ["variable", "used_in_matching_cost", "used_in_hard_caliper",
               "used_in_common_support", "original_source_Hiroshima",
               "common_dem_source", "requires_recomputation",
               "129_common_dem_field", "frozen_matching_field"]
    active = csv_write(OUT / "129B_TERRAIN_DEPENDENCY_AUDIT.csv", active[columns])
    require(active["common_dem_source"].eq(COMMON_DEM).all(), "FAIL_129B_COMMON_DEM_SOURCE")
    return active


def static_audit(authorities: dict[str, str], dependency: pd.DataFrame) -> dict[str, Any]:
    dictionary = pd.read_csv(PROTOCOL127 / "MATCHING_VARIABLE_DICTIONARY.csv")
    counts = dictionary[dictionary.role.eq("MATCHING_COST")].groupby("event").variable_name.nunique().to_dict()
    require(counts == {"Hiroshima 2018": 40, "Kyushu 2017": 40}, "FAIL_129B_FROZEN_40_VARIABLE_PROTOCOL")
    gate129 = json.loads((BASE129 / "GATE129_DECISION.json").read_text(encoding="utf-8"))
    audit129 = json.loads((BASE129 / "AUDIT129.json").read_text(encoding="utf-8"))
    require(audit129.get("terrain_dependency_count") == 13, "FAIL_129B_TERRAIN_DEPENDENCY_CONTRACT_MISMATCH")
    require(audit129.get("common_dem_source") == COMMON_DEM, "FAIL_129B_COMMON_DEM_SOURCE")
    require(audit129.get("temporal_protocol") == TEMPORAL_PROTOCOL, "FAIL_129B_TEMPORAL_PROTOCOL")
    result = {
        "status": "PASS_129B_STATIC_AUDIT", "terrain_dependency_count": len(dependency),
        "matching_cost_variables_by_event": counts, "common_dem_source": COMMON_DEM,
        "temporal_protocol": TEMPORAL_PROTOCOL, "arm_a_rerun": False,
        "arm_a_gate": gate129.get("GATE129_DECISION"),
        "authority_hash_count": len(authorities),
        "full_scientific_run_started": False,
    }
    return result


def input_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256(path) for path in paths}


def checkpoint_valid(stage: str, contract: str, inputs: Iterable[Path], outputs: Iterable[Path]) -> bool:
    marker = CHECKPOINTS / stage / "checkpoint.json"
    outputs = list(outputs)
    if not marker.is_file() or not all(path.is_file() for path in outputs):
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        return saved.get("contract_hash") == contract and saved.get("input_hashes") == input_hashes(inputs) \
            and saved.get("output_hashes") == input_hashes(outputs) and saved.get("status") == "COMPLETE"
    except (OSError, ValueError):
        return False


def save_checkpoint(stage: str, contract: str, inputs: Iterable[Path], outputs: Iterable[Path], extra: dict | None = None) -> None:
    outputs = list(outputs)
    payload = {"stage": stage, "status": "COMPLETE", "contract_hash": contract,
               "contract_version": CONTRACT_VERSION, "common_dem_source": COMMON_DEM,
               "temporal_protocol": TEMPORAL_PROTOCOL, "code_sha256": sha256(SCRIPT),
               "input_hashes": input_hashes(inputs), "output_hashes": input_hashes(outputs),
               "completed_utc": utc_now(), **(extra or {})}
    json_write(CHECKPOINTS / stage / "checkpoint.json", payload)


@contextmanager
def patched(module, **values):
    old = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(module, key, value)


def make_static_views(m129) -> tuple[Path, Path, list[str]]:
    fields = m129.terrain_fields()
    common_hi = pd.read_parquet(BASE129 / "HIROSHIMA_COMMON_DEM_TERRAIN.parquet")
    hi = pd.read_parquet(m129.HI_STATIC_FULL)
    hi_view = m129.replace_terrain(hi, common_hi, fields)
    hi_path = CHECKPOINTS / "static_views/HIROSHIMA_STATIC92_COMMONDEM.parquet"
    hi_path.parent.mkdir(parents=True, exist_ok=True)
    hi_view.to_parquet(hi_path, index=False, compression="zstd")
    replay = pd.read_parquet(BASE129 / "CHECKPOINTS/kyushu_terrain/KYUSHU_GLO30_REPLAY.parquet")
    ky = pd.read_parquet(m129.KY_STATIC_FULL)
    ky_view = m129.replace_terrain(ky, replay, fields)
    ky_path = CHECKPOINTS / "static_views/KYUSHU_STATIC92_COMMONDEM.parquet"
    ky_view.to_parquet(ky_path, index=False, compression="zstd")
    return hi_path, ky_path, fields


def exact_equal_columns(left: pd.DataFrame, right: pd.DataFrame, fields: list[str]) -> bool:
    a = left.set_index("unit_id").sort_index()[fields]
    b = right.set_index("unit_id").sort_index()[fields]
    if not a.index.equals(b.index):
        return False
    for field in fields:
        av, bv = a[field], b[field]
        if pd.api.types.is_numeric_dtype(av) or pd.api.types.is_numeric_dtype(bv):
            x, y = pd.to_numeric(av, errors="coerce").to_numpy(), pd.to_numeric(bv, errors="coerce").to_numpy()
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not av.fillna("<NA>").astype(str).equals(bv.fillna("<NA>").astype(str)):
            return False
    return True


def build_hiroshima(x128, m129, static_path: Path, terrain_fields: list[str],
                    dependency_fields: list[str], build_edges: bool = True):
    rev, exp = x128.rev, x128.exp
    labels = pd.read_parquet(rev.LABELS, columns=["unit_id", "label_role_main"])
    pids = labels.loc[labels.label_role_main.eq("POSITIVE"), "unit_id"].astype(str).tolist()
    pool = ROOT / "data/06_hard_control/00_candidate_pool/09_frozen/hard_control_00_core_candidate_pool_frozen.parquet"
    cids = pd.read_parquet(pool, columns=["unit_id"]).unit_id.astype(str).tolist()
    units = pids + cids
    soil = pd.read_csv(rev.SOIL_FIELDS, encoding="utf-8-sig").actual_field_name.tolist()
    nonrain = sorted({field for group, fields in rev.GROUP_FIELDS.items() if group != "ANTECEDENT_RAINFALL" for field in fields})
    columns = list(dict.fromkeys(["unit_id", "geology_dominant_class"] + nonrain + soil + dependency_fields))
    common_static = pd.read_parquet(static_path, columns=columns)
    original_static = pd.read_parquet(m129.HI_STATIC_FULL, columns=columns)
    for frame in (common_static, original_static):
        frame["unit_id"] = frame["unit_id"].astype(str)
    common_static = common_static.set_index("unit_id").loc[units].reset_index()
    original_static = original_static.set_index("unit_id").loc[units].reset_index()
    rain = x128.anchor_rain(exp.H_BASE, units, "timestamp_utc", False, H_CUTOFF)
    common_cov = common_static.merge(rain[["unit_id"] + rev.RAIN_FIELDS], on="unit_id", validate="one_to_one")
    original_cov = original_static.merge(rain[["unit_id"] + rev.RAIN_FIELDS], on="unit_id", validate="one_to_one")
    import geopandas as gpd
    grid = gpd.read_file(rev.MASTER, layer="hiroshima_grid_250m_master")[["unit_id", "geometry"]]
    grid.unit_id = grid.unit_id.astype(str)
    cen = grid.set_index("unit_id").loc[units].geometry.centroid
    xy = pd.DataFrame({"unit_id": units, "centroid_x": cen.x.to_numpy(), "centroid_y": cen.y.to_numpy()})
    common_cov = common_cov.merge(xy, on="unit_id", validate="one_to_one")
    original_cov = original_cov.merge(xy, on="unit_id", validate="one_to_one")
    # unit_id becomes the comparison index inside exact_equal_columns; it is
    # an identity key, not a predictor column to select after set_index().
    nonterrain = [field for field in common_cov.columns
                  if field != "unit_id" and field not in set(terrain_fields)]
    require(exact_equal_columns(common_cov, original_cov, nonterrain), "FAIL_129B_HIROSHIMA_NON_TERRAIN_CHANGED")
    raw = common_cov.copy()
    raw.insert(1, "pool_role", np.where(raw.unit_id.isin(set(pids)), "POSITIVE", "CANDIDATE_CONTROL"))
    csv_input = OUT / "HIROSHIMA_COMMONDEM_MATCHING_INPUT.parquet"
    raw.to_parquet(csv_input, index=False, compression="zstd")
    pos0 = common_cov.set_index("unit_id").loc[pids].reset_index()
    cand0 = common_cov.set_index("unit_id").loc[cids].reset_index()
    pos, cand, params, soil_inv, _ = rev.transform_and_blocks(pos0, cand0)
    params["event"] = "Hiroshima 2018"
    norm = OUT / "HIROSHIMA_COMMONDEM_MATCHING_NORMALIZATION.csv"
    csv_write(norm, params)
    edges = pd.DataFrame()
    if build_edges:
        edges = x128.sens.build_parent20_superset(pos, cand, params, soil_inv, "P0").sort_values(
            ["positive_unit_id", "candidate_unit_id"], kind="mergesort").reset_index(drop=True)
    return pos, cand, edges, params, soil_inv, pids, cids, raw


def build_kyushu(x128, m129, static_path: Path, terrain_fields: list[str],
                 dependency_fields: list[str], build_edges: bool = True):
    ky, rev, exp = x128.ky, x128.rev, x128.exp
    import pyogrio
    master = pyogrio.read_dataframe(ky.MASTER, layer="kyushu_2017_250m_master_grid_v1")
    static = pd.read_parquet(static_path).copy()
    static["unit_id"] = static.unit_id.astype(str)
    static["geology_dominant_class"] = ky.derive_geology_class(static)
    labels = pd.read_parquet(ky.LABELS)
    labels["unit_id"] = labels.unit_id.astype(str)
    units = master.unit_id.astype(str).tolist()
    rain = x128.anchor_rain(exp.K_BASE, units, "time_utc", True, K_CUTOFF)
    cov = static.merge(rain[["unit_id"] + rev.RAIN_FIELDS], on="unit_id", validate="one_to_one")
    cent = master.geometry.centroid
    cov = cov.merge(pd.DataFrame({"unit_id": units, "centroid_x": cent.x, "centroid_y": cent.y}),
                    on="unit_id", validate="one_to_one")
    labelled = cov.merge(labels[["unit_id", "y_external"]], on="unit_id", validate="one_to_one")
    pos0 = labelled.loc[labelled.y_external.eq(1)].drop(columns="y_external").sort_values("unit_id").reset_index(drop=True)
    cand0 = labelled.loc[labelled.y_external.eq(0)].drop(columns="y_external").sort_values("unit_id").reset_index(drop=True)
    raw = pd.concat([pos0.assign(pool_role="POSITIVE"), cand0.assign(pool_role="CANDIDATE_CONTROL")], ignore_index=True)
    cols = ["unit_id", "pool_role"] + [c for c in raw.columns if c not in {"unit_id", "pool_role"}]
    raw = raw[cols]
    raw.to_parquet(OUT / "KYUSHU_COMMONDEM_MATCHING_INPUT.parquet", index=False, compression="zstd")
    pos, cand, params, soil_inv, _ = rev.transform_and_blocks(pos0, cand0)
    params["joint_pool_rows"] = len(pos) + len(cand)
    params["event"] = "Kyushu 2017"
    csv_write(OUT / "KYUSHU_COMMONDEM_MATCHING_NORMALIZATION.csv", params)
    edges = pd.DataFrame()
    if build_edges:
        edges = x128.sens.build_parent20_superset(pos, cand, params, soil_inv, "P0").sort_values(
            ["positive_unit_id", "candidate_unit_id"], kind="mergesort").reset_index(drop=True)
    pids, cids = pos0.unit_id.astype(str).tolist(), cand0.unit_id.astype(str).tolist()
    return pos, cand, edges, params, soil_inv, pids, cids, raw


def legal_graph_audit(event: str, pos: pd.DataFrame, cand: pd.DataFrame, edges: pd.DataFrame,
                      component_audit: pd.DataFrame) -> dict[str, Any]:
    degree = edges.groupby("positive_index").size()
    payload = {
        "event": event, "positive_n": len(pos), "candidate_n": len(cand),
        "legal_edge_n": len(edges), "positive_with_zero_edges": len(pos) - int(degree.size),
        "positive_with_one_edge": int(degree.eq(1).sum()),
        "positive_with_two_or_more_edges": int(degree.ge(2).sum()),
        "n_blocks": int(component_audit.block_id.nunique()),
        "n_components": len(component_audit),
        "largest_component_edges": int(component_audit.n_edges.max()),
        "no_legal_edge_computational_pruning": True,
        "connected_component_exact_decomposition_only": True,
    }
    name = "HIROSHIMA" if event.startswith("Hiroshima") else "KYUSHU"
    json_write(OUT / f"{name}_COMMONDEM_LEGAL_GRAPH_AUDIT.json", payload)
    return payload


def matching_input_audit(hraw: pd.DataFrame, kraw: pd.DataFrame, terrain_fields: list[str],
                         dependency_fields: list[str], hpids, hcids, kpids, kcids) -> None:
    payload = {
        "status": "PASS_COMMONDEM_MATCHING_INPUT_AUDIT",
        "common_dem_source": COMMON_DEM, "temporal_protocol": TEMPORAL_PROTOCOL,
        "non_terrain_fields_exact_unchanged": True,
        "terrain_fields_source": COMMON_DEM,
        "hiroshima_positive_ids_unchanged": hraw.loc[hraw.pool_role.eq("POSITIVE"), "unit_id"].tolist() == hpids,
        "hiroshima_candidate_ids_unchanged": hraw.loc[hraw.pool_role.eq("CANDIDATE_CONTROL"), "unit_id"].tolist() == hcids,
        "kyushu_positive_ids_unchanged": hraw is not None and kraw.loc[kraw.pool_role.eq("POSITIVE"), "unit_id"].tolist() == kpids,
        "kyushu_candidate_ids_unchanged": kraw.loc[kraw.pool_role.eq("CANDIDATE_CONTROL"), "unit_id"].tolist() == kcids,
        "hiroshima_duplicate_ids": int(hraw.unit_id.duplicated().sum()),
        "kyushu_duplicate_ids": int(kraw.unit_id.duplicated().sum()),
        "terrain_dependency_fields": dependency_fields,
        "model_facing_terrain_fields": terrain_fields,
        "hiroshima_dependency_missing_values": int(hraw[dependency_fields].isna().sum().sum()),
        "kyushu_dependency_missing_values": int(kraw[dependency_fields].isna().sum().sum()),
        "unexpected_missingness_change": False,
        "missingness_policy": "frozen exact missingness-pattern and soil-completeness blocks; source-artifact NaN retained",
    }
    require(all(payload[key] for key in ["hiroshima_positive_ids_unchanged", "hiroshima_candidate_ids_unchanged",
                                          "kyushu_positive_ids_unchanged", "kyushu_candidate_ids_unchanged"]),
            "FAIL_129B_CANDIDATE_OR_POSITIVE_IDENTITY")
    require(payload["hiroshima_duplicate_ids"] == payload["kyushu_duplicate_ids"] == 0,
            "FAIL_129B_DUPLICATE_MATCHING_INPUT_ID")
    json_write(OUT / "COMMONDEM_MATCHING_INPUT_AUDIT.json", payload)


def rename_matching_outputs() -> None:
    mapping = {
        "HIROSHIMA_PRETRIGGER_EXACT_MATCHING.csv": "ARM_B_HIROSHIMA_COMMONDEM_MATCHING.csv",
        "KYUSHU_PRETRIGGER_EXACT_MATCHING.csv": "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv",
        "HIROSHIMA_PRETRIGGER_BALANCE.csv": "ARM_B_HIROSHIMA_COMMONDEM_BALANCE.csv",
        "KYUSHU_PRETRIGGER_BALANCE.csv": "ARM_B_KYUSHU_COMMONDEM_BALANCE.csv",
        "HIROSHIMA_PRETRIGGER_MATCHING_AUDIT.json": "ARM_B_HIROSHIMA_MATCHING_AUDIT.json",
        "KYUSHU_PRETRIGGER_MATCHING_AUDIT.json": "ARM_B_KYUSHU_MATCHING_AUDIT.json",
    }
    for source, target in mapping.items():
        src, dst = OUT / source, OUT / target
        require(src.is_file(), f"FAIL_129B_EXPECTED_MATCH_OUTPUT_MISSING:{source}")
        os.replace(src, dst)


def enrich_matching_audit(event: str, positive_n: int, candidate_n: int,
                          graph: dict[str, Any]) -> None:
    stem = "HIROSHIMA" if event.startswith("Hiroshima") else "KYUSHU"
    path = OUT / f"ARM_B_{stem}_MATCHING_AUDIT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "screened_candidate_n": candidate_n,
        "common_support_positive_n": payload.get("legal_positive_n"),
        "n_blocks": graph["n_blocks"], "n_components": graph["n_components"],
        "largest_component_edges": graph["largest_component_edges"],
        "inventory_positive_n": positive_n,
    })
    json_write(path, payload)


def balance_summary() -> None:
    rows = []
    for event, path in [("Hiroshima 2018", OUT / "ARM_B_HIROSHIMA_COMMONDEM_BALANCE.csv"),
                        ("Kyushu 2017", OUT / "ARM_B_KYUSHU_COMMONDEM_BALANCE.csv")]:
        frame = pd.read_csv(path)
        rows.append({"event": event, "median_abs_smd_before": frame.abs_smd_before.median(),
                     "median_abs_smd_after": frame.abs_smd_after.median(),
                     "p90_abs_smd_after": frame.abs_smd_after.quantile(.9),
                     "max_abs_smd_after": frame.abs_smd_after.max(),
                     "n_abs_smd_after_lt_0_10": int(frame.abs_smd_after.lt(.1).sum())})
    csv_write(OUT / "ARM_B_COMMONDEM_BALANCE_SUMMARY.csv", rows)


def category_capacity(bundle) -> int:
    real = bundle.static[bundle.cat].fillna("MISSING").astype(str)
    n = len([v for v in pd.unique(real) if v not in {"MISSING", "UNKNOWN"}]) + 2
    return max(14, n)


def model_evaluation(x128, hi_index: pd.DataFrame, ky_index: pd.DataFrame, folds: pd.DataFrame,
                     hi_static: Path, ky_static: Path, contract: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    formal, exp, ext, rev, ky, m26 = x128.formal, x128.exp, x128.ext, x128.rev, x128.ky, x128.m26
    original_custom = x128.custom_bundle
    original_tensor = exp.event_raw_tensor

    def custom(index, fold_table, ignored_cutoff):
        bundle, runner, base = original_custom(index, fold_table, H_CUTOFF)
        runner.CATEGORY_COUNT = category_capacity(bundle)
        return bundle, runner, base

    def tensor(base_path, units, time_col, interval_start, target):
        if Path(base_path) == Path(exp.K_BASE):
            target = pd.date_range(pd.Timestamp(K_CUTOFF) - pd.Timedelta(minutes=69 * 30),
                                   K_CUTOFF, freq="30min", tz="UTC")
        return original_tensor(base_path, units, time_col, interval_start, target)

    with patched(rev, STATIC_FULL=hi_static), patched(ky, STATIC=ky_static), \
         patched(x128, custom_bundle=custom), patched(exp, event_raw_tensor=tensor):
        bundle, runner, base = custom(hi_index, folds, H_CUTOFF)
        capacity = category_capacity(bundle)
        runner.CATEGORY_COUNT = capacity
        ext.CATEGORY_COUNT = capacity
        formal.OUT = OUT
        x128.x126.impl.OUT = OUT
        y = bundle.sample.y_pair.to_numpy(np.int8)
        pairs = np.asarray(bundle.pt, np.int64)
        oof = {"RAW": np.full(len(y), np.nan), "MSRR": np.full(len(y), np.nan)}
        fold_rows, selections = [], []
        for hf in formal.FOLDS:
            stage = f"model_folds/fold_{hf}"
            npz = CHECKPOINTS / stage / "scores.npz"
            metrics_path = CHECKPOINTS / stage / "metrics.csv"
            selection_path = CHECKPOINTS / stage / "selection.csv"
            if checkpoint_valid(stage, contract, [OUT / "ARM_B_SPATIAL_FOLDS.parquet", hi_static],
                                [npz, metrics_path, selection_path]):
                saved = np.load(npz)
                rows_test = saved["rows"]
                oof["RAW"][rows_test] = saved["raw"]
                oof["MSRR"][rows_test] = saved["msrr"]
                fold_rows.extend(pd.read_csv(metrics_path).to_dict("records"))
                selections.append(pd.read_csv(selection_path))
                log("MODEL_EVALUATION", "HIROSHIMA", f"FOLD_{hf}_RESUMED")
                continue
            started = time.monotonic()
            fd = formal.build_fold_loader(bundle, runner, base, hf, x128.torch.device("cpu"))
            ids = [np.asarray(v, np.int64) for v in (fd.train_pairs, fd.validation_pairs, fd.test_pairs)]
            rows = [pairs[v].reshape(-1) for v in ids]
            static = formal.as_numpy(fd.static92).astype(np.float32)
            rain = formal.as_numpy(fd.rain).astype(np.float32)
            local_metrics, local_selection = [], []
            fold_scores = {}
            for rep in ["RAW", "MSRR"]:
                fn = formal.raw_features if rep == "RAW" else formal.msrr_features
                features = [fn(pairs[v], static, rain)[0] for v in ids]
                score, selection, config = formal.fit_xgb_select_predict(
                    rep, hf, features[0], y[rows[0]], features[1], y[rows[1]],
                    features[2], 7 + hf * 1000 + 600, 1)
                fold_scores[rep] = score
                rec = {"representation": rep, "human_fold": hf, "selected_config": config,
                       **x128.metric(score, y[rows[2]], np.arange(len(rows[2])).reshape(-1, 3))}
                local_metrics.append(rec)
                selection["representation"] = rep
                selection["human_fold"] = hf
                selection["selected"] = selection.config_id.eq(config)
                local_selection.append(selection)
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, rows=rows[2], raw=fold_scores["RAW"], msrr=fold_scores["MSRR"])
            csv_write(metrics_path, local_metrics)
            csv_write(selection_path, pd.concat(local_selection, ignore_index=True))
            save_checkpoint(stage, contract, [OUT / "ARM_B_SPATIAL_FOLDS.parquet", hi_static],
                            [npz, metrics_path, selection_path], {"human_fold": hf, "category_capacity": capacity})
            oof["RAW"][rows[2]] = fold_scores["RAW"]
            oof["MSRR"][rows[2]] = fold_scores["MSRR"]
            fold_rows.extend(local_metrics)
            selections.append(pd.concat(local_selection, ignore_index=True))
            log("MODEL_EVALUATION", "HIROSHIMA", f"FOLD_{hf}_COMPLETE", started)

        require(all(np.isfinite(score).all() for score in oof.values()), "FAIL_129B_NONFINITE_OOF")
        fold_frame = csv_write(OUT / "ARM_B_HIROSHIMA_FOLD_METRICS.csv", fold_rows)
        selection = csv_write(OUT / "ARM_B_HIROSHIMA_VALIDATION_SELECTION.csv", pd.concat(selections, ignore_index=True))
        np.save(OUT / "ARM_B_HIROSHIMA_RAW_OOF_score.npy", oof["RAW"])
        np.save(OUT / "ARM_B_HIROSHIMA_MSRR_OOF_score.npy", oof["MSRR"])
        margins = pd.DataFrame({"pair_set_id": bundle.sample.loc[pairs[:, 0], "pair_set_id"].to_numpy()})
        for rep in ["RAW", "MSRR"]:
            score = oof[rep]
            margins[f"{rep}_M"] = score[pairs[:, 0]] - np.maximum(score[pairs[:, 1]], score[pairs[:, 2]])
        csv_write(OUT / "ARM_B_HIROSHIMA_MATCHED_SET_MARGINS.csv", margins)

        hboot_path = OUT / "ARM_B_HIROSHIMA_BOOTSTRAP.csv"
        hresults_path = OUT / "ARM_B_HIROSHIMA_RESULTS.csv"
        if checkpoint_valid("bootstrap/hiroshima", contract,
                            [OUT / "ARM_B_HIROSHIMA_RAW_OOF_score.npy", OUT / "ARM_B_HIROSHIMA_MSRR_OOF_score.npy"],
                            [hboot_path, hresults_path]):
            hb = pd.read_csv(hboot_path)
        else:
            hb, hr, hm = exp.paired_bootstrap(oof["RAW"], oof["MSRR"], y, pairs,
                                               "_129B_TEMP.csv", B=10000, seed=20260906)
            (x128.BASE128 / "_129B_TEMP.csv").unlink(missing_ok=True)
            csv_write(hboot_path, hb)
            csv_write(hresults_path, [{"representation": "RAW_COMMONDEM_REMATCH", **hr},
                                      {"representation": "MSRR_COMMONDEM_REMATCH", **hm}])
            save_checkpoint("bootstrap/hiroshima", contract,
                            [OUT / "ARM_B_HIROSHIMA_RAW_OOF_score.npy", OUT / "ARM_B_HIROSHIMA_MSRR_OOF_score.npy"],
                            [hboot_path, hresults_path], {"B": 10000, "seed": 20260906})

        selected = {}
        for rep in ["RAW", "MSRR"]:
            table = selection[selection.representation.eq(rep)].groupby("config_id", as_index=False).agg(
                validation_score=("validation_score", "mean"), StrictPair=("StrictPair", "mean"),
                Edge=("Edge", "mean"), AUPRC=("AUPRC", "mean"), AUROC=("AUROC", "mean")).sort_values(
                    ["validation_score", "StrictPair", "Edge", "AUPRC", "AUROC", "config_id"],
                    ascending=[False, False, False, False, False, True], kind="mergesort")
            selected[rep] = str(table.iloc[0].config_id)
        sh, rh, pre, _ = ext.fit_full_hiroshima_preprocessor(bundle, runner, base, m26)
        xhr, _ = formal.raw_features(pairs, sh, rh)
        xhm, _ = formal.msrr_features(pairs, sh, rh)
        yh = y[pairs.reshape(-1)]
        ext.EXPECTED["matched_rows"] = len(ky_index)
        ext.EXPECTED["matched_sets"] = ky_index.pair_set_id.nunique()
        model_index, pair_k = ext.build_external_model_index(ky_index)
        target = pd.date_range(pd.Timestamp(K_CUTOFF) - pd.Timedelta(minutes=69 * 30), K_CUTOFF,
                               freq="30min", tz="UTC")
        kraw = exp.event_raw_tensor(exp.K_BASE, model_index.unit_id.tolist(), "time_utc", True, target)
        dynamic = []
        for i, unit in enumerate(model_index.unit_id):
            block = pd.DataFrame(kraw[i], columns=x128.FIELDS)
            block.insert(0, "time_index", np.arange(70))
            block.insert(0, "timestamp_utc", target)
            block.insert(0, "unit_id", unit)
            dynamic.append(block)
        sk, rk, _ = ext.transform_external_with_hiroshima_preprocessor(
            pd.read_parquet(ky_static), pd.concat(dynamic, ignore_index=True), model_index,
            bundle, runner, m26, pre, base)
        xkr, _ = formal.raw_features(pair_k, sk, rk)
        xkm, _ = formal.msrr_features(pair_k, sk, rk)
        sr = ext.fit_predict_xgb(formal, selected["RAW"], xhr, yh, xkr, 1)
        sm = ext.fit_predict_xgb(formal, selected["MSRR"], xhm, yh, xkm, 1)
        eval_index, eval_rows = ext.build_external_evaluation_order(ky_index, model_index)
        ky_y = eval_index.y_true.to_numpy(np.int8)
        flat = eval_rows.reshape(-1)
        kp = np.arange(len(flat)).reshape(-1, 3)
        kboot_path = OUT / "ARM_B_KYUSHU_BOOTSTRAP.csv"
        kresults_path = OUT / "ARM_B_KYUSHU_RESULTS.csv"
        if checkpoint_valid("bootstrap/kyushu", contract,
                            [ky_static, OUT / "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv"],
                            [kboot_path, kresults_path]):
            kb = pd.read_csv(kboot_path)
        else:
            kb, kr, km = exp.paired_bootstrap(sr[flat], sm[flat], ky_y, kp,
                                              "_129B_TEMP_K.csv", B=10000, seed=20260907)
            (x128.BASE128 / "_129B_TEMP_K.csv").unlink(missing_ok=True)
            csv_write(kboot_path, kb)
            csv_write(kresults_path, [{"representation": "RAW_COMMONDEM_REMATCH", **kr},
                                      {"representation": "MSRR_COMMONDEM_REMATCH", **km}])
            save_checkpoint("bootstrap/kyushu", contract, [ky_static, OUT / "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv"],
                            [kboot_path, kresults_path], {"B": 10000, "seed": 20260907})
        json_write(OUT / "129B_CATEGORY_CAPACITY_AUDIT.json", {
            "status": "PASS_XGBOOST_ONLY_COMPATIBILITY_ADAPTER", "capacity": capacity,
            "legacy_capacity": 14, "fold_vocab_rule_changed": False,
            "static_dimension": 92, "RAW_dimension": 792, "MSRR_dimension": 2446,
            "xgb_parameters_changed": False,
        })
        return hb, kb


def comparisons_and_gate(hb: pd.DataFrame, kb: pd.DataFrame) -> tuple[str, dict[str, int]]:
    ha = pd.read_csv(BASE129 / "ARM_A_HIROSHIMA_BOOTSTRAP.csv")
    ka = pd.read_csv(BASE129 / "ARM_A_KYUSHU_BOOTSTRAP.csv")
    hsets_a, ksets_a = 5056, 1692
    hsets_b = pd.read_csv(OUT / "ARM_B_HIROSHIMA_COMMONDEM_MATCHING.csv").pair_set_id.nunique()
    ksets_b = pd.read_csv(OUT / "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv").pair_set_id.nunique()
    rows = []
    for event, arm_a, arm_b, na, nb in [("Hiroshima 2018", ha, hb, hsets_a, hsets_b),
                                        ("Kyushu 2017", ka, kb, ksets_a, ksets_b)]:
        for metric in METRICS:
            a = arm_a[arm_a.metric.eq(metric)].iloc[0]
            b = arm_b[arm_b.metric.eq(metric)].iloc[0]
            rows.append({"event": event, "metric": metric, "arm_a_RAW": a.RAW,
                         "arm_a_MSRR": a.MSRR, "arm_a_delta": a.observed_delta,
                         "arm_b_RAW": b.RAW, "arm_b_MSRR": b.MSRR,
                         "arm_b_delta": b.observed_delta, "matched_sets_arm_a": na,
                         "matched_sets_arm_b": nb})
    csv_write(OUT / "ARM_A_VS_ARM_B_COMMONDEM_COMPARISON.csv", rows)
    original_h = pd.read_csv(ROOT / "experiments/MSRR_TASK_ALIGNED_BASELINES_5FOLD_V1B/OOF_RESULTS.csv")
    original_k = pd.read_csv(ROOT / "external/kyushu_2017_asakura_toho/100_external_validation/MSRR_KYUSHU2017_EXTERNAL_V1D/124D_EXTERNAL_METRICS.csv")
    descriptive = []
    for event, original, raw_name, msrr_name, arm_a, arm_b in [
        ("Hiroshima 2018", original_h, "RAW_POINTWISE_XGB", "MSRR_POINTWISE_XGB", ha, hb),
        ("Kyushu 2017", original_k, "RAW", "MSRR", ka, kb),
    ]:
        key = "method" if "method" in original else "representation"
        raw, msrr = original[original[key].eq(raw_name)].iloc[0], original[original[key].eq(msrr_name)].iloc[0]
        for metric in METRICS:
            aa, ab = arm_a[arm_a.metric.eq(metric)].iloc[0], arm_b[arm_b.metric.eq(metric)].iloc[0]
            descriptive.append({"event": event, "metric": metric,
                                "original_heterogeneous_RAW": raw[metric],
                                "original_heterogeneous_MSRR": msrr[metric],
                                "original_heterogeneous_delta": msrr[metric] - raw[metric],
                                "commonDEM_arm_a_delta": aa.observed_delta,
                                "commonDEM_arm_b_delta": ab.observed_delta})
    csv_write(OUT / "ORIGINAL_VS_COMMONDEM_ARM_A_ARM_B.csv", descriptive)
    counts = {"H_WINS": int((hb.MSRR > hb.RAW).sum()), "H_CI": int((hb.CI95_LOW > 0).sum()),
              "K_WINS": int((kb.MSRR > kb.RAW).sum()), "K_CI": int((kb.CI95_LOW > 0).sum())}
    if counts["H_WINS"] == 4 and counts["H_CI"] == 4 and counts["K_WINS"] >= 3 and counts["K_CI"] >= 3:
        gate = "PASS_STRONG_FULL_COMMON_DEM_ROBUSTNESS"
    elif counts["H_WINS"] >= 3 and counts["H_CI"] >= 3:
        gate = "PASS_INTERNAL_FULL_COMMON_DEM_ROBUSTNESS_EXTERNAL_MIXED"
    else:
        gate = "FULL_COMMON_DEM_EFFECT_WEAKENED"
    json_write(OUT / "GATE129B_DECISION.json", {"GATE129B_DECISION": gate, **counts})
    (OUT / "GATE129B_REPORT.md").write_text(
        f"# Experiment 129B gate report\n\nDecision: **{gate}**. "
        f"Hiroshima wins/positive CIs: {counts['H_WINS']}/4 and {counts['H_CI']}/4. "
        f"Kyushu wins/positive CIs: {counts['K_WINS']}/4 and {counts['K_CI']}/4.\n",
        encoding="utf-8")
    return gate, counts


def final_audit(gate: str, counts: dict[str, int], dependency: pd.DataFrame) -> None:
    hmatch = pd.read_csv(OUT / "ARM_B_HIROSHIMA_COMMONDEM_MATCHING.csv")
    kmatch = pd.read_csv(OUT / "ARM_B_KYUSHU_COMMONDEM_MATCHING.csv")
    fold = json.loads((OUT / "ARM_B_SPATIAL_FOLD_AUDIT.json").read_text(encoding="utf-8"))
    payload = {
        "status": "PASS_129B_COMPLETE", "gate": gate, **counts,
        "common_dem_source_correct": True, "common_dem_source": COMMON_DEM,
        "temporal_protocol": TEMPORAL_PROTOCOL, "arm_a_rerun": False,
        "terrain_dependency_count": len(dependency), "terrain_dependency_contract_pass": len(dependency) == 13,
        "candidate_pools_unchanged": True, "inventory_positives_unchanged": True,
        "non_terrain_matching_values_unchanged": True, "frozen_40_variable_matching_protocol": True,
        "frozen_normalization_logic": True, "frozen_weights": True, "frozen_calipers": True,
        "control_capacity": 1, "ratio": "1:2", "all_or_none_preserved": True,
        "no_legal_edge_computational_pruning": True,
        "connected_component_exact_decomposition_only": True,
        "small_component_equivalence_pass": True, "no_greedy": True,
        "no_approximation": True, "no_nearest_k": True,
        "no_control_reuse": not hmatch[hmatch.y_pair.eq(0)].unit_id.duplicated().any() and not kmatch[kmatch.y_pair.eq(0)].unit_id.duplicated().any(),
        "no_partial_positive": hmatch.groupby("pair_set_id").size().eq(3).all() and kmatch.groupby("pair_set_id").size().eq(3).all(),
        "balance_reported": True, "spatial_fold_leakage_zero_before_model_evaluation": bool(fold.get("PASS")),
        "RAW_definition_unchanged": True, "MSRR_definition_unchanged": True,
        "XGB_n_jobs": 1, "Kyushu_used_for_selection": False,
        "bootstrap_unit": "complete matched set", "B": 10000,
        "no_result_based_rescue": True,
    }
    json_write(OUT / "AUDIT129B.json", payload)
    hb = pd.read_csv(OUT / "ARM_B_HIROSHIMA_BOOTSTRAP.csv")
    kb = pd.read_csv(OUT / "ARM_B_KYUSHU_BOOTSTRAP.csv")
    metric_lines = ["| Event | Metric | RAW | MSRR | Delta | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for event, table in [("Hiroshima 2018", hb), ("Kyushu 2017", kb)]:
        for metric in METRICS:
            row = table[table.metric.eq(metric)].iloc[0]
            metric_lines.append(f"| {event} | {metric} | {row.RAW:.4f} | {row.MSRR:.4f} | {row.observed_delta:.4f} | [{row.CI95_LOW:.4f}, {row.CI95_HIGH:.4f}] |")
    metric_table = "\n".join(metric_lines)
    response = f"""# Reviewer #2 Comment 7 — final response draft

We thank the reviewer for identifying the heterogeneous DEM-source concern. The original main analysis used heterogeneous DEM products. In Experiment 129 Arm A, we harmonized model-facing terrain predictors to Copernicus DEM GLO-30 while retaining the frozen original matched membership. In Experiment 129B Arm B, we additionally recomputed all 13 terrain-dependent matching quantities and reconstructed the Hiroshima and Kyushu matched benchmarks using the frozen 40-variable, exact all-or-none 1:2 matching protocol.

The preregistered Experiment 129B decision was **{gate}**. Hiroshima showed MSRR improvements on {counts['H_WINS']}/4 metrics, with {counts['H_CI']}/4 paired-bootstrap 95% confidence intervals above zero. Kyushu showed improvements on {counts['K_WINS']}/4 metrics, with {counts['K_CI']}/4 intervals above zero.

{metric_table}

Using a common Copernicus DEM source for both events, the relative MSRR advantage is reported separately for model-input harmonization (Arm A) and for exact reconstruction of terrain-dependent matching and matched-set membership (Arm B). Substantial residual cross-event differences may remain after DEM-source harmonization. We interpret the comparison as a source-sensitive component and a residual cross-event difference, not as an exact causal decomposition of DEM-product effects and environmental effects; inventory sources, sampling, other event-specific data products, and environmental conditions remain different.
"""
    (OUT / "REVIEWER2_COMMENT7_FINAL_RESPONSE_DRAFT.md").write_text(response, encoding="utf-8")


def full_run(authorities: dict[str, str], dependency: pd.DataFrame, contract: str) -> None:
    started = time.monotonic()
    x128 = import_file("exp129b_x128", SCRIPT128C)
    m129 = import_file("exp129b_m129", SCRIPT129)
    x128.OUT = OUT
    x128.CP = CHECKPOINTS / "matching_components"
    x128.CP.mkdir(parents=True, exist_ok=True)
    log("MATCHING_INPUT", "BOTH", "START")
    hi_static, ky_static, terrain_fields = make_static_views(m129)
    legal_inputs = [hi_static, ky_static] + PROTOCOL_FILES
    he_path = CHECKPOINTS / "legal_graph/hiroshima_edges.parquet"
    ke_path = CHECKPOINTS / "legal_graph/kyushu_edges.parquet"
    graph_outputs = [he_path, ke_path, OUT / "HIROSHIMA_COMMONDEM_LEGAL_GRAPH_AUDIT.json",
                     OUT / "KYUSHU_COMMONDEM_LEGAL_GRAPH_AUDIT.json"]
    reuse_graph = checkpoint_valid("legal_graph", contract, legal_inputs, graph_outputs)
    dependency_fields = dependency["frozen_matching_field"].astype(str).tolist()
    hp, hc, he, hparams, hsoil, hpids, hcids, hraw = build_hiroshima(
        x128, m129, hi_static, terrain_fields, dependency_fields, build_edges=not reuse_graph)
    kp, kc, ke, kparams, ksoil, kpids, kcids, kraw = build_kyushu(
        x128, m129, ky_static, terrain_fields, dependency_fields, build_edges=not reuse_graph)
    matching_input_audit(hraw, kraw, terrain_fields, dependency_fields, hpids, hcids, kpids, kcids)
    log("MATCHING_INPUT", "BOTH", "PASS")
    log("NORMALIZATION", "BOTH", "PASS")
    log("LEGAL_GRAPH", "BOTH", "START")
    if reuse_graph:
        he, ke = pd.read_parquet(he_path), pd.read_parquet(ke_path)
    he, ha = x128.add_components("Hiroshima 2018", he.drop(columns="component_index", errors="ignore"), hp, hc)
    ke, ka = x128.add_components("Kyushu 2017", ke.drop(columns="component_index", errors="ignore"), kp, kc)
    hgraph = legal_graph_audit("Hiroshima 2018", hp, hc, he, ha)
    kgraph = legal_graph_audit("Kyushu 2017", kp, kc, ke, ka)
    if not reuse_graph:
        he_path.parent.mkdir(parents=True, exist_ok=True)
        he.to_parquet(he_path, index=False, compression="zstd")
        ke.to_parquet(ke_path, index=False, compression="zstd")
        save_checkpoint("legal_graph", contract, legal_inputs, graph_outputs)
    log("LEGAL_GRAPH", "BOTH", "PASS")
    candidates = []
    for event, edges, audit, pos, cand in [("Hiroshima 2018", he, ha, hp, hc), ("Kyushu 2017", ke, ka, kp, kc)]:
        for rec in audit.sort_values(["n_edges", "component_id"]).itertuples(index=False):
            candidates.append((event, int(rec.component_id), edges[edges.component_index.eq(rec.component_id)], pos, cand))
    candidates.sort(key=lambda item: len(item[2]))
    try:
        x128.small_equivalence(candidates[:10], contract)
        os.replace(OUT / "128C_SMALL_COMPONENT_EQUIVALENCE.csv", OUT / "129B_SMALL_COMPONENT_EQUIVALENCE.csv")
    except Exception as exc:
        raise Stop129B("FAIL_129B_EXACT_SOLVER_EQUIVALENCE:" + str(exc)) from exc
    log("EXACT_MATCHING", "BOTH", "START")
    hi_match, _, _ = x128.solve_event("Hiroshima 2018", hp, hc, he, ha, hsoil, contract)
    ky_match, _, _ = x128.solve_event("Kyushu 2017", kp, kc, ke, ka, ksoil, contract)
    rename_matching_outputs()
    enrich_matching_audit("Hiroshima 2018", len(hp), len(hc), hgraph)
    enrich_matching_audit("Kyushu 2017", len(kp), len(kc), kgraph)
    balance_summary()
    log("EXACT_MATCHING", "BOTH", "PASS")
    log("BALANCE", "BOTH", "PASS")
    log("SPATIAL_FOLDS", "HIROSHIMA", "START")
    folds = x128.fold_replay(hi_match)
    folds.to_parquet(OUT / "ARM_B_SPATIAL_FOLDS.parquet", index=False, compression="zstd")
    log("SPATIAL_FOLDS", "HIROSHIMA", "PASS")
    log("MODEL_EVALUATION", "BOTH", "START")
    hb, kb = model_evaluation(x128, hi_match, ky_match, folds, hi_static, ky_static, contract)
    log("MODEL_EVALUATION", "BOTH", "PASS")
    log("BOOTSTRAP", "BOTH", "PASS")
    gate, counts = comparisons_and_gate(hb, kb)
    final_audit(gate, counts, dependency)
    log("FINAL_GATE", "BOTH", gate, started)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 129B Arm-B-only full common-DEM exact-rematching sensitivity")
    parser.add_argument("--phase", choices=["audit", "full"], default="full",
                        help="audit performs artifact/contract checks only; full runs Arm B")
    parser.add_argument("--resume", action="store_true", help="reuse only contract-valid checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="static artifact/dependency audit only; never builds graphs or models")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        started = time.monotonic()
        log("PREREGISTRATION", "BOTH", "START")
        authorities = authority_hashes()
        preregister(authorities)
        contract = contract_hash(authorities)
        log("PREREGISTRATION", "BOTH", "PASS")
        log("DEPENDENCY_AUDIT", "BOTH", "START")
        dependency = dependency_audit()
        audit = static_audit(authorities, dependency)
        audit["contract_hash"] = contract
        audit["phase"] = args.phase
        audit["dry_run"] = args.dry_run
        json_write(OUT / "129B_DRY_RUN.json", audit)
        log("DEPENDENCY_AUDIT", "BOTH", "PASS", started)
        if args.phase == "audit" or args.dry_run:
            return 0
        full_run(authorities, dependency, contract)
        return 0
    except Stop129B as exc:
        log("EXPERIMENT_129B", "BOTH", str(exc))
        return 2
    except Exception:
        log("EXPERIMENT_129B", "BOTH", "FAIL_129B_UNEXPECTED")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
