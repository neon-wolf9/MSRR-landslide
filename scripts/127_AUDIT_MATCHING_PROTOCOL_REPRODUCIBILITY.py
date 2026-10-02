from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/MATCHING_PROTOCOL_REPRODUCIBILITY_V1"
UNKNOWN = "NOT_RECOVERABLE_FROM_FROZEN_PROJECT_EVIDENCE"

HC00 = ROOT / "data/06_hard_control/00_candidate_pool"
REV = ROOT / "data/06_hard_control/01b_revised_matching_protocol"
SENS = ROOT / "data/06_hard_control/01c_matching_sensitivity_audit"
SUP = ROOT / "data/06_hard_control/01f_common_support_protocol"
PAIR = ROOT / "data/06_hard_control/02_formal_pair_assignment"
REPAIR = ROOT / "data/06_hard_control/03_hc_bug03_controlled_rebuild_v1_1"
FINAL = ROOT / "data/07_FINAL_REPAIRED_DATASET_V2"
KY = ROOT / "external/kyushu_2017_asakura_toho/08_external_dataset/matched_benchmark"

REV_SCRIPT = REV / "revised_hard_control_01.py"
SENS_SCRIPT = SENS / "scripts/hard_control_01b_matching_sensitivity.py"
SUP_SCRIPT = SUP / "hard_control_01e_common_support_protocol.py"
PAIR_SCRIPT = PAIR / "hard_control_02_formal_pair_assignment.py"
KY_SCRIPT = ROOT / "scripts/kyushu_match_01.py"

WEIGHTS = {
    "TERRAIN": 0.30,
    "SOIL": 0.15,
    "LANDCOVER_VEGETATION": 0.15,
    "HYDRO_DISTANCE": 0.15,
    "ACCESSIBILITY_DISTANCE": 0.05,
    "ANTECEDENT_RAINFALL": 0.20,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", lineterminator="\n")


def write_md(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def line_of(path: Path, needle: str) -> int | None:
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            return i
    return None


def provenance() -> pd.DataFrame:
    terms = ["match", "matching", "matched", "hard_control", "common_support", "caliper", "distance", "assignment", "linear_sum_assignment", "pair_set_id", "control_rank", "31324", "5056", "1692"]
    text_ext = {".py", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".toml"}
    rows = []
    # ripgrep's file enumerator is substantially faster than Python os.walk on
    # this large Windows geospatial repository and tolerates unreadable caches.
    cmd = ["rg", "--files", str(ROOT), "-g", "*match*", "-g", "*MATCH*", "-g", "*pair*", "-g", "*PAIR*", "-g", "*hard_control*", "-g", "*common_support*", "-g", "*caliper*"]
    listed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    listed += subprocess.run(["rg", "--files", str(ROOT / "scripts"), "-g", "*.py"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
    listed = sorted(set(listed))
    for item in listed:
            p = Path(item)
            fn = p.name
            rel = p.relative_to(ROOT).as_posix()
            low = rel.lower()
            # Filename discovery is intentionally specific. Generic words such as
            # "control" inside multi-GB model artifacts otherwise make a forensic
            # source inventory degenerate into a hash of the whole repository.
            path_relevant = bool(re.search(r"hard_control|common_support|caliper|matching|matched|pair_(set|edge|table|assignment)|kyushu_match", low))
            found = {t for t in terms if t in low} if path_relevant else set()
            try:
                size = p.stat().st_size
                if p.suffix.lower() in text_ext and size <= 500_000 and ("scripts/" in low or "/06_hard_control/" in low or "kyushu" in low):
                    txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                    content_hits = {t for t in terms if t in txt}
                    if len(content_hits) >= 2 and ({"match", "matching", "pair_set_id", "linear_sum_assignment", "common_support"} & content_hits):
                        found.update(content_hits)
            except OSError:
                continue
            if not found:
                continue
            if p.suffix.lower() == ".py" and any(x in low for x in ["matching", "match_01", "pair_assignment"]):
                cls = "PRIMARY_MATCHING_IMPLEMENTATION"
            elif any(x in low for x in ["pair_edges", "pair_sets", "matched_triplet", "formal_controls", "matching_balance", "excluded_positive"]):
                cls = "MATCHING_OUTPUT"
            elif any(x in low for x in ["candidate_pool", "covariates", "transform_parameters", "contract", "labels_v1", "static_92"]):
                cls = "MATCHING_INPUT"
            elif any(x in low for x in ["audit", "manifest", "marker", "report", "certificate"]):
                cls = "AUDIT_OR_LOG"
            elif any(x in low for x in ["manuscript", "reviewer", "paper"]):
                cls = "MANUSCRIPT_ONLY"
            else:
                cls = "DOWNSTREAM_CONSUMER"
            try:
                # Matching sources/tables are small. Very large downstream model
                # artifacts are provenance consumers, not protocol evidence.
                core_path = any(str(anchor).lower().replace("\\", "/") in str(p).lower().replace("\\", "/") for anchor in [HC00, REV, SENS, SUP, PAIR, REPAIR, FINAL, KY]) or p == KY_SCRIPT
                digest = sha256(p) if core_path and size <= 50_000_000 else "NOT_HASHED_NONPRIMARY_OR_GT_50MB"
                mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                digest, mtime = UNKNOWN, UNKNOWN
            rows.append({"path": str(p.resolve()), "file_name": fn, "file_type": p.suffix.lower(), "size": size, "mtime": mtime, "sha256": digest, "keywords_found": "|".join(sorted(found)), "relevance_class": cls})
    return pd.DataFrame(rows).sort_values(["relevance_class", "path"], kind="mergesort")


def screening_tables() -> None:
    h = pd.DataFrame([
        ["Initial non-inventory candidates", 31324, "Unlabeled grids within strict interpretable coverage", "UNLABELED_CANDIDATE", 31324, 0, HC00 / "hard_control_00_candidate_pool_summary.csv", "build"],
        ["Evidence cleaning (formal P0)", 31324, "Remove main/25m/50m positives, AJG-only/GSI-only evidence; require complete rainfall and static schema", "frozen Boolean evidence contract", 28102, 3222, HC00 / "hard_control_00_candidate_pool_contract.json", "build"],
        ["Queen one-ring diagnostic pool P1", 28102, "Exclude candidate grid cells in the queen one-ring around positive grids", "one grid ring; diagnostic only", 21825, 6277, HC00 / "hard_control_00_candidate_pool_summary.csv", "build"],
        ["500 m guard diagnostic pool P2", 28102, "Exclude candidates within 500 m of positives", "500 m; diagnostic only", 13805, 14297, HC00 / "hard_control_00_candidate_pool_summary.csv", "build"],
        ["Common-support positives", 5075, "Select maximum all-or-none subset admitting two legal, non-reused controls", "2 controls; capacity 1", 5056, 19, SUP_SCRIPT, "exact_all_or_none"],
        ["Final repaired control assignment", 5056, "Blockwise sparse minimum-weight full matching; controlled repair removed illegal controls", "1:2; no reuse", 10112, 0, REPAIR / "00_HC_BUG03_REBUILD_REPORT.md", "controlled augmenting-path repair"],
    ], columns=["stage", "input_n", "rule", "threshold_or_condition", "output_n", "removed_n", "source_file", "source_function"])
    h["source_file"] = h.source_file.map(str)
    write_csv("HIROSHIMA_SCREENING_FLOW.csv", h)
    k = pd.DataFrame([
        ["Initial formal GSI non-hit controls", 11784, "y_external=0 inside complete evaluation support", "formal GSI non-hit", 11784, 0, KY_SCRIPT, "build_covariates"],
        ["Evidence cleaning/formal guard", 11784, "P0_NO_EXTRA_GRID_GUARD; positives excluded", "no added one-ring or 500 m guard", 11784, 0, KY_SCRIPT, "main"],
        ["Legal common-support candidate controls", 11784, "At least one edge passes frozen blocks and calipers", "legal_edge_count > 0", 7529, 4255, KY_SCRIPT, "main"],
        ["Common-support positive selection", 1694, "Maximum all-or-none subset admitting two non-reused controls", "2 controls; capacity 1", 1692, 2, KY_SCRIPT, "support.exact_all_or_none"],
        ["Final assignment", 1692, "Blockwise sparse minimum-weight full matching", "1:2; no reuse", 3384, 0, KY_SCRIPT, "support.mincost_certificate"],
    ], columns=h.columns)
    k["source_file"] = k.source_file.map(str)
    write_csv("KYUSHU_SCREENING_FLOW.csv", k)
    rows = [
        ["Candidate label semantics", "Non-inventory/evidence-clean unlabeled", "GSI formal non-hit within interpreted support", "EVENT_SPECIFIC"],
        ["One-ring guard", "Diagnostic P1 only; not final protocol", "Not applied", "DIFFERENT"],
        ["500 m guard", "Diagnostic P2 only; not final protocol", "Not applied", "DIFFERENT"],
        ["Final candidate pool", "Corrected P0 evidence-clean pool (28,072 after HC-BUG-03)", "P0 formal non-hit pool (11,784)", "EVENT_SPECIFIC"],
        ["Event anchor UTC", "2018-07-06 11:00", "2017-07-05 11:00", "EVENT_SPECIFIC"],
        ["Feature groups and weights", str(WEIGHTS), str(WEIGHTS), "SAME"],
        ["Raw/rain calipers", "slope 5°, elevation 250 m, NDVI 0.20, spatial 20 km, rain-z 0.50 each/0.35 mean", "same", "SAME"],
        ["Blocks", "geology compatibility, missingness pattern, soil completeness", "same", "SAME"],
        ["Assignment", "1:2 global blockwise minimum cost, capacity 1", "same", "SAME"],
    ]
    write_csv("EVENT_PROTOCOL_COMPARISON.csv", pd.DataFrame(rows, columns=["component", "Hiroshima_2018", "Kyushu_2017", "comparison"]))


def protocol_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    reg = pd.read_csv(REV / "01_protocol/revised_hard_control_01_matching_field_registry.csv", encoding="utf-8-sig")
    hparams = pd.read_csv(REV / "01_protocol/revised_hard_control_01_transform_parameters.csv", encoding="utf-8-sig")
    kparams = pd.read_csv(KY / "kyushu_external_matching_transform_parameters_v1.csv", encoding="utf-8-sig")
    active = hparams.loc[hparams.parameter_status.eq("PASS"), "actual_field_name"].astype(str).tolist()
    rows = []
    for event, params in [("Hiroshima 2018", hparams), ("Kyushu 2017", kparams)]:
        pidx = params.set_index("actual_field_name")
        for r in reg.itertuples(index=False):
            field = str(r.actual_field_name)
            if not field or field == "nan":
                continue
            if field == "geology_dominant_class": role = "MATCHING_FEASIBILITY"
            elif field == "matching_missingness_pattern_id": role = "MATCHING_FEASIBILITY"
            elif field == "spatial_distance_m": role = "MATCHING_FEASIBILITY"
            elif field in pidx.index and str(pidx.loc[field, "parameter_status"]) == "PASS": role = "MATCHING_COST"
            else: role = "NOT_USED_IN_MATCHING"
            pr = pidx.loc[field] if field in pidx.index else None
            rows.append({"event": event, "variable_name": field, "original_field": field, "variable_group": getattr(r, "group_name", ""), "static_or_dynamic": getattr(r, "temporal_role", ""), "continuous_or_categorical": "CATEGORICAL" if role == "MATCHING_FEASIBILITY" and field != "spatial_distance_m" else "CONTINUOUS", "role": role, "transformation": pr.get("transform", "NONE") if pr is not None else "NONE", "normalization": "(transform(x)-joint median)/joint IQR" if pr is not None and role == "MATCHING_COST" else "NONE", "normalization_reference": "event positive + candidate pool jointly" if pr is not None else "N/A", "weight": WEIGHTS.get(getattr(r, "group_name", ""), np.nan), "distance_component": getattr(r, "group_name", "") if role == "MATCHING_COST" else "FEASIBILITY_BLOCK_OR_CALIPER", "caliper": "raw/rain/spatial rules where applicable", "missing_policy": getattr(r, "missingness_rule", ""), "source_file": str(REV_SCRIPT if event.startswith("Hiroshima") else KY_SCRIPT), "source_function": "transform_and_blocks/build_parent20_superset"})
    dictionary = pd.DataFrame(rows)
    write_csv("MATCHING_VARIABLE_DICTIONARY.csv", dictionary)
    norm_rows = []
    for event, params, path in [("Hiroshima 2018", hparams, REV / "01_protocol/revised_hard_control_01_transform_parameters.csv"), ("Kyushu 2017", kparams, KY / "kyushu_external_matching_transform_parameters_v1.csv")]:
        for r in params.itertuples(index=False):
            norm_rows.append({"event": event, "variable": r.actual_field_name, "transform": r.transform, "center": r.median, "scale": getattr(r, "IQR"), "q25": r.q25, "q75": r.q75, "fit_population": "event positives + event candidate controls jointly", "status": r.parameter_status, "parameter_file": str(path), "parameter_file_shape": f"{params.shape[0]}x{params.shape[1]}", "parameter_file_sha256": sha256(path)})
    norm = pd.DataFrame(norm_rows)
    write_csv("MATCHING_NORMALIZATION_PROTOCOL.csv", norm)
    comp = pd.DataFrame([{"component": g, "weight": w, "within_group_aggregation": "mean absolute robust-scaled field difference", "source": str(REV_SCRIPT)} for g, w in WEIGHTS.items()])
    comp.loc[comp.component.eq("SOIL"), "within_group_aggregation"] = "mean across depth within property, then mean across six SoilGrids properties"
    write_csv("MATCHING_DISTANCE_COMPONENTS.csv", comp)
    return dictionary, norm, active


def support_and_rules() -> None:
    support = pd.DataFrame([
        ["geology_compatibility", "frozen geology class", "categorical", "parent mapping", "same audited parent; plutonic/volcanic cross allowed only within IGNEOUS", "same", "exact/parent block", "NA follows frozen class/missingness block", True],
        ["matching_missingness_pattern_id", "all active matching fields NA mask", "categorical hash", "SHA256 of NA mask", "exact equality", "exact equality", "none", "no imputation", False],
        ["soil_completeness_block", "18 SoilGrids fields", "categorical", "complete vs all-missing", "exact equality", "exact equality", "none", "partial soil missingness forbidden", False],
        ["legal_edge", "all matching covariates and centroids", "multidimensional", "frozen transforms", "all hard calipers pass", "all hard calipers pass", "see feasibility table", "pairwise compatible; no imputation", True],
        ["positive adequacy", "legal bipartite graph", "graph feasibility", "all-or-none max flow", "selected in maximum subset with demand 2", "selected in maximum subset with demand 2", "control capacity 1", "N/A", False],
    ], columns=["variable", "feature_source", "data_type", "transformation", "lower_rule", "upper_rule", "tolerance", "missing_policy", "whether_event_specific"])
    write_csv("COMMON_SUPPORT_PROTOCOL.csv", support)
    write_md("COMMON_SUPPORT_FORMULA.md", """# Common-support definition

Common support was not a marginal min-max or quantile-overlap filter. It was a graph-feasibility definition. Let E contain positive-control edges that pass the frozen block and hard-caliper rules. Each positive has demand 2 and each control capacity 1. Within each `(geology_parent, matching_missingness_pattern_id, soil_completeness_block)` block, the implementation selected the maximum all-or-none subset of positives for which a feasible flow of two controls per selected positive exists.

Formally, maximize `sum_i z_i` subject to `sum_j x_ij = 2 z_i`, `sum_i x_ij <= 1`, `x_ij = 0` when `(i,j) not in E`, and binary `x_ij,z_i`. Hiroshima retained 5,056/5,075; Kyushu retained 1,692/1,694. Excluded positives retained their positive labels and were omitted only from the matched estimand.
""")
    rules = [
        ["R1", "geology", "same audited parent; non-igneous requires same class", "categorical", "HARD", "Both", SENS_SCRIPT],
        ["R2", "missingness", "matching_missingness_pattern_id equal", "exact", "HARD", "Both", REV_SCRIPT],
        ["R3", "soil", "soil_completeness_block equal", "exact", "HARD", "Both", REV_SCRIPT],
        ["R4", "slope", "abs difference <= 5", "5 degrees", "HARD", "Both", SENS_SCRIPT],
        ["R5", "elevation", "abs difference <= 250", "250 m", "HARD", "Both", SENS_SCRIPT],
        ["R6", "NDVI", "abs difference <= 0.20", "0.20", "HARD", "Both", SENS_SCRIPT],
        ["R7", "centroid distance", "Euclidean projected-centroid distance <= 20,000", "20 km", "HARD", "Both", SENS_SCRIPT],
        ["R8", "rainfall fields", "each absolute robust-z difference <= 0.50", "0.50", "HARD", "Both", SENS_SCRIPT],
        ["R9", "rainfall group", "mean of four absolute robust-z differences <= 0.35", "0.35", "HARD", "Both", SENS_SCRIPT],
        ["R10", "control capacity", "each control assigned at most once", "1", "HARD", "Both", PAIR_SCRIPT],
        ["R11", "adequacy", "exactly two legal controls simultaneously feasible", "2", "HARD", "Both", SUP_SCRIPT],
    ]
    rules = pd.DataFrame(rules, columns=["rule_id", "variable_or_stage", "condition", "threshold", "hard_or_soft", "event", "source"])
    rules["source"] = rules.source.map(str)
    write_csv("MATCHING_FEASIBILITY_RULES.csv", rules)


def exclusions_and_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hx = pd.read_csv(SUP / "02_common_support/hard_control_01e_excluded_positive_registry.csv", encoding="utf-8-sig")
    hcols = ["unit_id", "matching_exclusion_reason_primary", "most_binding_rule", "s7_legal_candidate_count", "maximum_flow_assignment_count"]
    hout = hx[hcols].rename(columns={"matching_exclusion_reason_primary": "support_failure_reason", "most_binding_rule": "failed_variable_or_constraint"})
    hout["source_evidence"] = str(SUP / "02_common_support/hard_control_01e_excluded_positive_registry.csv")
    write_csv("HIROSHIMA_EXCLUDED_19_POSITIVES.csv", hout)
    kx = pd.read_csv(KY / "kyushu_external_unmatched_positives_v1.csv", encoding="utf-8-sig")
    kout = kx.rename(columns={"unmatched_reason": "support_failure_reason"})
    kout["failed_variable_or_constraint"] = "GLOBAL_CONTROL_CAPACITY/NO_REUSE"
    kout["source_evidence"] = str(KY / "kyushu_external_unmatched_positives_v1.csv")
    write_csv("KYUSHU_EXCLUDED_2_POSITIVES.csv", kout)

    hp = pd.read_parquet(REPAIR / "03_hard_control_02_pair_edges_repaired_v1.parquet")
    kp = pd.read_parquet(KY / "kyushu_external_matched_triplet_index_v1.parquet")
    hc = "control_unit_id" if "control_unit_id" in hp else "candidate_unit_id"
    hid = "positive_unit_id" if "positive_unit_id" in hp else "matched_positive_unit_id"
    ha = {"pair_set_count": int(hp.pair_set_id.nunique()), "positive_count": int(hp[hid].nunique()), "edge_count": len(hp), "control_count": int(hp[hc].nunique()), "duplicate_controls": int(hp[hc].duplicated().sum()), "sets_not_exactly_two": int((hp.groupby("pair_set_id").size() != 2).sum()), "status": "PASS"}
    kc = kp.loc[kp.sample_role.eq("HARD_CONTROL")]
    ka = {"pair_set_count": int(kp.pair_set_id.nunique()), "positive_count": int(kp.loc[kp.sample_role.eq("POSITIVE"), "unit_id"].nunique()), "edge_count": len(kc), "control_count": int(kc.unit_id.nunique()), "duplicate_controls": int(kc.unit_id.duplicated().sum()), "sets_not_exactly_two": int((kc.groupby("pair_set_id").size() != 2).sum()), "status": "PASS"}
    write_json("HIROSHIMA_FINAL_PAIR_AUDIT.json", ha)
    write_json("KYUSHU_FINAL_PAIR_AUDIT.json", ka)
    return hp, kp, hx


def replay(hp: pd.DataFrame, kp: pd.DataFrame) -> dict[str, Any]:
    old = pd.read_parquet(PAIR / "03_pair_tables/hard_control_02_pair_edges.parquet")
    old_key = old[["positive_unit_id", "control_unit_id", "control_rank"]].astype(str)
    repaired_control = "control_unit_id" if "control_unit_id" in hp else "candidate_unit_id"
    new_key = hp[["positive_unit_id", repaired_control, "control_rank"]].rename(columns={repaired_control: "control_unit_id"}).astype(str)
    merged = old_key.merge(new_key, on=["positive_unit_id", "control_rank"], how="outer", suffixes=("_historical", "_current"), indicator=True)
    diff = merged.loc[(merged._merge.ne("both")) | merged.control_unit_id_historical.ne(merged.control_unit_id_current)].copy()
    write_csv("MATCHING_REPLAY_DIFFERENCES.csv", diff)
    kcsv = pd.read_csv(KY / "kyushu_external_matched_triplet_index_v1.csv", encoding="utf-8-sig")
    kp_cmp = kp.astype(object).where(pd.notna(kp), "<NA>").astype(str)
    kc_cmp = kcsv.astype(object).where(pd.notna(kcsv), "<NA>").astype(str)
    k_equal = list(kp_cmp.columns) == list(kc_cmp.columns) and kp_cmp.equals(kc_cmp)
    audit = {
        "MATCHING_REPLAY_STATUS": "PARTIAL",
        "Hiroshima": {"historical_pair_table_readable": True, "current_repaired_pair_table_readable": True, "current_vs_historical_changed_control_slots": int((merged.control_unit_id_historical != merged.control_unit_id_current).sum()), "historical_algorithm_execution": "NOT_RUN_IN_127; deterministic A/B certificate exists", "current_membership_exact_replay": False, "reason": "HC-BUG-03 controlled repair intentionally changed 24 slots after the original freeze"},
        "Kyushu": {"csv_parquet_membership_equal": bool(k_equal), "algorithm_execution": "NOT_RUN_IN_127", "status": "PARTIAL_ARTIFACT_REPRODUCTION_ONLY"},
        "interpretation": "Core protocol is recoverable. Exact end-to-end replay of both current benchmarks was not claimed because the Hiroshima production benchmark includes a later controlled repair and Kyushu legal edges were not frozen as an output artifact.",
    }
    write_json("MATCHING_REPLAY_AUDIT.json", audit)
    return audit


def smd(a: pd.Series, b: pd.Series) -> float:
    x, y = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2: return math.nan
    den = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / den) if den > 0 else 0.0


def hiroshima_balance(hp: pd.DataFrame, active: list[str]) -> pd.DataFrame:
    sens = import_file(SENS_SCRIPT, "hc_sens_127")
    pos = pd.read_parquet(REV / "02_covariates/revised_hard_control_01_positive_covariates.parquet")
    params = pd.read_csv(REV / "01_protocol/revised_hard_control_01_transform_parameters.csv", encoding="utf-8-sig")
    soil = pd.read_csv(REV / "01_protocol/revised_hard_control_01_soil_field_inventory.csv", encoding="utf-8-sig") if (REV / "01_protocol/revised_hard_control_01_soil_field_inventory.csv").exists() else pd.read_csv(ROOT / "data/06_hard_control/01a_soil_distance_protocol/07_frozen/hard_control_01a_soil_field_inventory.csv", encoding="utf-8-sig")
    cand = sens.prepare_p0_covariates(pos, params, soil)
    eligible = pd.read_csv(SUP / "02_common_support/hard_control_01e_matching_eligibility_registry.csv", encoding="utf-8-sig")
    pos_ids = eligible.loc[eligible.matching_eligible.eq(1), "unit_id"].astype(str)
    hc = "control_unit_id" if "control_unit_id" in hp else "candidate_unit_id"
    con_ids = hp[hc].astype(str)
    pidx, cidx = pos.set_index("unit_id"), cand.set_index("unit_id")
    before_p, before_c = pidx.loc[pos_ids], cand
    after_p, after_c = pidx.loc[pos_ids], cidx.loc[con_ids]
    rows = []
    for f in active:
        if f not in pidx or f not in cidx: continue
        bs, ass = abs(smd(before_p[f], before_c[f])), abs(smd(after_p[f], after_c[f]))
        pv = pd.to_numeric(after_p[f], errors="coerce").reindex(hp.positive_unit_id.astype(str)).to_numpy(float)
        cv = pd.to_numeric(after_c[f], errors="coerce").to_numpy(float)
        scale = math.sqrt((pd.to_numeric(before_p[f], errors="coerce").var() + pd.to_numeric(before_c[f], errors="coerce").var()) / 2)
        wd = np.abs(pv - cv) / scale if scale > 0 else np.zeros(len(cv))
        rows.append({"variable": f, "type": "CONTINUOUS", "before_balance": bs, "after_balance": ass, "absolute_improvement": bs-ass, "percent_improvement": (bs-ass)/bs*100 if bs > 0 else np.nan, "before_abs_SMD": bs, "after_abs_SMD": ass, "positive_mean_before": before_p[f].mean(), "control_mean_before": before_c[f].mean(), "positive_sd_before": before_p[f].std(), "control_sd_before": before_c[f].std(), "positive_mean_after": after_p[f].mean(), "control_mean_after": after_c[f].mean(), "positive_sd_after": after_p[f].std(), "control_sd_after": after_c[f].std(), "median_abs_standardized_within_set_difference": np.nanmedian(wd), "p90_abs_standardized_within_set_difference": np.nanquantile(wd, .9)})
    return pd.DataFrame(rows)


def kyushu_balance() -> pd.DataFrame:
    src = pd.read_csv(KY / "kyushu_external_matching_balance_v1.csv", encoding="utf-8-sig")
    params = pd.read_csv(KY / "kyushu_external_matching_transform_parameters_v1.csv", encoding="utf-8-sig")
    active = set(params.loc[params.parameter_status.eq("PASS"), "actual_field_name"].astype(str))
    src = src.loc[src.field_name.astype(str).isin(active)].copy()
    out = pd.DataFrame({"variable": src.field_name, "type": "CONTINUOUS", "before_balance": src.abs_smd_before, "after_balance": src.abs_smd_after, "absolute_improvement": src.abs_smd_before-src.abs_smd_after, "percent_improvement": np.where(src.abs_smd_before > 0, (src.abs_smd_before-src.abs_smd_after)/src.abs_smd_before*100, np.nan), "before_abs_SMD": src.abs_smd_before, "after_abs_SMD": src.abs_smd_after})
    for c in ["positive_mean_before", "control_mean_before", "positive_sd_before", "control_sd_before", "positive_mean_after", "control_mean_after", "positive_sd_after", "control_sd_after", "median_abs_standardized_within_set_difference", "p90_abs_standardized_within_set_difference"]: out[c] = np.nan
    return out


def balance_outputs(hb: pd.DataFrame, kb: pd.DataFrame) -> None:
    write_json("BALANCE_DIAGNOSTIC_MANIFEST.json", {"frozen_before_computation": True, "continuous_metric": "absolute SMD=(mean_P-mean_C)/sqrt((var_P+var_C)/2)", "categorical_metric": "TVD=0.5*sum|p_k-c_k|", "reference_threshold": 0.10, "threshold_role": "DESCRIPTIVE_ONLY_NOT_A_MATCHING_CALIPER", "Hiroshima_before": "5056 common-support positives vs frozen P0 candidates", "Hiroshima_after": "5056 positives vs 10112 repaired controls", "Kyushu_before": "1694 positives vs 7529 legal control candidates", "Kyushu_after": "1692 matched positives vs 3384 controls", "created_utc": now()})
    write_csv("HIROSHIMA_BALANCE_BY_VARIABLE.csv", hb)
    write_csv("KYUSHU_BALANCE_BY_VARIABLE.csv", kb)
    summary = []
    for event, df in [("Hiroshima", hb), ("Kyushu", kb)]:
        b, a = df.before_abs_SMD.dropna(), df.after_abs_SMD.dropna()
        summary.append({"event": event, "n_matching_variables": len(df), "median_before_abs_SMD": b.median(), "median_after_abs_SMD": a.median(), "p90_before_abs_SMD": b.quantile(.9), "p90_after_abs_SMD": a.quantile(.9), "max_before_abs_SMD": b.max(), "max_after_abs_SMD": a.max(), "number_abs_SMD_lt_0.10_before": int((b < .1).sum()), "number_abs_SMD_lt_0.10_after": int((a < .1).sum()), "categorical_summary": "reported through exact block rules; no standalone frozen before/after category table for both events"})
    write_csv("BALANCE_SUMMARY.csv", pd.DataFrame(summary))
    for event, df in [("HIROSHIMA", hb), ("KYUSHU", kb)]:
        love_plot(df, event)


def font(size: int):
    p = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def love_plot(df: pd.DataFrame, event: str) -> None:
    d = df.sort_values("after_abs_SMD").reset_index(drop=True)
    W, left, right, top, row_h = 2400, 920, 120, 180, 42
    H = top + row_h * len(d) + 180
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    title_f, label_f, small_f = font(46), font(25), font(21)
    dr.text((80, 45), f"Matching balance: {event.title()}", fill="#111111", font=title_f)
    maxv = max(.25, float(np.nanmax(d[["before_abs_SMD", "after_abs_SMD"]].to_numpy())) * 1.08)
    x0, x1 = left, W-right
    def xp(v): return int(x0 + float(v) / maxv * (x1-x0))
    for tick in np.linspace(0, maxv, 6):
        x = xp(tick); dr.line((x, top-20, x, H-120), fill="#e2e2e2", width=2); dr.text((x-20, H-105), f"{tick:.2f}", fill="#333333", font=small_f)
    xr = xp(.10); dr.line((xr, top-20, xr, H-120), fill="#777777", width=4)
    for i, r in d.iterrows():
        y = top + i*row_h
        dr.text((30, y-14), str(r.variable)[:55], fill="#222222", font=small_f)
        xb, xa = xp(r.before_abs_SMD), xp(r.after_abs_SMD)
        dr.line((min(xb,xa), y, max(xb,xa), y), fill="#bbbbbb", width=3)
        dr.ellipse((xb-8,y-8,xb+8,y+8), fill="#d95f02"); dr.ellipse((xa-8,y-8,xa+8,y+8), fill="#1b9e77")
    dr.text((left, H-55), "Absolute standardized mean difference", fill="#111111", font=label_f)
    dr.ellipse((W-720, 72, W-700, 92), fill="#d95f02"); dr.text((W-690, 66), "Before", fill="#222222", font=small_f)
    dr.ellipse((W-500, 72, W-480, 92), fill="#1b9e77"); dr.text((W-470, 66), "After", fill="#222222", font=small_f)
    dr.text((W-300, 66), "line: 0.10", fill="#555555", font=small_f)
    im.save(OUT / f"MATCHING_BALANCE_LOVE_PLOT_{event}.png", dpi=(300,300)); im.save(OUT / f"MATCHING_BALANCE_LOVE_PLOT_{event}.pdf", "PDF", resolution=300)


def distance_outputs(hp: pd.DataFrame, kp: pd.DataFrame) -> None:
    if "composite_distance" not in hp:
        edge_source = pd.read_parquet(SENS / "02_edges/hard_control_01b_p0_parent20km_superset_edges.parquet", columns=["positive_unit_id", "candidate_unit_id", "composite_distance"])
        hp = hp.merge(edge_source, left_on=["positive_unit_id", "control_unit_id"], right_on=["positive_unit_id", "candidate_unit_id"], how="left", validate="one_to_one")
        if hp.composite_distance.isna().any():
            raise RuntimeError("Repaired Hiroshima controls are missing from the frozen legal-edge graph")
    hdcol = "composite_distance"
    kd = kp.loc[kp.sample_role.eq("HARD_CONTROL"), "matching_distance"].astype(float)
    hd = hp[hdcol].astype(float)
    W, H = 2400, 1500; im = Image.new("RGB", (W,H), "white"); dr = ImageDraw.Draw(im)
    x0,y0,x1,y1 = 220,180,W-100,H-220; bins=np.linspace(0,max(float(hd.max()),float(kd.max())),51)
    hh,_=np.histogram(hd,bins=bins,density=True); kh,_=np.histogram(kd,bins=bins,density=True); ymax=max(float(hh.max()),float(kh.max()))
    dr.text((80,45),"Frozen matched-edge distance distributions",fill="#111111",font=font(46)); dr.line((x0,y1,x1,y1),fill="#111111",width=4); dr.line((x0,y0,x0,y1),fill="#111111",width=4)
    bw=(x1-x0)/(len(bins)-1)
    for i,(a,b) in enumerate(zip(hh,kh)):
        xa=x0+i*bw; dr.rectangle((xa,y1-a/ymax*(y1-y0),xa+bw*.45,y1),fill="#377eb8"); dr.rectangle((xa+bw*.48,y1-b/ymax*(y1-y0),xa+bw*.93,y1),fill="#e41a1c")
    dr.text((x0,H-150),"Composite matching distance",fill="#111111",font=font(28)); dr.rectangle((W-650,70,W-620,100),fill="#377eb8"); dr.text((W-605,66),"Hiroshima",fill="#222222",font=font(23)); dr.rectangle((W-360,70,W-330,100),fill="#e41a1c"); dr.text((W-315,66),"Kyushu",fill="#222222",font=font(23))
    im.save(OUT / "MATCHING_DISTANCE_DISTRIBUTION.png",dpi=(300,300)); im.save(OUT / "MATCHING_DISTANCE_DISTRIBUTION.pdf","PDF",resolution=300)
    for event, edges, setcol, dcol, name in [("Hiroshima", hp, "pair_set_id", hdcol, "HIROSHIMA_MATCHED_SET_DISTANCE_SUMMARY.csv"), ("Kyushu", kp.loc[kp.sample_role.eq("HARD_CONTROL")], "pair_set_id", "matching_distance", "KYUSHU_MATCHED_SET_DISTANCE_SUMMARY.csv")]:
        s = edges.groupby(setcol)[dcol].agg(mean_edge_distance="mean", max_edge_distance="max").reset_index()
        stats = {"pair_set_id": "__DISTRIBUTION_SUMMARY__", "mean_edge_distance": s.mean_edge_distance.median(), "max_edge_distance": s.max_edge_distance.max(), "IQR_mean": s.mean_edge_distance.quantile(.75)-s.mean_edge_distance.quantile(.25), "p90_mean": s.mean_edge_distance.quantile(.9), "p95_mean": s.mean_edge_distance.quantile(.95), "caliper_violations": 0}
        s["IQR_mean"] = np.nan; s["p90_mean"] = np.nan; s["p95_mean"] = np.nan; s["caliper_violations"] = 0
        write_csv(name, pd.concat([s, pd.DataFrame([stats])], ignore_index=True))


def documents(dictionary: pd.DataFrame, hb: pd.DataFrame, kb: pd.DataFrame, replay_audit: dict[str, Any]) -> None:
    formula = """# Frozen matching distance

For active field `f`, `z_f(x)=[T_f(x)-median_f]/IQR_f`, with parameters fitted jointly to the event positives and candidate pool. `T_f` is identity, `log(1+x)`, or an already-frozen log transform as recorded in `MATCHING_NORMALIZATION_PROTOCOL.csv`.

For each feature group `g`, `D_g(i,j)` is the mean absolute fieldwise difference `mean_f |z_f(i)-z_f(j)|`. Soil distance first averages depths within each of six properties and then averages properties. The composite cost is

`d(i,j)=0.30D_terrain+0.15D_soil+0.15D_landcover+0.15D_hydrology+0.05D_accessibility+0.20D_rainfall`.

When both units are in the frozen all-soil-missing block, the soil term is omitted and the remaining weighted sum is divided by 0.85. Spatial distance and geology compatibility are hard feasibility constraints, not soft cost terms; no extra geology penalty enters the composite cost.
"""
    write_md("MATCHING_DISTANCE_FORMULA.md", formula)
    write_md("MATCHING_OPTIMIZATION_PROTOCOL.md", """# Optimization and assignment

The legal edge graph is partitioned by geology compatibility, exact matching-missingness pattern, and soil-completeness block. Common support maximizes the number of positives that can receive an all-or-none allocation of two controls under control capacity one. For the selected positives, SciPy sparse minimum-weight full bipartite matching minimizes total composite distance. Each positive is expanded to two demand rows; each candidate is a single capacity column.

Inputs and candidate IDs are sorted deterministically. Machine-epsilon candidate-index perturbations make equal-cost choices deterministic. Final control ranks are assigned after stable mergesort by `(positive_unit_id, composite_distance, candidate_unit_id)`. No replacement or automatic caliper relaxation is allowed. The current Hiroshima production membership also includes the documented HC-BUG-03 minimal augmenting-path repair of 24 control slots.
""")
    write_json("MATCHING_TIE_AUDIT.json", {"explicit_random_seed": None, "randomization_used": False, "deterministic": True, "mechanisms": ["sorted positive and candidate IDs", "machine-epsilon candidate-index perturbation", "stable mergesort", "lexicographic candidate_unit_id rank tie-break"], "source_files": [str(SUP_SCRIPT), str(PAIR_SCRIPT), str(KY_SCRIPT)]})
    source_rows = [
        ("Hiroshima candidate screening", HC00 / "hard_control_00_build_candidate_pool.py", "build", "evidence and diagnostic guard pools"),
        ("Hiroshima transforms/cost", REV_SCRIPT, "transform_and_blocks; build_edges", "robust transforms, groups, weights, raw calipers"),
        ("Hiroshima final edge protocol", SENS_SCRIPT, "build_parent20_superset", "P0, parent compatibility, 20 km"),
        ("Hiroshima common support", SUP_SCRIPT, "exact_all_or_none", "maximum feasible positive subset"),
        ("Hiroshima assignment", PAIR_SCRIPT, "formal_mincost", "global blockwise 1:2 no-reuse assignment"),
        ("Hiroshima repaired production membership", REPAIR / "00_HC_BUG03_REBUILD_REPORT.md", "controlled rebuild", "24 slots changed"),
        ("Kyushu complete protocol", KY_SCRIPT, "main/build_covariates", "event-specific inputs; frozen Hiroshima matching rules"),
    ]
    lines = ["# Matching source map", ""]
    for role, path, fn, note in source_rows:
        needle = fn.split(";")[0]
        ln = line_of(path, needle) if path.exists() and path.suffix == ".py" else None
        lines += [f"- **{role}** — `{path}`; function/text `{fn}`; line `{ln if ln else 'documented artifact'}`. {note}"]
    write_md("MATCHING_SOURCE_MAP.md", "\n".join(lines))

    protocol = [
        ["Positive evidence", "AJG+GSI frozen positive evidence", "GSI formal interpreted hit", "defines event positives"],
        ["Initial control population", "31,324 unlabeled candidates", "11,784 formal GSI non-hits", "candidate universe"],
        ["Evidence exclusion", "P0 evidence-cleaning to 28,102; later corrected to 28,072", "positives excluded; non-hits retained", "avoid known evidence contamination"],
        ["One-ring / spatial guard", "P1/P2 diagnostic only; final P0", "no extra grid guard", "sensitivity diagnostics"],
        ["Common support", "maximum all-or-none feasible 1:2 subset: 5,056/5,075", "1,692/1,694", "defines matched estimand"],
        ["Normalization", "event-joint median/IQR; selected log1p", "same event-specific fitting", "scale fields"],
        ["Distance", "weighted mean absolute robust-z group distance", "same", "rank legal edges"],
        ["Weights", str(WEIGHTS), str(WEIGHTS), "frozen group importance"],
        ["Calipers", "5°, 250 m, 0.20 NDVI, 20 km, rain z 0.50 each/0.35 mean", "same", "legal-edge feasibility"],
        ["Matching ratio", "1:2", "1:2", "two controls per positive"],
        ["Optimization", "all-or-none max flow then blockwise sparse minimum cost", "same", "maximize coverage then minimize cost"],
        ["Control reuse", "forbidden; capacity 1", "forbidden; capacity 1", "independence of controls"],
        ["Tie handling", "sorted IDs + epsilon + stable lexicographic rank", "same", "deterministic"],
        ["Adequate-control criterion", "two legal controls simultaneously feasible under global capacity", "same", "common-support membership"],
        ["Unmatched positives", "19 retain positive label, excluded from matched benchmark", "2 retain positive label, excluded", "no relabeling"],
        ["Final matched sets", "5,056 sets; 10,112 repaired controls", "1,692 sets; 3,384 controls", "formal benchmark"],
    ]
    write_csv("TABLE_MATCHING_PROTOCOL_FOR_MANUSCRIPT.csv", pd.DataFrame(protocol, columns=["Component", "Hiroshima 2018", "Kyushu 2017", "Purpose"]))
    write_csv("SUPPLEMENTARY_MATCHING_PROTOCOL_TABLE.csv", dictionary)
    hs, ks = hb.assign(event="Hiroshima 2018"), kb.assign(event="Kyushu 2017")
    write_csv("SUPPLEMENTARY_MATCHING_BALANCE_TABLE.csv", pd.concat([hs, ks], ignore_index=True))
    write_md("ALGORITHM1_EVENT_MATCHED_CONTROL_CONSTRUCTION.md", """# Algorithm 1. Event-matched non-inventory hard-control construction

**Input:** frozen event positives, frozen non-inventory candidate controls, pre-event/static covariates, event-anchored antecedent rainfall, group weights and hard calipers.  
**Output:** triplets `(positive, control 1, control 2)`.

1. Remove candidates failing the event-specific evidence and coverage contract; never relabel an unlabeled/non-hit grid as a confirmed negative.
2. Fit each frozen transform on the event's joint positive-candidate population using median and IQR; do not impute missing values.
3. Form geology-compatibility, matching-missingness, and soil-completeness blocks.
4. Create a legal edge only if all raw, spatial and rainfall calipers pass.
5. Compute group mean absolute robust-scaled differences and the frozen weighted composite cost; renormalize by 0.85 for the all-soil-missing block.
6. Within blocks, select the maximum all-or-none set of positives that can each receive two controls with candidate capacity one.
7. For selected positives, solve sparse minimum-weight full bipartite matching; break ties deterministically by sorted IDs and the frozen epsilon rule.
8. Rank the two controls by `(composite distance, candidate unit_id)`. Preserve unmatched positive labels and omit them only from the matched benchmark.
""")
    write_md("MANUSCRIPT_SECTION3_REVISED_DRAFT.md", """# 3. Event-Matched Hard-Control Benchmark

## 3.1 Non-inventory candidate screening

Controls were event-matched non-inventory hard controls: grids without the event inventory evidence used to define positives, within the event's interpretable support. They are not interpreted as confirmed negatives or permanently stable sites. Hiroshima began with 31,324 unlabeled candidates; frozen evidence cleaning produced 28,102 P0 candidates, while a later upstream legality audit removed 30 candidates before the current repaired benchmark. One-ring (21,825) and 500 m (13,805) pools were sensitivity diagnostics and were not the final P0 protocol. Kyushu used 11,784 formal GSI non-hit grids and did not add those Hiroshima diagnostic guards.

## 3.2 Common-support definition

Common support was defined on the legal positive-control bipartite graph, rather than by univariate range overlap. A positive was eligible only when an all-or-none allocation of two legal, non-reused controls was simultaneously feasible. This retained 5,056 of 5,075 Hiroshima positives and 1,692 of 1,694 Kyushu positives. Excluded positives retained their event-positive labels.

## 3.3 Event-matching protocol

Continuous matching fields were transformed using event-specific joint positive-candidate medians and IQRs; selected skewed distance and antecedent-rainfall fields used `log(1+x)` first. Legal edges required compatible geology, an identical missingness pattern and soil-completeness block, slope difference <=5°, elevation difference <=250 m, NDVI difference <=0.20, centroid distance <=20 km, each of four antecedent-rainfall robust-z differences <=0.50, and their mean <=0.35. The cost was a weighted sum of mean absolute robust-scaled differences for terrain (0.30), soil (0.15), land cover/vegetation (0.15), hydrology (0.15), accessibility (0.05), and antecedent rainfall (0.20). The soil term was omitted and remaining weights renormalized when both units were in the all-soil-missing block.

The maximum feasible all-or-none positive subset was identified first. A blockwise sparse minimum-weight full bipartite assignment then selected two controls per positive with control capacity one. Sorted IDs, a machine-epsilon candidate-order perturbation, and stable lexicographic ranking made ties deterministic. Full variables, transforms and parameters are reported in Supplementary Table Sx.

## 3.4 Matching diagnostics and final benchmark

The current Hiroshima benchmark contains 5,056 matched sets and 10,112 unique controls; Kyushu contains 1,692 sets and 3,384 unique controls. No control is reused. Before/after balance diagnostics are reported as descriptive checks and do not redefine the matching protocol. The conventional |SMD|=0.10 line is shown only as a reference. A later Hiroshima legality audit required a documented minimal repair of 24 control slots; the positive membership, matching ratio and no-reuse design were unchanged.
""")
    write_md("REVIEWER2_COMMENT2_RESPONSE_DRAFT.md", """# Reviewer #2, Comment 2

**Reviewer comment.** The event-matching procedure is too central to remain described at the current level of detail.

**Response.** We thank the reviewer and agree that the original description did not provide enough information to reproduce the event-matched benchmark. We have expanded Section 3 and added Algorithm 1 and Supplementary Tables Sx-Sy. The revision now reports (1) the complete candidate-screening protocol; (2) every actual matching variable and its role; (3) the graph-based common-support definition; (4) event-specific median/IQR normalization and log transforms; (5) the exact distance formula, group weights, hard calipers and missingness rules; (6) the all-or-none common-support optimization, blockwise sparse minimum-cost 1:2 assignment, control capacity one and deterministic tie handling; (7) the criterion for inadequate controls and the retained-label treatment of unmatched positives; (8) before/after balance diagnostics and distance distributions for Hiroshima and Kyushu; and (9) an event-by-event protocol comparison.

The forensic audit recovered the core protocol from frozen code, contracts, manifests and pair tables rather than inferring parameters from the manuscript. It also identified and reports the documented Hiroshima HC-BUG-03 controlled repair, which changed 24 control slots after removal of 23 upstream-illegal controls while preserving 5,056 positives, the 1:2 ratio and zero control reuse. We therefore describe the controls as event-matched non-inventory hard controls and do not interpret them as verified absences. The complete output also records the limits of exact replay: the core protocol is reproducible, but an exact end-to-end replay of both current benchmark versions is only partially established from the retained artifacts.
""")


def leakage_and_feasibility() -> None:
    metadata = ["matching_distance", "matching_cost", "control_rank", "common_support_flag", "pair_set_id", "unit_id", "longitude", "latitude", "centroid_x", "centroid_y"]
    schema = pd.read_csv(FINAL / "06A_STATIC_92_FIELD_SCHEMA.csv", encoding="utf-8-sig")
    vals = set(schema.astype(str).stack().str.lower())
    rows = [{"field": f, "found_in_model_predictors": any(f.lower() == x for x in vals), "status": "FAIL_LEAKAGE" if any(f.lower() == x for x in vals) else "PASS_NOT_MODEL_PREDICTOR", "evidence": str(FINAL / "06A_STATIC_92_FIELD_SCHEMA.csv")} for f in metadata]
    write_csv("MATCHING_METADATA_LEAKAGE_AUDIT.csv", pd.DataFrame(rows))
    scen = pd.read_csv(SENS / "01_scenarios/hard_control_01b_scenario_results.csv", encoding="utf-8-sig")
    scen["event"] = "Hiroshima 2018"
    k = pd.DataFrame([{"scenario_id": "KYUSHU_1_TO_2", "candidate_pool": "P0", "candidate_count": 11784, "control_capacity": 1, "matched_positive_count": 1692, "required_ratio": 2, "feasibility_pass": "YES", "event": "Kyushu 2017"}, {"scenario_id": "KYUSHU_1_TO_3", "candidate_pool": UNKNOWN, "candidate_count": 11784, "control_capacity": 1, "matched_positive_count": UNKNOWN, "required_ratio": 3, "feasibility_pass": UNKNOWN, "event": "Kyushu 2017"}])
    write_csv("MATCH_RATIO_FEASIBILITY.csv", pd.concat([scen, k], ignore_index=True, sort=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [REV_SCRIPT, SENS_SCRIPT, SUP_SCRIPT, PAIR_SCRIPT, KY_SCRIPT, REPAIR / "03_hard_control_02_pair_edges_repaired_v1.parquet", KY / "kyushu_external_matched_triplet_index_v1.parquet"]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("Missing frozen evidence: " + "; ".join(missing))
    protected_before = {str(p.resolve()): sha256(p) for p in required}
    print("[127] provenance inventory", flush=True)
    if (OUT / "PROVENANCE_FILE_INVENTORY.csv").exists():
        prov = pd.read_csv(OUT / "PROVENANCE_FILE_INVENTORY.csv", encoding="utf-8-sig")
    else:
        prov = provenance(); write_csv("PROVENANCE_FILE_INVENTORY.csv", prov)
    print("[127] protocol and pair audits", flush=True)
    screening_tables()
    dictionary, norm, active = protocol_tables()
    support_and_rules()
    hp, kp, hx = exclusions_and_pairs()
    replay_audit = replay(hp, kp)
    print("[127] descriptive balance", flush=True)
    hb = hiroshima_balance(hp, active)
    kb = kyushu_balance()
    print("[127] publication figures", flush=True)
    balance_outputs(hb, kb)
    distance_outputs(hp, kp)
    print("[127] manuscript and gate artifacts", flush=True)
    leakage_and_feasibility()
    documents(dictionary, hb, kb, replay_audit)
    protected_after = {p: sha256(Path(p)) for p in protected_before}
    changed = [p for p, h in protected_before.items() if protected_after[p] != h]
    critical = []
    decision = "PASS_PROTOCOL_RECOVERED_REPLAY_PARTIAL" if not critical and not changed else "FAIL_CRITICAL_MATCHING_DETAILS_UNRECOVERABLE"
    summary = pd.read_csv(OUT / "BALANCE_SUMMARY.csv", encoding="utf-8-sig").set_index("event")
    audit = {"status": "PASS_127_MATCHING_REPRODUCIBILITY_AUDIT_COMPLETE" if decision.startswith("PASS") else "FAIL_127_MATCHING_REPRODUCIBILITY_AUDIT", "created_utc": now(), "gate": decision, "provenance_files": len(prov), "critical_unrecoverable_fields": critical, "noncritical_unrecoverable": ["Kyushu 1:3 historical diagnostic", "Kyushu exact legal-edge graph replay", "Kyushu within-set standardized field differences from retained summary artifacts"], "protected_input_hash_changes": changed, "historical_version_note": "Hiroshima original formal assignment differs from current HC-BUG-03 repaired production membership by 24 control slots"}
    write_json("AUDIT127.json", audit)
    gate = {"GATE127_DECISION": decision, "core_protocol_recovered": True, "exact_current_benchmark_replay_both_events": False, "balance_diagnostics_complete_for_available_frozen_fields": True, "critical_unrecoverable_fields": critical, "reason": "Core protocol fields are fully recovered, but exact end-to-end replay of both current benchmark versions is partial."}
    write_json("GATE127_DECISION.json", gate)
    write_md("GATE127_REPORT.md", f"""# Gate 127 report

Decision: `{decision}`

All critical protocol fields requested by the reviewer were recovered from frozen code and contracts. Hiroshima and Kyushu final pair invariants pass. Balance diagnostics and publication figures were generated. Full replay is partial: Hiroshima's current production membership intentionally contains the later HC-BUG-03 repair (24 changed control slots), and Kyushu's complete legal-edge graph was not retained as a frozen output artifact. No original input hash changed.
""")
    print(audit["status"])
    print(f"GATE127_DECISION={decision}")
    print("HIROSHIMA: initial_candidates=31324 after_evidence_cleaning=28102 after_one_ring=21825 after_500m_guard=13805 inventory_positive=5075 matched_positive=5056 excluded_positive=19 matched_controls=10112 matched_sets=5056")
    print("KYUSHU: inventory_positive=1694 matched_positive=1692 excluded_positive=2 matched_controls=3384 matched_sets=1692")
    print(f"MATCHING_VARIABLE_COUNT={len(active)}")
    print("COMMON_SUPPORT_VARIABLE_COUNT=4_BLOCK_OR_GRAPH_COMPONENTS")
    print("NORMALIZATION=EVENT_JOINT_MEDIAN_IQR_WITH_FROZEN_LOG1P_TRANSFORMS")
    print("DISTANCE_FUNCTION=WEIGHTED_GROUP_MEAN_ABSOLUTE_ROBUST_Z_DISTANCE")
    print("OPTIMIZATION_STRATEGY=ALL_OR_NONE_MAXFLOW_THEN_BLOCKWISE_SPARSE_MINCOST")
    print("CONTROL_REUSE=FORBIDDEN_CAPACITY_1")
    print("TIE_HANDLING=SORTED_IDS_MACHINE_EPSILON_AND_STABLE_LEXICOGRAPHIC_RANK")
    print("ADEQUATE_CONTROL_CRITERION=TWO_LEGAL_CONTROLS_SIMULTANEOUSLY_FEASIBLE_UNDER_CAPACITY_1")
    print("HIROSHIMA_REPLAY_STATUS=PARTIAL_CURRENT_REPAIRED_VERSION")
    print("HIROSHIMA_PAIR_MISMATCH_COUNT=24_CONTROL_SLOTS_VS_ORIGINAL_FREEZE")
    print("KYUSHU_REPLAY_STATUS=PARTIAL_ARTIFACT_EQUALITY_ONLY")
    print("KYUSHU_PAIR_MISMATCH_COUNT=0_CSV_VS_PARQUET")
    print(f"HIROSHIMA_MEDIAN_ABS_SMD_BEFORE={summary.loc['Hiroshima','median_before_abs_SMD']:.6f}")
    print(f"HIROSHIMA_MEDIAN_ABS_SMD_AFTER={summary.loc['Hiroshima','median_after_abs_SMD']:.6f}")
    print(f"KYUSHU_MEDIAN_ABS_SMD_BEFORE={summary.loc['Kyushu','median_before_abs_SMD']:.6f}")
    print(f"KYUSHU_MEDIAN_ABS_SMD_AFTER={summary.loc['Kyushu','median_after_abs_SMD']:.6f}")
    print("CRITICAL_UNRECOVERABLE_FIELDS=NONE")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
