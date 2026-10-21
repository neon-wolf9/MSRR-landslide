# -*- coding: utf-8 -*-
r"""
17_FINALIZE_REPAIRED_DATASET_V2.py
=================================

ONE-SHOT FINAL DATASET CLOSURE for the repository root.

Goal
----
End the internal data-repair stage in one run.

This script does NOT overwrite the old frozen dataset.
It creates a new authoritative repaired dataset bundle:

data/07_FINAL_REPAIRED_DATASET_V2

The final bundle contains:
- repaired 5,056 x (1 positive + 2 hard controls);
- 15,168-row training cohort;
- canonical positive-anchored spatial-group / fold assignment;
- model-ready pair tensor;
- frozen static 92-feature raw table;
- frozen dynamic 70 x 10 tensor;
- dynamic timestamp registry;
- corrected HC00 membership audit;
- P0 pair-edge legality audit;
- no-reuse / pair-integrity / fold anti-leakage QA;
- manifest + hashes + final decision;
- DATA_SECTION_CLOSED.flag ONLY IF every hard QA passes.

IMPORTANT SEMANTICS
-------------------
1. Formal positives are NOT changed.
2. Repaired hard controls come from the already completed
   HC-BUG-03 controlled rebuild.
3. New controls are NOT required to have an old pre-match spatial_group_id.
4. Spatial group/fold is a PAIR / positive-anchor split property.
5. Since pair_set_id and positive_unit_id are unchanged, the canonical
   pair -> spatial_group_id / fold mapping remains the authoritative
   split mapping. Every member of a repaired pair inherits that mapping.
6. Static/dynamic source features remain the already frozen upstream
   EO/GIS and rainfall assets.
7. No model training is performed here.

Expected final invariants
-------------------------
pair_sets                     = 5,056
positives                     = 5,056
hard_controls                 = 10,112
unique_hard_controls          = 10,112
training_rows                 = 15,168
changed_control_slots         = 24
corrected_hc00_core           = 28,072
static_fields                 = 92
dynamic_fields                = 10
dynamic_timesteps             = 70
dynamic tensor shape          = (15168, 70, 10)
pair tensor shape             = (5056, 3)
pair cross fold               = 0
spatial group cross fold      = 0
P0 edge failures              = 0
controls outside corrected HC00 = 0
control reuse                 = 0

Run
---
python scripts/17_FINALIZE_REPAIRED_DATASET_V2.py

This source-to-frozen-data builder is not part of the lightweight review tests.
It requires the original private upstream inputs and must not be run merely
because the script is present in a release candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# GLOBAL CONTRACT
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT = ROOT / "data" / "07_FINAL_REPAIRED_DATASET_V2"
STAGE_ROOT = ROOT / "data" / "07_FINAL_REPAIRED_DATASET_V2__STAGING"

REBUILD_ROOT = (
    ROOT
    / "data"
    / "06_hard_control"
    / "03_hc_bug03_controlled_rebuild_v1_1"
)

ORIGINAL_PAIR_SETS = (
    ROOT
    / "data"
    / "06_hard_control"
    / "02_formal_pair_assignment"
    / "03_pair_tables"
    / "hard_control_02_pair_sets.parquet"
)

ORIGINAL_FORMAL_LABEL = (
    ROOT
    / "data"
    / "06_hard_control"
    / "02_formal_pair_assignment"
    / "04_label_layer"
    / "hard_control_02_full_grid_label_layer.parquet"
)

CANON_PAIR_FOLD = (
    ROOT
    / "data"
    / "05_final_dataset_assembly"
    / "01_spatial_split_protocol"
    / "dataset_01_pair_set_fold_crosswalk.parquet"
)

CANON_UNIT_FOLD = (
    ROOT
    / "data"
    / "05_final_dataset_assembly"
    / "01_spatial_split_protocol"
    / "dataset_01_unit_fold_crosswalk.parquet"
)

CANON_PRIMARY_ASSIGN = (
    ROOT
    / "data"
    / "05_final_dataset_assembly"
    / "01_spatial_split_protocol"
    / "dataset_01_primary_5fold_assignment.parquet"
)

OLD_FEATURE_WHITELIST = (
    ROOT
    / "data"
    / "05_final_dataset_assembly"
    / "00_dataset_inventory_and_split_feasibility"
    / "dataset_00_model_feature_whitelist.csv"
)

STATIC_MODEL_VIEW = (
    ROOT
    / "data"
    / "03_static_eogis_features"
    / "90_third_layer_assembly"
    / "04_static_02_assembly"
    / "03_model_candidate_view"
    / "static_third_layer_model_candidate_view.parquet"
)

STATIC_FIELDS_FROZEN_CANDIDATES = [
    ROOT
    / "data"
    / "03_static_eogis_features"
    / "90_third_layer_assembly"
    / "04_static_02_assembly"
    / "06_frozen"
    / "static_third_layer_model_fields_frozen.csv",

    ROOT
    / "data"
    / "03_static_eogis_features"
    / "90_third_layer_assembly"
    / "04_static_02_assembly"
    / "03_model_candidate_view"
    / "static_third_layer_model_fields_frozen.csv",
]

RAIN_DATASET = (
    ROOT
    / "data"
    / "04_dynamic_rainfall_features"
    / "07_rain_04_final_audit_freeze"
    / "07_frozen"
    / "dynamic_rainfall_features_frozen.parquet_dataset"
)

P0_EDGES = (
    ROOT
    / "data"
    / "06_hard_control"
    / "01c_matching_sensitivity_audit"
    / "02_edges"
    / "hard_control_01b_p0_parent20km_superset_edges.parquet"
)

ELIGIBILITY_CANDIDATES = [
    ROOT
    / "data"
    / "02_reliable_observation_domain"
    / "11_hard_control_candidate_eligibility_pool_frozen_v1_1"
    / "01_frozen_full_registry"
    / "hard_control_eligibility_full_grid_250m_frozen_v1_1.parquet",

    ROOT
    / "data"
    / "02_reliable_observation_domain"
    / "11_hard_control_candidate_eligibility_pool_frozen"
    / "01_frozen_full_registry"
    / "hard_control_eligibility_full_grid_250m_frozen_v1.parquet",
]

HC00_CORE_CANDIDATES = [
    ROOT
    / "data"
    / "06_hard_control"
    / "00_candidate_pool"
    / "hard_control_00_core_candidate_pool.csv",

    ROOT
    / "data"
    / "06_hard_control"
    / "00_candidate_pool"
    / "09_frozen"
    / "hard_control_00_core_candidate_pool_frozen.csv",

    ROOT
    / "data"
    / "06_hard_control"
    / "00_candidate_pool"
    / "hard_control_00_core_candidate_pool.parquet",
]

N_PAIRS = 5056
N_POS = 5056
N_CONTROLS = 10112
N_ROWS = 15168
N_CHANGED_SLOTS = 24
N_CORE = 28072
N_STATIC = 92
N_DYNAMIC = 10
N_TIME = 70

DYNAMIC_FIELDS = [
    "rain_30m_mm",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_12h_mm",
    "rain_24h_mm",
    "rain_48h_mm",
    "rain_72h_mm",
    "rain_120h_mm",
    "api_k090_step30m_120h",
]

MODEL_START_UTC = pd.Timestamp("2018-07-05T00:30:00Z")
MODEL_END_UTC = pd.Timestamp("2018-07-06T11:00:00Z")

FORBIDDEN_FEATURE_TOKENS = {
    "pair_set_id",
    "control_rank",
    "sample_role",
    "outer_fold",
    "outer_fold_id",
    "spatial_group_id",
    "evidence_component_id",
    "evidence_component_ids",
    "y_main",
    "y_pair",
    "matching_distance",
    "matching_cost",
    "composite_distance",
    "positive_label_flag",
    "hard_control_flag",
    "ajg",
    "gsi",
    "future_rainfall",
}

SOURCE_VERSION = "FINAL_REPAIRED_DATASET_V2_20260818"


# =====================================================================
# Utilities
# =====================================================================

def log(msg: str) -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (STAGE_ROOT / "FINALIZE.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def semantic_hash_frame(df: pd.DataFrame, keys: list[str] | None = None) -> str:
    x = df.copy()
    if keys:
        use = [k for k in keys if k in x.columns]
        if use:
            x = x.sort_values(use, kind="mergesort", na_position="last")
    x = x.reindex(sorted(x.columns), axis=1)
    payload = x.to_csv(
        index=False,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_schema(path: Path) -> list[str]:
    if path.is_dir():
        import pyarrow.dataset as ds
        return list(ds.dataset(str(path), format="parquet").schema.names)

    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)

    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)

    return []


def read_table(path: Path, columns=None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise RuntimeError(f"Unsupported table: {path}")


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def truthy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({"YES", "TRUE", "1", "Y", "T"})
    )


def norm_pair(v) -> str | None:
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s


def clean_text(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null", "<na>", "nat"}:
        return None
    return s


def choose_fold_col(cols: list[str]) -> str | None:
    for c in ["outer_fold", "fold", "fold_5", "cv_fold", "split_fold"]:
        if c in cols:
            return c
    candidates = [
        c for c in cols
        if "fold" in c.lower()
        and not any(b in c.lower() for b in ("count", "score", "metric"))
    ]
    return candidates[0] if candidates else None


def safe_remove_staging() -> None:
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")


# =====================================================================
# Step 0 — Prerequisites
# =====================================================================

def prerequisite_audit() -> dict:
    required = {
        "rebuild_root": REBUILD_ROOT,
        "original_pair_sets": ORIGINAL_PAIR_SETS,
        "original_formal_label": ORIGINAL_FORMAL_LABEL,
        "canonical_pair_fold": CANON_PAIR_FOLD,
        "canonical_unit_fold": CANON_UNIT_FOLD,
        "static_model_view": STATIC_MODEL_VIEW,
        "rain_dataset": RAIN_DATASET,
        "p0_edges": P0_EDGES,
    }
    missing = []
    for name, p in required.items():
        if not p.exists():
            missing.append({"asset": name, "path": str(p)})

    if missing:
        raise RuntimeError(
            "Missing prerequisites:\n" + json.dumps(missing, indent=2)
        )

    rebuild_qa = REBUILD_ROOT / "06_FINAL_REBUILD_QA.json"
    rebuild_decision = REBUILD_ROOT / "08_REBUILD_DECISION.json"

    qa_payload = None
    decision_payload = None

    if rebuild_qa.exists():
        qa_payload = json.loads(rebuild_qa.read_text(encoding="utf-8-sig"))
    if rebuild_decision.exists():
        decision_payload = json.loads(
            rebuild_decision.read_text(encoding="utf-8-sig")
        )

    return {
        "status": "PASS",
        "rebuild_qa": qa_payload,
        "rebuild_decision": decision_payload,
        "required_assets": {k: str(v) for k, v in required.items()},
    }


# =====================================================================
# Step 1 — Discover authoritative repaired 10,112-slot assignment
# =====================================================================

PAIR_NAMES = ["pair_set_id", "pair_id"]
RANK_NAMES = ["control_rank", "rank", "control_slot"]
CONTROL_NAMES = [
    "control_unit_id",
    "assigned_control_unit_id",
    "final_control_unit_id",
    "new_control_unit_id",
    "replacement_unit_id",
    "unit_id",
]
POS_NAMES = [
    "positive_unit_id",
    "matched_positive_unit_id",
]


def first_col(cols: list[str], names: list[str]) -> str | None:
    lowmap = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lowmap:
            return lowmap[n.lower()]
    return None


def parse_assignment_candidate(path: Path) -> pd.DataFrame | None:
    try:
        cols = get_schema(path)
    except Exception:
        return None

    pcol = first_col(cols, PAIR_NAMES)
    rcol = first_col(cols, RANK_NAMES)
    ccol = first_col(cols, CONTROL_NAMES)
    poscol = first_col(cols, POS_NAMES)

    # Long 10,112-row assignment.
    if pcol and rcol and ccol:
        try:
            q = read_table(
                path,
                [c for c in [pcol, rcol, ccol, poscol] if c is not None],
            )
        except Exception:
            return None

        q = q.rename(
            columns={
                pcol: "pair_set_id",
                rcol: "control_rank",
                ccol: "control_unit_id",
                **({poscol: "positive_unit_id"} if poscol else {}),
            }
        )

        q["pair_set_id"] = q["pair_set_id"].map(norm_pair)
        q["control_rank"] = pd.to_numeric(
            q["control_rank"], errors="coerce"
        )

        q = q[
            q["pair_set_id"].notna()
            & q["control_rank"].isin([1, 2])
            & q["control_unit_id"].notna()
        ].copy()

        q = q.drop_duplicates(["pair_set_id", "control_rank"])

        if len(q) == N_CONTROLS and q["pair_set_id"].nunique() == N_PAIRS:
            return q[
                ["pair_set_id", "control_rank", "control_unit_id"]
                + (["positive_unit_id"] if "positive_unit_id" in q.columns else [])
            ].copy()

    # Wide 5,056-row pair table.
    if pcol:
        wide_control_pairs = [
            ("control_unit_id_1", "control_unit_id_2"),
            ("control_1_unit_id", "control_2_unit_id"),
            ("control1_unit_id", "control2_unit_id"),
        ]
        lowmap = {c.lower(): c for c in cols}

        for a, b in wide_control_pairs:
            if a.lower() in lowmap and b.lower() in lowmap:
                ca, cb = lowmap[a.lower()], lowmap[b.lower()]
                use = [pcol, ca, cb]
                if poscol:
                    use.append(poscol)
                try:
                    w = read_table(path, use)
                except Exception:
                    return None

                if len(w) != N_PAIRS:
                    continue

                rows = []
                for r in w.itertuples(index=False):
                    d = r._asdict()
                    pair = norm_pair(d[pcol])
                    pos = d.get(poscol) if poscol else None
                    rows.append(
                        {
                            "pair_set_id": pair,
                            "control_rank": 1,
                            "control_unit_id": d[ca],
                            "positive_unit_id": pos,
                        }
                    )
                    rows.append(
                        {
                            "pair_set_id": pair,
                            "control_rank": 2,
                            "control_unit_id": d[cb],
                            "positive_unit_id": pos,
                        }
                    )

                q = pd.DataFrame(rows)
                if (
                    len(q) == N_CONTROLS
                    and q["control_unit_id"].notna().all()
                ):
                    return q

    return None


def discover_repaired_assignment() -> tuple[pd.DataFrame, dict]:
    candidates = []

    for p in REBUILD_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".parquet", ".csv"}:
            continue

        score = 0
        name = p.name.lower()
        path_low = str(p).lower()

        if "assign" in name:
            score += 50
        if "control" in name:
            score += 30
        if "pair" in name:
            score += 20
        if "repaired" in name or "rebuild" in name or "final" in name:
            score += 30
        if "changed" in name or "removed" in name or "new_" in name:
            score -= 40
        if "audit" in name or "qa" in name:
            score -= 15

        candidates.append((score, p))

    candidates.sort(key=lambda x: (-x[0], str(x[1])))

    audit_rows = []

    for score, path in candidates:
        try:
            q = parse_assignment_candidate(path)
            audit_rows.append(
                {
                    "path": str(path),
                    "score": score,
                    "accepted": bool(q is not None),
                    "rows": int(len(q)) if q is not None else None,
                }
            )
            if q is not None:
                pd.DataFrame(audit_rows).to_csv(
                    STAGE_ROOT / "01_REPAIRED_ASSIGNMENT_DISCOVERY.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                return q, {
                    "source_path": str(path),
                    "source_mode": "DISCOVERED_COMPLETE_REPAIRED_ASSIGNMENT",
                }
        except Exception as e:
            audit_rows.append(
                {
                    "path": str(path),
                    "score": score,
                    "accepted": False,
                    "error": repr(e),
                }
            )

    pd.DataFrame(audit_rows).to_csv(
        STAGE_ROOT / "01_REPAIRED_ASSIGNMENT_DISCOVERY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Fallback: original wide pair table + changed 24 slots.
    changed_candidates = [
        REBUILD_ROOT / "04B_changed_control_slots_v1.csv",
        REBUILD_ROOT / "04_changed_control_slots.csv",
    ]
    changed_path = first_existing(changed_candidates)

    if changed_path is None:
        raise RuntimeError(
            "No complete repaired 10,112-slot assignment was discovered "
            "and no changed-slot table was found."
        )

    orig = pd.read_parquet(ORIGINAL_PAIR_SETS)
    pcol = first_col(list(orig.columns), PAIR_NAMES)
    poscol = first_col(list(orig.columns), POS_NAMES)

    if pcol is None:
        raise RuntimeError("Original pair-set table lacks pair_set_id")

    # Detect wide controls.
    wide = None
    for a, b in [
        ("control_unit_id_1", "control_unit_id_2"),
        ("control_1_unit_id", "control_2_unit_id"),
        ("control1_unit_id", "control2_unit_id"),
    ]:
        if a in orig.columns and b in orig.columns:
            wide = (a, b)
            break
    if wide is None:
        raise RuntimeError("Cannot resolve original wide control columns")

    rows = []
    for r in orig.itertuples(index=False):
        d = r._asdict()
        pair = norm_pair(d[pcol])
        rows.append(
            {
                "pair_set_id": pair,
                "control_rank": 1,
                "control_unit_id": d[wide[0]],
                "positive_unit_id": d[poscol] if poscol else None,
            }
        )
        rows.append(
            {
                "pair_set_id": pair,
                "control_rank": 2,
                "control_unit_id": d[wide[1]],
                "positive_unit_id": d[poscol] if poscol else None,
            }
        )

    base = pd.DataFrame(rows)

    ch = pd.read_csv(changed_path, low_memory=False)
    cp = first_col(list(ch.columns), PAIR_NAMES)
    cr = first_col(list(ch.columns), RANK_NAMES)

    newc = first_col(
        list(ch.columns),
        [
            "new_control_unit_id",
            "replacement_unit_id",
            "final_control_unit_id",
            "assigned_control_unit_id",
        ],
    )

    if not cp or not cr or not newc:
        raise RuntimeError(
            f"Changed-slot schema not sufficient: {list(ch.columns)}"
        )

    ch["PAIR_KEY"] = ch[cp].map(norm_pair)
    ch["RANK_KEY"] = pd.to_numeric(ch[cr], errors="coerce")

    if len(ch) != N_CHANGED_SLOTS:
        raise RuntimeError(
            f"Changed-slot rows={len(ch)}, expected {N_CHANGED_SLOTS}"
        )

    replace_map = {
        (r.PAIR_KEY, int(r.RANK_KEY)): getattr(r, newc)
        for r in ch.itertuples(index=False)
    }

    base["control_unit_id"] = [
        replace_map.get((p, int(r)), c)
        for p, r, c in zip(
            base["pair_set_id"],
            base["control_rank"],
            base["control_unit_id"],
        )
    ]

    return base, {
        "source_path": str(changed_path),
        "source_mode": "RECONSTRUCTED_FROM_ORIGINAL_PLUS_24_CHANGED_SLOTS",
    }


# =====================================================================
# Step 2 — Canonical original pairs and positive anchors
# =====================================================================

def load_original_pairs_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    orig = pd.read_parquet(ORIGINAL_PAIR_SETS)

    pcol = first_col(list(orig.columns), PAIR_NAMES)
    poscol = first_col(list(orig.columns), POS_NAMES)
    if not pcol or not poscol:
        raise RuntimeError("Cannot resolve original pair / positive columns")

    wide = None
    for a, b in [
        ("control_unit_id_1", "control_unit_id_2"),
        ("control_1_unit_id", "control_2_unit_id"),
        ("control1_unit_id", "control2_unit_id"),
    ]:
        if a in orig.columns and b in orig.columns:
            wide = (a, b)
            break

    if wide is None:
        raise RuntimeError("Cannot resolve original control columns")

    pair = orig[[pcol, poscol, wide[0], wide[1]]].copy()
    pair.columns = [
        "pair_set_id",
        "positive_unit_id",
        "control_unit_id_1",
        "control_unit_id_2",
    ]
    pair["pair_set_id"] = pair["pair_set_id"].map(norm_pair)

    if len(pair) != N_PAIRS or pair["pair_set_id"].nunique() != N_PAIRS:
        raise RuntimeError("Original pair table contract failed")

    long = pd.concat(
        [
            pair[
                ["pair_set_id", "positive_unit_id", "control_unit_id_1"]
            ].rename(columns={"control_unit_id_1": "control_unit_id"})
            .assign(control_rank=1),
            pair[
                ["pair_set_id", "positive_unit_id", "control_unit_id_2"]
            ].rename(columns={"control_unit_id_2": "control_unit_id"})
            .assign(control_rank=2),
        ],
        ignore_index=True,
    )

    long = long[
        ["pair_set_id", "positive_unit_id", "control_rank", "control_unit_id"]
    ]

    return pair, long


# =====================================================================
# Step 3 — Corrected HC00
# =====================================================================

def load_corrected_core() -> tuple[set[str], dict]:
    ep = first_existing(ELIGIBILITY_CANDIDATES)
    cp = first_existing(HC00_CORE_CANDIDATES)

    if ep is None:
        raise RuntimeError("Frozen eligibility table not found")
    if cp is None:
        raise RuntimeError("HC00 core table not found")

    ecols = get_schema(ep)

    wanted = [
        c
        for c in [
            "unit_id",
            "primary_eligibility_status",
            "control_exclusion_flag",
            "inventory_evidence_exclusion_flag",
        ]
        if c in ecols
    ]

    elig = read_table(ep, wanted)

    if "primary_eligibility_status" not in elig.columns:
        raise RuntimeError("Eligibility status field missing")

    status = (
        elig["primary_eligibility_status"]
        .fillna("")
        .astype(str)
        .str.upper()
    )
    eligible = status.str.startswith("ELIGIBLE")

    explicit_exclusion = pd.Series(False, index=elig.index)
    for c in [
        "control_exclusion_flag",
        "inventory_evidence_exclusion_flag",
    ]:
        if c in elig.columns:
            explicit_exclusion |= truthy(elig[c])

    eligible_ids = set(
        elig.loc[eligible & ~explicit_exclusion, "unit_id"].astype(str)
    )

    core = read_table(cp, ["unit_id"])
    core_ids = set(core["unit_id"].astype(str))

    corrected = core_ids & eligible_ids

    if len(corrected) != N_CORE:
        raise RuntimeError(
            f"Corrected HC00 core={len(corrected)}, expected {N_CORE}"
        )

    meta = {
        "eligibility_table": str(ep),
        "core_table": str(cp),
        "eligible_count": int(len(eligible_ids)),
        "current_core_count": int(len(core_ids)),
        "corrected_core_count": int(len(corrected)),
    }
    return corrected, meta


# =====================================================================
# Step 4 — Build repaired pair tables / cohort
# =====================================================================

def build_repaired_pairs(
    repaired_controls: pd.DataFrame,
    original_pair: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    rc = repaired_controls.copy()
    rc["pair_set_id"] = rc["pair_set_id"].map(norm_pair)
    rc["control_rank"] = pd.to_numeric(
        rc["control_rank"], errors="coerce"
    ).astype("Int64")
    rc["control_unit_id"] = rc["control_unit_id"].astype(str)

    rc = rc[["pair_set_id", "control_rank", "control_unit_id"]].copy()

    if len(rc) != N_CONTROLS:
        raise RuntimeError(f"Repaired control rows={len(rc)}")
    if rc[["pair_set_id", "control_rank"]].duplicated().any():
        raise RuntimeError("Duplicate repaired control slot")
    if rc["pair_set_id"].nunique() != N_PAIRS:
        raise RuntimeError("Repaired pair_set count mismatch")
    if rc["control_unit_id"].nunique() != N_CONTROLS:
        raise RuntimeError("Control reuse detected in repaired assignment")

    counts = rc.groupby("pair_set_id")["control_rank"].agg(
        ["count", "nunique"]
    )
    if not ((counts["count"] == 2) & (counts["nunique"] == 2)).all():
        raise RuntimeError("Not every pair has exactly ranks {1,2}")

    pos = original_pair[
        ["pair_set_id", "positive_unit_id"]
    ].drop_duplicates()

    rc = rc.merge(
        pos,
        on="pair_set_id",
        how="left",
        validate="many_to_one",
    )

    if rc["positive_unit_id"].isna().any():
        raise RuntimeError("Repaired control references unknown pair_set_id")

    c1 = (
        rc[rc["control_rank"].eq(1)][
            ["pair_set_id", "control_unit_id"]
        ]
        .rename(columns={"control_unit_id": "control_unit_id_1"})
    )
    c2 = (
        rc[rc["control_rank"].eq(2)][
            ["pair_set_id", "control_unit_id"]
        ]
        .rename(columns={"control_unit_id": "control_unit_id_2"})
    )

    pair = (
        pos.merge(c1, on="pair_set_id", validate="one_to_one")
        .merge(c2, on="pair_set_id", validate="one_to_one")
    )

    pair = pair.sort_values("pair_set_id", kind="mergesort").reset_index(
        drop=True
    )

    # Compare changed slots.
    orig_long = pd.concat(
        [
            original_pair[
                ["pair_set_id", "positive_unit_id", "control_unit_id_1"]
            ]
            .rename(columns={"control_unit_id_1": "original_control_unit_id"})
            .assign(control_rank=1),
            original_pair[
                ["pair_set_id", "positive_unit_id", "control_unit_id_2"]
            ]
            .rename(columns={"control_unit_id_2": "original_control_unit_id"})
            .assign(control_rank=2),
        ],
        ignore_index=True,
    )

    cmp = orig_long.merge(
        rc[
            [
                "pair_set_id",
                "positive_unit_id",
                "control_rank",
                "control_unit_id",
            ]
        ],
        on=["pair_set_id", "positive_unit_id", "control_rank"],
        how="outer",
        validate="one_to_one",
    )

    cmp["changed"] = (
        cmp["original_control_unit_id"].astype(str)
        != cmp["control_unit_id"].astype(str)
    )

    changed = cmp[cmp["changed"]].copy()

    # Long 15,168-row training cohort.
    positive_rows = pair[
        ["pair_set_id", "positive_unit_id"]
    ].rename(columns={"positive_unit_id": "unit_id"})
    positive_rows["sample_role"] = "MATCHED_POSITIVE"
    positive_rows["y_pair"] = 1
    positive_rows["control_rank"] = pd.Series(
        [pd.NA] * len(positive_rows), dtype="Int64"
    )

    control_rows = rc[
        ["pair_set_id", "positive_unit_id", "control_rank", "control_unit_id"]
    ].rename(columns={"control_unit_id": "unit_id"})
    control_rows["sample_role"] = "HARD_CONTROL"
    control_rows["y_pair"] = 0

    cohort = pd.concat(
        [
            positive_rows[
                ["pair_set_id", "unit_id", "sample_role", "y_pair", "control_rank"]
            ],
            control_rows[
                ["pair_set_id", "unit_id", "sample_role", "y_pair", "control_rank"]
            ],
        ],
        ignore_index=True,
    )

    role_order = np.where(
        cohort["sample_role"].eq("MATCHED_POSITIVE"),
        0,
        cohort["control_rank"].astype("Int64").fillna(0).astype(int),
    )
    cohort["pair_position"] = role_order

    cohort = cohort.sort_values(
        ["pair_set_id", "pair_position", "unit_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    cohort["sample_index"] = np.arange(
        len(cohort), dtype=np.int64
    )

    qa = {
        "pair_sets": int(pair["pair_set_id"].nunique()),
        "positives": int(pair["positive_unit_id"].nunique()),
        "hard_controls": int(len(rc)),
        "unique_hard_controls": int(rc["control_unit_id"].nunique()),
        "training_rows": int(len(cohort)),
        "changed_control_slots": int(len(changed)),
        "positive_unit_ids_unchanged": bool(
            pair["positive_unit_id"].astype(str).tolist()
            == original_pair.sort_values(
                "pair_set_id", kind="mergesort"
            )["positive_unit_id"].astype(str).tolist()
        ),
    }

    return pair, rc, cohort, changed, qa


# =====================================================================
# Step 5 — Canonical pair -> spatial group / fold semantics
# =====================================================================

def load_canonical_pair_split(
    repaired_pair: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    cols = get_schema(CANON_PAIR_FOLD)

    if "pair_set_id" not in cols:
        raise RuntimeError("Canonical pair fold table lacks pair_set_id")

    fold_col = choose_fold_col(cols)

    spatial_col = (
        "spatial_group_id"
        if "spatial_group_id" in cols
        else None
    )

    # Sometimes pair crosswalk may not carry spatial group. Fall back to primary.
    use_source = CANON_PAIR_FOLD

    if not spatial_col:
        pcols = get_schema(CANON_PRIMARY_ASSIGN)
        if "pair_set_id" in pcols and "spatial_group_id" in pcols:
            use_source = CANON_PRIMARY_ASSIGN
            cols = pcols
            spatial_col = "spatial_group_id"
            fold_col = choose_fold_col(cols)

    if not spatial_col or not fold_col:
        raise RuntimeError(
            "Could not resolve pair_set_id -> spatial_group_id / fold"
        )

    split = read_table(
        use_source,
        ["pair_set_id", spatial_col, fold_col],
    )

    split["pair_set_id"] = split["pair_set_id"].map(norm_pair)
    split = split.rename(
        columns={
            spatial_col: "spatial_group_id",
            fold_col: "outer_fold",
        }
    )

    split["spatial_group_id"] = split["spatial_group_id"].map(clean_text)
    split["outer_fold"] = split["outer_fold"].astype(str).str.strip()

    split = split[
        ["pair_set_id", "spatial_group_id", "outer_fold"]
    ].drop_duplicates()

    if len(split) != N_PAIRS:
        raise RuntimeError(
            f"Canonical pair split rows={len(split)}, expected {N_PAIRS}"
        )

    if split["spatial_group_id"].isna().any():
        raise RuntimeError("Canonical pair split has missing spatial group")

    if split["pair_set_id"].duplicated().any():
        raise RuntimeError("Canonical pair split duplicate pair_set_id")

    repaired_pairs = set(repaired_pair["pair_set_id"])
    split_pairs = set(split["pair_set_id"])

    if repaired_pairs != split_pairs:
        raise RuntimeError(
            "Repaired pair_set_id universe differs from canonical split"
        )

    group_fold_n = (
        split.groupby("spatial_group_id")["outer_fold"].nunique()
    )
    group_cross_fold = int((group_fold_n > 1).sum())

    fold_counts = (
        split.groupby("outer_fold")["pair_set_id"]
        .nunique()
        .sort_index()
        .to_dict()
    )

    qa = {
        "source": str(use_source),
        "pair_sets": int(len(split)),
        "spatial_groups": int(split["spatial_group_id"].nunique()),
        "group_cross_fold_count": group_cross_fold,
        "fold_pair_counts": {str(k): int(v) for k, v in fold_counts.items()},
    }

    return split, qa


def attach_split_to_cohort(
    cohort: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    out = cohort.merge(
        split,
        on="pair_set_id",
        how="left",
        validate="many_to_one",
    )

    if out[["spatial_group_id", "outer_fold"]].isna().any().any():
        raise RuntimeError("Missing split metadata after cohort merge")

    # pair cannot cross fold by construction, but audit it.
    pair_fold_n = out.groupby("pair_set_id")["outer_fold"].nunique()
    pair_group_n = out.groupby("pair_set_id")["spatial_group_id"].nunique()
    group_fold_n = out.groupby("spatial_group_id")["outer_fold"].nunique()

    qa = {
        "pair_cross_fold_count": int((pair_fold_n > 1).sum()),
        "pair_cross_spatial_group_count": int((pair_group_n > 1).sum()),
        "spatial_group_cross_fold_count": int((group_fold_n > 1).sum()),
        "unique_spatial_groups": int(out["spatial_group_id"].nunique()),
    }

    return out, qa


# =====================================================================
# Step 6 — Corrected core + P0 edge legality
# =====================================================================

def audit_control_legality(
    repaired_controls: pd.DataFrame,
    corrected_core: set[str],
) -> tuple[pd.DataFrame, dict]:

    q = repaired_controls.copy()
    q["control_unit_id"] = q["control_unit_id"].astype(str)
    q["positive_unit_id"] = q["positive_unit_id"].astype(str)

    q["in_corrected_core"] = q["control_unit_id"].isin(corrected_core)

    outside = q[~q["in_corrected_core"]].copy()

    # P0 edge legality.
    import pyarrow.dataset as ds

    dset = ds.dataset(str(P0_EDGES), format="parquet")
    schema = set(dset.schema.names)

    if not {"positive_unit_id", "candidate_unit_id"}.issubset(schema):
        raise RuntimeError("P0 edge graph schema missing required IDs")

    pos_ids = sorted(set(q["positive_unit_id"]))

    tab = dset.to_table(
        columns=["positive_unit_id", "candidate_unit_id"],
        filter=ds.field("positive_unit_id").isin(pos_ids),
    )

    edges = tab.to_pandas()
    edge_keys = set(
        zip(
            edges["positive_unit_id"].astype(str),
            edges["candidate_unit_id"].astype(str),
        )
    )

    q["p0_edge_legal"] = [
        (p, c) in edge_keys
        for p, c in zip(
            q["positive_unit_id"],
            q["control_unit_id"],
        )
    ]

    failures = q[~q["p0_edge_legal"]].copy()

    qa = {
        "controls": int(len(q)),
        "controls_outside_corrected_core": int(len(outside)),
        "p0_edge_failures": int(len(failures)),
        "control_reuse": int(
            len(q) - q["control_unit_id"].nunique()
        ),
    }

    q.to_parquet(
        STAGE_ROOT / "05_CONTROL_LEGALITY_AUDIT.parquet",
        index=False,
    )

    return q, qa


# =====================================================================
# Step 7 — Static 92
# =====================================================================

def resolve_static_fields() -> tuple[list[str], dict]:
    fields = []

    if OLD_FEATURE_WHITELIST.exists():
        w = pd.read_csv(OLD_FEATURE_WHITELIST, low_memory=False)

        if "field_name" in w.columns:
            mask = pd.Series(True, index=w.index)

            if "model_input_group" in w.columns:
                mask &= (
                    w["model_input_group"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .eq("STATIC")
                )

            if "allowed" in w.columns:
                mask &= (
                    w["allowed"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .isin({"YES", "TRUE", "1"})
                )

            fields = (
                w.loc[mask, "field_name"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

    if len(fields) != N_STATIC:
        fp = first_existing(STATIC_FIELDS_FROZEN_CANDIDATES)
        if fp is not None:
            f = pd.read_csv(fp, low_memory=False)
            c = None
            for name in [
                "field_name",
                "final_field_name",
                "feature_name",
            ]:
                if name in f.columns:
                    c = name
                    break
            if c:
                fields = (
                    f[c].dropna().astype(str).drop_duplicates().tolist()
                )

    # Final fallback: old static view minus unit_id is NOT safe if >92.
    # Stop rather than invent.
    if len(fields) != N_STATIC:
        raise RuntimeError(
            f"Resolved static fields={len(fields)}, expected exactly {N_STATIC}. "
            "Do not guess feature whitelist."
        )

    schema = set(get_schema(STATIC_MODEL_VIEW))
    missing = [f for f in fields if f not in schema]
    if missing:
        raise RuntimeError(
            f"Static whitelist fields missing from model view: {missing[:20]}"
        )

    forbidden_hits = [
        f
        for f in fields
        if f.lower() in FORBIDDEN_FEATURE_TOKENS
        or any(
            token in f.lower()
            for token in [
                "pair_set",
                "control_rank",
                "matching_cost",
                "matching_distance",
                "evidence_component",
                "future_rainfall",
            ]
        )
    ]

    if forbidden_hits:
        raise RuntimeError(
            f"Leakage-like fields in static whitelist: {forbidden_hits}"
        )

    return fields, {
        "field_count": len(fields),
        "source_whitelist": str(OLD_FEATURE_WHITELIST),
        "static_model_view": str(STATIC_MODEL_VIEW),
    }


def build_static_raw(
    cohort: pd.DataFrame,
    fields: list[str],
) -> tuple[pd.DataFrame, dict]:

    unit_ids = cohort["unit_id"].astype(str).tolist()
    unique_ids = sorted(set(unit_ids))

    # Use pyarrow dataset filtering to avoid loading unnecessary columns.
    import pyarrow.dataset as ds

    dset = ds.dataset(str(STATIC_MODEL_VIEW), format="parquet")

    tab = dset.to_table(
        columns=["unit_id"] + fields,
        filter=ds.field("unit_id").isin(unique_ids),
    )

    static = tab.to_pandas()
    static["unit_id"] = static["unit_id"].astype(str)

    if static["unit_id"].duplicated().any():
        raise RuntimeError("Static model view has duplicate unit_id")

    raw = cohort[
        [
            "sample_index",
            "pair_set_id",
            "unit_id",
            "sample_role",
            "control_rank",
            "outer_fold",
            "spatial_group_id",
        ]
    ].copy()

    raw["unit_id"] = raw["unit_id"].astype(str)

    raw = raw.merge(
        static,
        on="unit_id",
        how="left",
        validate="many_to_one",
    )

    feature_missing_all = [
        f for f in fields if raw[f].isna().all()
    ]

    missing_unit = raw[
        fields
    ].isna().all(axis=1)

    # We do NOT reject legitimate per-feature missing values because the old
    # model contract already handles static missingness. We only reject units
    # for which all 92 fields are absent, which indicates a failed join.
    qa = {
        "rows": int(len(raw)),
        "static_fields": int(len(fields)),
        "units_with_all_92_static_missing": int(missing_unit.sum()),
        "fields_all_missing": feature_missing_all,
        "unit_id_unique_in_source": int(static["unit_id"].nunique()),
    }

    raw.to_parquet(
        STAGE_ROOT / "06_STATIC_92_RAW_BY_SAMPLE.parquet",
        index=False,
    )

    # Save schema.
    field_meta = []
    for f in fields:
        field_meta.append(
            {
                "field_name": f,
                "dtype": str(raw[f].dtype),
                "missing_count": int(raw[f].isna().sum()),
                "missing_fraction": float(raw[f].isna().mean()),
            }
        )

    pd.DataFrame(field_meta).to_csv(
        STAGE_ROOT / "06A_STATIC_92_FIELD_SCHEMA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return raw, qa


# =====================================================================
# Step 8 — Dynamic 70 x 10
# =====================================================================

def build_dynamic_tensor(
    cohort: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict]:

    import pyarrow.dataset as ds

    dset = ds.dataset(
        str(RAIN_DATASET),
        format="parquet",
        partitioning="hive",
    )

    schema = set(dset.schema.names)

    required = {"unit_id", "timestamp_utc", *DYNAMIC_FIELDS}
    missing = sorted(required - schema)
    if missing:
        raise RuntimeError(
            f"Frozen rainfall dataset missing fields: {missing}"
        )

    unique_units = sorted(
        set(cohort["unit_id"].astype(str))
    )

    # Explicit model window: exactly 70 half-hour observations.
    expected_times = pd.date_range(
        start=MODEL_START_UTC,
        end=MODEL_END_UTC,
        freq="30min",
        tz="UTC",
    )

    if len(expected_times) != N_TIME:
        raise RuntimeError(
            f"Internal expected timestamp count={len(expected_times)}"
        )

    log(
        "Reading frozen rainfall for "
        f"{len(unique_units):,} units, 70 timestamps..."
    )

    filt = (
        ds.field("unit_id").isin(unique_units)
        & (ds.field("timestamp_utc") >= MODEL_START_UTC.to_pydatetime())
        & (ds.field("timestamp_utc") <= MODEL_END_UTC.to_pydatetime())
    )

    tab = dset.to_table(
        columns=["unit_id", "timestamp_utc"] + DYNAMIC_FIELDS,
        filter=filt,
    )

    rain = tab.to_pandas()

    rain["unit_id"] = rain["unit_id"].astype(str)
    rain["timestamp_utc"] = pd.to_datetime(
        rain["timestamp_utc"], utc=True
    )

    rain = rain[
        rain["timestamp_utc"].isin(expected_times)
    ].copy()

    if rain.duplicated(["unit_id", "timestamp_utc"]).any():
        dup = rain[
            rain.duplicated(["unit_id", "timestamp_utc"], keep=False)
        ].head(20)
        raise RuntimeError(
            "Duplicate rainfall unit/timestamp rows:\n"
            + dup.to_string(index=False)
        )

    counts = rain.groupby("unit_id")["timestamp_utc"].nunique()

    missing_units = sorted(
        set(unique_units) - set(counts.index)
    )

    bad_counts = counts[counts.ne(N_TIME)]

    if missing_units or len(bad_counts):
        raise RuntimeError(
            f"Dynamic coverage failure: missing_units={len(missing_units)}, "
            f"bad_70_count_units={len(bad_counts)}"
        )

    # Verify exact timestamp set per unit.
    time_set = set(expected_times)
    bad_time_units = []

    for uid, g in rain.groupby("unit_id", sort=False):
        if set(g["timestamp_utc"]) != time_set:
            bad_time_units.append(uid)
            if len(bad_time_units) >= 20:
                break

    if bad_time_units:
        raise RuntimeError(
            f"Units with incorrect 70-time registry: {bad_time_units}"
        )

    unit_to_sample = dict(
        zip(
            cohort["unit_id"].astype(str),
            cohort["sample_index"].astype(int),
        )
    )

    rain["sample_index"] = rain["unit_id"].map(unit_to_sample)

    if rain["sample_index"].isna().any():
        raise RuntimeError("Dynamic rows failed sample_index mapping")

    time_to_index = {
        t: i for i, t in enumerate(expected_times)
    }
    rain["time_index"] = rain["timestamp_utc"].map(time_to_index)

    rain = rain.sort_values(
        ["sample_index", "time_index"],
        kind="mergesort",
    )

    values = rain[DYNAMIC_FIELDS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    if values.shape[0] != N_ROWS * N_TIME:
        raise RuntimeError(
            f"Dynamic flattened rows={values.shape[0]}, "
            f"expected {N_ROWS*N_TIME}"
        )

    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise RuntimeError(f"Dynamic NaN/Inf count={bad}")

    if (values < 0).any():
        bad = int((values < 0).sum())
        raise RuntimeError(f"Dynamic negative count={bad}")

    tensor = values.reshape(N_ROWS, N_TIME, N_DYNAMIC)

    np.save(
        STAGE_ROOT / "07_DYNAMIC_70x10_FLOAT32.npy",
        tensor,
        allow_pickle=False,
    )

    ts = pd.DataFrame(
        {
            "time_index": np.arange(N_TIME, dtype=np.int16),
            "timestamp_utc": expected_times,
        }
    )
    ts.to_csv(
        STAGE_ROOT / "07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "dynamic_index": np.arange(N_DYNAMIC, dtype=np.int16),
            "field_name": DYNAMIC_FIELDS,
        }
    ).to_csv(
        STAGE_ROOT / "07B_DYNAMIC_10_FIELD_ORDER.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa = {
        "tensor_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(np.isfinite(tensor).all()),
        "negative_count": int((tensor < 0).sum()),
        "start_utc": str(expected_times[0]),
        "end_utc": str(expected_times[-1]),
        "timesteps": int(N_TIME),
        "dynamic_fields": int(N_DYNAMIC),
        "future_timesteps_read": 0,
    }

    return tensor, ts, qa


# =====================================================================
# Step 9 — Pair tensor
# =====================================================================

def build_pair_tensor(
    cohort: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict]:

    rows = []

    for pair_id, g in cohort.groupby("pair_set_id", sort=True):
        pos = g[g["sample_role"].eq("MATCHED_POSITIVE")]
        c1 = g[g["control_rank"].eq(1)]
        c2 = g[g["control_rank"].eq(2)]

        if len(pos) != 1 or len(c1) != 1 or len(c2) != 1:
            raise RuntimeError(
                f"Pair structure failure: {pair_id}"
            )

        rows.append(
            {
                "pair_set_id": pair_id,
                "positive_sample_index": int(pos.iloc[0]["sample_index"]),
                "control1_sample_index": int(c1.iloc[0]["sample_index"]),
                "control2_sample_index": int(c2.iloc[0]["sample_index"]),
                "outer_fold": pos.iloc[0]["outer_fold"],
                "spatial_group_id": pos.iloc[0]["spatial_group_id"],
            }
        )

    reg = pd.DataFrame(rows)

    tensor = reg[
        [
            "positive_sample_index",
            "control1_sample_index",
            "control2_sample_index",
        ]
    ].to_numpy(dtype=np.int64)

    if tensor.shape != (N_PAIRS, 3):
        raise RuntimeError(
            f"Pair tensor shape={tensor.shape}, expected {(N_PAIRS,3)}"
        )

    np.save(
        STAGE_ROOT / "08_PAIR_TENSOR_INDEX_INT64.npy",
        tensor,
        allow_pickle=False,
    )

    reg.to_parquet(
        STAGE_ROOT / "08A_PAIR_TENSOR_REGISTRY.parquet",
        index=False,
    )

    qa = {
        "shape": list(tensor.shape),
        "pair_sets": int(len(reg)),
        "all_sample_indices_in_range": bool(
            tensor.min() >= 0 and tensor.max() < N_ROWS
        ),
    }

    return tensor, reg, qa


# =====================================================================
# Step 10 — Optional full label snapshot for training roles
# =====================================================================

def build_training_label_snapshot(
    cohort: pd.DataFrame,
) -> pd.DataFrame:

    out = cohort[
        [
            "sample_index",
            "pair_set_id",
            "unit_id",
            "sample_role",
            "y_pair",
            "control_rank",
            "pair_position",
            "spatial_group_id",
            "outer_fold",
        ]
    ].copy()

    out["positive_label_flag"] = out["sample_role"].eq(
        "MATCHED_POSITIVE"
    )
    out["hard_control_flag"] = out["sample_role"].eq(
        "HARD_CONTROL"
    )

    positive_map = (
        out[out["sample_role"].eq("MATCHED_POSITIVE")]
        .set_index("pair_set_id")["unit_id"]
        .to_dict()
    )

    out["matched_positive_unit_id"] = out["pair_set_id"].map(
        positive_map
    )

    out.to_parquet(
        STAGE_ROOT / "09_FINAL_TRAINING_COHORT_LABELS.parquet",
        index=False,
    )

    return out


# =====================================================================
# Step 11 — Fold registries
# =====================================================================

def build_fold_registries(
    cohort: pd.DataFrame,
    pair_reg: pd.DataFrame,
) -> dict:

    fold_labels = sorted(
        cohort["outer_fold"].dropna().astype(str).unique()
    )

    fold_map = {f: i for i, f in enumerate(fold_labels)}

    cohort = cohort.copy()
    cohort["outer_fold_id"] = cohort["outer_fold"].map(fold_map).astype(
        np.int8
    )

    cohort.to_parquet(
        STAGE_ROOT / "04_FINAL_SAMPLE_INDEX.parquet",
        index=False,
    )

    fold_rows = []
    for fold in fold_labels:
        g = cohort[cohort["outer_fold"].astype(str).eq(fold)]
        fold_rows.append(
            {
                "outer_fold": fold,
                "outer_fold_id": fold_map[fold],
                "pair_sets": int(g["pair_set_id"].nunique()),
                "samples": int(len(g)),
                "positives": int(
                    g["sample_role"].eq("MATCHED_POSITIVE").sum()
                ),
                "controls": int(
                    g["sample_role"].eq("HARD_CONTROL").sum()
                ),
                "spatial_groups": int(
                    g["spatial_group_id"].nunique()
                ),
            }
        )

    fold_summary = pd.DataFrame(fold_rows)

    fold_summary.to_csv(
        STAGE_ROOT / "10_FOLD_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_cross = (
        cohort.groupby("pair_set_id")["outer_fold"].nunique() > 1
    ).sum()

    group_cross = (
        cohort.groupby("spatial_group_id")["outer_fold"].nunique() > 1
    ).sum()

    return {
        "fold_count": len(fold_labels),
        "fold_labels": fold_labels,
        "pair_cross_fold_count": int(pair_cross),
        "spatial_group_cross_fold_count": int(group_cross),
        "fold_pair_counts": {
            str(r.outer_fold): int(r.pair_sets)
            for r in fold_summary.itertuples(index=False)
        },
    }


# =====================================================================
# Step 12 — Final QA and freeze
# =====================================================================

def final_qa(
    pair_qa: dict,
    split_qa: dict,
    cohort_split_qa: dict,
    legality_qa: dict,
    static_qa: dict,
    dynamic_qa: dict,
    pair_tensor_qa: dict,
    fold_qa: dict,
    core_meta: dict,
) -> dict:

    checks = {
        "pair_sets_5056": pair_qa["pair_sets"] == N_PAIRS,
        "positives_5056": pair_qa["positives"] == N_POS,
        "controls_10112": pair_qa["hard_controls"] == N_CONTROLS,
        "unique_controls_10112": (
            pair_qa["unique_hard_controls"] == N_CONTROLS
        ),
        "training_rows_15168": pair_qa["training_rows"] == N_ROWS,
        "changed_slots_24": (
            pair_qa["changed_control_slots"] == N_CHANGED_SLOTS
        ),
        "positives_unchanged": pair_qa["positive_unit_ids_unchanged"],
        "corrected_core_28072": (
            core_meta["corrected_core_count"] == N_CORE
        ),
        "controls_inside_corrected_core": (
            legality_qa["controls_outside_corrected_core"] == 0
        ),
        "p0_edge_failures_zero": legality_qa["p0_edge_failures"] == 0,
        "control_reuse_zero": legality_qa["control_reuse"] == 0,
        "canonical_groups_500": split_qa["spatial_groups"] == 500,
        "canonical_group_crossfold_zero": (
            split_qa["group_cross_fold_count"] == 0
        ),
        "pair_crossfold_zero": (
            cohort_split_qa["pair_cross_fold_count"] == 0
            and fold_qa["pair_cross_fold_count"] == 0
        ),
        "spatial_group_crossfold_zero": (
            cohort_split_qa["spatial_group_cross_fold_count"] == 0
            and fold_qa["spatial_group_cross_fold_count"] == 0
        ),
        "static_fields_92": static_qa["static_fields"] == N_STATIC,
        "static_join_complete": (
            static_qa["units_with_all_92_static_missing"] == 0
        ),
        "dynamic_shape_15168_70_10": (
            dynamic_qa["tensor_shape"] == [N_ROWS, N_TIME, N_DYNAMIC]
        ),
        "dynamic_finite": dynamic_qa["finite"],
        "dynamic_negative_zero": dynamic_qa["negative_count"] == 0,
        "dynamic_future_read_zero": (
            dynamic_qa["future_timesteps_read"] == 0
        ),
        "pair_tensor_shape_5056_3": (
            pair_tensor_qa["shape"] == [N_PAIRS, 3]
        ),
        "pair_tensor_indices_valid": (
            pair_tensor_qa["all_sample_indices_in_range"]
        ),
        "five_folds": fold_qa["fold_count"] == 5,
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "status": status,
        "source_version": SOURCE_VERSION,
        "checks": checks,
        "failed_checks": [
            k for k, v in checks.items() if not v
        ],
    }


def build_manifest() -> pd.DataFrame:
    rows = []

    for p in sorted(STAGE_ROOT.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"FINALIZE.log"}:
            continue

        rel = p.relative_to(STAGE_ROOT).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "bytes": int(p.stat().st_size),
                "sha256": sha256(p),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(
        STAGE_ROOT / "98_FILE_MANIFEST_SHA256.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest


def promote() -> None:
    if FINAL_ROOT.exists():
        # Do not destroy an existing final dataset silently.
        existing_decision = FINAL_ROOT / "99_FINAL_DATASET_DECISION.json"
        if existing_decision.exists():
            payload = json.loads(
                existing_decision.read_text(encoding="utf-8-sig")
            )
            if payload.get("status") == "PASS":
                raise RuntimeError(
                    "A PASS final repaired dataset already exists at:\n"
                    f"{FINAL_ROOT}\n"
                    "Nothing was overwritten."
                )
        raise RuntimeError(
            f"Final output directory already exists: {FINAL_ROOT}\n"
            "Remove/rename it manually only if you intentionally want to rerun."
        )

    STAGE_ROOT.rename(FINAL_ROOT)


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    if FINAL_ROOT.exists():
        existing = FINAL_ROOT / "DATA_SECTION_CLOSED.flag"
        if existing.exists():
            print("=" * 110)
            print("DATA SECTION ALREADY CLOSED")
            print(FINAL_ROOT)
            print("=" * 110)
            return 0

    safe_remove_staging()

    try:
        log("FINAL repaired dataset V2 closure started.")

        # 0.
        log("Step 0/12: prerequisite audit...")
        pre = prerequisite_audit()
        write_json(STAGE_ROOT / "00_PREREQUISITE_AUDIT.json", pre)

        # 1.
        log("Step 1/12: discovering authoritative repaired control assignment...")
        repaired_controls, repair_source = discover_repaired_assignment()
        write_json(
            STAGE_ROOT / "01A_REPAIRED_ASSIGNMENT_SOURCE.json",
            repair_source,
        )

        # 2.
        log("Step 2/12: loading original pair anchors...")
        original_pair, original_long = load_original_pairs_long()

        # 3.
        log("Step 3/12: reconstructing corrected HC00 core...")
        corrected_core, core_meta = load_corrected_core()
        write_json(STAGE_ROOT / "02_CORRECTED_HC00_CORE.json", core_meta)

        # 4.
        log("Step 4/12: building repaired pair tables and 15,168-row cohort...")
        (
            repaired_pair,
            repaired_long,
            cohort,
            changed,
            pair_qa,
        ) = build_repaired_pairs(repaired_controls, original_pair)

        repaired_pair.to_parquet(
            STAGE_ROOT / "03_REPAIRED_PAIR_SETS.parquet",
            index=False,
        )
        repaired_long.to_parquet(
            STAGE_ROOT / "03A_REPAIRED_CONTROL_ASSIGNMENT_LONG.parquet",
            index=False,
        )
        changed.to_csv(
            STAGE_ROOT / "03B_CHANGED_24_CONTROL_SLOTS.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_json(STAGE_ROOT / "03C_PAIR_REBUILD_QA.json", pair_qa)

        # 5.
        log(
            "Step 5/12: attaching canonical positive-anchored "
            "spatial groups/folds..."
        )
        split, split_qa = load_canonical_pair_split(repaired_pair)
        split.to_parquet(
            STAGE_ROOT / "04_PAIR_SPATIAL_GROUP_FOLD_CROSSWALK.parquet",
            index=False,
        )
        write_json(STAGE_ROOT / "04A_CANONICAL_SPLIT_QA.json", split_qa)

        cohort, cohort_split_qa = attach_split_to_cohort(cohort, split)
        write_json(
            STAGE_ROOT / "04B_COHORT_SPLIT_QA.json",
            cohort_split_qa,
        )

        # 6.
        log("Step 6/12: auditing corrected-core and P0 edge legality...")
        repaired_long = repaired_long.merge(
            original_pair[["pair_set_id", "positive_unit_id"]],
            on="pair_set_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_anchor"),
        )
        if "positive_unit_id_anchor" in repaired_long.columns:
            repaired_long["positive_unit_id"] = repaired_long[
                "positive_unit_id_anchor"
            ]
            repaired_long = repaired_long.drop(
                columns=["positive_unit_id_anchor"]
            )

        legality_table, legality_qa = audit_control_legality(
            repaired_long,
            corrected_core,
        )
        write_json(
            STAGE_ROOT / "05A_CONTROL_LEGALITY_QA.json",
            legality_qa,
        )

        # 7.
        log("Step 7/12: rebuilding frozen static 92-feature sample table...")
        static_fields, static_meta = resolve_static_fields()
        write_json(
            STAGE_ROOT / "06B_STATIC_SOURCE_CONTRACT.json",
            static_meta,
        )
        static_raw, static_qa = build_static_raw(
            cohort,
            static_fields,
        )
        write_json(STAGE_ROOT / "06C_STATIC_92_QA.json", static_qa)

        # 8.
        log("Step 8/12: rebuilding frozen dynamic 70x10 tensor...")
        dynamic_tensor, ts_registry, dynamic_qa = build_dynamic_tensor(
            cohort
        )
        write_json(STAGE_ROOT / "07C_DYNAMIC_70x10_QA.json", dynamic_qa)

        # 9.
        log("Step 9/12: building 5,056 x 3 pair tensor index...")
        pair_tensor, pair_registry, pair_tensor_qa = build_pair_tensor(
            cohort
        )
        write_json(
            STAGE_ROOT / "08B_PAIR_TENSOR_QA.json",
            pair_tensor_qa,
        )

        # 10.
        log("Step 10/12: building final training-role label snapshot...")
        training_labels = build_training_label_snapshot(cohort)

        # 11.
        log("Step 11/12: building fold registries and anti-leakage QA...")
        fold_qa = build_fold_registries(cohort, pair_registry)
        write_json(STAGE_ROOT / "10A_FOLD_QA.json", fold_qa)

        # 12.
        log("Step 12/12: final hard QA...")
        qa = final_qa(
            pair_qa=pair_qa,
            split_qa=split_qa,
            cohort_split_qa=cohort_split_qa,
            legality_qa=legality_qa,
            static_qa=static_qa,
            dynamic_qa=dynamic_qa,
            pair_tensor_qa=pair_tensor_qa,
            fold_qa=fold_qa,
            core_meta=core_meta,
        )

        write_json(STAGE_ROOT / "99_FINAL_DATASET_DECISION.json", qa)

        report = f"""# FINAL REPAIRED DATASET V2

## Status

`{qa["status"]}`

## Source version

`{SOURCE_VERSION}`

## Repaired matching source

```json
{json.dumps(repair_source, ensure_ascii=False, indent=2)}
```

## Pair rebuild

```json
{json.dumps(pair_qa, ensure_ascii=False, indent=2)}
```

## Corrected HC00

```json
{json.dumps(core_meta, ensure_ascii=False, indent=2)}
```

## Spatial split

```json
{json.dumps(split_qa, ensure_ascii=False, indent=2)}
```

## Control legality

```json
{json.dumps(legality_qa, ensure_ascii=False, indent=2)}
```

## Static

```json
{json.dumps(static_qa, ensure_ascii=False, indent=2)}
```

## Dynamic

```json
{json.dumps(dynamic_qa, ensure_ascii=False, indent=2)}
```

## Pair tensor

```json
{json.dumps(pair_tensor_qa, ensure_ascii=False, indent=2)}
```

## Fold QA

```json
{json.dumps(fold_qa, ensure_ascii=False, indent=2)}
```

## Final hard checks

```json
{json.dumps(qa["checks"], ensure_ascii=False, indent=2)}
```

## Semantics

- Formal positive anchors are unchanged.
- Hard controls are repaired under corrected HC00 eligibility.
- Old spatial group/fold is NOT used as a pre-match candidate gate.
- After matching, all three members of each pair inherit the canonical
  positive-anchor pair split.
- Static and rainfall sources remain frozen upstream assets.
- No model was trained during this data finalization.
"""

        (STAGE_ROOT / "00_FINAL_DATASET_REPORT.md").write_text(
            report,
            encoding="utf-8",
        )

        if qa["status"] != "PASS":
            build_manifest()
            print("\n" + "=" * 110)
            print("FINAL DATA QA: FAIL")
            print("FAILED CHECKS:")
            for x in qa["failed_checks"]:
                print(" -", x)
            print("No final dataset was promoted.")
            print("Staging retained at:")
            print(STAGE_ROOT)
            print("=" * 110)
            return 2

        # Freeze marker before manifest.
        (STAGE_ROOT / "DATA_SECTION_CLOSED.flag").write_text(
            "PASS\n"
            f"version={SOURCE_VERSION}\n"
            "pair_sets=5056\n"
            "positives=5056\n"
            "controls=10112\n"
            "training_rows=15168\n"
            "static_fields=92\n"
            "dynamic_shape=15168x70x10\n"
            "pair_cross_fold=0\n"
            "spatial_group_cross_fold=0\n"
            "control_reuse=0\n"
            "p0_edge_failures=0\n"
            "controls_outside_corrected_hc00=0\n",
            encoding="utf-8",
        )

        manifest = build_manifest()

        # Add manifest hash summary after manifest generation.
        write_json(
            STAGE_ROOT / "98A_MANIFEST_SUMMARY.json",
            {
                "file_count_before_summary": int(len(manifest)),
                "manifest_sha256": sha256(
                    STAGE_ROOT / "98_FILE_MANIFEST_SHA256.csv"
                ),
                "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        # Re-write final decision after every core artifact exists.
        write_json(STAGE_ROOT / "99_FINAL_DATASET_DECISION.json", qa)

        log("All hard QA passed. Promoting staging to final repaired dataset V2...")
        promote()

        print("\n" + "=" * 118)
        print("FINAL REPAIRED DATASET V2 COMPLETE")
        print("=" * 118)
        print("FINAL QA: PASS")
        print()
        print("PAIR SETS:              ", pair_qa["pair_sets"])
        print("POSITIVES:               ", pair_qa["positives"])
        print("HARD CONTROLS:           ", pair_qa["hard_controls"])
        print("UNIQUE CONTROLS:         ", pair_qa["unique_hard_controls"])
        print("CHANGED CONTROL SLOTS:   ", pair_qa["changed_control_slots"])
        print("TRAINING ROWS:           ", pair_qa["training_rows"])
        print("CORRECTED HC00 CORE:     ", core_meta["corrected_core_count"])
        print("CONTROL REUSE:           ", legality_qa["control_reuse"])
        print("P0 EDGE FAILURES:        ", legality_qa["p0_edge_failures"])
        print(
            "CONTROLS OUTSIDE CORE:   ",
            legality_qa["controls_outside_corrected_core"],
        )
        print("SPATIAL GROUPS:          ", split_qa["spatial_groups"])
        print(
            "PAIR CROSS-FOLD:         ",
            fold_qa["pair_cross_fold_count"],
        )
        print(
            "GROUP CROSS-FOLD:        ",
            fold_qa["spatial_group_cross_fold_count"],
        )
        print("STATIC FIELDS:           ", static_qa["static_fields"])
        print("DYNAMIC SHAPE:           ", dynamic_qa["tensor_shape"])
        print("PAIR TENSOR SHAPE:       ", pair_tensor_qa["shape"])
        print()
        print("FINAL_DECISION: PASS_DATA_SECTION_CLOSED")
        print()
        print("Authoritative final dataset:")
        print(FINAL_ROOT)
        print()
        print("Main files:")
        for name in [
            "00_FINAL_DATASET_REPORT.md",
            "03_REPAIRED_PAIR_SETS.parquet",
            "03A_REPAIRED_CONTROL_ASSIGNMENT_LONG.parquet",
            "03B_CHANGED_24_CONTROL_SLOTS.csv",
            "04_FINAL_SAMPLE_INDEX.parquet",
            "04_PAIR_SPATIAL_GROUP_FOLD_CROSSWALK.parquet",
            "05_CONTROL_LEGALITY_AUDIT.parquet",
            "06_STATIC_92_RAW_BY_SAMPLE.parquet",
            "07_DYNAMIC_70x10_FLOAT32.npy",
            "07A_DYNAMIC_70_TIMESTAMP_REGISTRY.csv",
            "07B_DYNAMIC_10_FIELD_ORDER.csv",
            "08_PAIR_TENSOR_INDEX_INT64.npy",
            "08A_PAIR_TENSOR_REGISTRY.parquet",
            "09_FINAL_TRAINING_COHORT_LABELS.parquet",
            "10_FOLD_SUMMARY.csv",
            "98_FILE_MANIFEST_SHA256.csv",
            "99_FINAL_DATASET_DECISION.json",
            "DATA_SECTION_CLOSED.flag",
        ]:
            print(FINAL_ROOT / name)

        print("=" * 118)
        return 0

    except Exception as e:
        err = {
            "status": "FAIL_EXCEPTION",
            "error_type": type(e).__name__,
            "error": str(e),
        }
        write_json(STAGE_ROOT / "99_EXCEPTION.json", err)

        print("\n" + "=" * 110)
        print("FINAL DATASET CLOSURE STOPPED SAFELY")
        print("=" * 110)
        print(type(e).__name__ + ":", str(e))
        print()
        print("No old frozen dataset was modified.")
        print("No final PASS directory was promoted.")
        print("Staging/debug output:")
        print(STAGE_ROOT)
        print("=" * 110)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
