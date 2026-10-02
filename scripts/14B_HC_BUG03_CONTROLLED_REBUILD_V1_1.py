# -*- coding: utf-8 -*-
r"""
14B_HC_BUG03_CONTROLLED_REBUILD_V1_1.py
====================================
HC-BUG-03 V1.1: versioned controlled rebuild after HC00 exclusion-propagation bug.

IMPORTANT
---------
This script DOES NOT overwrite the frozen/original formal dataset.

It creates a new repair version under:

<PROJECT_ROOT>\data\06_hard_control\
03_hc_bug03_controlled_rebuild_v1

The rebuild is based on the already-audited HC-BUG chain:

HC-BUG-01
    30 upstream-excluded units incorrectly entered HC00 core;
    23 became formal hard controls.

HC-BUG-02C
    22/23 invalid controls have direct legal unused replacements.

HC-BUG-02D
    the final blocked slot is repairable by a depth-1 augmenting path
    touching two slots/two pair sets.

HC-BUG-02D final QA:
    10,112 matched control slots
    10,112 unique controls
    no reuse
    zero controls outside corrected core
    5,056 pair sets
    exactly two controls per pair
    zero illegal path edges

Formal positive semantics remain unchanged:
    AJG1 == 1 AND GSI == 1

This is a DATA REBUILD, not model training.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Project configuration
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "data"
    / "06_hard_control"
    / "03_hc_bug03_controlled_rebuild_v1_1"
)

AUDIT_OUT = (
    ROOT
    / "experiments"
    / "HC_BUG_03_CONTROLLED_REBUILD_V1_1_1_AUDIT"
)

FORMAL_LABEL = (
    ROOT
    / "data"
    / "06_hard_control"
    / "02_formal_pair_assignment"
    / "04_label_layer"
    / "hard_control_02_full_grid_label_layer.parquet"
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

    ROOT
    / "data"
    / "02_reliable_observation_domain"
    / "10_hard_control_candidate_eligibility_pool"
    / "01_full_grid_registry"
    / "hard_control_eligibility_full_grid_250m_v1.parquet",
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

HC02C_PLAN = (
    ROOT
    / "experiments"
    / "HC_BUG_02C_LOCKED_P0_REPLACEMENT_AUDIT"
    / "09_DRYRUN_23_REPLACEMENT_PLAN.csv"
)

HC02D_PATH = (
    ROOT
    / "experiments"
    / "HC_BUG_02D_AUGMENTING_PATH_AUDIT"
    / "04_AUGMENTING_PATH.csv"
)

HC02D_ASSIGNMENT = (
    ROOT
    / "experiments"
    / "HC_BUG_02D_AUGMENTING_PATH_AUDIT"
    / "05_DRYRUN_REPAIRED_CONTROL_ASSIGNMENT.csv"
)

HC02D_FINAL = (
    ROOT
    / "experiments"
    / "HC_BUG_02D_AUGMENTING_PATH_AUDIT"
    / "07_FINAL_DECISION.json"
)

# Exact state already audited.
EXPECTED_MASTER_ROWS = 139364
EXPECTED_POSITIVES = 5075
EXPECTED_MATCHED_POSITIVES = 5056
EXPECTED_CONTROLS = 10112
EXPECTED_PAIRSETS = 5056

EXPECTED_UPSTREAM_ELIGIBLE = 29084
EXPECTED_CURRENT_CORE = 28102
EXPECTED_REMOVED_CORE = 30
EXPECTED_CORRECTED_CORE = 28072
EXPECTED_CHANGED_SLOTS = 24
EXPECTED_INVALID_CONTROLS = 23

# Rebuild will stop rather than silently overwrite a prior HC-BUG-03 result.
ALLOW_OVERWRITE_HCBUG03_OUTPUT = False


# =====================================================================
# Utilities
# =====================================================================

def log(msg: str):
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (AUDIT_OUT / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )

    line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{msg}"
    )

    print(
        line,
        flush=True,
    )

    with (
        AUDIT_OUT
        / "logs"
        / "rebuild.log"
    ).open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            line + "\n"
        )


def write_json(
    path: Path,
    obj,
):
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(
                8 * 1024 * 1024
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p

    return None


def get_schema(path: Path):
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return list(
            pq.ParquetFile(
                path
            ).schema.names
        )

    if path.suffix.lower() == ".csv":
        return list(
            pd.read_csv(
                path,
                nrows=0,
            ).columns
        )

    return []


def read_table(
    path: Path,
    columns=None,
):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(
            path,
            columns=columns,
        )

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            usecols=columns,
            low_memory=False,
        )

    raise RuntimeError(
        f"Unsupported table: {path}"
    )


def truthy(s: pd.Series):
    if pd.api.types.is_bool_dtype(
        s
    ):
        return (
            s.fillna(False)
            .astype(bool)
        )

    if pd.api.types.is_numeric_dtype(
        s
    ):
        return (
            pd.to_numeric(
                s,
                errors="coerce",
            )
            .fillna(0)
            > 0
        )

    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({
            "YES",
            "TRUE",
            "1",
            "Y",
            "T",
        })
    )


def normalize_pair_id(v):
    if pd.isna(v):
        return None

    if isinstance(
        v,
        (int, np.integer),
    ):
        return str(
            int(v)
        )

    if isinstance(
        v,
        (float, np.floating),
    ):
        if float(v).is_integer():
            return str(
                int(v)
            )

        return str(v)

    s = str(v).strip()

    if re.fullmatch(
        r"-?\d+\.0",
        s,
    ):
        return s[:-2]

    return s


def normalize_rank(v):
    if pd.isna(v):
        return None

    try:
        x = float(v)

        if x.is_integer():
            return int(x)

    except Exception:
        pass

    return str(v).strip()


def make_slot_key(
    pair_key,
    rank,
):
    return (
        f"{pair_key}"
        f"::CONTROL_RANK={rank}"
    )


def ensure_clean_output():
    if OUT.exists():
        existing = list(
            OUT.iterdir()
        )

        if existing:
            if not ALLOW_OVERWRITE_HCBUG03_OUTPUT:
                raise RuntimeError(
                    "HC-BUG-03 output folder already contains files.\n"
                    "The script will not overwrite a prior rebuild.\n"
                    f"Folder: {OUT}\n"
                    "Rename/remove the prior HC-BUG-03 folder only after "
                    "you have archived it, or set "
                    "ALLOW_OVERWRITE_HCBUG03_OUTPUT=True explicitly."
                )

            shutil.rmtree(
                OUT
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# Load authoritative corrected core
# =====================================================================

def load_corrected_core():
    eligibility_path = first_existing(
        ELIGIBILITY_CANDIDATES
    )

    core_path = first_existing(
        HC00_CORE_CANDIDATES
    )

    if eligibility_path is None:
        raise RuntimeError(
            "Frozen upstream eligibility table not found."
        )

    if core_path is None:
        raise RuntimeError(
            "Current HC00 core candidate table not found."
        )

    eligibility_schema = get_schema(
        eligibility_path
    )

    ecols = [
        c
        for c in [
            "unit_id",
            "primary_eligibility_status",
            "control_exclusion_flag",
            "inventory_evidence_exclusion_flag",
            "eligibility_reason_codes",
            "final_control_status",
        ]
        if c in eligibility_schema
    ]

    eligibility = read_table(
        eligibility_path,
        ecols,
    )

    if eligibility[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Upstream eligibility unit_id is not unique."
        )

    if (
        "primary_eligibility_status"
        not in eligibility.columns
    ):
        raise RuntimeError(
            "primary_eligibility_status missing."
        )

    status = (
        eligibility[
            "primary_eligibility_status"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    eligibility[
        "AUTHORITATIVE_ELIGIBLE"
    ] = status.str.startswith(
        "ELIGIBLE"
    )

    explicit_excluded = pd.Series(
        False,
        index=eligibility.index,
    )

    for c in (
        "control_exclusion_flag",
        "inventory_evidence_exclusion_flag",
    ):
        if c in eligibility.columns:
            explicit_excluded |= truthy(
                eligibility[c]
            )

    contradiction = (
        eligibility[
            "AUTHORITATIVE_ELIGIBLE"
        ]
        & explicit_excluded
    )

    if contradiction.any():
        raise RuntimeError(
            "Frozen eligibility contains "
            f"{int(contradiction.sum())} eligible/excluded contradictions."
        )

    core = read_table(
        core_path
    )

    if core[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Current HC00 core unit_id is not unique."
        )

    eligible_ids = set(
        eligibility.loc[
            eligibility[
                "AUTHORITATIVE_ELIGIBLE"
            ],
            "unit_id",
        ].tolist()
    )

    current_ids = set(
        core[
            "unit_id"
        ].tolist()
    )

    corrected_ids = (
        current_ids
        & eligible_ids
    )

    removed_ids = (
        current_ids
        - eligible_ids
    )

    corrected = core[
        core[
            "unit_id"
        ].isin(
            corrected_ids
        )
    ].copy()

    removed = (
        core[
            core[
                "unit_id"
            ].isin(
                removed_ids
            )
        ]
        .merge(
            eligibility,
            on="unit_id",
            how="left",
            suffixes=(
                "__hc00",
                "__upstream",
            ),
        )
    )

    state = {
        "upstream_eligible": len(
            eligible_ids
        ),
        "current_core": len(
            current_ids
        ),
        "removed_core": len(
            removed_ids
        ),
        "corrected_core": len(
            corrected_ids
        ),
    }

    expected = {
        "upstream_eligible": (
            EXPECTED_UPSTREAM_ELIGIBLE
        ),
        "current_core": (
            EXPECTED_CURRENT_CORE
        ),
        "removed_core": (
            EXPECTED_REMOVED_CORE
        ),
        "corrected_core": (
            EXPECTED_CORRECTED_CORE
        ),
    }

    mismatch = {
        k: {
            "observed": state[k],
            "expected": expected[k],
        }
        for k
        in expected
        if state[k]
        != expected[k]
    }

    if mismatch:
        raise RuntimeError(
            "Corrected-core state no longer matches the audited state:\n"
            + json.dumps(
                mismatch,
                ensure_ascii=False,
                indent=2,
            )
        )

    return {
        "eligibility_path": (
            eligibility_path
        ),
        "core_path": (
            core_path
        ),
        "eligibility": (
            eligibility
        ),
        "current_core": (
            core
        ),
        "corrected_core": (
            corrected
        ),
        "corrected_ids": (
            corrected_ids
        ),
        "removed": (
            removed
        ),
        "removed_ids": (
            removed_ids
        ),
        "state": (
            state
        ),
    }


# =====================================================================
# Load original formal label layer and reconstruct control slots
# =====================================================================

def load_formal():
    if not FORMAL_LABEL.exists():
        raise RuntimeError(
            f"Missing formal label layer:\n{FORMAL_LABEL}"
        )

    formal = pd.read_parquet(
        FORMAL_LABEL
    )

    if len(
        formal
    ) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MASTER_ROWS} full-grid rows, got {len(formal)}."
        )

    if formal[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Formal full-grid label layer unit_id is not unique."
        )

    if (
        "positive_label_flag"
        in formal.columns
    ):
        is_positive = truthy(
            formal[
                "positive_label_flag"
            ]
        )

    elif "y_main" in formal.columns:
        is_positive = (
            pd.to_numeric(
                formal[
                    "y_main"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0
        )

    else:
        raise RuntimeError(
            "Cannot identify formal positive rows."
        )

    if (
        "hard_control_flag"
        in formal.columns
    ):
        is_hard = truthy(
            formal[
                "hard_control_flag"
            ]
        )

    elif (
        "sample_role"
        in formal.columns
    ):
        is_hard = (
            formal[
                "sample_role"
            ]
            .fillna("")
            .astype(str)
            .eq(
                "HARD_CONTROL"
            )
        )

    else:
        raise RuntimeError(
            "Cannot identify formal hard-control rows."
        )

    formal[
        "__IS_POSITIVE"
    ] = is_positive

    formal[
        "__IS_HARD"
    ] = is_hard

    positive_count = int(
        is_positive.sum()
    )

    control_count = int(
        is_hard.sum()
    )

    if positive_count != EXPECTED_POSITIVES:
        raise RuntimeError(
            f"Expected {EXPECTED_POSITIVES} positives, got {positive_count}."
        )

    if control_count != EXPECTED_CONTROLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROLS} hard controls, got {control_count}."
        )

    hard = formal[
        formal[
            "__IS_HARD"
        ]
    ].copy()

    required = {
        "pair_set_id",
        "matched_positive_unit_id",
        "control_rank",
    }

    if not required.issubset(
        set(
            hard.columns
        )
    ):
        raise RuntimeError(
            "Hard controls missing fields: "
            f"{required - set(hard.columns)}"
        )

    hard[
        "__PAIR_KEY"
    ] = hard[
        "pair_set_id"
    ].map(
        normalize_pair_id
    )

    hard[
        "__RANK_KEY"
    ] = hard[
        "control_rank"
    ].map(
        normalize_rank
    )

    hard[
        "__SLOT_KEY"
    ] = [
        make_slot_key(
            p,
            r,
        )
        for p, r
        in zip(
            hard[
                "__PAIR_KEY"
            ],
            hard[
                "__RANK_KEY"
            ],
        )
    ]

    if hard[
        "__SLOT_KEY"
    ].duplicated().any():
        raise RuntimeError(
            "Original control slot key is not unique."
        )

    pair_counts = (
        hard.groupby(
            "__PAIR_KEY"
        )
        .size()
    )

    if len(
        pair_counts
    ) != EXPECTED_PAIRSETS:
        raise RuntimeError(
            "Original pair-set count mismatch."
        )

    if (
        pair_counts != 2
    ).any():
        raise RuntimeError(
            "Original formal pairing is not exactly 1:2."
        )

    matched_positive_ids = set(
        str(x)
        for x
        in hard[
            "matched_positive_unit_id"
        ].dropna()
    )

    if len(
        matched_positive_ids
    ) != EXPECTED_MATCHED_POSITIVES:
        raise RuntimeError(
            "Expected 5,056 matched positives, got "
            f"{len(matched_positive_ids)}."
        )

    return (
        formal,
        hard,
    )


# =====================================================================
# Read HC-BUG-02D final dry-run assignment
# =====================================================================

def load_02d_assignment():
    if not HC02D_FINAL.exists():
        raise RuntimeError(
            f"Missing HC-BUG-02D final decision:\n{HC02D_FINAL}"
        )

    final = json.loads(
        HC02D_FINAL.read_text(
            encoding="utf-8"
        )
    )

    decision = final.get(
        "FINAL_DECISION",
        "",
    )

    if not decision.startswith(
        "PASS_SHORT_AUGMENTING_PATH_FOUND"
    ) and not decision.startswith(
        "PASS_LONG_AUGMENTING_PATH_FOUND"
    ):
        raise RuntimeError(
            "HC-BUG-02D is not a PASS state:\n"
            f"{decision}"
        )

    if not HC02D_ASSIGNMENT.exists():
        raise RuntimeError(
            f"Missing HC-BUG-02D repaired assignment:\n{HC02D_ASSIGNMENT}"
        )

    assignment = pd.read_csv(
        HC02D_ASSIGNMENT,
        low_memory=False,
    )

    required = {
        "slot_key",
        "pair_set_id",
        "control_rank",
        "positive_unit_id",
        "original_control_unit_id",
        "dryrun_control_unit_id",
        "changed_from_original",
    }

    if not required.issubset(
        set(
            assignment.columns
        )
    ):
        raise RuntimeError(
            "02D repaired assignment missing: "
            f"{required - set(assignment.columns)}"
        )

    if len(
        assignment
    ) != EXPECTED_CONTROLS:
        raise RuntimeError(
            "02D repaired assignment should have 10,112 control slots."
        )

    if assignment[
        "slot_key"
    ].duplicated().any():
        raise RuntimeError(
            "02D repaired assignment slot_key is not unique."
        )

    if assignment[
        "dryrun_control_unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "02D repaired assignment contains control reuse."
        )

    changed = truthy(
        assignment[
            "changed_from_original"
        ]
    )

    if int(
        changed.sum()
    ) != EXPECTED_CHANGED_SLOTS:
        raise RuntimeError(
            "Expected 24 changed slots from HC-BUG-02D, got "
            f"{int(changed.sum())}."
        )

    assignment[
        "__PAIR_KEY"
    ] = assignment[
        "pair_set_id"
    ].map(
        normalize_pair_id
    )

    assignment[
        "__RANK_KEY"
    ] = assignment[
        "control_rank"
    ].map(
        normalize_rank
    )

    assignment[
        "__SLOT_KEY"
    ] = [
        make_slot_key(
            p,
            r,
        )
        for p, r
        in zip(
            assignment[
                "__PAIR_KEY"
            ],
            assignment[
                "__RANK_KEY"
            ],
        )
    ]

    if not (
        assignment[
            "__SLOT_KEY"
        ]
        == assignment[
            "slot_key"
        ]
    ).all():
        raise RuntimeError(
            "02D slot_key does not match pair/rank metadata."
        )

    return (
        assignment,
        final,
    )


# =====================================================================
# Validate new control assignments against corrected core + P0 edges
# =====================================================================

def validate_assignment_graph(
    assignment,
    corrected_ids,
):
    if not P0_EDGES.exists():
        raise RuntimeError(
            f"Locked P0 candidate graph missing:\n{P0_EDGES}"
        )

    new_controls = set(
        assignment[
            "dryrun_control_unit_id"
        ].tolist()
    )

    outside = (
        new_controls
        - corrected_ids
    )

    if outside:
        raise RuntimeError(
            f"{len(outside)} repaired controls are outside corrected core."
        )

    # Check only the 10,112 final edges against P0 candidate universe.
    formal_keys = (
        assignment[
            [
                "positive_unit_id",
                "dryrun_control_unit_id",
            ]
        ]
        .rename(
            columns={
                "dryrun_control_unit_id": (
                    "candidate_unit_id"
                ),
            }
        )
        .copy()
    )

    formal_keys[
        "positive_unit_id"
    ] = formal_keys[
        "positive_unit_id"
    ].astype(str)

    import pyarrow.dataset as ds

    dataset = ds.dataset(
        str(
            P0_EDGES
        ),
        format="parquet",
    )

    positive_ids = sorted(
        formal_keys[
            "positive_unit_id"
        ]
        .unique()
        .tolist()
    )

    # Query only current matched positives.
    table = dataset.to_table(
        columns=[
            "positive_unit_id",
            "candidate_unit_id",
        ],
        filter=(
            ds.field(
                "positive_unit_id"
            ).isin(
                positive_ids
            )
        ),
    )

    p0 = (
        table
        .to_pandas()
        .drop_duplicates(
            [
                "positive_unit_id",
                "candidate_unit_id",
            ]
        )
    )

    p0[
        "positive_unit_id"
    ] = p0[
        "positive_unit_id"
    ].astype(str)

    q = formal_keys.merge(
        p0.assign(
            __IN_P0=1
        ),
        on=[
            "positive_unit_id",
            "candidate_unit_id",
        ],
        how="left",
    )

    missing = q[
        q[
            "__IN_P0"
        ].isna()
    ].copy()

    if len(
        missing
    ):
        raise RuntimeError(
            f"{len(missing)} rebuilt pair edges are absent from locked P0 graph."
        )

    return {
        "final_edges": int(
            len(
                formal_keys
            )
        ),
        "P0_edge_coverage": 1.0,
        "controls_outside_corrected_core": 0,
    }


# =====================================================================
# Build repaired pair edges / pair table
# =====================================================================

def build_pair_assets(
    assignment,
):
    pair_edges = assignment[
        [
            "pair_set_id",
            "positive_unit_id",
            "control_rank",
            "original_control_unit_id",
            "dryrun_control_unit_id",
            "changed_from_original",
        ]
    ].copy()

    pair_edges = pair_edges.rename(
        columns={
            "dryrun_control_unit_id": (
                "control_unit_id"
            ),
            "changed_from_original": (
                "hc_bug03_changed_flag"
            ),
        }
    )

    pair_edges[
        "hc_bug03_repair_version"
    ] = (
        "HC_BUG_03_CONTROLLED_REBUILD_V1_1"
    )

    pair_edges[
        "hc_bug03_repair_reason"
    ] = np.where(
        truthy(
            pair_edges[
                "hc_bug03_changed_flag"
            ]
        ),
        "HC00_UPSTREAM_EXCLUSION_PROPAGATION_REPAIR",
        "UNCHANGED_FROM_ORIGINAL_FORMAL_ASSIGNMENT",
    )

    # Stable order.
    pair_edges[
        "__PAIR_KEY"
    ] = pair_edges[
        "pair_set_id"
    ].map(
        normalize_pair_id
    )

    pair_edges[
        "__RANK_KEY"
    ] = pair_edges[
        "control_rank"
    ].map(
        normalize_rank
    )

    pair_edges = (
        pair_edges.sort_values(
            [
                "__PAIR_KEY",
                "__RANK_KEY",
            ]
        )
        .drop(
            columns=[
                "__PAIR_KEY",
                "__RANK_KEY",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pair_counts = (
        pair_edges.groupby(
            "pair_set_id",
            dropna=False,
        )
        .size()
    )

    if (
        pair_counts != 2
    ).any():
        raise RuntimeError(
            "Repaired pair edges do not have exactly two controls per pair."
        )

    if pair_edges[
        "control_unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Repaired pair edges reuse a control."
        )

    # One-row-per-pair table.
    p1 = pair_edges[
        pair_edges[
            "control_rank"
        ].map(
            normalize_rank
        )
        == 1
    ][
        [
            "pair_set_id",
            "positive_unit_id",
            "control_unit_id",
        ]
    ].rename(
        columns={
            "control_unit_id": (
                "control_1_unit_id"
            )
        }
    )

    p2 = pair_edges[
        pair_edges[
            "control_rank"
        ].map(
            normalize_rank
        )
        == 2
    ][
        [
            "pair_set_id",
            "positive_unit_id",
            "control_unit_id",
        ]
    ].rename(
        columns={
            "control_unit_id": (
                "control_2_unit_id"
            )
        }
    )

    pair_table = p1.merge(
        p2,
        on=[
            "pair_set_id",
            "positive_unit_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        pair_table
    ) != EXPECTED_PAIRSETS:
        raise RuntimeError(
            "Repaired pair table does not contain 5,056 pair sets."
        )

    if (
        pair_table[
            "control_1_unit_id"
        ]
        == pair_table[
            "control_2_unit_id"
        ]
    ).any():
        raise RuntimeError(
            "A repaired pair contains the same control twice."
        )

    return (
        pair_edges,
        pair_table,
    )


# =====================================================================
# Rebuild full-grid label layer
# =====================================================================

PAIR_ASSIGNMENT_FIELDS = [
    "sample_role",
    "y_pair",
    "hard_control_flag",
    "matching_eligible",
    "pair_set_id",
    "matched_positive_unit_id",
    "control_rank",
    "matching_protocol_id",
    "training_pair_candidate_flag",
]


def find_excluded_unlabeled_template(
    formal,
    eligibility,
):
    """
    Identify existing non-positive, non-hard-control rows that upstream marks
    as evidence-excluded. These are the closest semantic template for the 23
    invalid controls after removal from HARD_CONTROL role.

    We do NOT copy unit-specific evidence fields. We only infer the modal
    values of pair-role/matching fields.
    """

    elig_excluded_ids = set(
        eligibility.loc[
            ~eligibility[
                "AUTHORITATIVE_ELIGIBLE"
            ],
            "unit_id",
        ].tolist()
    )

    q = formal[
        formal[
            "unit_id"
        ].isin(
            elig_excluded_ids
        )
        & ~formal[
            "__IS_POSITIVE"
        ]
        & ~formal[
            "__IS_HARD"
        ]
    ].copy()

    if q.empty:
        raise RuntimeError(
            "No existing upstream-excluded unlabeled rows found to infer "
            "pair-role reset semantics."
        )

    template = {}

    audit_rows = []

    for c in PAIR_ASSIGNMENT_FIELDS:
        if c not in formal.columns:
            continue

        s = q[c]

        # Prefer actual modal value, including NA when overwhelmingly modal.
        counts = (
            s.astype("object")
            .where(
                ~s.isna(),
                other="<NA>",
            )
            .value_counts(
                dropna=False
            )
        )

        if counts.empty:
            continue

        top_value = counts.index[
            0
        ]

        top_count = int(
            counts.iloc[
                0
            ]
        )

        frac = (
            top_count
            / len(
                q
            )
        )

        if top_value == "<NA>":
            value = np.nan

        else:
            value = top_value

        template[
            c
        ] = value

        audit_rows.append({
            "field": c,
            "modal_value": str(
                top_value
            ),
            "modal_count": (
                top_count
            ),
            "template_rows": int(
                len(
                    q
                )
            ),
            "modal_fraction": float(
                frac
            ),
        })

    audit = pd.DataFrame(
        audit_rows
    )

    return (
        template,
        audit,
        q,
    )


def rebuild_label_layer(
    formal,
    hard_original,
    assignment,
    eligibility,
):
    repaired = formal.drop(
        columns=[
            "__IS_POSITIVE",
            "__IS_HARD",
        ],
        errors="ignore",
    ).copy()

    # Original control slot metadata.
    slot_rows = hard_original[
        [
            c
            for c in hard_original.columns
            if not c.startswith(
                "__"
            )
        ]
    ].copy()

    slot_rows[
        "__PAIR_KEY"
    ] = hard_original[
        "__PAIR_KEY"
    ].values

    slot_rows[
        "__RANK_KEY"
    ] = hard_original[
        "__RANK_KEY"
    ].values

    slot_rows[
        "__SLOT_KEY"
    ] = hard_original[
        "__SLOT_KEY"
    ].values

    slot_template = {
        r[
            "__SLOT_KEY"
        ]: r
        for _, r
        in slot_rows.iterrows()
    }

    # Units that were formal controls but are no longer used.
    original_control_ids = set(
        hard_original[
            "unit_id"
        ].tolist()
    )

    repaired_control_ids = set(
        assignment[
            "dryrun_control_unit_id"
        ].tolist()
    )

    removed_control_ids = (
        original_control_ids
        - repaired_control_ids
    )

    new_control_ids = (
        repaired_control_ids
        - original_control_ids
    )

    if len(
        removed_control_ids
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Expected exactly 23 old controls to leave the formal cohort, "
            f"got {len(removed_control_ids)}."
        )

    if len(
        new_control_ids
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Expected exactly 23 new controls to enter the formal cohort, "
            f"got {len(new_control_ids)}."
        )

    # Infer how an upstream-excluded non-positive row is represented when it
    # is NOT a hard control.
    template, template_audit, template_rows = (
        find_excluded_unlabeled_template(
            formal,
            eligibility,
        )
    )

    # Reset the 23 invalid old controls to existing excluded-unlabeled
    # pair-role semantics.
    reset_mask = repaired[
        "unit_id"
    ].isin(
        removed_control_ids
    )

    for c, value in template.items():
        if c not in repaired.columns:
            continue

        if pd.isna(
            value
        ):
            repaired.loc[
                reset_mask,
                c,
            ] = np.nan

        else:
            repaired.loc[
                reset_mask,
                c,
            ] = value

    # Safety overrides for role semantics regardless of modal representation.
    if (
        "hard_control_flag"
        in repaired.columns
    ):
        repaired.loc[
            reset_mask,
            "hard_control_flag",
        ] = 0

    if (
        "training_pair_candidate_flag"
        in repaired.columns
    ):
        repaired.loc[
            reset_mask,
            "training_pair_candidate_flag",
        ] = 0

    if (
        "matching_eligible"
        in repaired.columns
    ):
        repaired.loc[
            reset_mask,
            "matching_eligible",
        ] = 0

    for c in [
        "pair_set_id",
        "matched_positive_unit_id",
        "control_rank",
    ]:
        if c in repaired.columns:
            repaired.loc[
                reset_mask,
                c,
            ] = np.nan

    # Assign every repaired control unit to the slot it now occupies.
    # Copy ONLY slot-level pair/matching fields from the original slot.
    pair_fields_present = [
        c
        for c in PAIR_ASSIGNMENT_FIELDS
        if c in repaired.columns
    ]

    unit_to_index = pd.Series(
        repaired.index,
        index=repaired[
            "unit_id"
        ],
    )

    update_rows = []

    for _, a in assignment.iterrows():
        slot = a[
            "slot_key"
        ]

        new_unit = a[
            "dryrun_control_unit_id"
        ]

        if new_unit not in unit_to_index.index:
            raise RuntimeError(
                f"Replacement control missing from full-grid label layer: {new_unit}"
            )

        if slot not in slot_template:
            raise RuntimeError(
                f"Slot missing from original formal controls: {slot}"
            )

        template_slot = slot_template[
            slot
        ]

        idx = unit_to_index[
            new_unit
        ]

        for c in pair_fields_present:
            repaired.at[
                idx,
                c,
            ] = template_slot[
                c
            ]

        # Explicit safety role.
        if (
            "sample_role"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "sample_role",
            ] = "HARD_CONTROL"

        if (
            "hard_control_flag"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "hard_control_flag",
            ] = 1

        if (
            "matching_eligible"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "matching_eligible",
            ] = 1

        if (
            "training_pair_candidate_flag"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "training_pair_candidate_flag",
            ] = 1

        if (
            "pair_set_id"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "pair_set_id",
            ] = template_slot[
                "pair_set_id"
            ]

        if (
            "matched_positive_unit_id"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "matched_positive_unit_id",
            ] = template_slot[
                "matched_positive_unit_id"
            ]

        if (
            "control_rank"
            in repaired.columns
        ):
            repaired.at[
                idx,
                "control_rank",
            ] = template_slot[
                "control_rank"
            ]

        # Do NOT alter y_main / positive_label_flag on replacement units.
        # They are not formal positives.
        update_rows.append({
            "slot_key": slot,
            "unit_id": new_unit,
            "pair_set_id": template_slot[
                "pair_set_id"
            ],
            "matched_positive_unit_id": template_slot[
                "matched_positive_unit_id"
            ],
            "control_rank": template_slot[
                "control_rank"
            ],
            "changed_from_original": bool(
                a[
                    "changed_from_original"
                ]
            ),
        })

    control_update_audit = pd.DataFrame(
        update_rows
    )

    return {
        "repaired": (
            repaired
        ),
        "removed_control_ids": (
            removed_control_ids
        ),
        "new_control_ids": (
            new_control_ids
        ),
        "template": (
            template
        ),
        "template_audit": (
            template_audit
        ),
        "control_update_audit": (
            control_update_audit
        ),
    }


# =====================================================================
# Final dataset QA
# =====================================================================

def final_qa(
    original_formal,
    repaired,
    pair_edges,
    pair_table,
    corrected_ids,
    removed_control_ids,
    new_control_ids,
):
    if len(
        repaired
    ) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            "Repaired label layer row count changed."
        )

    if repaired[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Repaired label layer has duplicate unit_id."
        )

    if (
        "positive_label_flag"
        in repaired.columns
    ):
        repaired_positive = truthy(
            repaired[
                "positive_label_flag"
            ]
        )

        original_positive = truthy(
            original_formal[
                "positive_label_flag"
            ]
        )

    else:
        repaired_positive = (
            pd.to_numeric(
                repaired[
                    "y_main"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0
        )

        original_positive = (
            pd.to_numeric(
                original_formal[
                    "y_main"
                ],
                errors="coerce",
            )
            .fillna(0)
            > 0
        )

    positive_ids_before = set(
        original_formal.loc[
            original_positive,
            "unit_id",
        ].tolist()
    )

    positive_ids_after = set(
        repaired.loc[
            repaired_positive,
            "unit_id",
        ].tolist()
    )

    positive_semantics_unchanged = (
        positive_ids_before
        == positive_ids_after
    )

    if not positive_semantics_unchanged:
        raise RuntimeError(
            "Positive unit_id set changed during HC-BUG-03 rebuild."
        )

    if (
        "hard_control_flag"
        in repaired.columns
    ):
        repaired_hard = truthy(
            repaired[
                "hard_control_flag"
            ]
        )

    else:
        repaired_hard = (
            repaired[
                "sample_role"
            ]
            .fillna("")
            .astype(str)
            .eq(
                "HARD_CONTROL"
            )
        )

    control_ids = set(
        repaired.loc[
            repaired_hard,
            "unit_id",
        ].tolist()
    )

    control_outside_core = (
        control_ids
        - corrected_ids
    )

    removed_still_controls = (
        control_ids
        & removed_control_ids
    )

    pair_counts = (
        pair_edges.groupby(
            "pair_set_id"
        )
        .size()
    )

    qa = {
        "status": "PASS",

        "master_rows": int(
            len(
                repaired
            )
        ),

        "positive_count": int(
            repaired_positive.sum()
        ),

        "positive_semantics_unchanged": bool(
            positive_semantics_unchanged
        ),

        "hard_control_count": int(
            repaired_hard.sum()
        ),

        "unique_hard_controls": len(
            control_ids
        ),

        "pair_set_count": int(
            len(
                pair_table
            )
        ),

        "pair_edges": int(
            len(
                pair_edges
            )
        ),

        "pair_sets_not_exactly_two_controls": int(
            (
                pair_counts != 2
            ).sum()
        ),

        "pair_edge_control_reuse_count": int(
            pair_edges[
                "control_unit_id"
            ]
            .duplicated()
            .sum()
        ),

        "controls_outside_corrected_core": int(
            len(
                control_outside_core
            )
        ),

        "removed_illegal_controls_still_used": int(
            len(
                removed_still_controls
            )
        ),

        "old_controls_removed": int(
            len(
                removed_control_ids
            )
        ),

        "new_controls_entered": int(
            len(
                new_control_ids
            )
        ),

        "changed_control_slots": int(
            truthy(
                pair_edges[
                    "hc_bug03_changed_flag"
                ]
            )
            .sum()
        ),

        "expected": {
            "master_rows": (
                EXPECTED_MASTER_ROWS
            ),
            "positive_count": (
                EXPECTED_POSITIVES
            ),
            "hard_control_count": (
                EXPECTED_CONTROLS
            ),
            "unique_hard_controls": (
                EXPECTED_CONTROLS
            ),
            "pair_set_count": (
                EXPECTED_PAIRSETS
            ),
            "pair_edges": (
                EXPECTED_CONTROLS
            ),
            "changed_control_slots": (
                EXPECTED_CHANGED_SLOTS
            ),
        },
    }

    hard_fail = (
        qa[
            "master_rows"
        ]
        != EXPECTED_MASTER_ROWS
        or qa[
            "positive_count"
        ]
        != EXPECTED_POSITIVES
        or not qa[
            "positive_semantics_unchanged"
        ]
        or qa[
            "hard_control_count"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "unique_hard_controls"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "pair_set_count"
        ]
        != EXPECTED_PAIRSETS
        or qa[
            "pair_edges"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "pair_sets_not_exactly_two_controls"
        ]
        != 0
        or qa[
            "pair_edge_control_reuse_count"
        ]
        != 0
        or qa[
            "controls_outside_corrected_core"
        ]
        != 0
        or qa[
            "removed_illegal_controls_still_used"
        ]
        != 0
        or qa[
            "old_controls_removed"
        ]
        != EXPECTED_INVALID_CONTROLS
        or qa[
            "new_controls_entered"
        ]
        != EXPECTED_INVALID_CONTROLS
        or qa[
            "changed_control_slots"
        ]
        != EXPECTED_CHANGED_SLOTS
    )

    if hard_fail:
        qa[
            "status"
        ] = "FAIL"

    return qa


# =====================================================================
# Main
# =====================================================================

def main():
    ensure_clean_output()

    log(
        "HC-BUG-03 controlled rebuild started"
    )

    # -------------------------------------------------------------
    # 0. Source manifest before any rebuild.
    # -------------------------------------------------------------
    required_files = [
        FORMAL_LABEL,
        P0_EDGES,
        HC02C_PLAN,
        HC02D_PATH,
        HC02D_ASSIGNMENT,
        HC02D_FINAL,
    ]

    missing = [
        str(
            p
        )
        for p
        in required_files
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required audited inputs:\n"
            + "\n".join(
                missing
            )
        )

    core_info = load_corrected_core()

    manifest_paths = {
        "formal_label_original": (
            FORMAL_LABEL
        ),
        "P0_candidate_edges": (
            P0_EDGES
        ),
        "upstream_eligibility": (
            core_info[
                "eligibility_path"
            ]
        ),
        "HC00_core_original": (
            core_info[
                "core_path"
            ]
        ),
        "HC02C_plan": (
            HC02C_PLAN
        ),
        "HC02D_augmenting_path": (
            HC02D_PATH
        ),
        "HC02D_assignment": (
            HC02D_ASSIGNMENT
        ),
        "HC02D_final": (
            HC02D_FINAL
        ),
    }

    manifest = {}

    for name, path in manifest_paths.items():
        manifest[
            name
        ] = {
            "path": str(
                path
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
            "sha256": sha256_file(
                path
            ),
        }

    write_json(
        OUT
        / "00_SOURCE_MANIFEST.json",
        manifest,
    )

    # -------------------------------------------------------------
    # 1. Freeze corrected HC00 core v1.
    # -------------------------------------------------------------
    corrected_core = core_info[
        "corrected_core"
    ].copy()

    corrected_core[
        "hc_bug03_repair_version"
    ] = (
        "HC_BUG_03_CONTROLLED_REBUILD_V1_1"
    )

    corrected_core[
        "hc_bug03_upstream_eligibility_gate"
    ] = True

    corrected_core.to_parquet(
        OUT
        / "01_hard_control_00_core_candidate_pool_corrected_v1.parquet",
        index=False,
    )

    corrected_core.to_csv(
        OUT
        / "01_hard_control_00_core_candidate_pool_corrected_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    core_info[
        "removed"
    ].to_csv(
        OUT
        / "01B_removed_illegal_HC00_core_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        OUT
        / "01C_corrected_core_state.json",
        core_info[
            "state"
        ],
    )

    # -------------------------------------------------------------
    # 2. Load original formal + audited repaired slot assignment.
    # -------------------------------------------------------------
    formal, hard_original = (
        load_formal()
    )

    assignment, hc02d_final = (
        load_02d_assignment()
    )

    # Ensure slot universe is exactly the original slot universe.
    original_slots = set(
        hard_original[
            "__SLOT_KEY"
        ]
    )

    repaired_slots = set(
        assignment[
            "slot_key"
        ]
    )

    if original_slots != repaired_slots:
        raise RuntimeError(
            "HC-BUG-02D repaired assignment does not cover the exact "
            "original 10,112 control slots."
        )

    # -------------------------------------------------------------
    # 3. Validate final repaired assignment against corrected core / P0.
    # -------------------------------------------------------------
    graph_qa = validate_assignment_graph(
        assignment,
        core_info[
            "corrected_ids"
        ],
    )

    write_json(
        OUT
        / "02_repaired_assignment_graph_QA.json",
        graph_qa,
    )

    # -------------------------------------------------------------
    # 4. Build versioned pair edges / pair table.
    # -------------------------------------------------------------
    pair_edges, pair_table = (
        build_pair_assets(
            assignment
        )
    )

    pair_edges.to_csv(
        OUT
        / "03_hard_control_02_pair_edges_repaired_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_edges.to_parquet(
        OUT
        / "03_hard_control_02_pair_edges_repaired_v1.parquet",
        index=False,
    )

    pair_table.to_csv(
        OUT
        / "04_hard_control_02_pair_table_repaired_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_table.to_parquet(
        OUT
        / "04_hard_control_02_pair_table_repaired_v1.parquet",
        index=False,
    )

    changed_slots = pair_edges[
        truthy(
            pair_edges[
                "hc_bug03_changed_flag"
            ]
        )
    ].copy()

    changed_slots.to_csv(
        OUT
        / "04B_changed_control_slots_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # 5. Rebuild full-grid formal label layer.
    # -------------------------------------------------------------
    label_result = (
        rebuild_label_layer(
            formal,
            hard_original,
            assignment,
            core_info[
                "eligibility"
            ],
        )
    )

    repaired_label = (
        label_result[
            "repaired"
        ]
    )

    repaired_label.to_parquet(
        OUT
        / "05_hard_control_02_full_grid_label_layer_repaired_v1.parquet",
        index=False,
    )

    repaired_label.to_csv(
        OUT
        / "05_hard_control_02_full_grid_label_layer_repaired_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    label_result[
        "template_audit"
    ].to_csv(
        OUT
        / "05B_excluded_unlabeled_template_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    label_result[
        "control_update_audit"
    ].to_csv(
        OUT
        / "05C_control_slot_update_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "removed_illegal_control_unit_id": sorted(
                label_result[
                    "removed_control_ids"
                ]
            )
        }
    ).to_csv(
        OUT
        / "05D_removed_formal_controls.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "new_replacement_control_unit_id": sorted(
                label_result[
                    "new_control_ids"
                ]
            )
        }
    ).to_csv(
        OUT
        / "05E_new_formal_controls.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # 6. Final integrated QA.
    # -------------------------------------------------------------
    qa = final_qa(
        original_formal=formal,
        repaired=repaired_label,
        pair_edges=pair_edges,
        pair_table=pair_table,
        corrected_ids=core_info[
            "corrected_ids"
        ],
        removed_control_ids=label_result[
            "removed_control_ids"
        ],
        new_control_ids=label_result[
            "new_control_ids"
        ],
    )

    qa[
        "graph_QA"
    ] = graph_qa

    qa[
        "HC_BUG_02D_decision"
    ] = hc02d_final.get(
        "FINAL_DECISION"
    )

    write_json(
        OUT
        / "06_FINAL_REBUILD_QA.json",
        qa,
    )

    # -------------------------------------------------------------
    # 7. Freeze audit hashes for outputs.
    # -------------------------------------------------------------
    output_hashes = {}

    for p in sorted(
        OUT.iterdir()
    ):
        if not p.is_file():
            continue

        if p.name in {
            "07_OUTPUT_HASHES.json",
            "08_REBUILD_DECISION.json",
        }:
            continue

        output_hashes[
            p.name
        ] = {
            "size_bytes": int(
                p.stat().st_size
            ),
            "sha256": sha256_file(
                p
            ),
        }

    write_json(
        OUT
        / "07_OUTPUT_HASHES.json",
        output_hashes,
    )

    # -------------------------------------------------------------
    # 8. Final decision.
    # -------------------------------------------------------------
    if qa[
        "status"
    ] == "PASS":
        decision = (
            "PASS_HC_BUG03_CONTROLLED_REBUILD_V1_1__"
            "PAIR_AND_LABEL_LAYER_READY_FOR_DATASET_INDEX_REGENERATION"
        )

    else:
        decision = (
            "STOP_HC_BUG03_REBUILD_QA_FAILED__"
            "DO_NOT_PROMOTE_REPAIRED_ASSETS"
        )

    final = {
        "FINAL_DECISION": (
            decision
        ),

        "positive_semantics": (
            "AJG1 == 1 AND GSI == 1"
        ),

        "original_assets_overwritten": False,

        "new_version_directory": str(
            OUT
        ),

        "corrected_core": core_info[
            "state"
        ],

        "rebuild_QA": (
            qa
        ),

        "changed_slots": int(
            len(
                changed_slots
            )
        ),

        "removed_illegal_controls": int(
            len(
                label_result[
                    "removed_control_ids"
                ]
            )
        ),

        "new_replacement_controls": int(
            len(
                label_result[
                    "new_control_ids"
                ]
            )
        ),

        "next_step_if_pass": (
            "Regenerate DATASET-01/02 fold/training indices by preserving the "
            "existing pair_set_id-to-fold assignment and replacing only the "
            "control unit_ids according to the repaired pair edges. Then rerun "
            "all leakage/count/dynamic-sequence audits before model training."
        ),

        "important_rule": (
            "Do not delete or overwrite the old frozen hard-control assets. "
            "Keep old and repaired versions side-by-side for provenance."
        ),
    }

    write_json(
        OUT
        / "08_REBUILD_DECISION.json",
        final,
    )

    # -------------------------------------------------------------
    # 9. Human-readable report.
    # -------------------------------------------------------------
    report = f"""# HC-BUG-03 Controlled Rebuild V1

## Final decision

`{decision}`

## Positive-label semantics

Unchanged:

`AJG1 == 1 AND GSI == 1`

No positive unit was relabeled.

## Rebuild scope

The original frozen assets were NOT overwritten.

New version:

`{OUT}`

## Corrected HC00 core

- upstream eligible: {core_info['state']['upstream_eligible']}
- original HC00 core: {core_info['state']['current_core']}
- illegal candidates removed: {core_info['state']['removed_core']}
- corrected HC00 core: {core_info['state']['corrected_core']}

## Pair repair

- pair sets: {len(pair_table)}
- pair edges / hard controls: {len(pair_edges)}
- changed control slots vs original: {len(changed_slots)}
- old illegal controls removed: {len(label_result['removed_control_ids'])}
- new replacement controls entered: {len(label_result['new_control_ids'])}

## Integrated QA

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```

## Production interpretation

This is a controlled minimal-change repair derived from the audited production
P0 candidate graph and the proven HC-BUG-02D augmenting path.

The 23 upstream-illegal hard controls are removed from formal hard-control
roles. The repaired design retains:

- 5,056 pair sets;
- exactly two controls per pair;
- 10,112 unique controls;
- no control reuse;
- zero controls outside corrected HC00 core;
- unchanged formal positive set.

## Next step

Do NOT train models from the old DATASET-02 index.

Regenerate the pair/fold/training indices using this repaired pair assignment
while preserving the existing pair_set_id-to-fold map, then rerun the complete
DATASET leakage and dynamic-sequence audits.
"""

    (
        OUT
        / "00_HC_BUG03_REBUILD_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------
    print(
        "\n"
        + "=" * 118
    )

    print(
        "HC-BUG-03 CONTROLLED REBUILD V1 COMPLETE"
    )

    print(
        "=" * 118
    )

    print(
        "ORIGINAL ASSETS OVERWRITTEN: FALSE"
    )

    print(
        "NEW VERSION:",
        OUT,
    )

    print()

    print(
        "HC00 CORE:",
        EXPECTED_CURRENT_CORE,
        "->",
        EXPECTED_CORRECTED_CORE,
    )

    print(
        "REMOVED ILLEGAL CORE:",
        EXPECTED_REMOVED_CORE,
    )

    print()

    print(
        "PAIR SETS:",
        len(
            pair_table
        ),
    )

    print(
        "HARD CONTROLS:",
        len(
            pair_edges
        ),
    )

    print(
        "CHANGED CONTROL SLOTS:",
        len(
            changed_slots
        ),
    )

    print(
        "REMOVED ILLEGAL FORMAL CONTROLS:",
        len(
            label_result[
                "removed_control_ids"
            ]
        ),
    )

    print(
        "NEW REPLACEMENT CONTROLS:",
        len(
            label_result[
                "new_control_ids"
            ]
        ),
    )

    print()

    print(
        "POSITIVE SEMANTICS UNCHANGED:",
        qa[
            "positive_semantics_unchanged"
        ],
    )

    print(
        "CONTROL REUSE:",
        qa[
            "pair_edge_control_reuse_count"
        ],
    )

    print(
        "CONTROLS OUTSIDE CORRECTED CORE:",
        qa[
            "controls_outside_corrected_core"
        ],
    )

    print(
        "FINAL QA:",
        qa[
            "status"
        ],
    )

    print()

    print(
        "FINAL_DECISION:",
        decision,
    )

    print()

    print(
        "Please send:"
    )

    for name in [
        "00_HC_BUG03_REBUILD_REPORT.md",
        "01C_corrected_core_state.json",
        "02_repaired_assignment_graph_QA.json",
        "04B_changed_control_slots_v1.csv",
        "05B_excluded_unlabeled_template_audit.csv",
        "05D_removed_formal_controls.csv",
        "05E_new_formal_controls.csv",
        "06_FINAL_REBUILD_QA.json",
        "08_REBUILD_DECISION.json",
    ]:
        print(
            OUT
            / name
        )

    print(
        "=" * 118
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
