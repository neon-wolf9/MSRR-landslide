# -*- coding: utf-8 -*-
r"""
16A_HC_BUG03S_SPATIAL_GROUP_RECOVERY_AND_REMATCH.py
===================================================
HC-BUG-03S / 16A

Goal
----
Repair the hard-control cohort AGAIN from the ORIGINAL formal assignment,
this time preserving the frozen DATASET-01 spatial-group/fold constraint.

Why
---
HC-BUG-03 fixed the HC00 exclusion-propagation bug, but the later 15C audit
showed that the minimal 23-control repair was not spatially admissible:

- the original dataset_01_unit_fold_crosswalk contains spatial_group_id for
  the OLD 15,168 training units;
- the repaired cohort contains 23 incoming replacement controls absent from
  that old crosswalk;
- one repaired spatial group was split across folds;
- one repaired row conflicted with the original group->fold assignment.

Therefore the HC-BUG-03 replacement plan must NOT be promoted as the final
formal dataset.

This script is READ ONLY with respect to frozen/formal assets. It creates only
experiment/audit outputs.

Two gated phases
----------------
PHASE A -- Recover spatial-group lineage

Search the project for a more complete table containing the EXACT column:
    unit_id
    spatial_group_id

A candidate is trusted ONLY if:
1. every old canonical training unit is present;
2. old spatial_group_id reproduces 15,168 / 15,168 EXACTLY;
3. every original group maps to exactly one canonical fold;
4. it covers all units needed by the repaired/candidate audit.

If no such registry exists, STOP.
The script also searches source code for the canonical spatial-group generator
so the next step can replay the true generation logic rather than inventing a
new grouping rule.

PHASE B -- Spatial-constrained rematching

Start from the ORIGINAL formal 10,112 control slots.

Remove the 23 controls that are outside the corrected HC00 core.

Then find a complete matching subject to:
- candidate belongs to corrected HC00 core;
- candidate is present in the locked production P0 edge graph for that pair;
- candidate is not a formal positive;
- candidate's spatial_group_id belongs to the SAME canonical outer fold as
  the pair;
- control reuse = 0.

The solver first tries direct free replacements for the 23 gaps, then uses
slot-level augmenting paths if displacement of existing valid controls is
necessary.

A PASS proves that a full 5,056 x 1:2 matching exists under the ORIGINAL
spatial-group anti-leakage protocol.

A PASS still DOES NOT overwrite formal data. It outputs a dry-run assignment
for the next controlled rebuild.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Paths and frozen expectations
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "experiments"
    / "HC_BUG_03S_SPATIAL_GROUP_RECOVERY_AND_REMATCH_V1"
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

ORIGINAL_FORMAL_LABEL = (
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

EXPECTED_MASTER_ROWS = 139364
EXPECTED_OLD_TRAIN_UNITS = 15168
EXPECTED_PAIRSETS = 5056
EXPECTED_CONTROLS = 10112
EXPECTED_INVALID_CONTROLS = 23
EXPECTED_CORRECTED_CORE = 28072

EXPECTED_FOLD_COUNTS = {
    "FOLD_1": 1011,
    "FOLD_2": 1011,
    "FOLD_3": 1011,
    "FOLD_4": 1011,
    "FOLD_5": 1012,
}

# Strict default: only groups already present in the original canonical
# training split are admissible. Do not silently assign a new/unseen group
# to a fold.
ALLOW_GROUPS_UNSEEN_IN_OLD_TRAINING = False

SCHEMA_SCAN_LIMIT = 12000
SOURCE_CODE_SCAN_LIMIT = 12000
SOURCE_SNIPPET_RADIUS = 700

P0_QUERY_BATCH = 128
INF = 1e18


# =====================================================================
# Utilities
# =====================================================================

def log(msg: str):
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUT / "logs").mkdir(
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
        OUT
        / "logs"
        / "16A.log"
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


def get_schema(path: Path):
    try:
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

    except Exception:
        return []

    return []


def get_row_count(path: Path):
    try:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            return int(
                pq.ParquetFile(
                    path
                ).metadata.num_rows
            )

        if path.suffix.lower() == ".csv":
            if path.stat().st_size > 1_000_000_000:
                return None

            with path.open(
                "rb"
            ) as f:
                return max(
                    sum(
                        1 for _ in f
                    )
                    - 1,
                    0,
                )

    except Exception:
        return None

    return None


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


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p

    return None


def truthy(s: pd.Series):
    if pd.api.types.is_bool_dtype(
        s
    ):
        return (
            s.fillna(
                False
            )
            .astype(
                bool
            )
        )

    if pd.api.types.is_numeric_dtype(
        s
    ):
        return (
            pd.to_numeric(
                s,
                errors="coerce",
            )
            .fillna(
                0
            )
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


def normalize_pair(v):
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
        if float(
            v
        ).is_integer():
            return str(
                int(v)
            )

        return str(
            v
        )

    s = str(
        v
    ).strip()

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
        x = float(
            v
        )

        if x.is_integer():
            return int(
                x
            )

    except Exception:
        pass

    return str(
        v
    ).strip()


def normalize_fold(v):
    if pd.isna(v):
        return None

    return str(
        v
    ).strip()


def make_slot_key(
    pair_key,
    rank,
):
    return (
        f"{pair_key}"
        f"::CONTROL_RANK={rank}"
    )


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            b = f.read(
                8
                * 1024
                * 1024
            )

            if not b:
                break

            h.update(
                b
            )

    return h.hexdigest()


# =====================================================================
# Canonical fold and old spatial-group anchor
# =====================================================================

def load_canonical_split():
    if not CANON_PAIR_FOLD.exists():
        raise RuntimeError(
            f"Missing canonical pair fold:\n{CANON_PAIR_FOLD}"
        )

    if not CANON_UNIT_FOLD.exists():
        raise RuntimeError(
            f"Missing canonical unit fold:\n{CANON_UNIT_FOLD}"
        )

    pair_cols = get_schema(
        CANON_PAIR_FOLD
    )

    if (
        "pair_set_id"
        not in pair_cols
    ):
        raise RuntimeError(
            "Canonical pair crosswalk lacks pair_set_id."
        )

    pair_fold_col = None

    for c in [
        "outer_fold",
        "fold",
        "cv_fold",
    ]:
        if c in pair_cols:
            pair_fold_col = c
            break

    if pair_fold_col is None:
        candidates = [
            c
            for c in pair_cols
            if "fold" in c.lower()
        ]

        if candidates:
            pair_fold_col = candidates[
                0
            ]

    if pair_fold_col is None:
        raise RuntimeError(
            "Canonical pair crosswalk lacks fold field."
        )

    pair = read_table(
        CANON_PAIR_FOLD,
        [
            "pair_set_id",
            pair_fold_col,
        ],
    )

    pair[
        "PAIR_KEY"
    ] = pair[
        "pair_set_id"
    ].map(
        normalize_pair
    )

    pair[
        "CANON_FOLD"
    ] = pair[
        pair_fold_col
    ].map(
        normalize_fold
    )

    pair_map = (
        pair[
            [
                "PAIR_KEY",
                "CANON_FOLD",
            ]
        ]
        .drop_duplicates()
    )

    if len(
        pair_map
    ) != EXPECTED_PAIRSETS:
        raise RuntimeError(
            "Canonical pair fold map does not contain 5,056 pairs."
        )

    if (
        pair.groupby(
            "PAIR_KEY"
        )[
            "CANON_FOLD"
        ]
        .nunique()
        > 1
    ).any():
        raise RuntimeError(
            "A canonical pair crosses folds."
        )

    counts = (
        pair_map[
            "CANON_FOLD"
        ]
        .value_counts()
        .to_dict()
    )

    if counts != EXPECTED_FOLD_COUNTS:
        raise RuntimeError(
            "Canonical fold counts changed.\n"
            f"{counts}"
        )

    unit_cols = get_schema(
        CANON_UNIT_FOLD
    )

    required = {
        "unit_id",
        "spatial_group_id",
    }

    if not required.issubset(
        set(
            unit_cols
        )
    ):
        raise RuntimeError(
            "Canonical old unit crosswalk lacks unit_id/spatial_group_id."
        )

    unit_fold_col = None

    for c in [
        "outer_fold",
        "fold",
        "cv_fold",
    ]:
        if c in unit_cols:
            unit_fold_col = c
            break

    if unit_fold_col is None:
        candidates = [
            c
            for c in unit_cols
            if "fold" in c.lower()
        ]

        if candidates:
            unit_fold_col = candidates[
                0
            ]

    if unit_fold_col is None:
        raise RuntimeError(
            "Canonical unit crosswalk lacks fold."
        )

    old = read_table(
        CANON_UNIT_FOLD,
        [
            "unit_id",
            "spatial_group_id",
            unit_fold_col,
        ],
    )

    if len(
        old
    ) != EXPECTED_OLD_TRAIN_UNITS:
        raise RuntimeError(
            "Canonical old unit fold crosswalk does not contain 15,168 rows."
        )

    if old[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Old unit fold crosswalk has duplicate unit_id."
        )

    old[
        "OLD_FOLD"
    ] = old[
        unit_fold_col
    ].map(
        normalize_fold
    )

    old[
        "GROUP_KEY"
    ] = old[
        "spatial_group_id"
    ].astype(
        str
    )

    group_fold_n = (
        old.groupby(
            "GROUP_KEY"
        )[
            "OLD_FOLD"
        ]
        .nunique()
    )

    if (
        group_fold_n
        > 1
    ).any():
        raise RuntimeError(
            "Original canonical spatial groups already cross folds."
        )

    group_to_fold = (
        old[
            [
                "GROUP_KEY",
                "OLD_FOLD",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "GROUP_KEY"
        )[
            "OLD_FOLD"
        ]
        .to_dict()
    )

    return {
        "pair_map": (
            pair_map
        ),
        "pair_to_fold": dict(
            zip(
                pair_map[
                    "PAIR_KEY"
                ],
                pair_map[
                    "CANON_FOLD"
                ],
            )
        ),
        "old_unit_groups": (
            old
        ),
        "old_group_to_fold": (
            group_to_fold
        ),
        "old_group_count": int(
            old[
                "GROUP_KEY"
            ].nunique()
        ),
    }


# =====================================================================
# PHASE A: exact spatial_group_id lineage recovery
# =====================================================================

def search_exact_group_registry_candidates():
    """
    IMPORTANT: exact column only.
    Do not treat arbitrary component/count fields as spatial groups.
    """

    rows = []
    seen = set()
    scanned = 0

    for p in ROOT.rglob(
        "*"
    ):
        if scanned >= SCHEMA_SCAN_LIMIT:
            break

        if (
            not p.is_file()
            or p.suffix.lower()
            not in {
                ".parquet",
                ".csv",
            }
        ):
            continue

        if OUT in p.parents:
            continue

        if p in seen:
            continue

        seen.add(
            p
        )

        cols = get_schema(
            p
        )

        scanned += 1

        if not {
            "unit_id",
            "spatial_group_id",
        }.issubset(
            set(
                cols
            )
        ):
            continue

        nrows = get_row_count(
            p
        )

        score = 0

        if nrows == EXPECTED_MASTER_ROWS:
            score += 500

        if (
            nrows is not None
            and nrows
            > EXPECTED_OLD_TRAIN_UNITS
        ):
            score += 120

        if p == CANON_UNIT_FOLD:
            # This is the anchor, not the full registry.
            score -= 300

        low = str(
            p
        ).lower()

        for token, pts in [
            (
                "spatial_split_protocol",
                100,
            ),
            (
                "spatial",
                30,
            ),
            (
                "group",
                30,
            ),
            (
                "crosswalk",
                20,
            ),
            (
                "registry",
                20,
            ),
            (
                "master",
                20,
            ),
            (
                "frozen",
                10,
            ),
        ]:
            if token in low:
                score += pts

        rows.append({
            "path": str(
                p
            ),
            "row_count": (
                nrows
            ),
            "score": (
                score
            ),
            "columns": "|".join(
                cols
            ),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "score",
                "row_count",
                "path",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )


def validate_exact_group_registry(
    path: Path,
    old_unit_groups: pd.DataFrame,
):
    try:
        q = read_table(
            path,
            [
                "unit_id",
                "spatial_group_id",
            ],
        )
    except Exception as e:
        return None, {
            "path": str(
                path
            ),
            "status": (
                "READ_FAIL"
            ),
            "error": (
                f"{type(e).__name__}: {e}"
            ),
        }

    q = q[
        q[
            "unit_id"
        ].notna()
    ].copy()

    conflict = (
        q.groupby(
            "unit_id"
        )[
            "spatial_group_id"
        ]
        .nunique(
            dropna=False
        )
    )

    multi_group = int(
        (
            conflict
            > 1
        ).sum()
    )

    if multi_group:
        return None, {
            "path": str(
                path
            ),
            "status": (
                "UNIT_MULTI_GROUP"
            ),
            "multi_group_units": (
                multi_group
            ),
        }

    q = (
        q[
            [
                "unit_id",
                "spatial_group_id",
            ]
        ]
        .drop_duplicates(
            "unit_id"
        )
        .copy()
    )

    q[
        "GROUP_KEY"
    ] = q[
        "spatial_group_id"
    ].astype(
        str
    )

    old = old_unit_groups[
        [
            "unit_id",
            "GROUP_KEY",
        ]
    ].rename(
        columns={
            "GROUP_KEY": (
                "OLD_GROUP_KEY"
            )
        }
    )

    joined = old.merge(
        q[
            [
                "unit_id",
                "GROUP_KEY",
            ]
        ],
        on="unit_id",
        how="left",
        validate="one_to_one",
    )

    old_covered = int(
        joined[
            "GROUP_KEY"
        ].notna()
        .sum()
    )

    exact_match = int(
        (
            joined[
                "GROUP_KEY"
            ].notna()
            & (
                joined[
                    "GROUP_KEY"
                ]
                == joined[
                    "OLD_GROUP_KEY"
                ]
            )
        ).sum()
    )

    mismatch = int(
        (
            joined[
                "GROUP_KEY"
            ].notna()
            & (
                joined[
                    "GROUP_KEY"
                ]
                != joined[
                    "OLD_GROUP_KEY"
                ]
            )
        ).sum()
    )

    status = (
        "PASS_EXACT_OLD_15168_REPRODUCTION"
        if (
            old_covered
            == EXPECTED_OLD_TRAIN_UNITS
            and exact_match
            == EXPECTED_OLD_TRAIN_UNITS
            and mismatch
            == 0
        )
        else "FAIL"
    )

    audit = {
        "path": str(
            path
        ),
        "status": (
            status
        ),
        "registry_rows": int(
            len(
                q
            )
        ),
        "old_units_covered": (
            old_covered
        ),
        "old_group_exact_match": (
            exact_match
        ),
        "old_group_mismatch": (
            mismatch
        ),
        "unique_groups": int(
            q[
                "GROUP_KEY"
            ].nunique()
        ),
    }

    return (
        q,
        audit,
    )


def source_code_lineage_search():
    """
    If a full exact registry does not exist, identify the production code
    that generated spatial_group_id. This is diagnostic only.
    """

    terms = [
        "spatial_group_id",
        "dataset_01_unit_fold_crosswalk",
        "dataset_01_pair_set_fold_crosswalk",
        "PRIMARY_BALANCED_SPATIAL_GROUPED_5FOLD",
        "spatial group",
    ]

    rows = []
    scanned = 0

    for p in ROOT.rglob(
        "*"
    ):
        if scanned >= SOURCE_CODE_SCAN_LIMIT:
            break

        if (
            not p.is_file()
            or p.suffix.lower()
            not in {
                ".py",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
            }
        ):
            continue

        try:
            if p.stat().st_size > 5_000_000:
                continue
        except Exception:
            continue

        scanned += 1

        try:
            text = p.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        low = text.lower()

        hits = []

        for term in terms:
            n = low.count(
                term.lower()
            )

            if n:
                hits.append(
                    (
                        term,
                        n,
                    )
                )

        if not hits:
            continue

        score = sum(
            n
            for _, n in hits
        )

        path_low = str(
            p
        ).lower()

        if (
            "05_final_dataset_assembly"
            in path_low
        ):
            score += 30

        if (
            "spatial_split_protocol"
            in path_low
        ):
            score += 50

        snippets = []

        for term, _ in hits:
            pos = low.find(
                term.lower()
            )

            if pos >= 0:
                a = max(
                    0,
                    pos
                    - SOURCE_SNIPPET_RADIUS,
                )

                b = min(
                    len(
                        text
                    ),
                    pos
                    + SOURCE_SNIPPET_RADIUS,
                )

                snippet = re.sub(
                    r"\s+",
                    " ",
                    text[
                        a:b
                    ],
                ).strip()

                snippets.append(
                    f"[{term}] {snippet}"
                )

        rows.append({
            "path": str(
                p
            ),
            "score": int(
                score
            ),
            "hits": "|".join(
                f"{term}:{n}"
                for term, n
                in hits
            ),
            "snippet": (
                "\n---\n".join(
                    snippets[
                        :5
                    ]
                )
            ),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "score",
                "path",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# =====================================================================
# Corrected HC00 core and ORIGINAL formal slots
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
            "Frozen upstream eligibility not found."
        )

    if core_path is None:
        raise RuntimeError(
            "HC00 core not found."
        )

    ecols = get_schema(
        eligibility_path
    )

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

    eligibility = read_table(
        eligibility_path,
        wanted,
    )

    if (
        "primary_eligibility_status"
        not in eligibility.columns
    ):
        raise RuntimeError(
            "Upstream eligibility lacks primary_eligibility_status."
        )

    if eligibility[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Eligibility unit_id not unique."
        )

    status = (
        eligibility[
            "primary_eligibility_status"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    eligible = status.str.startswith(
        "ELIGIBLE"
    )

    explicit_excluded = pd.Series(
        False,
        index=eligibility.index,
    )

    for c in [
        "control_exclusion_flag",
        "inventory_evidence_exclusion_flag",
    ]:
        if c in eligibility.columns:
            explicit_excluded |= truthy(
                eligibility[
                    c
                ]
            )

    contradiction = (
        eligible
        & explicit_excluded
    )

    if contradiction.any():
        raise RuntimeError(
            "Upstream eligibility internally contradictory."
        )

    eligible_ids = set(
        eligibility.loc[
            eligible,
            "unit_id",
        ].tolist()
    )

    core = read_table(
        core_path,
        [
            "unit_id",
        ],
    )

    current_core = set(
        core[
            "unit_id"
        ].tolist()
    )

    corrected_core = (
        current_core
        & eligible_ids
    )

    removed = (
        current_core
        - corrected_core
    )

    if len(
        corrected_core
    ) != EXPECTED_CORRECTED_CORE:
        raise RuntimeError(
            "Corrected HC00 core count changed: "
            f"{len(corrected_core)}"
        )

    return {
        "eligibility_path": (
            eligibility_path
        ),
        "core_path": (
            core_path
        ),
        "corrected_ids": (
            corrected_core
        ),
        "removed_ids": (
            removed
        ),
    }


def load_original_slots(
    corrected_ids,
    pair_to_fold,
):
    if not ORIGINAL_FORMAL_LABEL.exists():
        raise RuntimeError(
            f"Missing original formal label layer:\n{ORIGINAL_FORMAL_LABEL}"
        )

    schema = get_schema(
        ORIGINAL_FORMAL_LABEL
    )

    wanted = [
        c
        for c in [
            "unit_id",
            "sample_role",
            "hard_control_flag",
            "positive_label_flag",
            "y_main",
            "pair_set_id",
            "matched_positive_unit_id",
            "control_rank",
        ]
        if c in schema
    ]

    formal = read_table(
        ORIGINAL_FORMAL_LABEL,
        wanted,
    )

    if formal[
        "unit_id"
    ].duplicated().any():
        raise RuntimeError(
            "Original full-grid label unit_id not unique."
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

    else:
        is_hard = (
            formal[
                "sample_role"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq(
                "HARD_CONTROL"
            )
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

    elif (
        "y_main"
        in formal.columns
    ):
        is_positive = (
            pd.to_numeric(
                formal[
                    "y_main"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            > 0
        )

    else:
        raise RuntimeError(
            "Cannot identify original positives."
        )

    formal[
        "IS_HARD"
    ] = is_hard

    formal[
        "IS_POSITIVE"
    ] = is_positive

    hard = formal[
        formal[
            "IS_HARD"
        ]
    ].copy()

    if len(
        hard
    ) != EXPECTED_CONTROLS:
        raise RuntimeError(
            "Original hard controls != 10,112."
        )

    hard[
        "PAIR_KEY"
    ] = hard[
        "pair_set_id"
    ].map(
        normalize_pair
    )

    hard[
        "RANK_KEY"
    ] = hard[
        "control_rank"
    ].map(
        normalize_rank
    )

    hard[
        "SLOT_KEY"
    ] = [
        make_slot_key(
            p,
            r,
        )
        for p, r
        in zip(
            hard[
                "PAIR_KEY"
            ],
            hard[
                "RANK_KEY"
            ],
        )
    ]

    if hard[
        "SLOT_KEY"
    ].duplicated().any():
        raise RuntimeError(
            "Original control slot duplicated."
        )

    hard[
        "PAIR_FOLD"
    ] = hard[
        "PAIR_KEY"
    ].map(
        pair_to_fold
    )

    if hard[
        "PAIR_FOLD"
    ].isna().any():
        raise RuntimeError(
            "Some formal hard-control pair lacks canonical fold."
        )

    hard[
        "CURRENT_VALID"
    ] = hard[
        "unit_id"
    ].isin(
        corrected_ids
    )

    invalid = hard[
        ~hard[
            "CURRENT_VALID"
        ]
    ].copy()

    if len(
        invalid
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Expected 23 invalid original formal controls, got "
            f"{len(invalid)}."
        )

    positive_ids = set(
        formal.loc[
            formal[
                "IS_POSITIVE"
            ],
            "unit_id",
        ].tolist()
    )

    return (
        formal,
        hard,
        invalid,
        positive_ids,
    )


# =====================================================================
# Spatial group registry object
# =====================================================================

class GroupRegistry:
    def __init__(
        self,
        registry: pd.DataFrame,
        old_group_to_fold: dict,
    ):
        q = registry[
            [
                "unit_id",
                "GROUP_KEY",
            ]
        ].copy()

        if q[
            "unit_id"
        ].duplicated().any():
            raise RuntimeError(
                "Trusted group registry unit_id is not unique."
            )

        self.unit_to_group = dict(
            zip(
                q[
                    "unit_id"
                ],
                q[
                    "GROUP_KEY"
                ],
            )
        )

        self.group_to_fold = dict(
            old_group_to_fold
        )

    def group(
        self,
        unit_id,
    ):
        return self.unit_to_group.get(
            unit_id
        )

    def fold(
        self,
        unit_id,
    ):
        group = self.group(
            unit_id
        )

        if group is None:
            return None

        return self.group_to_fold.get(
            group
        )

    def is_legal_for_fold(
        self,
        unit_id,
        target_fold,
    ):
        group = self.group(
            unit_id
        )

        if group is None:
            return False

        known_fold = self.group_to_fold.get(
            group
        )

        if known_fold is None:
            if (
                ALLOW_GROUPS_UNSEEN_IN_OLD_TRAINING
            ):
                return True

            return False

        return (
            known_fold
            == target_fold
        )


# =====================================================================
# P0 candidate provider with spatial constraint
# =====================================================================

class CandidateProvider:
    def __init__(
        self,
        corrected_ids,
        positive_ids,
        group_registry: GroupRegistry,
    ):
        if not P0_EDGES.exists():
            raise RuntimeError(
                f"Locked P0 graph missing:\n{P0_EDGES}"
            )

        import pyarrow.dataset as ds

        self.ds_module = ds

        self.dataset = ds.dataset(
            str(
                P0_EDGES
            ),
            format="parquet",
        )

        cols = set(
            self.dataset.schema.names
        )

        required = {
            "positive_unit_id",
            "candidate_unit_id",
        }

        if not required.issubset(
            cols
        ):
            raise RuntimeError(
                "P0 graph missing positive/candidate ids."
            )

        self.order_col = (
            "composite_distance"
            if "composite_distance"
            in cols
            else None
        )

        self.columns = [
            "positive_unit_id",
            "candidate_unit_id",
        ]

        if self.order_col:
            self.columns.append(
                self.order_col
            )

        self.corrected_ids = (
            corrected_ids
        )

        self.positive_ids = (
            positive_ids
        )

        self.group_registry = (
            group_registry
        )

        self.cache = {}

        self.query_calls = 0
        self.raw_rows = 0
        self.legal_rows = 0

    def ensure(
        self,
        requests,
    ):
        """
        requests: iterable of (positive_id, target_fold)
        Cache key includes fold.
        """

        missing = []

        for positive_id, fold in requests:
            key = (
                str(
                    positive_id
                ),
                str(
                    fold
                ),
            )

            if key not in self.cache:
                missing.append(
                    key
                )

        if not missing:
            return

        # Query by positive in batches, then filter separately by requested fold.
        positives = sorted(
            set(
                p
                for p, _
                in missing
            )
        )

        raw_by_positive = defaultdict(
            list
        )

        for i in range(
            0,
            len(
                positives
            ),
            P0_QUERY_BATCH,
        ):
            batch = positives[
                i:
                i
                + P0_QUERY_BATCH
            ]

            filt = (
                self.ds_module
                .field(
                    "positive_unit_id"
                )
                .isin(
                    batch
                )
            )

            table = (
                self.dataset
                .to_table(
                    columns=(
                        self.columns
                    ),
                    filter=(
                        filt
                    ),
                )
            )

            df = table.to_pandas()

            self.query_calls += 1

            self.raw_rows += len(
                df
            )

            if df.empty:
                continue

            df[
                "positive_unit_id"
            ] = df[
                "positive_unit_id"
            ].astype(
                str
            )

            # Structural legality independent of fold.
            df = df[
                df[
                    "candidate_unit_id"
                ].isin(
                    self.corrected_ids
                )
            ].copy()

            df = df[
                ~df[
                    "candidate_unit_id"
                ].isin(
                    self.positive_ids
                )
            ].copy()

            if self.order_col:
                df[
                    "_ORDER"
                ] = (
                    pd.to_numeric(
                        df[
                            self.order_col
                        ],
                        errors="coerce",
                    )
                    .fillna(
                        INF
                        / 100
                    )
                )

            else:
                df[
                    "_ORDER"
                ] = 0.0

            df = (
                df.sort_values(
                    [
                        "positive_unit_id",
                        "_ORDER",
                        "candidate_unit_id",
                    ]
                )
                .drop_duplicates(
                    [
                        "positive_unit_id",
                        "candidate_unit_id",
                    ]
                )
            )

            for pos, g in df.groupby(
                "positive_unit_id"
            ):
                raw_by_positive[
                    pos
                ] = [
                    (
                        row[
                            "candidate_unit_id"
                        ],
                        float(
                            row[
                                "_ORDER"
                            ]
                        ),
                    )
                    for _, row in g.iterrows()
                ]

        for positive_id, fold in missing:
            vals = []

            for cand, cost in raw_by_positive.get(
                positive_id,
                [],
            ):
                if self.group_registry.is_legal_for_fold(
                    cand,
                    fold,
                ):
                    vals.append(
                        (
                            cand,
                            cost,
                        )
                    )

            self.cache[
                (
                    positive_id,
                    fold,
                )
            ] = vals

            self.legal_rows += len(
                vals
            )

    def get(
        self,
        positive_id,
        fold,
    ):
        key = (
            str(
                positive_id
            ),
            str(
                fold
            ),
        )

        self.ensure(
            [
                key
            ]
        )

        return self.cache[
            key
        ]

    def stats(
        self,
    ):
        return {
            "query_calls": (
                self.query_calls
            ),
            "raw_P0_rows_read": (
                self.raw_rows
            ),
            "spatial_legal_edges_cached": (
                self.legal_rows
            ),
            "cached_positive_fold_requests": (
                len(
                    self.cache
                )
            ),
            "order_col": (
                self.order_col
            ),
        }


# =====================================================================
# Matching helpers
# =====================================================================

def build_starting_matching(
    hard: pd.DataFrame,
):
    slot_meta = {}

    slot_to_control = {}

    control_to_slot = {}

    for _, r in hard.iterrows():
        slot = r[
            "SLOT_KEY"
        ]

        slot_meta[
            slot
        ] = {
            "pair_key": (
                r[
                    "PAIR_KEY"
                ]
            ),
            "rank_key": (
                r[
                    "RANK_KEY"
                ]
            ),
            "positive_unit_id": str(
                r[
                    "matched_positive_unit_id"
                ]
            ),
            "pair_fold": (
                r[
                    "PAIR_FOLD"
                ]
            ),
            "original_control_unit_id": (
                r[
                    "unit_id"
                ]
            ),
        }

        if bool(
            r[
                "CURRENT_VALID"
            ]
        ):
            ctrl = r[
                "unit_id"
            ]

            if ctrl in control_to_slot:
                raise RuntimeError(
                    "Original valid controls reuse a unit."
                )

            slot_to_control[
                slot
            ] = ctrl

            control_to_slot[
                ctrl
            ] = slot

        else:
            slot_to_control[
                slot
            ] = None

    unmatched = [
        s
        for s, c
        in slot_to_control.items()
        if c is None
    ]

    if len(
        unmatched
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Starting matching should have 23 unmatched slots."
        )

    return (
        slot_meta,
        slot_to_control,
        control_to_slot,
        unmatched,
    )


def direct_free_repair(
    unmatched_slots,
    slot_meta,
    slot_to_control,
    control_to_slot,
    provider,
):
    """
    Maximum bipartite matching between the 23 missing slots and currently
    FREE spatial-legal controls. Existing 10,089 valid controls remain fixed.
    """

    adjacency = {}

    requests = [
        (
            slot_meta[
                s
            ][
                "positive_unit_id"
            ],
            slot_meta[
                s
            ][
                "pair_fold"
            ],
        )
        for s
        in unmatched_slots
    ]

    provider.ensure(
        requests
    )

    occupied = set(
        control_to_slot
    )

    for slot in unmatched_slots:
        meta = slot_meta[
            slot
        ]

        vals = provider.get(
            meta[
                "positive_unit_id"
            ],
            meta[
                "pair_fold"
            ],
        )

        adjacency[
            slot
        ] = [
            (
                c,
                cost,
            )
            for c, cost
            in vals
            if c not in occupied
        ]

    # Most constrained first.
    order = sorted(
        unmatched_slots,
        key=lambda s: (
            len(
                adjacency[
                    s
                ]
            ),
            s,
        ),
    )

    candidate_owner = {}

    slot_choice = {}

    def dfs(
        slot,
        seen,
    ):
        for cand, cost in adjacency[
            slot
        ]:
            if cand in seen:
                continue

            seen.add(
                cand
            )

            if cand not in candidate_owner:
                candidate_owner[
                    cand
                ] = slot

                slot_choice[
                    slot
                ] = (
                    cand,
                    cost,
                )

                return True

            old_slot = candidate_owner[
                cand
            ]

            if dfs(
                old_slot,
                seen,
            ):
                candidate_owner[
                    cand
                ] = slot

                slot_choice[
                    slot
                ] = (
                    cand,
                    cost,
                )

                return True

        return False

    for slot in order:
        dfs(
            slot,
            set(),
        )

    # Apply direct matches.
    rows = []

    for slot, (
        cand,
        cost,
    ) in slot_choice.items():
        if cand in control_to_slot:
            raise RuntimeError(
                "Direct-free solver selected occupied control."
            )

        slot_to_control[
            slot
        ] = cand

        control_to_slot[
            cand
        ] = slot

        rows.append({
            "slot_key": (
                slot
            ),
            "pair_set_id": (
                slot_meta[
                    slot
                ][
                    "pair_key"
                ]
            ),
            "control_rank": (
                slot_meta[
                    slot
                ][
                    "rank_key"
                ]
            ),
            "positive_unit_id": (
                slot_meta[
                    slot
                ][
                    "positive_unit_id"
                ]
            ),
            "pair_fold": (
                slot_meta[
                    slot
                ][
                    "pair_fold"
                ]
            ),
            "replacement_unit_id": (
                cand
            ),
            "order_cost": (
                cost
            ),
            "mode": (
                "DIRECT_FREE_SPATIAL_LEGAL"
            ),
            "candidate_count_free": (
                len(
                    adjacency[
                        slot
                    ]
                )
            ),
        })

    still_unmatched = [
        s
        for s
        in unmatched_slots
        if slot_to_control[
            s
        ]
        is None
    ]

    count_rows = []

    for s in unmatched_slots:
        count_rows.append({
            "slot_key": (
                s
            ),
            "pair_set_id": (
                slot_meta[
                    s
                ][
                    "pair_key"
                ]
            ),
            "pair_fold": (
                slot_meta[
                    s
                ][
                    "pair_fold"
                ]
            ),
            "free_spatial_legal_candidates": (
                len(
                    adjacency[
                        s
                    ]
                )
            ),
            "direct_match_found": (
                s
                in slot_choice
            ),
        })

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            count_rows
        ),
        still_unmatched,
    )


def find_augmenting_path(
    root_slot,
    slot_meta,
    slot_to_control,
    control_to_slot,
    provider,
):
    """
    Exact slot-level BFS under the spatial-fold legality constraint.
    """

    parent_slot = {
        root_slot: (
            None
        )
    }

    parent_control = {
        root_slot: (
            None
        )
    }

    depth = {
        root_slot: (
            0
        )
    }

    queue = deque(
        [
            root_slot
        ]
    )

    visited_controls = set()

    while queue:
        current_depth = depth[
            queue[
                0
            ]
        ]

        layer = []

        while (
            queue
            and depth[
                queue[
                    0
                ]
            ]
            == current_depth
        ):
            layer.append(
                queue.popleft()
            )

        requests = [
            (
                slot_meta[
                    s
                ][
                    "positive_unit_id"
                ],
                slot_meta[
                    s
                ][
                    "pair_fold"
                ],
            )
            for s
            in layer
        ]

        provider.ensure(
            requests
        )

        for slot in layer:
            meta = slot_meta[
                slot
            ]

            current_control = (
                slot_to_control.get(
                    slot
                )
            )

            candidates = provider.get(
                meta[
                    "positive_unit_id"
                ],
                meta[
                    "pair_fold"
                ],
            )

            for cand, cost in candidates:
                if (
                    current_control
                    is not None
                    and cand
                    == current_control
                ):
                    continue

                if cand in visited_controls:
                    continue

                visited_controls.add(
                    cand
                )

                owner = control_to_slot.get(
                    cand
                )

                if owner is None:
                    return {
                        "found": (
                            True
                        ),
                        "root_slot": (
                            root_slot
                        ),
                        "terminal_slot": (
                            slot
                        ),
                        "terminal_free_control": (
                            cand
                        ),
                        "terminal_cost": (
                            cost
                        ),
                        "parent_slot": (
                            parent_slot
                        ),
                        "parent_control": (
                            parent_control
                        ),
                        "depth": (
                            depth
                        ),
                        "visited_slots": (
                            len(
                                parent_slot
                            )
                        ),
                        "visited_controls": (
                            len(
                                visited_controls
                            )
                        ),
                    }

                if owner in parent_slot:
                    continue

                parent_slot[
                    owner
                ] = slot

                parent_control[
                    owner
                ] = cand

                depth[
                    owner
                ] = (
                    depth[
                        slot
                    ]
                    + 1
                )

                queue.append(
                    owner
                )

    return {
        "found": (
            False
        ),
        "root_slot": (
            root_slot
        ),
        "visited_slots": (
            len(
                parent_slot
            )
        ),
        "visited_controls": (
            len(
                visited_controls
            )
        ),
    }


def reconstruct_and_apply_path(
    result,
    slot_meta,
    slot_to_control,
    control_to_slot,
):
    terminal = result[
        "terminal_slot"
    ]

    rev_slots = []

    cur = terminal

    while cur is not None:
        rev_slots.append(
            cur
        )

        cur = result[
            "parent_slot"
        ].get(
            cur
        )

    path_slots = list(
        reversed(
            rev_slots
        )
    )

    actions = []

    # Build all changes first.
    for i, slot in enumerate(
        path_slots
    ):
        old_control = slot_to_control.get(
            slot
        )

        if i < (
            len(
                path_slots
            )
            - 1
        ):
            child = path_slots[
                i
                + 1
            ]

            new_control = result[
                "parent_control"
            ][
                child
            ]

            action = (
                "TAKE_OCCUPIED_SPATIAL_LEGAL_CONTROL"
            )

        else:
            new_control = result[
                "terminal_free_control"
            ]

            action = (
                "TAKE_FREE_SPATIAL_LEGAL_CONTROL"
            )

        meta = slot_meta[
            slot
        ]

        actions.append({
            "path_step": (
                i
            ),
            "slot_key": (
                slot
            ),
            "pair_set_id": (
                meta[
                    "pair_key"
                ]
            ),
            "control_rank": (
                meta[
                    "rank_key"
                ]
            ),
            "positive_unit_id": (
                meta[
                    "positive_unit_id"
                ]
            ),
            "pair_fold": (
                meta[
                    "pair_fold"
                ]
            ),
            "old_control_unit_id": (
                old_control
            ),
            "new_control_unit_id": (
                new_control
            ),
            "action": (
                action
            ),
        })

    # Remove all old matched controls from touched slots first.
    for a in actions:
        old = a[
            "old_control_unit_id"
        ]

        if (
            old is not None
            and control_to_slot.get(
                old
            )
            == a[
                "slot_key"
            ]
        ):
            del control_to_slot[
                old
            ]

    # Apply new controls.
    for a in actions:
        slot = a[
            "slot_key"
        ]

        new = a[
            "new_control_unit_id"
        ]

        if new in control_to_slot:
            raise RuntimeError(
                "Augmenting path application would create control reuse."
            )

        slot_to_control[
            slot
        ] = new

        control_to_slot[
            new
        ] = slot

    return pd.DataFrame(
        actions
    )


def complete_spatial_matching(
    unmatched_slots,
    slot_meta,
    slot_to_control,
    control_to_slot,
    provider,
):
    all_paths = []

    # Repeated passes are safer when several left slots are initially unmatched.
    pass_no = 0

    while True:
        current_unmatched = [
            s
            for s in unmatched_slots
            if slot_to_control[
                s
            ]
            is None
        ]

        if not current_unmatched:
            break

        pass_no += 1

        progress = 0

        for root in list(
            current_unmatched
        ):
            if slot_to_control[
                root
            ] is not None:
                continue

            result = find_augmenting_path(
                root,
                slot_meta,
                slot_to_control,
                control_to_slot,
                provider,
            )

            if not result[
                "found"
            ]:
                all_paths.append({
                    "pass_no": (
                        pass_no
                    ),
                    "root_slot": (
                        root
                    ),
                    "found": (
                        False
                    ),
                    "path_depth": (
                        None
                    ),
                    "visited_slots": (
                        result[
                            "visited_slots"
                        ]
                    ),
                    "visited_controls": (
                        result[
                            "visited_controls"
                        ]
                    ),
                })

                continue

            actions = reconstruct_and_apply_path(
                result,
                slot_meta,
                slot_to_control,
                control_to_slot,
            )

            path_id = (
                f"PASS{pass_no}_"
                f"{root}"
            )

            actions[
                "augmenting_path_id"
            ] = path_id

            actions[
                "augmenting_path_depth"
            ] = (
                len(
                    actions
                )
                - 1
            )

            actions[
                "pass_no"
            ] = (
                pass_no
            )

            all_paths.append(
                {
                    "pass_no": (
                        pass_no
                    ),
                    "root_slot": (
                        root
                    ),
                    "found": (
                        True
                    ),
                    "path_depth": (
                        len(
                            actions
                        )
                        - 1
                    ),
                    "visited_slots": (
                        result[
                            "visited_slots"
                        ]
                    ),
                    "visited_controls": (
                        result[
                            "visited_controls"
                        ]
                    ),
                    "actions": (
                        actions
                    ),
                }
            )

            progress += 1

        if progress == 0:
            break

        if pass_no > EXPECTED_INVALID_CONTROLS + 2:
            raise RuntimeError(
                "Unexpected augmenting-path pass count."
            )

    action_frames = [
        x[
            "actions"
        ]
        for x in all_paths
        if x.get(
            "found"
        )
        and "actions"
        in x
    ]

    actions_all = (
        pd.concat(
            action_frames,
            ignore_index=True,
        )
        if action_frames
        else pd.DataFrame()
    )

    search_summary = pd.DataFrame(
        [
            {
                k: v
                for k, v
                in x.items()
                if k
                != "actions"
            }
            for x
            in all_paths
        ]
    )

    remaining = [
        s
        for s
        in unmatched_slots
        if slot_to_control[
            s
        ]
        is None
    ]

    return (
        actions_all,
        search_summary,
        remaining,
    )


# =====================================================================
# Final QA
# =====================================================================

def final_matching_qa(
    slot_meta,
    slot_to_control,
    corrected_ids,
    group_registry,
    provider,
):
    slots = list(
        slot_meta
    )

    unmatched = [
        s
        for s
        in slots
        if slot_to_control.get(
            s
        )
        is None
    ]

    controls = [
        slot_to_control[
            s
        ]
        for s
        in slots
        if slot_to_control.get(
            s
        )
        is not None
    ]

    duplicate_controls = (
        len(
            controls
        )
        - len(
            set(
                controls
            )
        )
    )

    outside_core = [
        c
        for c
        in controls
        if c not in corrected_ids
    ]

    rows = []

    edge_failures = []

    spatial_failures = []

    for slot in slots:
        meta = slot_meta[
            slot
        ]

        ctrl = slot_to_control.get(
            slot
        )

        group = (
            group_registry.group(
                ctrl
            )
            if ctrl
            is not None
            else None
        )

        group_fold = (
            group_registry.fold(
                ctrl
            )
            if ctrl
            is not None
            else None
        )

        changed = (
            ctrl
            != meta[
                "original_control_unit_id"
            ]
        )

        rows.append({
            "slot_key": (
                slot
            ),
            "pair_set_id": (
                meta[
                    "pair_key"
                ]
            ),
            "control_rank": (
                meta[
                    "rank_key"
                ]
            ),
            "positive_unit_id": (
                meta[
                    "positive_unit_id"
                ]
            ),
            "pair_fold": (
                meta[
                    "pair_fold"
                ]
            ),
            "original_control_unit_id": (
                meta[
                    "original_control_unit_id"
                ]
            ),
            "final_control_unit_id": (
                ctrl
            ),
            "spatial_group_id": (
                group
            ),
            "spatial_group_fold": (
                group_fold
            ),
            "changed_from_original": (
                changed
            ),
        })

        if ctrl is None:
            continue

        if not group_registry.is_legal_for_fold(
            ctrl,
            meta[
                "pair_fold"
            ],
        ):
            spatial_failures.append(
                {
                    "slot_key": (
                        slot
                    ),
                    "control_unit_id": (
                        ctrl
                    ),
                    "pair_fold": (
                        meta[
                            "pair_fold"
                        ]
                    ),
                    "group": (
                        group
                    ),
                    "group_fold": (
                        group_fold
                    ),
                }
            )

        legal_candidates = {
            c
            for c, _
            in provider.get(
                meta[
                    "positive_unit_id"
                ],
                meta[
                    "pair_fold"
                ],
            )
        }

        if ctrl not in legal_candidates:
            edge_failures.append(
                {
                    "slot_key": (
                        slot
                    ),
                    "positive_unit_id": (
                        meta[
                            "positive_unit_id"
                        ]
                    ),
                    "control_unit_id": (
                        ctrl
                    ),
                    "pair_fold": (
                        meta[
                            "pair_fold"
                        ]
                    ),
                    "reason": (
                        "FINAL_CONTROL_NOT_IN_SPATIAL_LEGAL_P0_EDGE_LIST"
                    ),
                }
            )

    assignment = pd.DataFrame(
        rows
    )

    pair_counts = (
        assignment.groupby(
            "pair_set_id"
        )[
            "final_control_unit_id"
        ]
        .count()
    )

    pair_unique = (
        assignment.groupby(
            "pair_set_id"
        )[
            "final_control_unit_id"
        ]
        .nunique()
    )

    group_cross_fold = (
        assignment[
            assignment[
                "spatial_group_id"
            ].notna()
        ]
        .groupby(
            "spatial_group_id"
        )[
            "pair_fold"
        ]
        .nunique()
    )

    qa = {
        "status": (
            "PASS"
        ),
        "control_slots": int(
            len(
                slots
            )
        ),
        "matched_slots": int(
            len(
                controls
            )
        ),
        "unmatched_slots": int(
            len(
                unmatched
            )
        ),
        "unique_controls": int(
            len(
                set(
                    controls
                )
            )
        ),
        "control_reuse": int(
            duplicate_controls
        ),
        "controls_outside_corrected_core": int(
            len(
                outside_core
            )
        ),
        "spatial_legality_failures": int(
            len(
                spatial_failures
            )
        ),
        "P0_edge_legality_failures": int(
            len(
                edge_failures
            )
        ),
        "pair_sets": int(
            assignment[
                "pair_set_id"
            ].nunique()
        ),
        "pairs_not_exactly_two_controls": int(
            (
                pair_counts
                != 2
            ).sum()
        ),
        "pairs_not_two_unique_controls": int(
            (
                pair_unique
                != 2
            ).sum()
        ),
        "spatial_groups_cross_fold": int(
            (
                group_cross_fold
                > 1
            ).sum()
        ),
        "changed_slots_vs_original": int(
            assignment[
                "changed_from_original"
            ].sum()
        ),
    }

    if (
        qa[
            "control_slots"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "matched_slots"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "unmatched_slots"
        ]
        != 0
        or qa[
            "unique_controls"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "control_reuse"
        ]
        != 0
        or qa[
            "controls_outside_corrected_core"
        ]
        != 0
        or qa[
            "spatial_legality_failures"
        ]
        != 0
        or qa[
            "P0_edge_legality_failures"
        ]
        != 0
        or qa[
            "pair_sets"
        ]
        != EXPECTED_PAIRSETS
        or qa[
            "pairs_not_exactly_two_controls"
        ]
        != 0
        or qa[
            "pairs_not_two_unique_controls"
        ]
        != 0
        or qa[
            "spatial_groups_cross_fold"
        ]
        != 0
    ):
        qa[
            "status"
        ] = "FAIL"

    return (
        assignment,
        pd.DataFrame(
            spatial_failures
        ),
        pd.DataFrame(
            edge_failures
        ),
        qa,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "HC-BUG-03S 16A spatial-group recovery + rematch started"
    )

    # -------------------------------------------------------------
    # A0. Canonical split anchor.
    # -------------------------------------------------------------
    split = load_canonical_split()

    write_json(
        OUT
        / "00_CANONICAL_SPATIAL_SPLIT_ANCHOR.json",
        {
            "pair_fold_source": str(
                CANON_PAIR_FOLD
            ),
            "unit_group_source": str(
                CANON_UNIT_FOLD
            ),
            "old_training_units": (
                EXPECTED_OLD_TRAIN_UNITS
            ),
            "old_spatial_groups": (
                split[
                    "old_group_count"
                ]
            ),
            "fold_pair_counts": (
                EXPECTED_FOLD_COUNTS
            ),
        },
    )

    # -------------------------------------------------------------
    # A1. Search exact spatial_group_id registry.
    # -------------------------------------------------------------
    log(
        "Searching exact unit_id + spatial_group_id registries..."
    )

    discovery = (
        search_exact_group_registry_candidates()
    )

    discovery.to_csv(
        OUT
        / "01_EXACT_SPATIAL_GROUP_REGISTRY_DISCOVERY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    registry_attempts = []

    valid_registries = []

    if not discovery.empty:
        for _, r in discovery.iterrows():
            q, audit = validate_exact_group_registry(
                Path(
                    r[
                        "path"
                    ]
                ),
                split[
                    "old_unit_groups"
                ],
            )

            audit[
                "discovery_score"
            ] = int(
                r[
                    "score"
                ]
            )

            audit[
                "discovery_row_count"
            ] = (
                None
                if pd.isna(
                    r[
                        "row_count"
                    ]
                )
                else int(
                    r[
                        "row_count"
                    ]
                )
            )

            registry_attempts.append(
                audit
            )

            if (
                q is not None
                and audit[
                    "status"
                ]
                == "PASS_EXACT_OLD_15168_REPRODUCTION"
                and len(
                    q
                )
                > EXPECTED_OLD_TRAIN_UNITS
            ):
                valid_registries.append(
                    (
                        audit,
                        q,
                    )
                )

    attempts_df = pd.DataFrame(
        registry_attempts
    )

    attempts_df.to_csv(
        OUT
        / "02_EXACT_REGISTRY_VALIDATION.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # A2. Always collect code lineage as provenance.
    # -------------------------------------------------------------
    log(
        "Searching source code lineage for spatial_group_id generation..."
    )

    code_hits = source_code_lineage_search()

    code_hits.to_csv(
        OUT
        / "03_SPATIAL_GROUP_SOURCE_CODE_LINEAGE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not valid_registries:
        final = {
            "FINAL_DECISION": (
                "STOP_NO_FULL_EXACT_SPATIAL_GROUP_REGISTRY__"
                "REPLAY_CANONICAL_GROUP_GENERATOR_REQUIRED"
            ),
            "exact_registry_candidates_found": int(
                len(
                    discovery
                )
            ),
            "exact_old_15168_reproducing_full_registries": (
                0
            ),
            "source_code_hits": int(
                len(
                    code_hits
                )
            ),
            "interpretation": (
                "Do not infer spatial_group_id from arbitrary component "
                "columns or nearest-neighbor heuristics. The exact canonical "
                "spatial-group generator must be replayed before rematching."
            ),
        }

        write_json(
            OUT
            / "99_FINAL_DECISION.json",
            final,
        )

        print(
            "\n"
            + "=" * 118
        )

        print(
            "HC-BUG-03S 16A STOPPED SAFELY"
        )

        print(
            "=" * 118
        )

        print(
            "FINAL_DECISION:",
            final[
                "FINAL_DECISION"
            ],
        )

        print()

        print(
            "Please send:"
        )

        for name in [
            "01_EXACT_SPATIAL_GROUP_REGISTRY_DISCOVERY.csv",
            "02_EXACT_REGISTRY_VALIDATION.csv",
            "03_SPATIAL_GROUP_SOURCE_CODE_LINEAGE.csv",
            "99_FINAL_DECISION.json",
        ]:
            print(
                OUT
                / name
            )

        print(
            "=" * 118
        )

        return 0

    # Prefer most complete/highest score exact lineage reproduction.
    valid_registries.sort(
        key=lambda x: (
            int(
                x[
                    0
                ].get(
                    "registry_rows",
                    0,
                )
            ),
            int(
                x[
                    0
                ].get(
                    "discovery_score",
                    0,
                )
            ),
        ),
        reverse=True,
    )

    registry_audit, registry = (
        valid_registries[
            0
        ]
    )

    write_json(
        OUT
        / "04_LOCKED_FULL_SPATIAL_GROUP_REGISTRY.json",
        registry_audit,
    )

    log(
        "Locked exact spatial-group registry: "
        f"{registry_audit['path']}"
    )

    # -------------------------------------------------------------
    # B1. Corrected core + original formal assignment.
    # -------------------------------------------------------------
    core = load_corrected_core()

    (
        formal,
        hard,
        invalid,
        positive_ids,
    ) = load_original_slots(
        core[
            "corrected_ids"
        ],
        split[
            "pair_to_fold"
        ],
    )

    invalid.to_csv(
        OUT
        / "05_ORIGINAL_23_INVALID_CONTROL_SLOTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # B2. Ensure trusted registry covers every original slot and all P0
    # candidates that we will actually use are assigned exact groups.
    # -------------------------------------------------------------
    group_registry = GroupRegistry(
        registry,
        split[
            "old_group_to_fold"
        ],
    )

    original_valid_controls = hard.loc[
        hard[
            "CURRENT_VALID"
        ],
        "unit_id",
    ].tolist()

    original_valid_missing_group = [
        c
        for c
        in original_valid_controls
        if group_registry.group(
            c
        )
        is None
    ]

    if original_valid_missing_group:
        raise RuntimeError(
            "Trusted registry fails to cover original valid controls: "
            f"{len(original_valid_missing_group)}"
        )

    # -------------------------------------------------------------
    # B3. Start from original 10,089 valid controls.
    # -------------------------------------------------------------
    (
        slot_meta,
        slot_to_control,
        control_to_slot,
        unmatched,
    ) = build_starting_matching(
        hard
    )

    provider = CandidateProvider(
        corrected_ids=(
            core[
                "corrected_ids"
            ]
        ),
        positive_ids=(
            positive_ids
        ),
        group_registry=(
            group_registry
        ),
    )

    # -------------------------------------------------------------
    # B4. Direct spatial-legal free replacements.
    # -------------------------------------------------------------
    log(
        "Solving direct free spatial-legal replacements for 23 gaps..."
    )

    (
        direct_plan,
        direct_counts,
        remaining,
    ) = direct_free_repair(
        unmatched,
        slot_meta,
        slot_to_control,
        control_to_slot,
        provider,
    )

    direct_counts.to_csv(
        OUT
        / "06_DIRECT_SPATIAL_LEGAL_CANDIDATE_COUNTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    direct_plan.to_csv(
        OUT
        / "07_DIRECT_SPATIAL_LEGAL_REPLACEMENTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"Direct spatial-legal replacements: "
        f"{len(direct_plan)}/{EXPECTED_INVALID_CONTROLS}; "
        f"remaining gaps={len(remaining)}"
    )

    # -------------------------------------------------------------
    # B5. Augmenting paths for any remaining gaps.
    # -------------------------------------------------------------
    if remaining:
        log(
            "Searching spatial-constrained slot-level augmenting paths..."
        )

    (
        augment_actions,
        augment_search,
        final_remaining,
    ) = complete_spatial_matching(
        unmatched_slots=(
            unmatched
        ),
        slot_meta=(
            slot_meta
        ),
        slot_to_control=(
            slot_to_control
        ),
        control_to_slot=(
            control_to_slot
        ),
        provider=(
            provider
        ),
    )

    augment_search.to_csv(
        OUT
        / "08_AUGMENTING_PATH_SEARCH_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    augment_actions.to_csv(
        OUT
        / "09_SPATIAL_CONSTRAINED_AUGMENTING_PATH_ACTIONS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # B6. Final exact matching QA.
    # -------------------------------------------------------------
    (
        assignment,
        spatial_fail,
        edge_fail,
        final_qa,
    ) = final_matching_qa(
        slot_meta=(
            slot_meta
        ),
        slot_to_control=(
            slot_to_control
        ),
        corrected_ids=(
            core[
                "corrected_ids"
            ]
        ),
        group_registry=(
            group_registry
        ),
        provider=(
            provider
        ),
    )

    assignment.to_csv(
        OUT
        / "10_DRYRUN_SPATIAL_CONSTRAINED_CONTROL_ASSIGNMENT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assignment.to_parquet(
        OUT
        / "10_DRYRUN_SPATIAL_CONSTRAINED_CONTROL_ASSIGNMENT.parquet",
        index=False,
    )

    spatial_fail.to_csv(
        OUT
        / "10B_SPATIAL_FAILURES.csv",
        index=False,
        encoding="utf-8-sig",
    )

    edge_fail.to_csv(
        OUT
        / "10C_P0_EDGE_FAILURES.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_qa[
        "remaining_unmatched_after_solver"
    ] = len(
        final_remaining
    )

    final_qa[
        "direct_replacements"
    ] = int(
        len(
            direct_plan
        )
    )

    final_qa[
        "augmenting_path_action_rows"
    ] = int(
        len(
            augment_actions
        )
    )

    final_qa[
        "candidate_provider"
    ] = provider.stats()

    write_json(
        OUT
        / "11_FINAL_SPATIAL_MATCHING_QA.json",
        final_qa,
    )

    if (
        len(
            final_remaining
        )
        == 0
        and final_qa[
            "status"
        ]
        == "PASS"
    ):
        decision = (
            "PASS_SPATIAL_CONSTRAINED_FULL_1TO2_MATCHING__"
            "READY_FOR_HC_BUG03S_CONTROLLED_REBUILD"
        )

    else:
        decision = (
            "STOP_SPATIAL_CONSTRAINED_MATCHING_INCOMPLETE__"
            "DO_NOT_PROMOTE_HC_BUG03_REPAIR"
        )

    final = {
        "FINAL_DECISION": (
            decision
        ),
        "trusted_spatial_group_registry": (
            registry_audit
        ),
        "formal_positive_semantics_unchanged": (
            "AJG1 == 1 AND GSI == 1"
        ),
        "corrected_HC00_core": int(
            len(
                core[
                    "corrected_ids"
                ]
            )
        ),
        "invalid_original_formal_controls": int(
            len(
                invalid
            )
        ),
        "direct_spatial_legal_replacements": int(
            len(
                direct_plan
            )
        ),
        "remaining_after_direct": int(
            len(
                remaining
            )
        ),
        "remaining_after_augmenting_solver": int(
            len(
                final_remaining
            )
        ),
        "final_matching_QA": (
            final_qa
        ),
        "old_frozen_assets_overwritten": (
            False
        ),
        "important": (
            "This dry-run supersedes the earlier non-spatial HC-BUG-03 "
            "replacement plan only if FINAL_DECISION is PASS."
        ),
    }

    write_json(
        OUT
        / "99_FINAL_DECISION.json",
        final,
    )

    # Human report.
    report = f"""# HC-BUG-03S Spatial-Constrained Recovery and Rematch

## Final decision

`{decision}`

## Trusted spatial-group lineage

```json
{json.dumps(registry_audit, ensure_ascii=False, indent=2)}
```

The registry was accepted only because it reproduced the old canonical
`spatial_group_id` values exactly for all 15,168 original training units.

## Corrected formal state

- corrected HC00 core: {len(core['corrected_ids'])}
- invalid original hard-control slots: {len(invalid)}
- pair sets: {EXPECTED_PAIRSETS}
- formal control slots: {EXPECTED_CONTROLS}

Positive semantics remain:

`AJG1 == 1 AND GSI == 1`

## Spatial-constrained rematching

- direct free replacements: {len(direct_plan)}
- remaining gaps after direct phase: {len(remaining)}
- remaining gaps after augmenting solver: {len(final_remaining)}

## Final QA

```json
{json.dumps(final_qa, ensure_ascii=False, indent=2)}
```

No formal dataset was overwritten.
"""

    (
        OUT
        / "00_HC_BUG03S_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "HC-BUG-03S / 16A COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "TRUSTED FULL GROUP REGISTRY:",
        registry_audit[
            "path"
        ],
    )

    print(
        "OLD GROUP REPRODUCTION:",
        registry_audit[
            "old_group_exact_match"
        ],
        "/",
        EXPECTED_OLD_TRAIN_UNITS,
    )

    print()

    print(
        "INVALID ORIGINAL CONTROLS:",
        len(
            invalid
        ),
    )

    print(
        "DIRECT SPATIAL-LEGAL REPLACEMENTS:",
        len(
            direct_plan
        ),
        "/",
        EXPECTED_INVALID_CONTROLS,
    )

    print(
        "REMAINING AFTER DIRECT:",
        len(
            remaining
        ),
    )

    print(
        "REMAINING AFTER AUGMENTING SOLVER:",
        len(
            final_remaining
        ),
    )

    print()

    print(
        "FINAL MATCHED SLOTS:",
        final_qa[
            "matched_slots"
        ],
    )

    print(
        "UNMATCHED:",
        final_qa[
            "unmatched_slots"
        ],
    )

    print(
        "CONTROL REUSE:",
        final_qa[
            "control_reuse"
        ],
    )

    print(
        "SPATIAL LEGALITY FAILURES:",
        final_qa[
            "spatial_legality_failures"
        ],
    )

    print(
        "SPATIAL GROUPS CROSS-FOLD:",
        final_qa[
            "spatial_groups_cross_fold"
        ],
    )

    print(
        "P0 EDGE FAILURES:",
        final_qa[
            "P0_edge_legality_failures"
        ],
    )

    print(
        "CHANGED SLOTS VS ORIGINAL:",
        final_qa[
            "changed_slots_vs_original"
        ],
    )

    print(
        "FINAL QA:",
        final_qa[
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
        "00_HC_BUG03S_REPORT.md",
        "01_EXACT_SPATIAL_GROUP_REGISTRY_DISCOVERY.csv",
        "02_EXACT_REGISTRY_VALIDATION.csv",
        "03_SPATIAL_GROUP_SOURCE_CODE_LINEAGE.csv",
        "04_LOCKED_FULL_SPATIAL_GROUP_REGISTRY.json",
        "06_DIRECT_SPATIAL_LEGAL_CANDIDATE_COUNTS.csv",
        "08_AUGMENTING_PATH_SEARCH_SUMMARY.csv",
        "09_SPATIAL_CONSTRAINED_AUGMENTING_PATH_ACTIONS.csv",
        "11_FINAL_SPATIAL_MATCHING_QA.json",
        "99_FINAL_DECISION.json",
    ]:
        p = OUT / name

        if p.exists():
            print(
                p
            )

    print(
        "=" * 120
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
