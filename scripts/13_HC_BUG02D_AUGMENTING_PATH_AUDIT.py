# -*- coding: utf-8 -*-
r"""
13_HC_BUG02D_AUGMENTING_PATH_AUDIT.py
=====================================
HC-BUG-02D: slot-level augmenting-path audit for the one blocked pair.

READ ONLY.
NO model training.
NO formal asset mutation.
NO pair-table overwrite.

Background
----------
HC-BUG-02C established:

- formal positive semantics remain:
      AJG1 == 1 AND GSI == 1

- current HC00 core: 28,102
- remove upstream-ineligible core candidates: 30
- corrected HC00 core: 28,072
- invalid formal hard controls: 23
- affected pair sets: 23
- 22/23 invalid controls have direct unused replacements
- exactly one pair has zero direct unused replacement while all other valid
  controls are frozen.

This script asks the next exact question:

Can the remaining missing control be repaired by an alternating exchange chain?

Example:
    blocked slot A needs control c1
    c1 is currently owned by slot B
    B switches to c2
    c2 is currently owned by slot C
    C switches to a free control c3

This is an augmenting path in the bipartite graph:
    left  = formal control slots (pair_set_id, control_rank)
    right = corrected legal control units
    edges = locked production P0 candidate edges

Why SLOT-level rather than pair-level
-------------------------------------
Each pair has two control slots. Treating only the pair as one node can
incorrectly forbid valid exchanges involving the pair's other current
control. Slot-level matching is the exact bipartite formulation.

Candidate universe
------------------
Locked production P0 graph:
<PROJECT_ROOT>\data\06_hard_control\
01c_matching_sensitivity_audit\02_edges\
hard_control_01b_p0_parent20km_superset_edges.parquet

Starting partial matching
-------------------------
1. Remove all 23 invalid formal controls.
2. Apply the 22 direct replacements found by HC-BUG-02C in memory.
3. Keep the one blocked slot unmatched.
4. Search an augmenting path from that unmatched slot.

Search protocol
---------------
Phase 1: shortest-path BFS up to depth 10.
Phase 2: if no short path exists, exhaustive BFS over every reachable formal
         slot in the locked corrected graph.

If exhaustive BFS finds no augmenting path, then the current partial matching
of size 10,111 is maximum in this exact corrected P0 graph. In a bipartite
matching problem, that means no complete 10,112-control / 5,056-pair 1:2
matching exists under this exact graph and candidate restrictions.

Even if PASS:
DO NOT write the dry-run chain directly into the frozen dataset.
Production repair must still patch HC00 and rerun the formal matching pipeline.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "experiments"
    / "HC_BUG_02D_AUGMENTING_PATH_AUDIT"
)

P0_EDGES = (
    ROOT
    / "data"
    / "06_hard_control"
    / "01c_matching_sensitivity_audit"
    / "02_edges"
    / "hard_control_01b_p0_parent20km_superset_edges.parquet"
)

FORMAL_LABEL = (
    ROOT
    / "data"
    / "06_hard_control"
    / "02_formal_pair_assignment"
    / "04_label_layer"
    / "hard_control_02_full_grid_label_layer.parquet"
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

HC02C_DIR = (
    ROOT
    / "experiments"
    / "HC_BUG_02C_LOCKED_P0_REPLACEMENT_AUDIT"
)

HC02C_PLAN = (
    HC02C_DIR
    / "09_DRYRUN_23_REPLACEMENT_PLAN.csv"
)

HC02C_FINAL = (
    HC02C_DIR
    / "10_FINAL_DECISION.json"
)

# State guards already established by HC-BUG-01/02/02C.
EXPECTED_UPSTREAM_ELIGIBLE = 29084
EXPECTED_CURRENT_CORE = 28102
EXPECTED_REMOVED_CORE = 30
EXPECTED_CORRECTED_CORE = 28072

EXPECTED_CONTROLS = 10112
EXPECTED_PAIRSETS = 5056
EXPECTED_INVALID_CONTROLS = 23
EXPECTED_DIRECT_REPLACEMENTS = 22
EXPECTED_UNMATCHED_SLOTS_AFTER_22 = 1

SHORT_DEPTH_LIMIT = 10

# Dataset query batch size. Each positive has ~hundreds of edges.
POSITIVE_QUERY_BATCH = 128

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
        OUT / "logs" / "audit.log"
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


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def get_schema(path: Path):
    if not path.exists():
        return []

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


def normalize_unique_unit(
    df,
    name,
):
    if "unit_id" not in df.columns:
        raise RuntimeError(
            f"{name} missing unit_id"
        )

    if df["unit_id"].isna().any():
        raise RuntimeError(
            f"{name} has null unit_id"
        )

    dup = int(
        df[
            "unit_id"
        ]
        .duplicated()
        .sum()
    )

    if dup:
        raise RuntimeError(
            f"{name} has {dup} duplicate unit_ids"
        )

    return df


# =====================================================================
# Load corrected core
# =====================================================================

def load_corrected_core():
    elig_path = first_existing(
        ELIGIBILITY_CANDIDATES
    )

    core_path = first_existing(
        HC00_CORE_CANDIDATES
    )

    if elig_path is None:
        raise RuntimeError(
            "Frozen upstream eligibility not found."
        )

    if core_path is None:
        raise RuntimeError(
            "Current HC00 core not found."
        )

    elig_cols = get_schema(
        elig_path
    )

    wanted = [
        c
        for c in [
            "unit_id",
            "primary_eligibility_status",
            "control_exclusion_flag",
            "inventory_evidence_exclusion_flag",
        ]
        if c in elig_cols
    ]

    eligibility = normalize_unique_unit(
        read_table(
            elig_path,
            wanted,
        ),
        "upstream eligibility",
    )

    if (
        "primary_eligibility_status"
        not in eligibility.columns
    ):
        raise RuntimeError(
            "Missing primary_eligibility_status"
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
            "Frozen eligibility internally contradictory: "
            f"{int(contradiction.sum())}"
        )

    core = normalize_unique_unit(
        read_table(
            core_path,
            ["unit_id"],
        ),
        "HC00 core",
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

    summary = {
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
        "eligibility_table": str(
            elig_path
        ),
        "core_table": str(
            core_path
        ),
    }

    return (
        corrected_ids,
        removed_ids,
        summary,
    )


# =====================================================================
# Load formal control slots
# =====================================================================

def load_formal_slots(
    corrected_ids,
):
    if not FORMAL_LABEL.exists():
        raise RuntimeError(
            f"Missing formal label layer: {FORMAL_LABEL}"
        )

    schema = get_schema(
        FORMAL_LABEL
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

    formal = normalize_unique_unit(
        read_table(
            FORMAL_LABEL,
            wanted,
        ),
        "formal label layer",
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
            "Formal positive flag unavailable."
        )

    formal[
        "IS_HARD_CONTROL"
    ] = is_hard

    formal[
        "IS_POSITIVE"
    ] = is_positive

    hard = formal[
        formal[
            "IS_HARD_CONTROL"
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
            "Formal hard controls lack slot metadata: "
            f"{required - set(hard.columns)}"
        )

    hard[
        "PAIR_KEY"
    ] = hard[
        "pair_set_id"
    ].map(
        normalize_pair_id
    )

    hard[
        "RANK_KEY"
    ] = hard[
        "control_rank"
    ].map(
        normalize_rank
    )

    if (
        hard[
            [
                "PAIR_KEY",
                "RANK_KEY",
                "matched_positive_unit_id",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Formal hard controls contain null pair/rank/positive metadata."
        )

    hard[
        "SLOT_KEY"
    ] = [
        make_slot_key(
            p,
            r,
        )
        for p, r in zip(
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
            "Formal control slot keys are not unique."
        )

    pair_counts = (
        hard.groupby(
            "PAIR_KEY"
        )
        .size()
    )

    if (
        pair_counts
        != 2
    ).any():
        bad = pair_counts[
            pair_counts
            != 2
        ]

        raise RuntimeError(
            "Formal pair does not have exactly two controls:\n"
            + bad.to_string()
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

    formal_positive_ids = set(
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
        formal_positive_ids,
    )


# =====================================================================
# Build the 10,111-edge partial matching from HC-BUG-02C
# =====================================================================

def load_02c_plan():
    if not HC02C_PLAN.exists():
        raise RuntimeError(
            f"Missing HC-BUG-02C plan:\n{HC02C_PLAN}"
        )

    plan = pd.read_csv(
        HC02C_PLAN,
        low_memory=False,
    )

    required = {
        "pair_set_id",
        "replacement_found",
        "replacement_unit_id",
        "illegal_control_unit_id",
        "control_rank",
    }

    if not required.issubset(
        set(
            plan.columns
        )
    ):
        raise RuntimeError(
            "02C plan missing fields: "
            f"{required - set(plan.columns)}"
        )

    plan[
        "PAIR_KEY"
    ] = plan[
        "pair_set_id"
    ].map(
        normalize_pair_id
    )

    plan[
        "RANK_KEY"
    ] = plan[
        "control_rank"
    ].map(
        normalize_rank
    )

    plan[
        "SLOT_KEY"
    ] = [
        make_slot_key(
            p,
            r,
        )
        for p, r in zip(
            plan[
                "PAIR_KEY"
            ],
            plan[
                "RANK_KEY"
            ],
        )
    ]

    found = truthy(
        plan[
            "replacement_found"
        ]
    )

    plan[
        "REPLACEMENT_FOUND_BOOL"
    ] = found

    return plan


def build_partial_matching(
    hard,
    invalid,
    plan,
    corrected_ids,
):
    if len(
        invalid
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Unexpected invalid control count: "
            f"{len(invalid)}"
        )

    if len(
        plan
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Unexpected 02C plan row count: "
            f"{len(plan)}"
        )

    invalid_slots = set(
        invalid[
            "SLOT_KEY"
        ].tolist()
    )

    plan_slots = set(
        plan[
            "SLOT_KEY"
        ].tolist()
    )

    if (
        invalid_slots
        != plan_slots
    ):
        missing = (
            invalid_slots
            - plan_slots
        )

        extra = (
            plan_slots
            - invalid_slots
        )

        raise RuntimeError(
            "02C plan does not exactly match invalid slots.\n"
            f"missing={missing}\n"
            f"extra={extra}"
        )

    direct = plan[
        plan[
            "REPLACEMENT_FOUND_BOOL"
        ]
    ].copy()

    blocked = plan[
        ~plan[
            "REPLACEMENT_FOUND_BOOL"
        ]
    ].copy()

    if len(
        direct
    ) != EXPECTED_DIRECT_REPLACEMENTS:
        raise RuntimeError(
            "Expected 22 direct replacements, got "
            f"{len(direct)}"
        )

    if len(
        blocked
    ) != EXPECTED_UNMATCHED_SLOTS_AFTER_22:
        raise RuntimeError(
            "Expected one blocked slot, got "
            f"{len(blocked)}"
        )

    # slot metadata for all 10,112 formal control slots.
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
            "pair_key": r[
                "PAIR_KEY"
            ],
            "rank_key": r[
                "RANK_KEY"
            ],
            "positive_unit_id": str(
                r[
                    "matched_positive_unit_id"
                ]
            ),
            "original_control_unit_id": r[
                "unit_id"
            ],
        }

        # Invalid controls are removed first.
        if bool(
            r[
                "CURRENT_VALID"
            ]
        ):
            ctrl = r[
                "unit_id"
            ]

            slot_to_control[
                slot
            ] = ctrl

            if ctrl in control_to_slot:
                raise RuntimeError(
                    "Formal valid control reuse detected before repair: "
                    f"{ctrl}"
                )

            control_to_slot[
                ctrl
            ] = slot

        else:
            slot_to_control[
                slot
            ] = None

    # Apply 22 direct replacements.
    direct_rows = []

    for _, r in direct.iterrows():
        slot = r[
            "SLOT_KEY"
        ]

        ctrl = r[
            "replacement_unit_id"
        ]

        if pd.isna(
            ctrl
        ):
            raise RuntimeError(
                f"Direct replacement missing unit_id for {slot}"
            )

        if ctrl not in corrected_ids:
            raise RuntimeError(
                f"02C direct replacement not in corrected core: {ctrl}"
            )

        if ctrl in control_to_slot:
            raise RuntimeError(
                "02C direct replacement already occupied by another slot: "
                f"{ctrl} -> {control_to_slot[ctrl]}"
            )

        slot_to_control[
            slot
        ] = ctrl

        control_to_slot[
            ctrl
        ] = slot

        direct_rows.append({
            "slot_key": slot,
            "pair_set_id": r[
                "PAIR_KEY"
            ],
            "control_rank": r[
                "RANK_KEY"
            ],
            "replacement_unit_id": ctrl,
        })

    unmatched_slots = [
        s
        for s, c
        in slot_to_control.items()
        if c is None
    ]

    if len(
        unmatched_slots
    ) != EXPECTED_UNMATCHED_SLOTS_AFTER_22:
        raise RuntimeError(
            "Partial matching should have exactly one unmatched slot; got "
            f"{len(unmatched_slots)}"
        )

    root_slot = unmatched_slots[
        0
    ]

    occupied_count = len(
        control_to_slot
    )

    if occupied_count != (
        EXPECTED_CONTROLS
        - 1
    ):
        raise RuntimeError(
            "Expected 10,111 occupied controls in partial matching, got "
            f"{occupied_count}"
        )

    if len(
        set(
            control_to_slot.keys()
        )
    ) != occupied_count:
        raise RuntimeError(
            "Duplicate control in partial matching."
        )

    return {
        "slot_meta": slot_meta,
        "slot_to_control": slot_to_control,
        "control_to_slot": control_to_slot,
        "root_slot": root_slot,
        "direct_rows": pd.DataFrame(
            direct_rows
        ),
        "blocked_row": blocked.iloc[
            0
        ].to_dict(),
    }


# =====================================================================
# Candidate provider: query the locked P0 edge graph lazily by positive.
# =====================================================================

class CandidateProvider:
    def __init__(
        self,
        path: Path,
        corrected_ids,
    ):
        if not path.exists():
            raise RuntimeError(
                f"P0 edge table missing: {path}"
            )

        import pyarrow.dataset as ds

        self.ds_module = ds

        self.dataset = ds.dataset(
            str(path),
            format="parquet",
        )

        schema = set(
            self.dataset.schema.names
        )

        required = {
            "positive_unit_id",
            "candidate_unit_id",
        }

        if not required.issubset(
            schema
        ):
            raise RuntimeError(
                "P0 candidate table missing: "
                f"{required - schema}"
            )

        self.order_col = (
            "composite_distance"
            if "composite_distance" in schema
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

        self.cache = {}

        self.query_calls = 0
        self.raw_rows_read = 0
        self.legal_rows_cached = 0

    def ensure(
        self,
        positive_ids,
    ):
        missing = [
            str(p)
            for p in positive_ids
            if str(p)
            not in self.cache
        ]

        if not missing:
            return

        # Batch predicate-pushdown queries.
        for i in range(
            0,
            len(missing),
            POSITIVE_QUERY_BATCH,
        ):
            batch = missing[
                i:
                i
                + POSITIVE_QUERY_BATCH
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
                    columns=self.columns,
                    filter=filt,
                )
            )

            df = (
                table
                .to_pandas()
            )

            self.query_calls += 1

            self.raw_rows_read += len(
                df
            )

            if df.empty:
                for p in batch:
                    self.cache[
                        p
                    ] = []

                continue

            df[
                "positive_unit_id"
            ] = (
                df[
                    "positive_unit_id"
                ]
                .astype(str)
            )

            df = df[
                df[
                    "candidate_unit_id"
                ]
                .isin(
                    self.corrected_ids
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
                        INF / 100
                    )
                )

            else:
                df[
                    "_ORDER"
                ] = 0.0

            # Deduplicate same positive-candidate edge.
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

            grouped = {
                p: list(
                    zip(
                        g[
                            "candidate_unit_id"
                        ].tolist(),
                        g[
                            "_ORDER"
                        ].astype(float)
                        .tolist(),
                    )
                )
                for p, g
                in df.groupby(
                    "positive_unit_id"
                )
            }

            for p in batch:
                vals = grouped.get(
                    p,
                    [],
                )

                self.cache[
                    p
                ] = vals

                self.legal_rows_cached += len(
                    vals
                )

    def get(
        self,
        positive_id,
    ):
        p = str(
            positive_id
        )

        self.ensure(
            [p]
        )

        return self.cache[
            p
        ]

    def stats(
        self,
    ):
        return {
            "query_calls": (
                self.query_calls
            ),
            "unique_positives_cached": len(
                self.cache
            ),
            "raw_rows_read": (
                self.raw_rows_read
            ),
            "legal_corrected_edges_cached": (
                self.legal_rows_cached
            ),
            "order_column": (
                self.order_col
            ),
        }


# =====================================================================
# Exact slot-level augmenting path search
# =====================================================================

def augmenting_path_bfs(
    root_slot,
    slot_meta,
    slot_to_control,
    control_to_slot,
    provider,
    max_depth=None,
):
    """
    BFS in alternating graph.

    Slot -> candidate control:
        any corrected P0 edge.

    Candidate control -> slot:
        if currently occupied, traverse to the slot that owns it.
        if free, augmenting path terminates.

    depth[slot] counts how many occupied controls have been displaced before
    arriving at that deficient slot.

    If root takes a free control directly:
        found_depth = 0
        touched_slots = 1

    If root takes B's control and B takes a free control:
        found_depth = 1
        touched_slots = 2
    """

    parent_slot = {
        root_slot: None
    }

    parent_control = {
        root_slot: None
    }

    depth = {
        root_slot: 0
    }

    queue = deque(
        [root_slot]
    )

    visited_controls = set()

    depth_visit_counts = defaultdict(
        int
    )

    terminal_slot = None
    terminal_free_control = None
    terminal_free_cost = None

    while queue:
        # Layer-batch load for better parquet pushdown efficiency.
        current_depth = depth[
            queue[0]
        ]

        if (
            max_depth is not None
            and current_depth > max_depth
        ):
            break

        layer_slots = []

        while (
            queue
            and depth[
                queue[0]
            ]
            == current_depth
        ):
            layer_slots.append(
                queue.popleft()
            )

        depth_visit_counts[
            current_depth
        ] += len(
            layer_slots
        )

        positives = {
            slot_meta[
                s
            ][
                "positive_unit_id"
            ]
            for s
            in layer_slots
        }

        provider.ensure(
            positives
        )

        for slot in layer_slots:
            d = depth[
                slot
            ]

            # We may search candidate controls at this slot if its depth is
            # within max_depth. Traversing to another occupied slot would
            # create depth d+1 and is disallowed when that exceeds max_depth.
            posid = slot_meta[
                slot
            ][
                "positive_unit_id"
            ]

            candidates = provider.get(
                posid
            )

            current_control = (
                slot_to_control.get(
                    slot
                )
            )

            for cand, cost in candidates:
                # If this slot is currently matched (because it became
                # deficient through an alternating path), selecting its own
                # current control merely reverses the previous step.
                if (
                    current_control
                    is not None
                    and cand
                    == current_control
                ):
                    continue

                # Candidate-control visits are global for BFS shortest path.
                if cand in visited_controls:
                    continue

                visited_controls.add(
                    cand
                )

                owner_slot = (
                    control_to_slot.get(
                        cand
                    )
                )

                if owner_slot is None:
                    terminal_slot = (
                        slot
                    )

                    terminal_free_control = (
                        cand
                    )

                    terminal_free_cost = (
                        float(
                            cost
                        )
                    )

                    return {
                        "found": True,
                        "found_depth": d,
                        "terminal_slot": (
                            terminal_slot
                        ),
                        "terminal_free_control": (
                            terminal_free_control
                        ),
                        "terminal_free_cost": (
                            terminal_free_cost
                        ),
                        "parent_slot": (
                            parent_slot
                        ),
                        "parent_control": (
                            parent_control
                        ),
                        "depth": depth,
                        "visited_slot_count": len(
                            depth
                        ),
                        "visited_control_count": len(
                            visited_controls
                        ),
                        "depth_visit_counts": dict(
                            sorted(
                                depth_visit_counts.items()
                            )
                        ),
                    }

                if owner_slot in parent_slot:
                    continue

                next_depth = (
                    d + 1
                )

                if (
                    max_depth is not None
                    and next_depth > max_depth
                ):
                    continue

                # parent slot takes 'cand' away from owner_slot,
                # making owner_slot the next deficient slot.
                parent_slot[
                    owner_slot
                ] = slot

                parent_control[
                    owner_slot
                ] = cand

                depth[
                    owner_slot
                ] = next_depth

                queue.append(
                    owner_slot
                )

    return {
        "found": False,
        "found_depth": None,
        "terminal_slot": None,
        "terminal_free_control": None,
        "terminal_free_cost": None,
        "parent_slot": parent_slot,
        "parent_control": parent_control,
        "depth": depth,
        "visited_slot_count": len(
            depth
        ),
        "visited_control_count": len(
            visited_controls
        ),
        "depth_visit_counts": dict(
            sorted(
                depth_visit_counts.items()
            )
        ),
    }


def reconstruct_augmenting_path(
    result,
    root_slot,
    slot_meta,
    slot_to_control,
):
    if not result[
        "found"
    ]:
        return pd.DataFrame()

    terminal = result[
        "terminal_slot"
    ]

    # Reconstruct slot path:
    # root -> ... -> terminal.
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

    if path_slots[
        0
    ] != root_slot:
        raise RuntimeError(
            "Augmenting path reconstruction did not start at root."
        )

    actions = []

    for i, slot in enumerate(
        path_slots
    ):
        old_control = (
            slot_to_control.get(
                slot
            )
        )

        if i < (
            len(
                path_slots
            )
            - 1
        ):
            child = path_slots[
                i + 1
            ]

            new_control = (
                result[
                    "parent_control"
                ][
                    child
                ]
            )

            endpoint_type = (
                "TAKE_OCCUPIED_CONTROL"
            )

        else:
            new_control = result[
                "terminal_free_control"
            ]

            endpoint_type = (
                "TAKE_FREE_CONTROL"
            )

        meta = slot_meta[
            slot
        ]

        actions.append({
            "path_step": i,
            "slot_key": slot,
            "pair_set_id": meta[
                "pair_key"
            ],
            "control_rank": meta[
                "rank_key"
            ],
            "positive_unit_id": meta[
                "positive_unit_id"
            ],
            "old_control_unit_id": (
                old_control
            ),
            "new_control_unit_id": (
                new_control
            ),
            "action_type": (
                endpoint_type
            ),
        })

    return pd.DataFrame(
        actions
    )


# =====================================================================
# Validate path and simulated repaired assignment
# =====================================================================

def validate_path_edges(
    path_df,
    provider,
):
    failures = []

    for _, r in path_df.iterrows():
        posid = str(
            r[
                "positive_unit_id"
            ]
        )

        cand = r[
            "new_control_unit_id"
        ]

        allowed = {
            c
            for c, _
            in provider.get(
                posid
            )
        }

        if cand not in allowed:
            failures.append({
                "slot_key": r[
                    "slot_key"
                ],
                "positive_unit_id": posid,
                "candidate_unit_id": cand,
                "reason": (
                    "NOT_IN_CORRECTED_P0_EDGE_LIST"
                ),
            })

    return pd.DataFrame(
        failures
    )


def apply_path_and_qa(
    path_df,
    slot_meta,
    slot_to_control,
    corrected_ids,
    provider,
):
    new_slot_to_control = dict(
        slot_to_control
    )

    for _, r in path_df.iterrows():
        new_slot_to_control[
            r[
                "slot_key"
            ]
        ] = r[
            "new_control_unit_id"
        ]

    unmatched = [
        s
        for s, c
        in new_slot_to_control.items()
        if c is None
    ]

    controls = [
        c
        for c
        in new_slot_to_control.values()
        if c is not None
    ]

    duplicate_count = (
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

    edge_fail = validate_path_edges(
        path_df,
        provider,
    )

    # Build a user-inspectable dry-run assignment table.
    rows = []

    for slot, meta in slot_meta.items():
        rows.append({
            "slot_key": slot,
            "pair_set_id": meta[
                "pair_key"
            ],
            "control_rank": meta[
                "rank_key"
            ],
            "positive_unit_id": meta[
                "positive_unit_id"
            ],
            "original_control_unit_id": meta[
                "original_control_unit_id"
            ],
            "dryrun_control_unit_id": new_slot_to_control[
                slot
            ],
            "changed_from_original": (
                meta[
                    "original_control_unit_id"
                ]
                != new_slot_to_control[
                    slot
                ]
            ),
        })

    assignment = pd.DataFrame(
        rows
    )

    pair_counts = (
        assignment.groupby(
            "pair_set_id"
        )[
            "dryrun_control_unit_id"
        ]
        .count()
    )

    pair_unique_counts = (
        assignment.groupby(
            "pair_set_id"
        )[
            "dryrun_control_unit_id"
        ]
        .nunique()
    )

    qa = {
        "status": "PASS",

        "slot_count": len(
            new_slot_to_control
        ),

        "matched_slot_count": len(
            controls
        ),

        "unmatched_slot_count": len(
            unmatched
        ),

        "unique_control_count": len(
            set(
                controls
            )
        ),

        "duplicate_control_count": (
            duplicate_count
        ),

        "control_outside_corrected_core_count": len(
            outside_core
        ),

        "path_edge_legality_failures": int(
            len(
                edge_fail
            )
        ),

        "pair_set_count": int(
            assignment[
                "pair_set_id"
            ]
            .nunique()
        ),

        "pairs_not_exactly_two_matched_slots": int(
            (
                pair_counts
                != 2
            ).sum()
        ),

        "pairs_not_two_unique_controls": int(
            (
                pair_unique_counts
                != 2
            ).sum()
        ),

        "total_changed_slots_vs_original": int(
            assignment[
                "changed_from_original"
            ]
            .sum()
        ),
    }

    if (
        qa[
            "slot_count"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "matched_slot_count"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "unmatched_slot_count"
        ]
        != 0
        or qa[
            "unique_control_count"
        ]
        != EXPECTED_CONTROLS
        or qa[
            "duplicate_control_count"
        ]
        != 0
        or qa[
            "control_outside_corrected_core_count"
        ]
        != 0
        or qa[
            "path_edge_legality_failures"
        ]
        != 0
        or qa[
            "pair_set_count"
        ]
        != EXPECTED_PAIRSETS
        or qa[
            "pairs_not_exactly_two_matched_slots"
        ]
        != 0
        or qa[
            "pairs_not_two_unique_controls"
        ]
        != 0
    ):
        qa[
            "status"
        ] = "FAIL"

    return (
        assignment,
        edge_fail,
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
        "HC-BUG-02D augmenting-path audit started"
    )

    if not P0_EDGES.exists():
        raise RuntimeError(
            f"Locked P0 edge table missing:\n{P0_EDGES}"
        )

    # -------------------------------------------------------------
    # 1. Corrected core.
    # -------------------------------------------------------------
    (
        corrected_ids,
        removed_ids,
        core_summary,
    ) = load_corrected_core()

    expected_core = {
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

    core_mismatch = {
        k: {
            "observed": core_summary[
                k
            ],
            "expected": v,
        }
        for k, v
        in expected_core.items()
        if core_summary[
            k
        ] != v
    }

    write_json(
        OUT
        / "00_CORE_STATE.json",
        {
            "core_summary": core_summary,
            "expected": expected_core,
            "mismatch": core_mismatch,
        },
    )

    if core_mismatch:
        final = {
            "FINAL_DECISION": (
                "STOP_CORE_STATE_CHANGED"
            ),
            "mismatch": (
                core_mismatch
            ),
        }

        write_json(
            OUT
            / "07_FINAL_DECISION.json",
            final,
        )

        print(
            "\nFINAL_DECISION:",
            final[
                "FINAL_DECISION"
            ],
        )

        return 0

    # -------------------------------------------------------------
    # 2. Formal 10,112 slots + 23 invalid controls.
    # -------------------------------------------------------------
    (
        formal,
        hard,
        invalid,
        formal_positive_ids,
    ) = load_formal_slots(
        corrected_ids
    )

    if len(
        hard
    ) != EXPECTED_CONTROLS:
        raise RuntimeError(
            "Unexpected formal control count: "
            f"{len(hard)}"
        )

    if hard[
        "PAIR_KEY"
    ].nunique() != EXPECTED_PAIRSETS:
        raise RuntimeError(
            "Unexpected formal pair-set count: "
            f"{hard['PAIR_KEY'].nunique()}"
        )

    if len(
        invalid
    ) != EXPECTED_INVALID_CONTROLS:
        raise RuntimeError(
            "Unexpected invalid formal control count: "
            f"{len(invalid)}"
        )

    # -------------------------------------------------------------
    # 3. Apply 22 direct 02C replacements, leave one slot unmatched.
    # -------------------------------------------------------------
    plan = load_02c_plan()

    partial = build_partial_matching(
        hard,
        invalid,
        plan,
        corrected_ids,
    )

    slot_meta = partial[
        "slot_meta"
    ]

    slot_to_control = partial[
        "slot_to_control"
    ]

    control_to_slot = partial[
        "control_to_slot"
    ]

    root_slot = partial[
        "root_slot"
    ]

    root_meta = slot_meta[
        root_slot
    ]

    partial_summary = {
        "formal_control_slots": len(
            slot_meta
        ),
        "occupied_controls_after_22_direct_replacements": len(
            control_to_slot
        ),
        "unmatched_slots": int(
            sum(
                c is None
                for c
                in slot_to_control.values()
            )
        ),
        "root_slot": (
            root_slot
        ),
        "root_pair_set_id": (
            root_meta[
                "pair_key"
            ]
        ),
        "root_control_rank": (
            root_meta[
                "rank_key"
            ]
        ),
        "root_positive_unit_id": (
            root_meta[
                "positive_unit_id"
            ]
        ),
        "root_original_invalid_control": (
            root_meta[
                "original_control_unit_id"
            ]
        ),
        "direct_replacements_applied": int(
            len(
                partial[
                    "direct_rows"
                ]
            )
        ),
    }

    write_json(
        OUT
        / "01_PARTIAL_MATCHING_STATE.json",
        partial_summary,
    )

    pd.DataFrame(
        [
            {
                "slot_key": (
                    root_slot
                ),
                **root_meta,
            }
        ]
    ).to_csv(
        OUT
        / "02_BLOCKED_ROOT_SLOT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    partial[
        "direct_rows"
    ].to_csv(
        OUT
        / "02B_22_DIRECT_REPLACEMENTS_APPLIED.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # 4. Locked candidate provider.
    # -------------------------------------------------------------
    provider = CandidateProvider(
        P0_EDGES,
        corrected_ids,
    )

    # -------------------------------------------------------------
    # 5. Short-chain search depth <= 10.
    # -------------------------------------------------------------
    log(
        "Searching shortest augmenting path with depth <= 10..."
    )

    short_result = augmenting_path_bfs(
        root_slot,
        slot_meta,
        slot_to_control,
        control_to_slot,
        provider,
        max_depth=SHORT_DEPTH_LIMIT,
    )

    short_summary = {
        "max_depth": (
            SHORT_DEPTH_LIMIT
        ),
        "found": bool(
            short_result[
                "found"
            ]
        ),
        "found_depth": (
            short_result[
                "found_depth"
            ]
        ),
        "visited_slot_count": (
            short_result[
                "visited_slot_count"
            ]
        ),
        "visited_control_count": (
            short_result[
                "visited_control_count"
            ]
        ),
        "depth_visit_counts": (
            short_result[
                "depth_visit_counts"
            ]
        ),
        "provider_stats_after_short_search": (
            provider.stats()
        ),
    }

    write_json(
        OUT
        / "03_SHORT_CHAIN_SEARCH.json",
        short_summary,
    )

    final_search = (
        short_result
    )

    search_mode = (
        "SHORT_BFS_DEPTH_LE_10"
    )

    # -------------------------------------------------------------
    # 6. If needed, exhaustive search of all reachable formal slots.
    # -------------------------------------------------------------
    if not short_result[
        "found"
    ]:
        log(
            "No augmenting path within depth 10. "
            "Running exhaustive reachable-slot BFS..."
        )

        full_result = augmenting_path_bfs(
            root_slot,
            slot_meta,
            slot_to_control,
            control_to_slot,
            provider,
            max_depth=None,
        )

        final_search = (
            full_result
        )

        search_mode = (
            "EXHAUSTIVE_REACHABLE_SLOT_BFS"
        )

        full_summary = {
            "found": bool(
                full_result[
                    "found"
                ]
            ),
            "found_depth": (
                full_result[
                    "found_depth"
                ]
            ),
            "visited_slot_count": (
                full_result[
                    "visited_slot_count"
                ]
            ),
            "visited_control_count": (
                full_result[
                    "visited_control_count"
                ]
            ),
            "depth_visit_counts": (
                full_result[
                    "depth_visit_counts"
                ]
            ),
            "provider_stats_after_full_search": (
                provider.stats()
            ),
        }

        write_json(
            OUT
            / "03B_EXHAUSTIVE_SEARCH.json",
            full_summary,
        )

    # -------------------------------------------------------------
    # 7. If found, reconstruct + simulate exact augmented matching.
    # -------------------------------------------------------------
    if final_search[
        "found"
    ]:
        path_df = reconstruct_augmenting_path(
            final_search,
            root_slot,
            slot_meta,
            slot_to_control,
        )

        path_df.to_csv(
            OUT
            / "04_AUGMENTING_PATH.csv",
            index=False,
            encoding="utf-8-sig",
        )

        (
            assignment,
            edge_fail,
            qa,
        ) = apply_path_and_qa(
            path_df,
            slot_meta,
            slot_to_control,
            corrected_ids,
            provider,
        )

        assignment.to_csv(
            OUT
            / "05_DRYRUN_REPAIRED_CONTROL_ASSIGNMENT.csv",
            index=False,
            encoding="utf-8-sig",
        )

        edge_fail.to_csv(
            OUT
            / "05B_PATH_EDGE_FAILURES.csv",
            index=False,
            encoding="utf-8-sig",
        )

        write_json(
            OUT
            / "06_FINAL_QA.json",
            qa,
        )

        touched_slots = len(
            path_df
        )

        touched_pairs = int(
            path_df[
                "pair_set_id"
            ].nunique()
        )

        if qa[
            "status"
        ] == "PASS":
            if (
                final_search[
                    "found_depth"
                ]
                <= SHORT_DEPTH_LIMIT
            ):
                decision = (
                    "PASS_SHORT_AUGMENTING_PATH_FOUND__"
                    "READY_FOR_HC_BUG03_CONTROLLED_REBUILD"
                )

            else:
                decision = (
                    "PASS_LONG_AUGMENTING_PATH_FOUND__"
                    "READY_FOR_HC_BUG03_CONTROLLED_REBUILD"
                )

        else:
            decision = (
                "AUGMENTING_PATH_FOUND_BUT_FINAL_QA_FAILED"
            )

        final = {
            "FINAL_DECISION": (
                decision
            ),

            "positive_semantics_unchanged": (
                "AJG1 == 1 AND GSI == 1"
            ),

            "search_mode": (
                search_mode
            ),

            "augmenting_path_found": True,

            "augmenting_path_depth": (
                final_search[
                    "found_depth"
                ]
            ),

            "touched_control_slots": (
                touched_slots
            ),

            "touched_pair_sets": (
                touched_pairs
            ),

            "root_blocked_slot": (
                partial_summary
            ),

            "final_QA": (
                qa
            ),

            "candidate_provider_stats": (
                provider.stats()
            ),

            "interpretation": (
                "A valid alternating exchange chain exists in the locked "
                "corrected P0 graph. The chain repairs the final blocked slot "
                "without control reuse and preserves exactly two controls per "
                "pair in the dry-run assignment."
            ),

            "production_rule": (
                "Do not write the dry-run path directly into frozen data. "
                "Patch HC00 to inherit upstream eligibility and rerun the "
                "production matching/pair-assignment pipeline."
            ),
        }

    else:
        qa = {
            "status": (
                "NO_AUGMENTING_PATH_FOUND"
            ),
            "visited_slots": (
                final_search[
                    "visited_slot_count"
                ]
            ),
            "total_formal_slots": (
                len(
                    slot_meta
                )
            ),
            "provider_stats": (
                provider.stats()
            ),
        }

        write_json(
            OUT
            / "06_FINAL_QA.json",
            qa,
        )

        decision = (
            "NO_AUGMENTING_PATH_IN_LOCKED_CORRECTED_P0_GRAPH__"
            "FULL_1TO2_MATCHING_INFEASIBLE_UNDER_THIS_EXACT_GRAPH"
        )

        final = {
            "FINAL_DECISION": (
                decision
            ),

            "positive_semantics_unchanged": (
                "AJG1 == 1 AND GSI == 1"
            ),

            "search_mode": (
                search_mode
            ),

            "augmenting_path_found": False,

            "root_blocked_slot": (
                partial_summary
            ),

            "visited_slot_count": (
                final_search[
                    "visited_slot_count"
                ]
            ),

            "total_formal_slots": (
                len(
                    slot_meta
                )
            ),

            "candidate_provider_stats": (
                provider.stats()
            ),

            "interpretation": (
                "The exhaustive alternating-graph search found no augmenting "
                "path from the sole unmatched control slot. Because the "
                "starting matching already has 10,111 valid matched slots, "
                "absence of an augmenting path means it is maximum in this "
                "exact corrected P0 bipartite graph. Therefore a complete "
                "10,112-control matching does not exist under these exact "
                "candidate-edge restrictions."
            ),

            "next_action": (
                "Do not silently weaken the protocol. Inspect which corrected "
                "constraint causes the Hall-type deficiency and decide whether "
                "the formal 1:2 protocol itself must be revised."
            ),
        }

    write_json(
        OUT
        / "07_FINAL_DECISION.json",
        final,
    )

    # -------------------------------------------------------------
    # 8. Human-readable report.
    # -------------------------------------------------------------
    if final_search[
        "found"
    ]:
        try:
            path_md = (
                path_df.to_markdown(
                    index=False
                )
            )
        except Exception:
            path_md = (
                path_df.to_string(
                    index=False
                )
            )

        path_section = f"""
## Augmenting path

{path_md}

## Final dry-run QA

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```
"""

    else:
        path_section = f"""
## Augmenting path

No augmenting path was found by exhaustive reachable-slot BFS.

## Search QA

```json
{json.dumps(qa, ensure_ascii=False, indent=2)}
```
"""

    report = f"""# HC-BUG-02D Slot-Level Augmenting-Path Audit

## Final decision

`{final['FINAL_DECISION']}`

## Frozen semantics

Formal positive remains:

`AJG1 == 1 AND GSI == 1`

Single-source evidence remains non-positive but excluded from trusted hard
controls under the existing conservative protocol.

## Corrected HC00 core

```json
{json.dumps(core_summary, ensure_ascii=False, indent=2)}
```

## Starting partial matching

```json
{json.dumps(partial_summary, ensure_ascii=False, indent=2)}
```

The starting state contains the 10,089 valid original controls plus the
22 direct HC-BUG-02C replacements, leaving exactly one unmatched control slot.

## Search mode

`{search_mode}`

Short-search result:

```json
{json.dumps(short_summary, ensure_ascii=False, indent=2)}
```

{path_section}

## Interpretation

The search operates on formal control slots `(pair_set_id, control_rank)`,
not only pair-set nodes. This preserves the exact two-control matching
structure and allows a valid chain to move either control slot when necessary.

No formal dataset was modified.
No model was trained.
"""

    (
        OUT
        / "00_HC_BUG02D_REPORT.md"
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
        "HC-BUG-02D AUGMENTING-PATH AUDIT COMPLETE"
    )

    print(
        "=" * 118
    )

    print(
        "CORRECTED HC00 CORE:",
        core_summary[
            "corrected_core"
        ],
    )

    print(
        "INVALID FORMAL CONTROLS:",
        len(
            invalid
        ),
    )

    print(
        "DIRECT 02C REPLACEMENTS APPLIED:",
        partial_summary[
            "direct_replacements_applied"
        ],
    )

    print(
        "UNMATCHED SLOTS BEFORE CHAIN:",
        partial_summary[
            "unmatched_slots"
        ],
    )

    print(
        "ROOT BLOCKED PAIR:",
        partial_summary[
            "root_pair_set_id"
        ],
    )

    print(
        "ROOT CONTROL RANK:",
        partial_summary[
            "root_control_rank"
        ],
    )

    print()

    print(
        "SEARCH MODE:",
        search_mode,
    )

    print(
        "AUGMENTING PATH FOUND:",
        final_search[
            "found"
        ],
    )

    print(
        "PATH DEPTH:",
        final_search[
            "found_depth"
        ],
    )

    print(
        "VISITED SLOTS:",
        final_search[
            "visited_slot_count"
        ],
        "/",
        len(
            slot_meta
        ),
    )

    print()

    if final_search[
        "found"
    ]:
        print(
            "TOUCHED SLOTS:",
            len(
                path_df
            ),
        )

        print(
            "TOUCHED PAIRS:",
            path_df[
                "pair_set_id"
            ].nunique(),
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
        final[
            "FINAL_DECISION"
        ],
    )

    print()

    print(
        "Please send:"
    )

    files = [
        "00_HC_BUG02D_REPORT.md",
        "01_PARTIAL_MATCHING_STATE.json",
        "02_BLOCKED_ROOT_SLOT.csv",
        "03_SHORT_CHAIN_SEARCH.json",
        "06_FINAL_QA.json",
        "07_FINAL_DECISION.json",
    ]

    if (
        OUT
        / "03B_EXHAUSTIVE_SEARCH.json"
    ).exists():
        files.append(
            "03B_EXHAUSTIVE_SEARCH.json"
        )

    if (
        OUT
        / "04_AUGMENTING_PATH.csv"
    ).exists():
        files.extend([
            "04_AUGMENTING_PATH.csv",
            "05_DRYRUN_REPAIRED_CONTROL_ASSIGNMENT.csv",
        ])

    for name in files:
        print(
            OUT / name
        )

    print(
        "=" * 118
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
