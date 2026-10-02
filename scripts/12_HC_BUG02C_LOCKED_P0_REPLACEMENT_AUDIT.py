# -*- coding: utf-8 -*-
r"""
12_HC_BUG02C_LOCKED_P0_REPLACEMENT_AUDIT.py
============================================
HC-BUG-02C: locked production candidate universe + 23-pair replacement audit.

READ ONLY.
NO model training.
NO formal data mutation.
NO pair-table overwrite.
NO fold regeneration.

Why HC-BUG-02C
--------------
HC-BUG-02B established two important facts:

1) The production formal assignment script explicitly defines:

   S7 = .../01c_matching_sensitivity_audit/02_edges/
        hard_control_01b_p0_parent20km_superset_edges.parquet

2) The candidate-lineage audit showed this P0 superset contains:
   - 2,909,329 candidate edges
   - 5,061 positives
   - median ~546 candidate edges / positive
   - 10,112 / 10,112 current formal edges (100% coverage)

HC-BUG-02B nevertheless stopped because its generic selector penalized paths
containing "sensitivity_audit". That was an audit-selector mistake.

HC-BUG-02B also accidentally expanded affected_pairs from the 23 bad controls
to almost all formal positives. HC-BUG-02C fixes that: affected pairs are
derived ONLY from the actually invalid formal hard-control rows.

Formal data semantics remain unchanged
--------------------------------------
Formal positive:
    AJG1 == 1 AND GSI == 1

Single-source evidence:
    not a formal positive,
    but excluded from trusted hard controls under the frozen conservative
    protocol.

Dry-run repair logic
--------------------
A. Corrected HC00 core:
       corrected_core =
           current_HC00_core ∩ upstream_frozen_eligible

B. Identify invalid formal hard controls:
       formal hard control NOT IN corrected_core

Expected from HC-BUG-01/02:
       current core      = 28,102
       remove illegal    = 30
       corrected core    = 28,072
       invalid controls  = 23
       affected pairs    = 23
       exactly one invalid control per affected pair

C. For each affected pair:
   - keep its other currently valid control fixed;
   - keep every valid control in all unaffected pairs fixed;
   - use ONLY candidate edges present in the locked production P0 superset;
   - candidate must belong to corrected_core;
   - candidate cannot already be any currently valid formal hard control;
   - candidate cannot be a formal positive;
   - candidate cannot be the pair's positive itself;
   - replacements cannot be reused across affected pairs.

D. Run no-reuse bipartite feasibility matching for 23 replacement slots.

Important interpretation
------------------------
PASS:
    Local 23-control repair is feasible.

FAIL/blocked:
    Local one-control-per-pair replacement is not feasible under the fixed
    current controls. This does NOT prove that a fresh global 1:2 matching
    rerun is infeasible; global rematching may rearrange more controls.

Even on PASS:
    DO NOT write the dry-run plan directly into the frozen dataset.
    Production repair must patch HC00 and rerun the original formal matching
    pipeline from the corrected candidate pool.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "HC_BUG_02C_LOCKED_P0_REPLACEMENT_AUDIT"

# ---------------------------------------------------------------------
# Locked production candidate universe.
# Confirmed by formal-assignment source code and prior 100% edge coverage.
# ---------------------------------------------------------------------
P0_EDGES = (
    ROOT / "data" / "06_hard_control"
    / "01c_matching_sensitivity_audit"
    / "02_edges"
    / "hard_control_01b_p0_parent20km_superset_edges.parquet"
)

FORMAL_ASSIGNMENT_SCRIPTS = [
    ROOT / "data" / "06_hard_control"
    / "02_formal_pair_assignment"
    / "hard_control_02_formal_pair_assignment.py",

    ROOT / "data" / "06_hard_control"
    / "02_formal_pair_assignment"
    / "scripts"
    / "hard_control_02_formal_pair_assignment.py",
]

FORMAL_LABEL = (
    ROOT / "data" / "06_hard_control"
    / "02_formal_pair_assignment"
    / "04_label_layer"
    / "hard_control_02_full_grid_label_layer.parquet"
)

FORMAL_EDGE_CANDIDATES = [
    ROOT / "data" / "06_hard_control"
    / "02_formal_pair_assignment"
    / "03_pair_tables"
    / "hard_control_02_pair_edges.parquet",

    ROOT / "data" / "06_hard_control"
    / "02_formal_pair_assignment"
    / "03_pair_tables"
    / "hard_control_02_pair_edges.csv",
]

ELIGIBILITY_CANDIDATES = [
    ROOT / "data" / "02_reliable_observation_domain"
    / "11_hard_control_candidate_eligibility_pool_frozen_v1_1"
    / "01_frozen_full_registry"
    / "hard_control_eligibility_full_grid_250m_frozen_v1_1.parquet",

    ROOT / "data" / "02_reliable_observation_domain"
    / "11_hard_control_candidate_eligibility_pool_frozen"
    / "01_frozen_full_registry"
    / "hard_control_eligibility_full_grid_250m_frozen_v1.parquet",

    ROOT / "data" / "02_reliable_observation_domain"
    / "10_hard_control_candidate_eligibility_pool"
    / "01_full_grid_registry"
    / "hard_control_eligibility_full_grid_250m_v1.parquet",
]

HC00_CORE_CANDIDATES = [
    ROOT / "data" / "06_hard_control"
    / "00_candidate_pool"
    / "hard_control_00_core_candidate_pool.csv",

    ROOT / "data" / "06_hard_control"
    / "00_candidate_pool"
    / "09_frozen"
    / "hard_control_00_core_candidate_pool_frozen.csv",

    ROOT / "data" / "06_hard_control"
    / "00_candidate_pool"
    / "hard_control_00_core_candidate_pool.parquet",
]

# ---------------------------------------------------------------------
# Safety expectations from HC-BUG-01 / HC-BUG-02.
# Script stops if the project state unexpectedly differs.
# ---------------------------------------------------------------------
EXPECTED_UPSTREAM_ELIGIBLE = 29084
EXPECTED_CURRENT_CORE = 28102
EXPECTED_REMOVED_CORE = 30
EXPECTED_CORRECTED_CORE = 28072
EXPECTED_FORMAL_CONTROLS = 10112
EXPECTED_FORMAL_PAIRSETS = 5056
EXPECTED_BAD_CONTROLS = 23
EXPECTED_AFFECTED_PAIRSETS = 23

# P0 lineage facts from uploaded audit; rechecked locally where practical.
EXPECTED_P0_FORMAL_EDGE_COVERAGE = 1.0

INF = 1e18


def log(msg: str):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)

    with (OUT / "logs" / "audit.log").open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(line + "\n")


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def get_schema(path: Path):
    try:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            return list(
                pq.ParquetFile(path).schema.names
            )

        if path.suffix.lower() == ".csv":
            return list(
                pd.read_csv(path, nrows=0).columns
            )
    except Exception:
        return []

    return []


def read_table(path: Path, columns=None):
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
        f"Unsupported table type: {path}"
    )


def truthy(s: pd.Series):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        return (
            pd.to_numeric(
                s,
                errors="coerce",
            ).fillna(0) > 0
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
    """
    Normalize pair-set keys only for in-memory joins.
    Keeps original pair_set_id values in output tables.
    """
    if pd.isna(v):
        return None

    if isinstance(v, (int, np.integer)):
        return str(int(v))

    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return str(v)

    s = str(v).strip()

    # Avoid "123.0" vs "123" mismatch from CSV/parquet dtype changes.
    if re.fullmatch(r"-?\d+\.0", s):
        return s[:-2]

    return s


def normalize_unique_unit(df, name):
    if "unit_id" not in df.columns:
        raise RuntimeError(
            f"{name} missing unit_id"
        )

    if df["unit_id"].isna().any():
        raise RuntimeError(
            f"{name} has null unit_id"
        )

    dup = int(
        df["unit_id"].duplicated().sum()
    )

    if dup:
        raise RuntimeError(
            f"{name} should be unit-level but has "
            f"{dup} duplicate unit_ids"
        )

    return df


# =====================================================================
# 1. Verify that production code really locks S7 to P0 superset
# =====================================================================

def verify_production_lineage():
    expected_filename = P0_EDGES.name
    expected_fragment = (
        "01c_matching_sensitivity_audit/02_edges/"
        "hard_control_01b_p0_parent20km_superset_edges.parquet"
    ).lower()

    rows = []

    for script in FORMAL_ASSIGNMENT_SCRIPTS:
        if not script.exists():
            continue

        text = script.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        low = text.lower().replace("\\", "/")

        exact_filename = (
            expected_filename.lower() in low
        )

        exact_path_fragment = (
            expected_fragment in low
        )

        s7_assignment = bool(
            re.search(
                r"\bS7\s*=",
                text,
            )
            and exact_filename
        )

        pos = low.find(
            expected_filename.lower()
        )

        snippet = ""

        if pos >= 0:
            a = max(
                0,
                pos - 650,
            )
            b = min(
                len(text),
                pos + 1100,
            )

            snippet = re.sub(
                r"\s+",
                " ",
                text[a:b],
            ).strip()

        rows.append({
            "script": str(script),
            "exact_filename_reference": exact_filename,
            "exact_path_fragment_reference": exact_path_fragment,
            "S7_assignment_detected": s7_assignment,
            "snippet": snippet,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df, False

    passed = bool(
        (
            df["exact_filename_reference"]
            & df["S7_assignment_detected"]
        ).any()
    )

    return df, passed


# =====================================================================
# 2. Load frozen upstream eligibility + current HC00 core
# =====================================================================

def load_eligibility():
    path = first_existing(
        ELIGIBILITY_CANDIDATES
    )

    if path is None:
        raise RuntimeError(
            "Frozen upstream eligibility table not found."
        )

    cols = get_schema(path)

    wanted = [
        c for c in [
            "unit_id",
            "primary_eligibility_status",
            "control_exclusion_flag",
            "inventory_evidence_exclusion_flag",
            "eligibility_reason_codes",
            "final_control_status",
        ]
        if c in cols
    ]

    df = normalize_unique_unit(
        read_table(
            path,
            wanted,
        ),
        "frozen upstream eligibility",
    )

    if "primary_eligibility_status" not in df.columns:
        raise RuntimeError(
            "Frozen eligibility lacks "
            "primary_eligibility_status"
        )

    status = (
        df["primary_eligibility_status"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    df["AUTHORITATIVE_ELIGIBLE"] = (
        status.str.startswith("ELIGIBLE")
    )

    explicit_excluded = pd.Series(
        False,
        index=df.index,
    )

    for c in (
        "control_exclusion_flag",
        "inventory_evidence_exclusion_flag",
    ):
        if c in df.columns:
            explicit_excluded |= truthy(
                df[c]
            )

    contradiction = (
        df["AUTHORITATIVE_ELIGIBLE"]
        & explicit_excluded
    )

    if contradiction.any():
        raise RuntimeError(
            "Upstream frozen eligibility is internally "
            "contradictory: "
            f"{int(contradiction.sum())} rows are both "
            "ELIGIBLE and explicitly evidence-excluded."
        )

    df["AUTHORITATIVE_EXCLUDED"] = (
        ~df["AUTHORITATIVE_ELIGIBLE"]
    )

    return path, df


def load_core():
    path = first_existing(
        HC00_CORE_CANDIDATES
    )

    if path is None:
        raise RuntimeError(
            "Current HC00 core candidate pool not found."
        )

    cols = get_schema(path)

    wanted = list(
        dict.fromkeys(
            ["unit_id"]
            + [
                c for c in [
                    "eligible_evidence_clean_candidate",
                    "primary_status",
                    "candidate_role",
                    "strict_coverage_status",
                ]
                if c in cols
            ]
        )
    )

    df = normalize_unique_unit(
        read_table(
            path,
            wanted,
        ),
        "current HC00 core",
    )

    return path, df


def corrected_core(
    eligibility,
    core,
):
    eligible_ids = set(
        eligibility.loc[
            eligibility["AUTHORITATIVE_ELIGIBLE"],
            "unit_id",
        ].tolist()
    )

    excluded_ids = set(
        eligibility.loc[
            eligibility["AUTHORITATIVE_EXCLUDED"],
            "unit_id",
        ].tolist()
    )

    current_ids = set(
        core["unit_id"].tolist()
    )

    corrected_ids = (
        current_ids & eligible_ids
    )

    removed_ids = (
        current_ids - eligible_ids
    )

    illegal_after_fix = (
        corrected_ids & excluded_ids
    )

    return {
        "eligible_ids": eligible_ids,
        "excluded_ids": excluded_ids,
        "current_ids": current_ids,
        "corrected_ids": corrected_ids,
        "removed_ids": removed_ids,
        "illegal_after_fix": illegal_after_fix,
    }


# =====================================================================
# 3. Load formal cohort and derive ONLY the truly affected 23 pairs
# =====================================================================

def load_formal():
    if not FORMAL_LABEL.exists():
        raise RuntimeError(
            f"Missing formal label layer: {FORMAL_LABEL}"
        )

    cols = get_schema(
        FORMAL_LABEL
    )

    wanted = [
        c for c in [
            "unit_id",
            "sample_role",
            "positive_label_flag",
            "y_main",
            "hard_control_flag",
            "pair_set_id",
            "matched_positive_unit_id",
            "control_rank",
            "matching_protocol_id",
        ]
        if c in cols
    ]

    df = normalize_unique_unit(
        read_table(
            FORMAL_LABEL,
            wanted,
        ),
        "formal label layer",
    )

    if "hard_control_flag" in df.columns:
        df["IS_HARD_CONTROL"] = truthy(
            df["hard_control_flag"]
        )
    else:
        df["IS_HARD_CONTROL"] = (
            df["sample_role"]
            .fillna("")
            .astype(str)
            .eq("HARD_CONTROL")
        )

    if "positive_label_flag" in df.columns:
        df["IS_POSITIVE"] = truthy(
            df["positive_label_flag"]
        )

    elif "y_main" in df.columns:
        df["IS_POSITIVE"] = (
            pd.to_numeric(
                df["y_main"],
                errors="coerce",
            ).fillna(0) > 0
        )

    else:
        raise RuntimeError(
            "Formal positive indicator unavailable."
        )

    return df


def derive_affected_pairs(
    formal,
    corrected_ids,
):
    hard = formal[
        formal["IS_HARD_CONTROL"]
    ].copy()

    invalid = hard[
        ~hard["unit_id"].isin(
            corrected_ids
        )
    ].copy()

    valid = hard[
        hard["unit_id"].isin(
            corrected_ids
        )
    ].copy()

    if "pair_set_id" not in invalid.columns:
        raise RuntimeError(
            "Formal hard controls lack pair_set_id."
        )

    if "matched_positive_unit_id" not in invalid.columns:
        raise RuntimeError(
            "Formal hard controls lack "
            "matched_positive_unit_id."
        )

    invalid["PAIR_KEY"] = invalid[
        "pair_set_id"
    ].map(normalize_pair_id)

    if invalid["PAIR_KEY"].isna().any():
        raise RuntimeError(
            "Invalid hard controls contain null pair_set_id."
        )

    # CRITICAL FIX VS HC-BUG-02B:
    # affected pair set is derived ONLY from invalid hard-control rows.
    pair_counts = (
        invalid.groupby(
            "PAIR_KEY"
        ).size()
    )

    multi_bad = pair_counts[
        pair_counts != 1
    ]

    if len(multi_bad):
        raise RuntimeError(
            "Expected exactly one invalid control per "
            "affected pair, but found:\n"
            + multi_bad.to_string()
        )

    affected_pairs = set(
        invalid["PAIR_KEY"].tolist()
    )

    pair_to_positive = {}

    for _, r in invalid.iterrows():
        pair = r["PAIR_KEY"]
        posid = r[
            "matched_positive_unit_id"
        ]

        if pd.isna(posid):
            raise RuntimeError(
                f"Affected pair {pair} has null "
                "matched_positive_unit_id."
            )

        pair_to_positive[pair] = str(
            posid
        )

    # Do NOT backfill unaffected positive pairs here.
    # This was the HC-BUG-02B pair-expansion bug.

    # Current valid controls are reserved globally,
    # including the good partner in each affected pair.
    reserved_valid_controls = set(
        valid["unit_id"].tolist()
    )

    return {
        "hard": hard,
        "invalid": invalid,
        "valid": valid,
        "affected_pairs": affected_pairs,
        "pair_to_positive": pair_to_positive,
        "reserved_valid_controls": reserved_valid_controls,
    }


# =====================================================================
# 4. Verify locked P0 candidate universe against current formal edges
# =====================================================================

def load_formal_edges_from_label(
    formal,
):
    hard = formal[
        formal["IS_HARD_CONTROL"]
    ].copy()

    return pd.DataFrame({
        "positive_unit_id": (
            hard["matched_positive_unit_id"]
            .astype(str)
        ),
        "control_unit_id": (
            hard["unit_id"]
        ),
    }).drop_duplicates()


def verify_p0_formal_edge_coverage(
    formal_edges,
):
    """
    Independent local safety recheck.

    Reads ONLY positive_unit_id + candidate_unit_id from the 2.9M-row P0
    parquet. This is much lighter than reading all matching features.
    """

    schema = get_schema(
        P0_EDGES
    )

    required = {
        "positive_unit_id",
        "candidate_unit_id",
    }

    if not required.issubset(
        set(schema)
    ):
        raise RuntimeError(
            "Locked P0 edge table lacks required columns: "
            f"{required - set(schema)}"
        )

    log(
        "Rechecking 10,112 formal edges against locked P0 "
        "candidate universe..."
    )

    keys = pd.read_parquet(
        P0_EDGES,
        columns=[
            "positive_unit_id",
            "candidate_unit_id",
        ],
    )

    keys = keys.rename(
        columns={
            "candidate_unit_id": "control_unit_id",
        }
    )

    keys["positive_unit_id"] = (
        keys["positive_unit_id"]
        .astype(str)
    )

    keys = keys.drop_duplicates(
        [
            "positive_unit_id",
            "control_unit_id",
        ]
    )

    q = formal_edges.merge(
        keys.assign(
            IN_P0=1
        ),
        on=[
            "positive_unit_id",
            "control_unit_id",
        ],
        how="left",
    )

    covered = int(
        q["IN_P0"]
        .fillna(0)
        .sum()
    )

    total = int(
        len(formal_edges)
    )

    fraction = (
        covered / total
        if total
        else np.nan
    )

    # Explicitly release the 2.9M-row key table.
    del keys

    return {
        "formal_edges_total": total,
        "formal_edges_covered": covered,
        "formal_edge_coverage_fraction": fraction,
    }


# =====================================================================
# 5. Read ONLY affected-positive rows from the P0 superset
# =====================================================================

def choose_order_column(schema):
    """
    Ordering is only used to make the dry-run deterministic.
    Feasibility does not depend on this choice.

    Prefer composite_distance because it is the combined matching-distance
    field in the P0 edge table. Fall back conservatively.
    """

    preferred = [
        "composite_distance",
        "terrain_distance",
        "centroid_distance_m",
        "candidate_index",
    ]

    for c in preferred:
        if c in schema:
            return c

    return None


def read_affected_p0_edges(
    affected_positive_ids,
):
    schema = get_schema(
        P0_EDGES
    )

    required = {
        "positive_unit_id",
        "candidate_unit_id",
    }

    if not required.issubset(
        set(schema)
    ):
        raise RuntimeError(
            "Locked P0 table lacks required edge columns."
        )

    order_col = choose_order_column(
        schema
    )

    columns = [
        "positive_unit_id",
        "candidate_unit_id",
    ]

    if order_col:
        columns.append(
            order_col
        )

    # Keep useful block/provenance columns when available.
    for c in [
        "candidate_pool",
        "geology_parent",
        "matching_missingness_pattern_id",
        "soil_completeness_block",
        "geology_exact_flag",
        "geology_penalty",
        "soil_group_unavailable_flag",
    ]:
        if (
            c in schema
            and c not in columns
        ):
            columns.append(c)

    log(
        f"Reading P0 candidate edges for exactly "
        f"{len(affected_positive_ids)} affected positives..."
    )

    # Prefer pyarrow dataset predicate pushdown.
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(
            str(P0_EDGES),
            format="parquet",
        )

        filt = ds.field(
            "positive_unit_id"
        ).isin(
            list(
                map(
                    str,
                    affected_positive_ids,
                )
            )
        )

        table = dataset.to_table(
            columns=columns,
            filter=filt,
        )

        df = table.to_pandas()

        read_mode = (
            "PYARROW_DATASET_FILTER_PUSHDOWN"
        )

    except Exception as e:
        log(
            "Predicate pushdown unavailable; "
            "falling back to selected-column pandas read. "
            f"{type(e).__name__}: {e}"
        )

        df = pd.read_parquet(
            P0_EDGES,
            columns=columns,
        )

        df = df[
            df["positive_unit_id"]
            .astype(str)
            .isin(
                set(
                    map(
                        str,
                        affected_positive_ids,
                    )
                )
            )
        ].copy()

        read_mode = (
            "PANDAS_SELECTED_COLUMNS_FALLBACK"
        )

    df["POS_STD"] = (
        df["positive_unit_id"]
        .astype(str)
    )

    df["CAND_STD"] = (
        df["candidate_unit_id"]
    )

    if order_col:
        x = pd.to_numeric(
            df[order_col],
            errors="coerce",
        )

        df["ORDER_COST"] = (
            x.fillna(
                INF / 100
            )
        )

        df["ORDER_SOURCE"] = (
            order_col
            + " (deterministic dry-run ordering only)"
        )

    else:
        df["ORDER_COST"] = 0.0
        df["ORDER_SOURCE"] = (
            "candidate_unit_id deterministic ordering only"
        )

    return df, {
        "read_mode": read_mode,
        "order_col": order_col,
        "rows_for_affected_positives": int(
            len(df)
        ),
        "affected_positives_covered": int(
            df["POS_STD"].nunique()
        ),
    }


# =====================================================================
# 6. Legal replacement filter
# =====================================================================

def build_legal_replacement_edges(
    p0_edges,
    pair_to_positive,
    corrected_ids,
    reserved_valid_controls,
    formal_positive_ids,
):
    reverse = {
        str(posid): pair
        for pair, posid
        in pair_to_positive.items()
    }

    q = p0_edges[
        p0_edges["POS_STD"].isin(
            set(reverse.keys())
        )
    ].copy()

    q["PAIR_KEY"] = (
        q["POS_STD"]
        .map(reverse)
    )

    if q["PAIR_KEY"].isna().any():
        raise RuntimeError(
            "Unexpected affected-positive to pair mapping failure."
        )

    q["IN_CORRECTED_CORE"] = (
        q["CAND_STD"]
        .isin(
            corrected_ids
        )
    )

    q["NOT_ALREADY_USED"] = (
        ~q["CAND_STD"]
        .isin(
            reserved_valid_controls
        )
    )

    q["NOT_FORMAL_POSITIVE"] = (
        ~q["CAND_STD"]
        .astype(str)
        .isin(
            set(
                map(
                    str,
                    formal_positive_ids,
                )
            )
        )
    )

    q["NOT_SELF"] = [
        str(cand)
        != pair_to_positive[pair]
        for cand, pair
        in zip(
            q["CAND_STD"],
            q["PAIR_KEY"],
        )
    ]

    q["LEGAL_REPLACEMENT"] = (
        q["IN_CORRECTED_CORE"]
        & q["NOT_ALREADY_USED"]
        & q["NOT_FORMAL_POSITIVE"]
        & q["NOT_SELF"]
    )

    return q


def replacement_count_table(
    legal_edges,
    affected_pairs,
):
    rows = []

    for pair in sorted(
        affected_pairs
    ):
        q = legal_edges[
            legal_edges["PAIR_KEY"].eq(
                pair
            )
            & legal_edges[
                "LEGAL_REPLACEMENT"
            ]
        ].copy()

        unique_q = q.drop_duplicates(
            "CAND_STD"
        )

        rows.append({
            "pair_set_id": pair,
            "legal_replacement_candidates": int(
                unique_q["CAND_STD"]
                .nunique()
            ),
            "best_order_cost": (
                float(
                    unique_q["ORDER_COST"]
                    .min()
                )
                if len(unique_q)
                else np.nan
            ),
            "order_source": (
                str(
                    unique_q[
                        "ORDER_SOURCE"
                    ].iloc[0]
                )
                if len(unique_q)
                else None
            ),
        })

    return pd.DataFrame(
        rows
    )


# =====================================================================
# 7. No-reuse matching
# =====================================================================

def bipartite_replacement_matching(
    legal_edges,
    affected_pairs,
):
    """
    Each affected pair needs exactly ONE replacement.

    Existing valid controls are already removed from legal_edges.
    This matcher only prevents replacement reuse among the 23 affected pairs.
    """

    adjacency = {}

    for pair in sorted(
        affected_pairs
    ):
        q = legal_edges[
            legal_edges["PAIR_KEY"].eq(
                pair
            )
            & legal_edges[
                "LEGAL_REPLACEMENT"
            ]
        ].copy()

        q = (
            q.sort_values(
                [
                    "ORDER_COST",
                    "CAND_STD",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .drop_duplicates(
                "CAND_STD"
            )
        )

        adjacency[pair] = [
            (
                row["CAND_STD"],
                float(
                    row["ORDER_COST"]
                ),
            )
            for _, row
            in q.iterrows()
        ]

    # Most constrained pair first.
    pair_order = sorted(
        affected_pairs,
        key=lambda p: (
            len(
                adjacency[p]
            ),
            p,
        ),
    )

    candidate_owner = {}
    pair_choice = {}

    def dfs(
        pair,
        seen_candidates,
    ):
        for cand, cost in adjacency[
            pair
        ]:
            if cand in seen_candidates:
                continue

            seen_candidates.add(
                cand
            )

            if cand not in candidate_owner:
                candidate_owner[
                    cand
                ] = pair

                pair_choice[
                    pair
                ] = (
                    cand,
                    cost,
                )

                return True

            old_pair = candidate_owner[
                cand
            ]

            if dfs(
                old_pair,
                seen_candidates,
            ):
                candidate_owner[
                    cand
                ] = pair

                pair_choice[
                    pair
                ] = (
                    cand,
                    cost,
                )

                return True

        return False

    matched = 0

    for pair in pair_order:
        if dfs(
            pair,
            set(),
        ):
            matched += 1

    rows = []

    for pair in pair_order:
        selected = pair_choice.get(
            pair
        )

        rows.append({
            "pair_set_id": pair,
            "replacement_found": (
                selected is not None
            ),
            "replacement_unit_id": (
                selected[0]
                if selected
                else None
            ),
            "replacement_order_cost": (
                selected[1]
                if selected
                else np.nan
            ),
            "legal_candidate_count": len(
                adjacency[pair]
            ),
        })

    return matched, pd.DataFrame(
        rows
    )


# =====================================================================
# 8. Simulated repaired cohort QA
# =====================================================================

def simulate_repaired_cohort(
    formal,
    invalid,
    plan,
    corrected_ids,
):
    hard = formal[
        formal["IS_HARD_CONTROL"]
    ].copy()

    invalid_ids = set(
        invalid["unit_id"].tolist()
    )

    unaffected = hard[
        ~hard["unit_id"].isin(
            invalid_ids
        )
    ].copy()

    if not bool(
        plan["replacement_found"].all()
    ):
        return None, {
            "status": (
                "BLOCKED_INCOMPLETE_REPLACEMENT_PLAN"
            )
        }

    invalid = invalid.copy()

    invalid["PAIR_KEY"] = (
        invalid["pair_set_id"]
        .map(
            normalize_pair_id
        )
    )

    old_by_pair = {
        r["PAIR_KEY"]: r
        for _, r
        in invalid.iterrows()
    }

    replacements = []

    for _, r in plan.iterrows():
        pair = r["pair_set_id"]
        old = old_by_pair[
            pair
        ]

        replacements.append({
            "unit_id": (
                r[
                    "replacement_unit_id"
                ]
            ),
            "pair_set_id": (
                old[
                    "pair_set_id"
                ]
            ),
            "matched_positive_unit_id": (
                old.get(
                    "matched_positive_unit_id"
                )
            ),
            "control_rank": (
                old.get(
                    "control_rank"
                )
            ),
            "sample_role": (
                "HARD_CONTROL"
            ),
            "hard_control_flag": 1,
        })

    rep = pd.DataFrame(
        replacements
    )

    keep = [
        c for c in [
            "unit_id",
            "pair_set_id",
            "matched_positive_unit_id",
            "control_rank",
            "sample_role",
            "hard_control_flag",
        ]
        if c in unaffected.columns
    ]

    sim = pd.concat(
        [
            unaffected[
                keep
            ],
            rep,
        ],
        ignore_index=True,
        sort=False,
    )

    pair_counts = (
        sim.groupby(
            "pair_set_id",
            dropna=False,
        )
        .size()
    )

    qa = {
        "status": "PASS",

        "simulated_control_rows": int(
            len(sim)
        ),

        "unique_control_unit_ids": int(
            sim["unit_id"].nunique()
        ),

        "duplicate_control_unit_ids": int(
            sim["unit_id"]
            .duplicated()
            .sum()
        ),

        "pair_set_count": int(
            sim[
                "pair_set_id"
            ].nunique()
        ),

        "pair_sets_not_exactly_two_controls": int(
            (
                pair_counts != 2
            ).sum()
        ),

        "replacement_units_not_in_corrected_core": int(
            (
                ~rep["unit_id"]
                .isin(
                    corrected_ids
                )
            ).sum()
        ),

        "replacement_reuse_count": int(
            rep[
                "unit_id"
            ].duplicated()
            .sum()
        ),
    }

    if (
        qa[
            "simulated_control_rows"
        ] != EXPECTED_FORMAL_CONTROLS
        or qa[
            "unique_control_unit_ids"
        ] != EXPECTED_FORMAL_CONTROLS
        or qa[
            "duplicate_control_unit_ids"
        ] != 0
        or qa[
            "pair_set_count"
        ] != EXPECTED_FORMAL_PAIRSETS
        or qa[
            "pair_sets_not_exactly_two_controls"
        ] != 0
        or qa[
            "replacement_units_not_in_corrected_core"
        ] != 0
        or qa[
            "replacement_reuse_count"
        ] != 0
    ):
        qa["status"] = "FAIL"

    return sim, qa


# =====================================================================
# MAIN
# =====================================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "HC-BUG-02C locked-P0 replacement audit started"
    )

    # -------------------------------------------------------------
    # A. Verify production-code lineage.
    # -------------------------------------------------------------
    lineage_df, lineage_pass = (
        verify_production_lineage()
    )

    lineage_df.to_csv(
        OUT
        / "00_PRODUCTION_P0_LINEAGE_CHECK.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not P0_EDGES.exists():
        raise RuntimeError(
            f"Locked production P0 candidate table missing:\n"
            f"{P0_EDGES}"
        )

    if not lineage_pass:
        final = {
            "FINAL_DECISION": (
                "STOP_PRODUCTION_P0_LINEAGE_NOT_CONFIRMED"
            ),
            "P0_EDGES": str(
                P0_EDGES
            ),
        }

        write_json(
            OUT
            / "10_FINAL_DECISION.json",
            final,
        )

        print(
            "\nFINAL_DECISION:",
            final["FINAL_DECISION"],
        )

        return 0

    # -------------------------------------------------------------
    # B. Corrected HC00 core.
    # -------------------------------------------------------------
    elig_path, eligibility = (
        load_eligibility()
    )

    core_path, core = (
        load_core()
    )

    cc = corrected_core(
        eligibility,
        core,
    )

    corrected_ids = cc[
        "corrected_ids"
    ]

    removed_ids = cc[
        "removed_ids"
    ]

    core_audit = {
        "upstream_eligible": len(
            cc[
                "eligible_ids"
            ]
        ),

        "current_HC00_core": len(
            cc[
                "current_ids"
            ]
        ),

        "removed_illegal_core": len(
            removed_ids
        ),

        "corrected_HC00_core": len(
            corrected_ids
        ),

        "illegal_overlap_after_fix": len(
            cc[
                "illegal_after_fix"
            ]
        ),

        "upstream_eligibility_table": str(
            elig_path
        ),

        "HC00_core_table": str(
            core_path
        ),
    }

    write_json(
        OUT
        / "01_CORRECTED_CORE_AUDIT.json",
        core_audit,
    )

    removed_detail = (
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

    removed_detail.to_csv(
        OUT
        / "01B_REMOVED_ILLEGAL_CORE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # C. True 23 affected formal controls / pairs only.
    # -------------------------------------------------------------
    formal = load_formal()

    aff = derive_affected_pairs(
        formal,
        corrected_ids,
    )

    hard = aff[
        "hard"
    ]
    invalid = aff[
        "invalid"
    ]
    affected_pairs = aff[
        "affected_pairs"
    ]
    pair_to_positive = aff[
        "pair_to_positive"
    ]
    reserved_valid_controls = aff[
        "reserved_valid_controls"
    ]

    invalid.to_csv(
        OUT
        / "02_INVALID_FORMAL_CONTROLS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_rows = []

    for _, r in invalid.iterrows():
        pair = normalize_pair_id(
            r[
                "pair_set_id"
            ]
        )

        pair_rows.append({
            "pair_set_id": pair,

            "positive_unit_id": (
                pair_to_positive[
                    pair
                ]
            ),

            "illegal_control_unit_id": (
                r[
                    "unit_id"
                ]
            ),

            "control_rank": (
                r.get(
                    "control_rank"
                )
            ),
        })

    affected_df = pd.DataFrame(
        pair_rows
    )

    affected_df.to_csv(
        OUT
        / "03_AFFECTED_23_PAIRSETS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # D. Hard safety-count check.
    # -------------------------------------------------------------
    state = {
        "upstream_eligible": len(
            cc[
                "eligible_ids"
            ]
        ),

        "current_core": len(
            cc[
                "current_ids"
            ]
        ),

        "removed_core": len(
            removed_ids
        ),

        "corrected_core": len(
            corrected_ids
        ),

        "formal_controls": int(
            len(
                hard
            )
        ),

        "formal_pair_sets": int(
            hard[
                "pair_set_id"
            ].nunique()
        ),

        "invalid_controls": int(
            len(
                invalid
            )
        ),

        "affected_pairs": int(
            len(
                affected_pairs
            )
        ),
    }

    expected = {
        "upstream_eligible": EXPECTED_UPSTREAM_ELIGIBLE,
        "current_core": EXPECTED_CURRENT_CORE,
        "removed_core": EXPECTED_REMOVED_CORE,
        "corrected_core": EXPECTED_CORRECTED_CORE,
        "formal_controls": EXPECTED_FORMAL_CONTROLS,
        "formal_pair_sets": EXPECTED_FORMAL_PAIRSETS,
        "invalid_controls": EXPECTED_BAD_CONTROLS,
        "affected_pairs": EXPECTED_AFFECTED_PAIRSETS,
    }

    mismatch = {
        k: {
            "observed": state[k],
            "expected": expected[k],
        }
        for k in expected
        if state[k] != expected[k]
    }

    write_json(
        OUT
        / "04_STATE_COUNT_GUARD.json",
        {
            "observed": state,
            "expected": expected,
            "mismatch": mismatch,
            "status": (
                "PASS"
                if not mismatch
                else "STOP"
            ),
        },
    )

    if mismatch:
        final = {
            "FINAL_DECISION": (
                "STOP_PROJECT_STATE_DIFFERS_FROM_AUDITED_HCBUG01_02"
            ),
            "mismatch": mismatch,
        }

        write_json(
            OUT
            / "10_FINAL_DECISION.json",
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
    # E. Independent P0 lineage outcome check: current 10,112 edges.
    # -------------------------------------------------------------
    formal_edges = (
        load_formal_edges_from_label(
            formal
        )
    )

    p0_coverage = (
        verify_p0_formal_edge_coverage(
            formal_edges
        )
    )

    write_json(
        OUT
        / "05_P0_FORMAL_EDGE_COVERAGE.json",
        p0_coverage,
    )

    if (
        abs(
            p0_coverage[
                "formal_edge_coverage_fraction"
            ]
            - EXPECTED_P0_FORMAL_EDGE_COVERAGE
        )
        > 1e-12
    ):
        final = {
            "FINAL_DECISION": (
                "STOP_LOCKED_P0_DOES_NOT_REPRODUCE_100_PERCENT_FORMAL_EDGE_COVERAGE"
            ),
            "P0_coverage": (
                p0_coverage
            ),
        }

        write_json(
            OUT
            / "10_FINAL_DECISION.json",
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
    # F. Read real P0 candidate edges for exactly 23 affected positives.
    # -------------------------------------------------------------
    affected_positive_ids = set(
        pair_to_positive.values()
    )

    p0_edges, p0_read_meta = (
        read_affected_p0_edges(
            affected_positive_ids
        )
    )

    write_json(
        OUT
        / "06_AFFECTED_P0_EDGE_READ_META.json",
        p0_read_meta,
    )

    if (
        p0_read_meta[
            "affected_positives_covered"
        ]
        != EXPECTED_AFFECTED_PAIRSETS
    ):
        final = {
            "FINAL_DECISION": (
                "STOP_P0_CANDIDATE_TABLE_DOES_NOT_COVER_ALL_23_AFFECTED_POSITIVES"
            ),
            "read_meta": (
                p0_read_meta
            ),
        }

        write_json(
            OUT
            / "10_FINAL_DECISION.json",
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
    # G. Legal replacement filter.
    # -------------------------------------------------------------
    formal_positive_ids = set(
        formal.loc[
            formal[
                "IS_POSITIVE"
            ],
            "unit_id",
        ].tolist()
    )

    legal_edges = (
        build_legal_replacement_edges(
            p0_edges,
            pair_to_positive,
            corrected_ids,
            reserved_valid_controls,
            formal_positive_ids,
        )
    )

    # Save only affected rows; typically ~10k, not 2.9M.
    legal_edges.to_csv(
        OUT
        / "07_TRUE_REPLACEMENT_EDGE_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        replacement_count_table(
            legal_edges,
            affected_pairs,
        )
    )

    counts.to_csv(
        OUT
        / "08_REPLACEMENT_CANDIDATE_COUNTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    zero_pairs = int(
        (
            counts[
                "legal_replacement_candidates"
            ]
            == 0
        ).sum()
    )

    # -------------------------------------------------------------
    # H. Unique no-reuse 23-pair assignment.
    # -------------------------------------------------------------
    matched_n, plan = (
        bipartite_replacement_matching(
            legal_edges,
            affected_pairs,
        )
    )

    plan = plan.merge(
        affected_df,
        on="pair_set_id",
        how="left",
        validate="one_to_one",
    )

    plan.to_csv(
        OUT
        / "09_DRYRUN_23_REPLACEMENT_PLAN.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # I. Full-control cohort QA.
    # -------------------------------------------------------------
    if (
        zero_pairs == 0
        and matched_n
        == EXPECTED_AFFECTED_PAIRSETS
    ):
        simulated, cohort_qa = (
            simulate_repaired_cohort(
                formal,
                invalid,
                plan,
                corrected_ids,
            )
        )

        if simulated is not None:
            simulated.to_csv(
                OUT
                / "09B_SIMULATED_REPAIRED_CONTROLS.csv",
                index=False,
                encoding="utf-8-sig",
            )

    else:
        simulated = None

        cohort_qa = {
            "status": (
                "BLOCKED_LOCAL_REPLACEMENT_INCOMPLETE"
            ),
            "zero_candidate_pairs": zero_pairs,
            "matched_pairs": matched_n,
            "required_pairs": (
                EXPECTED_AFFECTED_PAIRSETS
            ),
        }

    write_json(
        OUT
        / "09C_SIMULATED_COHORT_QA.json",
        cohort_qa,
    )

    # -------------------------------------------------------------
    # J. Final decision.
    # -------------------------------------------------------------
    if zero_pairs > 0:
        decision = (
            "LOCAL_REPAIR_BLOCKED_SOME_PAIRS_HAVE_NO_UNUSED_P0_REPLACEMENT__"
            "GLOBAL_REMATCH_STILL_UNTESTED"
        )

    elif matched_n < EXPECTED_AFFECTED_PAIRSETS:
        decision = (
            "LOCAL_REPAIR_BLOCKED_BY_REPLACEMENT_COLLISION__"
            "GLOBAL_REMATCH_STILL_UNTESTED"
        )

    elif cohort_qa.get(
        "status"
    ) != "PASS":
        decision = (
            "LOCAL_REPLACEMENTS_FOUND_BUT_SIMULATED_COHORT_QA_FAILED"
        )

    else:
        decision = (
            "PASS_LOCAL_23_CONTROL_REPAIR_FEASIBLE__"
            "READY_FOR_HC_BUG03_CONTROLLED_REBUILD"
        )

    final = {
        "FINAL_DECISION": decision,

        "positive_semantics_unchanged": (
            "AJG1 == 1 AND GSI == 1"
        ),

        "locked_production_candidate_table": str(
            P0_EDGES
        ),

        "production_lineage_verified": bool(
            lineage_pass
        ),

        "P0_formal_edge_coverage": (
            p0_coverage
        ),

        "state": state,

        "replacement_feasibility": {
            "zero_candidate_pairs": (
                zero_pairs
            ),

            "no_reuse_matched_pairs": (
                matched_n
            ),

            "required_pairs": (
                EXPECTED_AFFECTED_PAIRSETS
            ),

            "min_legal_candidates_per_pair": (
                int(
                    counts[
                        "legal_replacement_candidates"
                    ].min()
                )
                if len(
                    counts
                )
                else 0
            ),

            "median_legal_candidates_per_pair": (
                float(
                    counts[
                        "legal_replacement_candidates"
                    ].median()
                )
                if len(
                    counts
                )
                else np.nan
            ),

            "max_legal_candidates_per_pair": (
                int(
                    counts[
                        "legal_replacement_candidates"
                    ].max()
                )
                if len(
                    counts
                )
                else 0
            ),
        },

        "simulated_cohort_QA": (
            cohort_qa
        ),

        "interpretation": (
            "PASS proves a local 23-control replacement exists while keeping "
            "all currently valid controls fixed. A blocked local repair does "
            "NOT prove the full 5,056-pair 1:2 problem is infeasible; a global "
            "rerun may rearrange controls."
        ),

        "production_rule_if_pass": (
            "Do not copy the dry-run replacement plan into production. "
            "Patch HC00 so core eligibility inherits upstream frozen "
            "eligibility, then rerun the original formal matching pipeline "
            "using the locked P0 S7 candidate universe."
        ),
    }

    write_json(
        OUT
        / "10_FINAL_DECISION.json",
        final,
    )

    # -------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------
    try:
        count_md = (
            counts.to_markdown(
                index=False
            )
        )

        plan_md = (
            plan.to_markdown(
                index=False
            )
        )

    except Exception:
        count_md = (
            counts.to_string(
                index=False
            )
        )

        plan_md = (
            plan.to_string(
                index=False
            )
        )

    report = f"""# HC-BUG-02C Locked-P0 Replacement Audit

## Final decision

`{decision}`

## Frozen semantics

Formal positive remains:

`AJG1 == 1 AND GSI == 1`

Single-source evidence is not relabeled as positive; it is simply excluded
from trusted hard controls according to the existing conservative protocol.

## Production candidate lineage

Locked table:

`{P0_EDGES}`

Production source-code reference verified:

`{lineage_pass}`

Current formal-edge coverage in P0:

- total: {p0_coverage['formal_edges_total']}
- covered: {p0_coverage['formal_edges_covered']}
- fraction: {p0_coverage['formal_edge_coverage_fraction']}

## Corrected HC00 state

```json
{json.dumps(state, ensure_ascii=False, indent=2)}
```

## Replacement candidate counts

{count_md}

## Dry-run unique replacement plan

{plan_md}

## Simulated cohort QA

```json
{json.dumps(cohort_qa, ensure_ascii=False, indent=2)}
```

## Interpretation

If PASS, a local one-for-one repair of the 23 invalid controls is feasible
while all 10,089 currently valid controls remain fixed.

If blocked, this only means that this minimal local repair is blocked. It does
not establish that a fresh global 5,056-pair 1:2 optimization is infeasible.

No formal dataset was modified.
No model was trained.
"""

    (
        OUT
        / "00_HC_BUG02C_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------
    print(
        "\n"
        + "=" * 116
    )

    print(
        "HC-BUG-02C LOCKED-P0 REPLACEMENT AUDIT COMPLETE"
    )

    print(
        "=" * 116
    )

    print(
        "PRODUCTION P0 LINEAGE:",
        "PASS"
        if lineage_pass
        else "FAIL",
    )

    print(
        "P0 FORMAL EDGE COVERAGE:",
        f"{p0_coverage['formal_edges_covered']}"
        f"/{p0_coverage['formal_edges_total']}",
        f"({p0_coverage['formal_edge_coverage_fraction']:.6f})",
    )

    print()

    print(
        "UPSTREAM ELIGIBLE:",
        state[
            "upstream_eligible"
        ],
    )

    print(
        "CURRENT HC00 CORE:",
        state[
            "current_core"
        ],
    )

    print(
        "REMOVED ILLEGAL CORE:",
        state[
            "removed_core"
        ],
    )

    print(
        "CORRECTED HC00 CORE:",
        state[
            "corrected_core"
        ],
    )

    print()

    print(
        "INVALID FORMAL CONTROLS:",
        state[
            "invalid_controls"
        ],
    )

    print(
        "AFFECTED PAIRS:",
        state[
            "affected_pairs"
        ],
    )

    print()

    print(
        "MIN LEGAL REPLACEMENTS / PAIR:",
        (
            int(
                counts[
                    "legal_replacement_candidates"
                ].min()
            )
            if len(
                counts
            )
            else 0
        ),
    )

    print(
        "MEDIAN LEGAL REPLACEMENTS / PAIR:",
        (
            float(
                counts[
                    "legal_replacement_candidates"
                ].median()
            )
            if len(
                counts
            )
            else 0
        ),
    )

    print(
        "ZERO-CANDIDATE PAIRS:",
        zero_pairs,
    )

    print(
        "NO-REUSE MATCHED PAIRS:",
        matched_n,
        "/",
        EXPECTED_AFFECTED_PAIRSETS,
    )

    print(
        "SIMULATED COHORT QA:",
        cohort_qa.get(
            "status"
        ),
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
        "00_HC_BUG02C_REPORT.md",
        "01_CORRECTED_CORE_AUDIT.json",
        "03_AFFECTED_23_PAIRSETS.csv",
        "05_P0_FORMAL_EDGE_COVERAGE.json",
        "06_AFFECTED_P0_EDGE_READ_META.json",
        "08_REPLACEMENT_CANDIDATE_COUNTS.csv",
        "09_DRYRUN_23_REPLACEMENT_PLAN.csv",
        "09C_SIMULATED_COHORT_QA.json",
        "10_FINAL_DECISION.json",
    ]:
        print(
            OUT / name
        )

    print(
        "=" * 116
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
