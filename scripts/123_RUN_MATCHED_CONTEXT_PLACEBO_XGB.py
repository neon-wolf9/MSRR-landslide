#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
123_RUN_MATCHED_CONTEXT_PLACEBO_XGB.py

Formal internal experiment:
123_MATCHED_CONTEXT_PLACEBO

Scientific question
-------------------
Does the frozen MSRR gain come from the *correct matched-set context*,
rather than merely from:
  (a) increasing feature dimension, or
  (b) applying a generic centering / deviation transform?

This script deliberately reuses the frozen Experiment 120B implementation.
It DOES NOT retrain RAW or TRUE_MSRR. Instead, it reuses the official
XGBoost RAW/MSRR OOF margins already produced by 120B and trains only two
placebo representations under the exact same frozen XGBoost candidate grid,
sample weights, spatial folds, seed convention, and validation-only selection:

    P0_RAW               frozen 120B OOF, 792 dims
    P1_TRUE_MSRR         frozen 120B OOF, 2446 dims
    P2_GLOBAL_CENTERED   newly trained placebo, 2446 dims
    P3_SHUFFLED_CONTEXT  newly trained placebo, 2446 dims

P2_GLOBAL_CENTERED
------------------
Uses the TRAIN-fold global mean only.
Validation/test transforms use that same train-derived mean.

P3_SHUFFLED_CONTEXT
-------------------
Computes matched-set means normally but assigns each target set an alien
context from another set within the SAME partition (train / validation / test).
The permutation is label-blind and is a strict derangement (zero self-matches).
All three members P/C1/C2 of a target set receive the same alien context.
This preserves context-distribution and feature dimension while destroying the
correct set-context correspondence.

Frozen gate
-----------
STRONG_MATCHED_CONTEXT_SUPPORT if:
  1) TRUE_MSRR > GLOBAL_CENTERED on all 4 pooled OOF metrics;
  2) TRUE_MSRR > SHUFFLED_CONTEXT on all 4 pooled OOF metrics;
  3) TRUE_MSRR beats each placebo on StrictPair in >=4/5 folds;
  4) TRUE_MSRR beats each placebo on Edge in >=4/5 folds.

MATCHED_CONTEXT_SUPPORT if:
  TRUE_MSRR beats each placebo on >=3/4 pooled OOF metrics
  but the strong gate above is not fully met.

Otherwise:
  WEAK_OR_NO_CONTEXT_SPECIFIC_SUPPORT

IMPORTANT
---------
- No post-result rescue.
- No outer-test model selection.
- No feature/representation modification after seeing results.
- pair_set_id / sample_role / control_rank / unit_id are metadata only and
  NEVER enter predictive features.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# =============================================================================
# 0. FROZEN PATHS / CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUT120B = (
    ROOT
    / "experiments"
    / "MSRR_CROSS_BACKBONE_5FOLD_V1B"
)

OUT = (
    ROOT
    / "experiments"
    / "MSRR_123_MATCHED_CONTEXT_PLACEBO"
)

SEED = 7
PLACEBO_SEED = 20260829
FOLDS = [1, 2, 3, 4, 5]
METRICS4 = ["AUROC", "AUPRC", "StrictPair", "Edge"]

VARIANTS = [
    "RAW",
    "TRUE_MSRR",
    "GLOBAL_CENTERED",
    "SHUFFLED_CONTEXT",
]

EXPECTED_N = 15168
EXPECTED_PAIRS = 5056

EXPECTED_DIMS = {
    "RAW": 792,
    "TRUE_MSRR": 2446,
    "GLOBAL_CENTERED": 2446,
    "SHUFFLED_CONTEXT": 2446,
}

# Official 120B pooled XGBoost values, only for recovery audit.
REFERENCE_120B = {
    "RAW": {
        "AUROC": 0.671284,
        "AUPRC": 0.489412,
        "StrictPair": 0.559731,
        "Edge": 0.709059,
    },
    "TRUE_MSRR": {
        "AUROC": 0.881482,
        "AUPRC": 0.791908,
        "StrictPair": 0.750198,
        "Edge": 0.846420,
    },
}

REFERENCE_TOL = 5e-4


# =============================================================================
# 1. BASIC UTILITIES
# =============================================================================

def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "123_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(code: str, msg: str) -> None:
    log("")
    log("=" * 120)
    log(code)
    log(msg)
    log("=" * 120)
    raise RuntimeError(msg)


def jwrite(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_sample_column(sample: pd.DataFrame, aliases: List[str]):
    lower = {str(c).lower(): c for c in sample.columns}
    for a in aliases:
        if a.lower() in lower:
            return sample[lower[a.lower()]].to_numpy()
    return None


# =============================================================================
# 2. LOAD THE ORIGINAL 120B SCRIPT AS A MODULE
# =============================================================================

def locate_original_120b_script() -> Path:
    preferred = (
        ROOT
        / "scripts"
        / "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"
    )

    if preferred.exists():
        return preferred

    candidates = sorted(
        (ROOT / "scripts").glob(
            "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT*.py"
        )
    )

    # Do not accidentally select this 123 script.
    candidates = [
        p for p in candidates
        if "123_RUN_MATCHED_CONTEXT_PLACEBO" not in p.name
    ]

    if not candidates:
        raise FileNotFoundError(
            "Cannot locate the frozen 120B script under "
            f"{ROOT / 'scripts'}"
        )

    return candidates[0]


def import_120b_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_120b_module",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 120B script: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Prevent the imported 120B fitting functions from appending logs
    # to the original formal 120B directory.
    mod.log = log
    mod.OUT = OUT

    return mod


# =============================================================================
# 3. FROZEN OOF RECOVERY
# =============================================================================

def locate_margin_file(backbone: str, representation: str) -> Path:
    wanted = (
        f"{backbone}__{representation}"
        .replace("-", "_")
        .replace("__", "_")
        + "_OOF_margin.npy"
    )

    exact = OUT120B / wanted
    if exact.exists():
        return exact

    # Robust fallback using normalized names.
    token_backbone = backbone.lower().replace("-", "").replace("_", "")
    token_rep = representation.lower()

    candidates = list(OUT120B.glob("*OOF_margin.npy"))

    matches = []
    for p in candidates:
        n = p.name.lower().replace("-", "").replace("_", "")
        if token_backbone in n and token_rep in n:
            matches.append(p)

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate OOF margin for "
            f"{backbone}/{representation}; matches={matches}"
        )

    return matches[0]


def load_frozen_xgb_oof(
    m120,
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:

    if not OUT120B.exists():
        fail(
            "FAIL_123_120B_OUTPUT_NOT_FOUND",
            str(OUT120B),
        )

    ref_csv = OUT120B / "OOF_RESULTS.csv"
    if not ref_csv.exists():
        fail(
            "FAIL_123_120B_OOF_RESULTS_NOT_FOUND",
            str(ref_csv),
        )

    ref_df = pd.read_csv(ref_csv, low_memory=False)

    frozen = {}

    mapping = {
        "RAW": ("XGBoost", "RAW"),
        "TRUE_MSRR": ("XGBoost", "MSRR"),
    }

    audit_rows = []

    for variant, (backbone, rep) in mapping.items():
        p = locate_margin_file(backbone, rep)
        arr = np.load(p).astype(np.float64, copy=False)

        if arr.shape != (len(y_all),):
            fail(
                "FAIL_123_FROZEN_MARGIN_SHAPE",
                f"{variant}: {arr.shape}, expected {(len(y_all),)}",
            )

        if not np.isfinite(arr).all():
            fail(
                "FAIL_123_FROZEN_MARGIN_NONFINITE",
                variant,
            )

        metrics = m120.metrics_from_full_oof(
            arr,
            y_all,
            all_pair_rows,
        )

        # Compare against both the official CSV and printed 120B reference.
        row = ref_df[
            (ref_df["backbone"].astype(str) == "XGBoost")
            &
            (ref_df["representation"].astype(str) == rep)
        ]

        if len(row) != 1:
            fail(
                "FAIL_123_120B_REFERENCE_ROW",
                f"Expected exactly one 120B reference row for XGBoost/{rep}.",
            )

        row = row.iloc[0]

        for metric in METRICS4:
            csv_ref = float(row[metric])
            diff_csv = abs(metrics[metric] - csv_ref)
            diff_printed = abs(
                metrics[metric] - REFERENCE_120B[variant][metric]
            )

            if diff_csv > REFERENCE_TOL or diff_printed > REFERENCE_TOL:
                fail(
                    "FAIL_123_FROZEN_OOF_RECALC_MISMATCH",
                    (
                        f"{variant}/{metric}: recalculated={metrics[metric]:.9f}, "
                        f"csv_ref={csv_ref:.9f}, "
                        f"printed_ref={REFERENCE_120B[variant][metric]:.9f}"
                    ),
                )

        frozen[variant] = arr

        audit_rows.append(
            {
                "variant": variant,
                "source_file": str(p),
                "source_sha256": sha256_file(p),
                **metrics,
                "status": "PASS",
            }
        )

    return frozen, pd.DataFrame(audit_rows)


# =============================================================================
# 4. PLACEBO REPRESENTATIONS
# =============================================================================

def global_train_context(
    tr_pairs: np.ndarray,
    static92: np.ndarray,
    rain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Training-fold global mean using TRAIN rows only.
    """
    anchors = np.asarray(tr_pairs, dtype=np.int64).reshape(-1)

    s_mu = static92[anchors].mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    r_mu = rain[anchors].mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    if s_mu.shape != (92,):
        raise RuntimeError(f"GLOBAL_STATIC_MEAN_BAD={s_mu.shape}")

    if r_mu.shape != (70, 10):
        raise RuntimeError(f"GLOBAL_RAIN_MEAN_BAD={r_mu.shape}")

    return s_mu, r_mu


def global_centered_features(
    pair_rows: np.ndarray,
    static92: np.ndarray,
    rain: np.ndarray,
    s_mu_train: np.ndarray,
    r_mu_train: np.ndarray,
    m120,
) -> Tuple[np.ndarray, np.ndarray]:

    tri = np.asarray(pair_rows, dtype=np.int64)
    anchors = tri.reshape(-1)

    s = static92[anchors].astype(np.float32, copy=False)
    r = rain[anchors].astype(np.float32, copy=False)

    s_rel = (
        s - s_mu_train.reshape(1, 92)
    ).astype(np.float32, copy=False)

    r_rel = (
        r - r_mu_train.reshape(1, 70, 10)
    ).astype(np.float32, copy=False)

    n = len(anchors)

    x = np.concatenate(
        [
            s.reshape(n, 92),
            s_rel.reshape(n, 92),
            np.abs(s_rel).reshape(n, 92),
            r.reshape(n, 700),
            r_rel.reshape(n, 700),
            np.abs(r_rel).reshape(n, 700),
            m120.dynsum7(r_rel.reshape(n, 70, 10)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if x.shape[1] != 2446:
        raise RuntimeError(
            f"GLOBAL_CENTERED_DIM_BAD={x.shape[1]} expected=2446"
        )

    if not np.isfinite(x).all():
        raise RuntimeError("NONFINITE_GLOBAL_CENTERED_FEATURES")

    return x, anchors


def sattolo_derangement(n: int, seed: int) -> np.ndarray:
    """
    Sattolo's algorithm generates one cycle, hence no fixed points for n > 1.
    """
    if n <= 1:
        raise RuntimeError(
            f"Cannot derange partition with n_sets={n}"
        )

    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int64)

    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]

    if np.any(perm == np.arange(n)):
        raise RuntimeError("DERANGEMENT_SELF_MATCH_FOUND")

    if len(np.unique(perm)) != n:
        raise RuntimeError("DERANGEMENT_NOT_PERMUTATION")

    return perm


def shuffled_context_features(
    pair_rows: np.ndarray,
    formal_pair_ids: np.ndarray,
    static92: np.ndarray,
    rain: np.ndarray,
    seed: int,
    m120,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    tri = np.asarray(pair_rows, dtype=np.int64)
    formal_pair_ids = np.asarray(formal_pair_ids, dtype=np.int64)

    if len(tri) != len(formal_pair_ids):
        raise RuntimeError("SHUFFLE_PAIR_ID_LENGTH_MISMATCH")

    anchors = tri.reshape(-1)

    s_ctx = static92[tri].astype(
        np.float32,
        copy=False,
    )
    r_ctx = rain[tri].astype(
        np.float32,
        copy=False,
    )

    own_s_mean = s_ctx.mean(
        axis=1,
        keepdims=False,
    )
    own_r_mean = r_ctx.mean(
        axis=1,
        keepdims=False,
    )

    perm = sattolo_derangement(
        len(tri),
        seed,
    )

    alien_s_mean = own_s_mean[perm]
    alien_r_mean = own_r_mean[perm]

    s_rel = (
        s_ctx - alien_s_mean[:, None, :]
    ).astype(np.float32, copy=False)

    r_rel = (
        r_ctx - alien_r_mean[:, None, :, :]
    ).astype(np.float32, copy=False)

    n = len(anchors)

    x = np.concatenate(
        [
            s_ctx.reshape(n, 92),
            s_rel.reshape(n, 92),
            np.abs(s_rel).reshape(n, 92),
            r_ctx.reshape(n, 700),
            r_rel.reshape(n, 700),
            np.abs(r_rel).reshape(n, 700),
            m120.dynsum7(r_rel.reshape(n, 70, 10)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if x.shape[1] != 2446:
        raise RuntimeError(
            f"SHUFFLED_CONTEXT_DIM_BAD={x.shape[1]} expected=2446"
        )

    if not np.isfinite(x).all():
        raise RuntimeError("NONFINITE_SHUFFLED_CONTEXT_FEATURES")

    mapping = pd.DataFrame(
        {
            "target_pair_index": formal_pair_ids,
            "alien_context_pair_index": formal_pair_ids[perm],
            "self_match": (
                formal_pair_ids
                == formal_pair_ids[perm]
            ),
        }
    )

    if mapping["self_match"].any():
        raise RuntimeError("SHUFFLED_CONTEXT_SELF_MATCH_NONZERO")

    return x, anchors, mapping


# =============================================================================
# 5. METADATA FOR ROW-LEVEL OUTPUT
# =============================================================================

def recover_metadata(
    bundle,
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
) -> Dict[str, Any]:

    n = len(y_all)

    recovered_pair_id = np.empty(n, dtype=np.int64)
    recovered_role = np.empty(n, dtype=object)
    recovered_rank = np.full(n, np.nan, dtype=np.float64)

    for pair_i, tri in enumerate(all_pair_rows):
        recovered_pair_id[tri] = pair_i
        recovered_role[tri[0]] = "positive"
        recovered_role[tri[1]] = "control"
        recovered_role[tri[2]] = "control"
        recovered_rank[tri[1]] = 1.0
        recovered_rank[tri[2]] = 2.0

    sample = bundle.sample

    unit_id = get_sample_column(
        sample,
        ["unit_id", "grid_id", "sample_id"],
    )
    if unit_id is None or len(unit_id) != n:
        unit_id = np.arange(n, dtype=np.int64).astype(str)
        unit_id_source = "fallback_sample_index"
    else:
        unit_id_source = "formal_sample_table"

    formal_pair_id = get_sample_column(
        sample,
        ["pair_set_id", "pair_id", "matched_set_id", "set_id"],
    )

    pair_set_id = recovered_pair_id.astype(object)
    pair_id_source = "recovered_from_bundle.pt"

    if formal_pair_id is not None and len(formal_pair_id) == n:
        valid = True
        seen = []
        for tri in all_pair_rows:
            vals = pd.unique(
                pd.Series(formal_pair_id[tri]).astype(str)
            )
            if len(vals) != 1:
                valid = False
                break
            seen.append(vals[0])

        if valid and len(set(seen)) == EXPECTED_PAIRS:
            pair_set_id = formal_pair_id.astype(object)
            pair_id_source = "formal_sample_table"

    formal_role = get_sample_column(
        sample,
        ["sample_role", "role"],
    )
    if formal_role is not None and len(formal_role) == n:
        sample_role = formal_role.astype(object)
        role_source = "formal_sample_table"
    else:
        sample_role = recovered_role
        role_source = "recovered_from_pair_slot"

    formal_rank = get_sample_column(
        sample,
        ["control_rank", "ctrl_rank"],
    )
    if formal_rank is not None and len(formal_rank) == n:
        control_rank = formal_rank
        rank_source = "formal_sample_table"
    else:
        control_rank = recovered_rank
        rank_source = "recovered_from_pair_slot"

    return {
        "unit_id": unit_id,
        "pair_set_id": pair_set_id,
        "sample_role": sample_role,
        "control_rank": control_rank,
        "unit_id_source": unit_id_source,
        "pair_id_source": pair_id_source,
        "role_source": role_source,
        "rank_source": rank_source,
    }


# =============================================================================
# 6. FOLD COMPARISON
# =============================================================================

def fold_win_count(
    fold_df: pd.DataFrame,
    winner: str,
    comparator: str,
    metric: str,
) -> int:

    a = (
        fold_df[
            fold_df["variant"] == winner
        ]
        .set_index("human_fold")[metric]
    )

    b = (
        fold_df[
            fold_df["variant"] == comparator
        ]
        .set_index("human_fold")[metric]
    )

    common = sorted(
        set(a.index) & set(b.index)
    )

    return int(
        sum(
            float(a.loc[f]) > float(b.loc[f])
            for f in common
        )
    )


# =============================================================================
# 7. FIGURE
# =============================================================================

def make_figure(oof_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"WARNING: matplotlib unavailable; figure skipped: {exc}")
        return

    plot = oof_df.set_index("variant").loc[
        VARIANTS,
        METRICS4,
    ]

    x = np.arange(len(METRICS4), dtype=float)
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, variant in enumerate(VARIANTS):
        ax.bar(
            x + (i - 1.5) * width,
            plot.loc[variant].to_numpy(dtype=float),
            width=width,
            label=variant,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS4)
    ax.set_ylabel("OOF metric")
    ax.set_title(
        "Matched-context placebo test: frozen XGBoost protocol"
    )
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "123_MATCHED_CONTEXT_PLACEBO.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "123_MATCHED_CONTEXT_PLACEBO.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 8. MAIN
# =============================================================================

def main(argv=None) -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xgb-jobs",
        type=int,
        default=8,
        help="Use the same XGBoost parallelism as the formal 120B run.",
    )

    args = parser.parse_args(argv)

    # Strict output protection.
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite existing non-empty formal output: "
            f"{OUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fresh log.
    log_path = OUT / "123_RUN.log"
    if log_path.exists():
        log_path.unlink()

    original_script = locate_original_120b_script()
    m120 = import_120b_module(original_script)

    m120.require_xgboost()
    set_seed(SEED)

    manifest = {
        "experiment": "123_MATCHED_CONTEXT_PLACEBO",
        "status_at_start": "FROZEN_BEFORE_RESULTS",
        "original_120b_script": str(original_script),
        "original_120b_script_sha256": sha256_file(original_script),
        "seed": SEED,
        "placebo_seed": PLACEBO_SEED,
        "folds": FOLDS,
        "learner": "XGBoost",
        "xgb_jobs": args.xgb_jobs,
        "xgb_candidates": m120.XGB_CANDIDATES,
        "variants": VARIANTS,
        "expected_dimensions": EXPECTED_DIMS,
        "raw_and_true_msrr_retrained": False,
        "raw_and_true_msrr_source": "frozen 120B OOF margins",
        "placebos_validation_only_selected": True,
        "outer_test_used_for_selection": False,
        "post_result_rescue_allowed": False,
        "forbidden_metadata_used_as_predictive_features": False,
        "global_center_definition": (
            "training-fold global mean only; applied unchanged to validation/test"
        ),
        "shuffled_context_definition": (
            "label-blind within-partition strict derangement of matched-set means; "
            "all three members of each target set share the same alien context"
        ),
        "gate": {
            "STRONG_MATCHED_CONTEXT_SUPPORT": (
                "TRUE_MSRR > both placebo variants on all 4 pooled OOF metrics "
                "AND TRUE_MSRR wins StrictPair and Edge >=4/5 folds against each placebo"
            ),
            "MATCHED_CONTEXT_SUPPORT": (
                "TRUE_MSRR > each placebo on >=3/4 pooled OOF metrics "
                "but strong gate not fully met"
            ),
            "otherwise": "WEAK_OR_NO_CONTEXT_SPECIFIC_SUPPORT",
        },
    }

    jwrite(
        OUT / "123_RUN_MANIFEST.json",
        manifest,
    )

    log("=" * 120)
    log("START 123 MATCHED-CONTEXT PLACEBO")
    log("=" * 120)
    log(f"ORIGINAL_120B_SCRIPT={original_script}")
    log(f"ORIGINAL_120B_SHA256={manifest['original_120b_script_sha256']}")
    log(f"XGB_JOBS={args.xgb_jobs}")
    log(f"PLACEBO_SEED={PLACEBO_SEED}")
    log("RAW_AND_TRUE_MSRR_RETRAINED=NO")

    # -------------------------------------------------------------------------
    # Formal dataset
    # -------------------------------------------------------------------------
    bundle, runner, base = m120.load_dataset_loader()

    y_all = (
        bundle.sample.y_pair
        .to_numpy(np.int8)
    )

    all_pair_rows = np.asarray(
        bundle.pt,
        dtype=np.int64,
    )

    if len(y_all) != EXPECTED_N:
        fail(
            "FAIL_123_SAMPLE_COUNT",
            f"Expected {EXPECTED_N}, got {len(y_all)}",
        )

    if all_pair_rows.shape != (EXPECTED_PAIRS, 3):
        fail(
            "FAIL_123_PAIR_SHAPE",
            f"Expected {(EXPECTED_PAIRS, 3)}, got {all_pair_rows.shape}",
        )

    m120.verify_pair_order(
        all_pair_rows,
        y_all,
        "GLOBAL",
    )

    metadata = recover_metadata(
        bundle,
        y_all,
        all_pair_rows,
    )

    # -------------------------------------------------------------------------
    # Load official frozen 120B XGB OOF for RAW and TRUE_MSRR.
    # -------------------------------------------------------------------------
    frozen_oof, recovery_audit = load_frozen_xgb_oof(
        m120,
        y_all,
        all_pair_rows,
    )

    recovery_audit.to_csv(
        OUT / "123_FROZEN_120B_RECOVERY_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("FROZEN_120B_RAW_MSRR_RECOVERY=PASS")

    # -------------------------------------------------------------------------
    # Reference selected configs from 120B.
    # -------------------------------------------------------------------------
    fold_ref_path = OUT120B / "FOLD_METRICS.csv"
    if not fold_ref_path.exists():
        fail(
            "FAIL_123_120B_FOLD_METRICS_MISSING",
            str(fold_ref_path),
        )

    fold_ref = pd.read_csv(
        fold_ref_path,
        low_memory=False,
    )

    xgb_ref = fold_ref[
        fold_ref["backbone"].astype(str) == "XGBoost"
    ].copy()

    required_ref = {
        "human_fold",
        "representation",
        "selected_config",
        *METRICS4,
    }

    if not required_ref.issubset(set(xgb_ref.columns)):
        fail(
            "FAIL_123_120B_FOLD_REFERENCE_COLUMNS",
            f"Missing columns: {sorted(required_ref - set(xgb_ref.columns))}",
        )

    # -------------------------------------------------------------------------
    # Prepare new OOF arrays.
    # -------------------------------------------------------------------------
    oof = {
        "RAW": frozen_oof["RAW"].copy(),
        "TRUE_MSRR": frozen_oof["TRUE_MSRR"].copy(),
        "GLOBAL_CENTERED": np.full(
            len(y_all),
            np.nan,
            dtype=np.float64,
        ),
        "SHUFFLED_CONTEXT": np.full(
            len(y_all),
            np.nan,
            dtype=np.float64,
        ),
    }

    outer_fold = np.full(
        len(y_all),
        -1,
        dtype=np.int16,
    )

    fold_rows: List[Dict[str, Any]] = []
    selection_tables = []
    selected_config_rows = []
    rep_audit_rows = []
    shuffle_audits = []

    # -------------------------------------------------------------------------
    # Frozen 5-fold loop.
    # -------------------------------------------------------------------------
    for hf in FOLDS:

        log("")
        log("=" * 120)
        log(f"START FOLD{hf}")
        log("=" * 120)

        fold_seed = SEED + hf * 1000
        set_seed(fold_seed)

        fd = m120.build_fold_loader(
            bundle,
            runner,
            base,
            hf,
            torch.device("cpu"),
        )

        static92 = m120.as_numpy(
            fd.static92
        ).astype(
            np.float32,
            copy=False,
        )

        rain = m120.as_numpy(
            fd.rain
        ).astype(
            np.float32,
            copy=False,
        )

        if (
            static92.ndim != 2
            or static92.shape[1] != 92
        ):
            fail(
                "FAIL_123_STATIC_SHAPE",
                f"FOLD{hf}: {static92.shape}",
            )

        if (
            rain.ndim != 3
            or rain.shape[1:] != (70, 10)
        ):
            fail(
                "FAIL_123_RAIN_SHAPE",
                f"FOLD{hf}: {rain.shape}",
            )

        tr_pair_ids = np.asarray(
            fd.train_pairs,
            dtype=np.int64,
        )
        va_pair_ids = np.asarray(
            fd.validation_pairs,
            dtype=np.int64,
        )
        te_pair_ids = np.asarray(
            fd.test_pairs,
            dtype=np.int64,
        )

        tr_pairs = all_pair_rows[tr_pair_ids]
        va_pairs = all_pair_rows[va_pair_ids]
        te_pairs = all_pair_rows[te_pair_ids]

        m120.verify_pair_order(
            tr_pairs,
            y_all,
            f"FOLD{hf}_TRAIN",
        )
        m120.verify_pair_order(
            va_pairs,
            y_all,
            f"FOLD{hf}_VAL",
        )
        m120.verify_pair_order(
            te_pairs,
            y_all,
            f"FOLD{hf}_TEST",
        )

        tr_idx = tr_pairs.reshape(-1)
        va_idx = va_pairs.reshape(-1)
        te_idx = te_pairs.reshape(-1)

        if np.intersect1d(tr_idx, va_idx).size:
            fail(
                "FAIL_123_TRAIN_VAL_OVERLAP",
                f"FOLD{hf}",
            )
        if np.intersect1d(tr_idx, te_idx).size:
            fail(
                "FAIL_123_TRAIN_TEST_OVERLAP",
                f"FOLD{hf}",
            )
        if np.intersect1d(va_idx, te_idx).size:
            fail(
                "FAIL_123_VAL_TEST_OVERLAP",
                f"FOLD{hf}",
            )

        if np.any(outer_fold[te_idx] != -1):
            fail(
                "FAIL_123_OUTER_FOLD_OVERLAP",
                f"FOLD{hf}",
            )
        outer_fold[te_idx] = hf

        ytr = y_all[tr_idx]
        yva = y_all[va_idx]
        yte = y_all[te_idx]

        # ---------------------------------------------------------------------
        # FROZEN RAW / TRUE MSRR fold metrics from frozen official OOF.
        # ---------------------------------------------------------------------
        for variant, ref_rep in [
            ("RAW", "RAW"),
            ("TRUE_MSRR", "MSRR"),
        ]:
            margin = oof[variant][te_idx]

            m = m120.metrics_from_ordered_rows(
                margin,
                yte,
            )

            ref_row = xgb_ref[
                (xgb_ref["human_fold"].astype(int) == hf)
                &
                (xgb_ref["representation"].astype(str) == ref_rep)
            ]

            if len(ref_row) != 1:
                fail(
                    "FAIL_123_FOLD_REFERENCE_ROW",
                    f"FOLD{hf} {ref_rep}",
                )

            ref_row = ref_row.iloc[0]

            for metric in METRICS4:
                if abs(
                    m[metric] - float(ref_row[metric])
                ) > REFERENCE_TOL:
                    fail(
                        "FAIL_123_FROZEN_FOLD_METRIC_MISMATCH",
                        (
                            f"FOLD{hf}/{variant}/{metric}: "
                            f"recalc={m[metric]:.9f}, "
                            f"reference={float(ref_row[metric]):.9f}"
                        ),
                    )

            cfg = str(ref_row["selected_config"])

            fold_rows.append(
                {
                    "human_fold": hf,
                    "variant": variant,
                    "selected_config": cfg,
                    "source": "FROZEN_120B_OOF",
                    **m,
                }
            )

            selected_config_rows.append(
                {
                    "human_fold": hf,
                    "variant": variant,
                    "selected_config": cfg,
                    "selection_source": "FROZEN_120B",
                }
            )

        # ---------------------------------------------------------------------
        # GLOBAL CENTERED placebo
        # ---------------------------------------------------------------------
        s_mu_train, r_mu_train = global_train_context(
            tr_pairs,
            static92,
            rain,
        )

        xtr_g, atr_g = global_centered_features(
            tr_pairs,
            static92,
            rain,
            s_mu_train,
            r_mu_train,
            m120,
        )
        xva_g, ava_g = global_centered_features(
            va_pairs,
            static92,
            rain,
            s_mu_train,
            r_mu_train,
            m120,
        )
        xte_g, ate_g = global_centered_features(
            te_pairs,
            static92,
            rain,
            s_mu_train,
            r_mu_train,
            m120,
        )

        if not np.array_equal(atr_g, tr_idx):
            fail("FAIL_123_GLOBAL_TRAIN_ORDER", f"FOLD{hf}")
        if not np.array_equal(ava_g, va_idx):
            fail("FAIL_123_GLOBAL_VAL_ORDER", f"FOLD{hf}")
        if not np.array_equal(ate_g, te_idx):
            fail("FAIL_123_GLOBAL_TEST_ORDER", f"FOLD{hf}")

        rep_audit_rows.append(
            {
                "human_fold": hf,
                "variant": "GLOBAL_CENTERED",
                "train_dim": xtr_g.shape[1],
                "validation_dim": xva_g.shape[1],
                "test_dim": xte_g.shape[1],
                "finite_train": bool(np.isfinite(xtr_g).all()),
                "finite_validation": bool(np.isfinite(xva_g).all()),
                "finite_test": bool(np.isfinite(xte_g).all()),
                "context_source": "TRAIN_GLOBAL_MEAN_ONLY",
                "forbidden_feature_count": 0,
            }
        )

        g_margin, g_sel, g_cfg = m120.fit_xgb_select_predict(
            "GLOBAL_CENTERED",
            hf,
            xtr_g,
            ytr,
            xva_g,
            yva,
            xte_g,
            fold_seed + 600,
            args.xgb_jobs,
        )

        oof["GLOBAL_CENTERED"][te_idx] = g_margin

        g_m = m120.metrics_from_ordered_rows(
            g_margin,
            yte,
        )

        fold_rows.append(
            {
                "human_fold": hf,
                "variant": "GLOBAL_CENTERED",
                "selected_config": g_cfg,
                "source": "NEW_PLACEBO_TRAINING",
                **g_m,
            }
        )

        selected_config_rows.append(
            {
                "human_fold": hf,
                "variant": "GLOBAL_CENTERED",
                "selected_config": g_cfg,
                "selection_source": "VALIDATION_ONLY",
            }
        )

        g_sel = g_sel.copy()
        g_sel["variant"] = "GLOBAL_CENTERED"
        selection_tables.append(g_sel)

        del xtr_g, xva_g, xte_g, atr_g, ava_g, ate_g, g_margin
        gc.collect()

        # ---------------------------------------------------------------------
        # SHUFFLED CONTEXT placebo
        # ---------------------------------------------------------------------
        xtr_s, atr_s, map_tr = shuffled_context_features(
            tr_pairs,
            tr_pair_ids,
            static92,
            rain,
            PLACEBO_SEED + hf * 100 + 1,
            m120,
        )

        xva_s, ava_s, map_va = shuffled_context_features(
            va_pairs,
            va_pair_ids,
            static92,
            rain,
            PLACEBO_SEED + hf * 100 + 2,
            m120,
        )

        xte_s, ate_s, map_te = shuffled_context_features(
            te_pairs,
            te_pair_ids,
            static92,
            rain,
            PLACEBO_SEED + hf * 100 + 3,
            m120,
        )

        if not np.array_equal(atr_s, tr_idx):
            fail("FAIL_123_SHUFFLE_TRAIN_ORDER", f"FOLD{hf}")
        if not np.array_equal(ava_s, va_idx):
            fail("FAIL_123_SHUFFLE_VAL_ORDER", f"FOLD{hf}")
        if not np.array_equal(ate_s, te_idx):
            fail("FAIL_123_SHUFFLE_TEST_ORDER", f"FOLD{hf}")

        for partition, mapping in [
            ("TRAIN", map_tr),
            ("VALIDATION", map_va),
            ("TEST", map_te),
        ]:
            mapping = mapping.copy()
            mapping.insert(0, "human_fold", hf)
            mapping.insert(1, "partition", partition)

            if mapping["self_match"].any():
                fail(
                    "FAIL_123_SHUFFLE_SELF_MATCH",
                    f"FOLD{hf}/{partition}",
                )

            shuffle_audits.append(mapping)

        rep_audit_rows.append(
            {
                "human_fold": hf,
                "variant": "SHUFFLED_CONTEXT",
                "train_dim": xtr_s.shape[1],
                "validation_dim": xva_s.shape[1],
                "test_dim": xte_s.shape[1],
                "finite_train": bool(np.isfinite(xtr_s).all()),
                "finite_validation": bool(np.isfinite(xva_s).all()),
                "finite_test": bool(np.isfinite(xte_s).all()),
                "context_source": "LABEL_BLIND_WITHIN_PARTITION_DERANGEMENT",
                "forbidden_feature_count": 0,
            }
        )

        s_margin, s_sel, s_cfg = m120.fit_xgb_select_predict(
            "SHUFFLED_CONTEXT",
            hf,
            xtr_s,
            ytr,
            xva_s,
            yva,
            xte_s,
            fold_seed + 600,
            args.xgb_jobs,
        )

        oof["SHUFFLED_CONTEXT"][te_idx] = s_margin

        s_m = m120.metrics_from_ordered_rows(
            s_margin,
            yte,
        )

        fold_rows.append(
            {
                "human_fold": hf,
                "variant": "SHUFFLED_CONTEXT",
                "selected_config": s_cfg,
                "source": "NEW_PLACEBO_TRAINING",
                **s_m,
            }
        )

        selected_config_rows.append(
            {
                "human_fold": hf,
                "variant": "SHUFFLED_CONTEXT",
                "selected_config": s_cfg,
                "selection_source": "VALIDATION_ONLY",
            }
        )

        s_sel = s_sel.copy()
        s_sel["variant"] = "SHUFFLED_CONTEXT"
        selection_tables.append(s_sel)

        del (
            xtr_s,
            xva_s,
            xte_s,
            atr_s,
            ava_s,
            ate_s,
            map_tr,
            map_va,
            map_te,
            s_margin,
            static92,
            rain,
            fd,
        )
        gc.collect()

        log(f"COMPLETE FOLD{hf}")

    # -------------------------------------------------------------------------
    # Coverage audit
    # -------------------------------------------------------------------------
    if np.any(outer_fold < 1):
        fail(
            "FAIL_123_OUTER_FOLD_COVERAGE",
            f"Unassigned rows={int(np.sum(outer_fold < 1))}",
        )

    for variant in VARIANTS:
        arr = oof[variant]
        if not np.isfinite(arr).all():
            fail(
                "FAIL_123_OOF_COVERAGE",
                (
                    f"{variant}: missing="
                    f"{int((~np.isfinite(arr)).sum())}"
                ),
            )

        np.save(
            OUT / f"{variant}_OOF_margin.npy",
            arr,
        )

    # -------------------------------------------------------------------------
    # Pooled OOF metrics
    # -------------------------------------------------------------------------
    pooled_rows = []

    for variant in VARIANTS:
        m = m120.metrics_from_full_oof(
            oof[variant],
            y_all,
            all_pair_rows,
        )

        pooled_rows.append(
            {
                "variant": variant,
                "feature_dim": EXPECTED_DIMS[variant],
                **m,
            }
        )

    oof_df = pd.DataFrame(pooled_rows)

    oof_df.to_csv(
        OUT / "123_OOF_METRICS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_df = pd.DataFrame(fold_rows).sort_values(
        ["human_fold", "variant"]
    )

    fold_df.to_csv(
        OUT / "123_FOLD_METRICS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(selected_config_rows).to_csv(
        OUT / "123_SELECTED_CONFIGS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if selection_tables:
        pd.concat(
            selection_tables,
            ignore_index=True,
            sort=False,
        ).to_csv(
            OUT / "123_VALIDATION_SELECTION_PLACEBOS.csv",
            index=False,
            encoding="utf-8-sig",
        )

    rep_audit_df = pd.DataFrame(rep_audit_rows)
    rep_audit_df.to_csv(
        OUT / "123_REPRESENTATION_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shuffle_audit_df = pd.concat(
        shuffle_audits,
        ignore_index=True,
    )

    shuffle_audit_df.to_csv(
        OUT / "123_CONTEXT_SHUFFLE_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Dimension / feature audit
    # -------------------------------------------------------------------------
    if not (
        (rep_audit_df["train_dim"] == 2446).all()
        and (rep_audit_df["validation_dim"] == 2446).all()
        and (rep_audit_df["test_dim"] == 2446).all()
    ):
        fail(
            "FAIL_123_PLACEBO_DIMENSION_AUDIT",
            "Placebo feature dimensions are not all 2446.",
        )

    if int(rep_audit_df["forbidden_feature_count"].sum()) != 0:
        fail(
            "FAIL_123_FORBIDDEN_FEATURE_AUDIT",
            "Forbidden metadata entered predictive features.",
        )

    if shuffle_audit_df["self_match"].any():
        fail(
            "FAIL_123_SHUFFLE_AUDIT",
            "At least one shuffled context self-match exists.",
        )

    # -------------------------------------------------------------------------
    # Build row-level OOF file for future figure/error analysis.
    # -------------------------------------------------------------------------
    frames = []
    sample_index = np.arange(len(y_all), dtype=np.int64)

    cfg_lookup = {
        (
            int(r["human_fold"]),
            str(r["variant"]),
        ): str(r["selected_config"])
        for r in selected_config_rows
    }

    for variant in VARIANTS:
        selected_cfg = [
            cfg_lookup.get(
                (int(f), variant),
                "",
            )
            for f in outer_fold
        ]

        frames.append(
            pd.DataFrame(
                {
                    "sample_index": sample_index,
                    "unit_id": metadata["unit_id"],
                    "pair_set_id": metadata["pair_set_id"],
                    "sample_role": metadata["sample_role"],
                    "control_rank": metadata["control_rank"],
                    "outer_fold": outer_fold,
                    "seed": SEED,
                    "y_true": y_all,
                    "backbone": "XGBoost",
                    "variant": variant,
                    "risk_score": oof[variant],
                    "probability": m120.sigmoid(oof[variant]),
                    "selected_config": selected_cfg,
                }
            )
        )

    row_oof = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_rows = EXPECTED_N * len(VARIANTS)

    if len(row_oof) != expected_rows:
        fail(
            "FAIL_123_ROW_OOF_COUNT",
            f"Expected {expected_rows}, got {len(row_oof)}",
        )

    row_oof.to_parquet(
        OUT / "123_OOF_PREDICTIONS.parquet",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Formal comparisons / frozen gate
    # -------------------------------------------------------------------------
    metric_map = (
        oof_df
        .set_index("variant")[METRICS4]
        .to_dict(orient="index")
    )

    true = metric_map["TRUE_MSRR"]

    comparison_rows = []

    pooled_wins = {}

    for comparator in [
        "GLOBAL_CENTERED",
        "SHUFFLED_CONTEXT",
    ]:
        cmp = metric_map[comparator]

        pooled_wins[comparator] = int(
            sum(
                float(true[m]) > float(cmp[m])
                for m in METRICS4
            )
        )

        for metric in METRICS4:
            comparison_rows.append(
                {
                    "comparison": f"TRUE_MSRR_vs_{comparator}",
                    "metric": metric,
                    "TRUE_MSRR": float(true[metric]),
                    "COMPARATOR": float(cmp[metric]),
                    "DELTA_TRUE_MINUS_COMPARATOR": float(
                        true[metric] - cmp[metric]
                    ),
                    "TRUE_FOLD_WINS_OUT_OF_5": fold_win_count(
                        fold_df,
                        "TRUE_MSRR",
                        comparator,
                        metric,
                    ),
                }
            )

    comp_df = pd.DataFrame(comparison_rows)

    comp_df.to_csv(
        OUT / "123_TRUE_VS_PLACEBO_COMPARISONS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def foldwins(comparator: str, metric: str) -> int:
        return int(
            comp_df[
                (comp_df["comparison"] == f"TRUE_MSRR_vs_{comparator}")
                &
                (comp_df["metric"] == metric)
            ]["TRUE_FOLD_WINS_OUT_OF_5"].iloc[0]
        )

    strong = (
        pooled_wins["GLOBAL_CENTERED"] == 4
        and pooled_wins["SHUFFLED_CONTEXT"] == 4
        and foldwins("GLOBAL_CENTERED", "StrictPair") >= 4
        and foldwins("GLOBAL_CENTERED", "Edge") >= 4
        and foldwins("SHUFFLED_CONTEXT", "StrictPair") >= 4
        and foldwins("SHUFFLED_CONTEXT", "Edge") >= 4
    )

    moderate = (
        pooled_wins["GLOBAL_CENTERED"] >= 3
        and pooled_wins["SHUFFLED_CONTEXT"] >= 3
    )

    if strong:
        gate = "STRONG_MATCHED_CONTEXT_SUPPORT"
    elif moderate:
        gate = "MATCHED_CONTEXT_SUPPORT"
    else:
        gate = "WEAK_OR_NO_CONTEXT_SPECIFIC_SUPPORT"

    decision = {
        "status": "PASS_123_MATCHED_CONTEXT_PLACEBO_COMPLETE",
        "gate123_decision": gate,
        "pooled_metric_wins_true_vs_global_out_of_4": (
            pooled_wins["GLOBAL_CENTERED"]
        ),
        "pooled_metric_wins_true_vs_shuffled_out_of_4": (
            pooled_wins["SHUFFLED_CONTEXT"]
        ),
        "fold_wins": {
            "TRUE_vs_GLOBAL_StrictPair": foldwins(
                "GLOBAL_CENTERED",
                "StrictPair",
            ),
            "TRUE_vs_GLOBAL_Edge": foldwins(
                "GLOBAL_CENTERED",
                "Edge",
            ),
            "TRUE_vs_SHUFFLED_StrictPair": foldwins(
                "SHUFFLED_CONTEXT",
                "StrictPair",
            ),
            "TRUE_vs_SHUFFLED_Edge": foldwins(
                "SHUFFLED_CONTEXT",
                "Edge",
            ),
        },
        "results": metric_map,
        "post_result_rescue_allowed": False,
        "next_step": (
            "BUILD_SPATIAL_MSRR_FIGURE_AND_CONTINUE_EXTERNAL_DATASET"
        ),
    }

    jwrite(
        OUT / "123_RESULT.json",
        decision,
    )

    make_figure(oof_df)

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------
    log("")
    log("=" * 120)
    log("PASS_123_MATCHED_CONTEXT_PLACEBO_COMPLETE")
    log(f"GATE123_DECISION={gate}")
    log(
        "POOLED_WINS_TRUE_VS_GLOBAL="
        f"{pooled_wins['GLOBAL_CENTERED']}/4"
    )
    log(
        "POOLED_WINS_TRUE_VS_SHUFFLED="
        f"{pooled_wins['SHUFFLED_CONTEXT']}/4"
    )
    log("=" * 120)

    for variant in VARIANTS:
        m = metric_map[variant]
        log(
            f"{variant}: "
            f"AUROC={m['AUROC']:.6f} "
            f"AUPRC={m['AUPRC']:.6f} "
            f"StrictPair={m['StrictPair']:.6f} "
            f"Edge={m['Edge']:.6f}"
        )

    log("")
    for comparator in [
        "GLOBAL_CENTERED",
        "SHUFFLED_CONTEXT",
    ]:
        log(f"TRUE_MSRR_MINUS_{comparator}:")
        for metric in METRICS4:
            r = comp_df[
                (comp_df["comparison"] == f"TRUE_MSRR_vs_{comparator}")
                &
                (comp_df["metric"] == metric)
            ].iloc[0]
            log(
                f"  {metric}: "
                f"delta={r['DELTA_TRUE_MINUS_COMPARATOR']:+.6f} "
                f"fold_wins={int(r['TRUE_FOLD_WINS_OUT_OF_5'])}/5"
            )

    log("")
    log(
        "SHUFFLED_SELF_MATCHES="
        f"{int(shuffle_audit_df['self_match'].sum())}"
    )
    log("FORBIDDEN_FEATURE_COUNT=0")
    log(
        "OOF_PREDICTIONS="
        f"{OUT / '123_OOF_PREDICTIONS.parquet'}"
    )
    log(f"OUTPUT={OUT}")
    log(
        "NEXT_STEP="
        "BUILD_SPATIAL_MSRR_FIGURE_AND_CONTINUE_EXTERNAL_DATASET"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("")
        print("=" * 120)
        print("123 EXPERIMENT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 120)
        traceback.print_exc()
        raise SystemExit(1)
