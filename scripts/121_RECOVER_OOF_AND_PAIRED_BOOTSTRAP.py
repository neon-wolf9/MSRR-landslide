#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP.py
=======================================

Purpose
-------
Recover row-level OOF predictions from the EIGHT frozen *_OOF_margin.npy files
already written by Experiment 120B, reconstruct fold/sample metadata from the
formal dataset loader, verify that the recovered scores reproduce the published
120B OOF metrics, and then run a matched-set paired bootstrap.

IMPORTANT
---------
* NO model retraining.
* NO representation rebuilding.
* NO hyperparameter reselection.
* NO modification of the frozen 120B results directory.
* Bootstrap unit = matched set (one positive + two controls).
* RAW and MSRR always use the SAME bootstrap draw.

Expected source directory
-------------------------
<PROJECT_ROOT>\\experiments\\MSRR_CROSS_BACKBONE_5FOLD_V1B

Expected frozen source artifacts include
----------------------------------------
OOF_RESULTS.csv
FOLD_METRICS.csv
ElasticNet_Logistic_RAW_OOF_margin.npy
ElasticNet_Logistic_MSRR_OOF_margin.npy
MLP_RAW_OOF_margin.npy
MLP_MSRR_OOF_margin.npy
HistGradientBoosting_RAW_OOF_margin.npy
HistGradientBoosting_MSRR_OOF_margin.npy
XGBoost_RAW_OOF_margin.npy
XGBoost_MSRR_OOF_margin.npy

Outputs
-------
<PROJECT_ROOT>\\experiments\\MSRR_121_PAIRED_BOOTSTRAP

Key outputs:
120B_RECOVERED_ROW_LEVEL_OOF.parquet
121_OOF_RECOVERY_AUDIT.csv
121_BOOTSTRAP_SUMMARY.csv
121_BOOTSTRAP_ALL_REPLICATES.parquet
121_PAIRED_BOOTSTRAP_FOREST.png/.pdf
121_RESULT.json
121_RUN_MANIFEST.json
121_RUN.log
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# 0. FROZEN CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC120B = ROOT / "experiments" / "MSRR_CROSS_BACKBONE_5FOLD_V1B"
OUT = ROOT / "experiments" / "MSRR_121_PAIRED_BOOTSTRAP"

SEED = 7
FOLDS = [1, 2, 3, 4, 5]
BOOTSTRAP_SEED = 20260829
B = 10_000

EXPECTED_N = 15_168
EXPECTED_PAIRS = 5_056
EXPECTED_ROWS_COMBINED = EXPECTED_N * 8
METRICS4 = ["AUROC", "AUPRC", "StrictPair", "Edge"]

BACKBONES = [
    "ElasticNet-Logistic",
    "MLP",
    "HistGradientBoosting",
    "XGBoost",
]
REPRESENTATIONS = ["RAW", "MSRR"]

METHODS = [
    f"{b}__{r}"
    for b in BACKBONES
    for r in REPRESENTATIONS
]

# Exact file naming logic used by 120B.
def margin_filename(method: str) -> str:
    return (
        method.replace("-", "_")
        .replace("__", "_")
        + "_OOF_margin.npy"
    )

# Tolerance is only for reproduction audit. Because the SAME stored margins are
# used, differences should normally be near machine precision.
REPRO_TOL = 1e-9


# =============================================================================
# 1. HELPERS
# =============================================================================

def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "121_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(code: str, msg: str) -> None:
    log("")
    log("=" * 120)
    log(code)
    log(msg)
    log("=" * 120)
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def jwrite(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def to_numpy_1d(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    elif hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr).reshape(-1)


def get_sample_column(sample: Any, candidates: Sequence[str]) -> Optional[np.ndarray]:
    """Best-effort extraction from the formal sample table/object."""
    # pandas-like __getitem__
    for c in candidates:
        try:
            if hasattr(sample, "columns") and c in list(sample.columns):
                return to_numpy_1d(sample[c])
        except Exception:
            pass

    # attribute access
    for c in candidates:
        try:
            if hasattr(sample, c):
                return to_numpy_1d(getattr(sample, c))
        except Exception:
            pass

    return None


def metrics_from_full_oof(
    margin_full: np.ndarray,
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
) -> Dict[str, float]:
    margin_full = np.asarray(margin_full, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.int8)
    pair_rows = np.asarray(all_pair_rows, dtype=np.int64)

    if margin_full.shape != y_all.shape:
        raise RuntimeError(
            f"MARGIN_LABEL_SHAPE_MISMATCH margin={margin_full.shape} y={y_all.shape}"
        )
    if not np.isfinite(margin_full).all():
        raise RuntimeError("INCOMPLETE_OOF_MARGIN")

    s3 = margin_full[pair_rows]
    y3 = y_all[pair_rows]

    if not np.all(y3[:, 0] == 1):
        raise RuntimeError("METRIC_POSITIVE_NOT_SLOT0")
    if not np.all(y3[:, 1:] == 0):
        raise RuntimeError("METRIC_CONTROL_LABEL_BAD")

    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]
    p = sigmoid(margin_full)

    return {
        "AUROC": float(roc_auc_score(y_all, p)),
        "AUPRC": float(average_precision_score(y_all, p)),
        "StrictPair": float((e1 & e2).mean()),
        "Edge": float(np.stack([e1, e2], axis=1).mean()),
    }


def pair_success_arrays(
    margin_full: np.ndarray,
    all_pair_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    s3 = np.asarray(margin_full, dtype=np.float64)[
        np.asarray(all_pair_rows, dtype=np.int64)
    ]
    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]
    strict = (e1 & e2).astype(np.float64)
    edge_count = (e1.astype(np.int8) + e2.astype(np.int8)).astype(np.float64)
    return strict, edge_count


# =============================================================================
# 2. LOAD FORMAL DATASET STRUCTURE (NO TRAINING)
# =============================================================================

def load_formal_structure():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.TRAIN_SEHC_V2_TEMPLATE import build_fold_loader, load_dataset_loader
    import torch

    log("Loading formal dataset structure (no model training)...")
    bundle, runner, base = load_dataset_loader()

    y_all = bundle.sample.y_pair.to_numpy(np.int8)
    all_pair_rows = np.asarray(bundle.pt, dtype=np.int64)

    if len(y_all) != EXPECTED_N:
        fail(
            "FAIL_121_UNEXPECTED_SAMPLE_COUNT",
            f"Expected {EXPECTED_N}, got {len(y_all)}",
        )

    if all_pair_rows.shape != (EXPECTED_PAIRS, 3):
        fail(
            "FAIL_121_UNEXPECTED_PAIR_SHAPE",
            f"Expected ({EXPECTED_PAIRS},3), got {all_pair_rows.shape}",
        )

    y3 = y_all[all_pair_rows]
    if not np.all(y3[:, 0] == 1) or not np.all(y3[:, 1:] == 0):
        fail(
            "FAIL_121_FORMAL_PAIR_ORDER",
            "Formal pair rows are not ordered [positive, control1, control2].",
        )

    # ---------------------------------------------------------
    # Reconstruct outer-test fold membership from the SAME
    # frozen fold loader used by 120B. No feature/model fitting.
    # ---------------------------------------------------------
    outer_fold = np.full(len(y_all), -1, dtype=np.int16)

    for hf in FOLDS:
        log(f"Recovering formal outer-test membership for fold {hf}...")
        fd = build_fold_loader(
            bundle,
            runner,
            base,
            hf,
            torch.device("cpu"),
        )
        te_pair_ids = np.asarray(fd.test_pairs, dtype=np.int64)
        te_rows = all_pair_rows[te_pair_ids].reshape(-1)

        if np.any(outer_fold[te_rows] != -1):
            fail(
                "FAIL_121_FOLD_OVERLAP",
                f"Fold {hf} overlaps prior outer-test assignment.",
            )
        outer_fold[te_rows] = hf
        del fd
        gc.collect()

    if np.any(outer_fold < 1):
        fail(
            "FAIL_121_FOLD_COVERAGE",
            f"Unassigned outer-test rows={int(np.sum(outer_fold < 1))}",
        )

    # ---------------------------------------------------------
    # Derive guaranteed matched-set metadata from all_pair_rows.
    # Use original columns if present and structurally valid.
    # ---------------------------------------------------------
    recovered_pair_id = np.empty(len(y_all), dtype=np.int64)
    recovered_role = np.empty(len(y_all), dtype=object)
    recovered_rank = np.full(len(y_all), np.nan, dtype=np.float64)

    for pair_i, tri in enumerate(all_pair_rows):
        recovered_pair_id[tri] = pair_i
        recovered_role[tri[0]] = "positive"
        recovered_role[tri[1]] = "control"
        recovered_role[tri[2]] = "control"
        recovered_rank[tri[1]] = 1.0
        recovered_rank[tri[2]] = 2.0

    sample = bundle.sample

    unit_id = get_sample_column(sample, ["unit_id", "grid_id", "sample_id"])
    if unit_id is None or len(unit_id) != len(y_all):
        log("WARNING: formal unit_id not found; using row_index as fallback.")
        unit_id = np.arange(len(y_all), dtype=np.int64).astype(str)
        unit_id_source = "fallback_row_index"
    else:
        unit_id_source = "formal_sample_table"

    formal_pair_id = get_sample_column(
        sample,
        ["pair_set_id", "pair_id", "matched_set_id", "set_id"],
    )

    pair_id_source = "recovered_from_bundle.pt"
    pair_set_id = recovered_pair_id.astype(object)

    if formal_pair_id is not None and len(formal_pair_id) == len(y_all):
        valid = True
        seen = []
        for tri in all_pair_rows:
            vals = pd.unique(pd.Series(formal_pair_id[tri]).astype(str))
            if len(vals) != 1:
                valid = False
                break
            seen.append(vals[0])
        if valid and len(set(seen)) == EXPECTED_PAIRS:
            pair_set_id = formal_pair_id.astype(object)
            pair_id_source = "formal_sample_table"
        else:
            log(
                "WARNING: formal pair_set_id exists but failed structural audit; "
                "using pair IDs recovered from bundle.pt."
            )

    formal_role = get_sample_column(sample, ["sample_role", "role"])
    sample_role = recovered_role.copy()
    role_source = "recovered_from_pair_slot"
    if formal_role is not None and len(formal_role) == len(y_all):
        sample_role = formal_role.astype(object)
        role_source = "formal_sample_table"

    formal_rank = get_sample_column(sample, ["control_rank", "ctrl_rank"])
    control_rank = recovered_rank.copy()
    rank_source = "recovered_from_pair_slot"
    if formal_rank is not None and len(formal_rank) == len(y_all):
        control_rank = formal_rank
        rank_source = "formal_sample_table"

    meta = {
        "unit_id": unit_id,
        "pair_set_id": pair_set_id,
        "sample_role": sample_role,
        "control_rank": control_rank,
        "outer_fold": outer_fold,
        "unit_id_source": unit_id_source,
        "pair_id_source": pair_id_source,
        "role_source": role_source,
        "rank_source": rank_source,
    }

    return bundle, runner, base, y_all, all_pair_rows, meta


# =============================================================================
# 3. RECOVER FROZEN MARGINS + VERIFY 120B EXACTLY
# =============================================================================

def load_reference_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    oof_path = SRC120B / "OOF_RESULTS.csv"
    fold_path = SRC120B / "FOLD_METRICS.csv"

    if not oof_path.exists():
        fail("FAIL_121_REFERENCE_OOF_RESULTS_MISSING", str(oof_path))
    if not fold_path.exists():
        fail("FAIL_121_REFERENCE_FOLD_METRICS_MISSING", str(fold_path))

    oof_ref = pd.read_csv(oof_path, low_memory=False)
    fold_ref = pd.read_csv(fold_path, low_memory=False)

    required_oof = {"backbone", "representation", *METRICS4}
    if not required_oof.issubset(set(oof_ref.columns)):
        fail(
            "FAIL_121_BAD_REFERENCE_OOF_COLUMNS",
            f"columns={list(oof_ref.columns)}",
        )

    return oof_ref, fold_ref


def recover_margins(
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
    oof_ref: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, str]]:
    margins: Dict[str, np.ndarray] = {}
    audit_rows = []
    source_hashes: Dict[str, str] = {}

    for method in METHODS:
        backbone, rep = method.split("__", 1)
        path = SRC120B / margin_filename(method)

        if not path.exists():
            fail(
                "FAIL_121_FROZEN_MARGIN_MISSING",
                f"Missing frozen OOF margin: {path}",
            )

        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)

        if len(arr) != EXPECTED_N:
            fail(
                "FAIL_121_MARGIN_LENGTH",
                f"{path.name}: expected {EXPECTED_N}, got {len(arr)}",
            )
        if not np.isfinite(arr).all():
            fail(
                "FAIL_121_MARGIN_NONFINITE",
                f"{path.name}: nonfinite={int((~np.isfinite(arr)).sum())}",
            )

        calc = metrics_from_full_oof(arr, y_all, all_pair_rows)

        ref_row = oof_ref[
            (oof_ref["backbone"] == backbone)
            & (oof_ref["representation"] == rep)
        ]
        if len(ref_row) != 1:
            fail(
                "FAIL_121_REFERENCE_CELL",
                f"Expected one reference row for {backbone}/{rep}, got {len(ref_row)}",
            )
        ref_row = ref_row.iloc[0]

        row = {
            "backbone": backbone,
            "representation": rep,
            "margin_file": str(path),
            "n_rows": len(arr),
        }

        for metric in METRICS4:
            ref = float(ref_row[metric])
            got = float(calc[metric])
            diff = abs(got - ref)
            row[f"{metric}_reference"] = ref
            row[f"{metric}_recovered"] = got
            row[f"{metric}_abs_diff"] = diff
            if diff > REPRO_TOL:
                fail(
                    "FAIL_121_RECOVERY_METRIC_MISMATCH",
                    f"{backbone}/{rep}/{metric}: recovered={got:.12f}, "
                    f"reference={ref:.12f}, abs_diff={diff:.3e}",
                )

        margins[method] = arr
        source_hashes[path.name] = sha256_file(path)
        audit_rows.append(row)

        log(
            f"RECOVERED {backbone} {rep}: "
            f"AUROC={calc['AUROC']:.6f} AUPRC={calc['AUPRC']:.6f} "
            f"StrictPair={calc['StrictPair']:.6f} Edge={calc['Edge']:.6f}"
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        OUT / "121_OOF_RECOVERY_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return margins, audit_df, source_hashes


# =============================================================================
# 4. CREATE ROW-LEVEL OOF FILE WITHOUT RETRAINING
# =============================================================================

def config_map_from_fold_metrics(fold_ref: pd.DataFrame) -> Dict[Tuple[int, str, str], str]:
    needed = {"human_fold", "backbone", "representation"}
    if not needed.issubset(set(fold_ref.columns)):
        return {}

    cfg_col = "selected_config" if "selected_config" in fold_ref.columns else None
    if cfg_col is None:
        return {}

    out = {}
    for _, r in fold_ref.iterrows():
        out[(int(r["human_fold"]), str(r["backbone"]), str(r["representation"]))] = str(
            r[cfg_col]
        )
    return out


def build_row_level_oof(
    y_all: np.ndarray,
    meta: Dict[str, Any],
    margins: Dict[str, np.ndarray],
    fold_ref: pd.DataFrame,
) -> pd.DataFrame:
    cfg_map = config_map_from_fold_metrics(fold_ref)

    frames = []
    n = len(y_all)
    idx = np.arange(n, dtype=np.int64)

    for method in METHODS:
        backbone, rep = method.split("__", 1)
        margin = margins[method]
        prob = sigmoid(margin)
        fold_arr = np.asarray(meta["outer_fold"], dtype=np.int16)

        selected_config = np.asarray(
            [
                cfg_map.get((int(f), backbone, rep), "")
                for f in fold_arr
            ],
            dtype=object,
        )

        frame = pd.DataFrame(
            {
                "sample_index": idx,
                "unit_id": meta["unit_id"],
                "pair_set_id": meta["pair_set_id"],
                "sample_role": meta["sample_role"],
                "control_rank": meta["control_rank"],
                "outer_fold": fold_arr,
                "seed": SEED,
                "y_true": y_all,
                "backbone": backbone,
                "representation": rep,
                "risk_score": margin,
                "probability": prob,
                "selected_config": selected_config,
            }
        )
        frames.append(frame)

    out_df = pd.concat(frames, ignore_index=True)

    if len(out_df) != EXPECTED_ROWS_COMBINED:
        fail(
            "FAIL_121_COMBINED_OOF_ROWS",
            f"Expected {EXPECTED_ROWS_COMBINED}, got {len(out_df)}",
        )

    # 8 cells x 15168 rows.
    counts = (
        out_df.groupby(["backbone", "representation"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    if not np.all(counts["n"].to_numpy() == EXPECTED_N):
        fail(
            "FAIL_121_CELL_ROW_COUNTS",
            counts.to_string(index=False),
        )

    # Matched-set structure must be 1 positive + 2 controls in every method cell.
    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            cell = out_df[
                (out_df["backbone"] == backbone)
                & (out_df["representation"] == rep)
            ]
            g = cell.groupby("pair_set_id")["y_true"].agg(["size", "sum"])
            if len(g) != EXPECTED_PAIRS or (g["size"] != 3).any() or (g["sum"] != 1).any():
                fail(
                    "FAIL_121_ROW_FILE_PAIR_STRUCTURE",
                    f"Bad pair structure in {backbone}/{rep}",
                )

    parquet_path = OUT / "120B_RECOVERED_ROW_LEVEL_OOF.parquet"
    out_df.to_parquet(parquet_path, index=False)

    # Small audit table, not a 121k-row CSV copy.
    counts.to_csv(
        OUT / "121_ROW_LEVEL_OOF_COUNTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log(f"ROW_LEVEL_OOF_SAVED={parquet_path}")
    log(f"ROW_LEVEL_OOF_ROWS={len(out_df)}")
    return out_df


# =============================================================================
# 5. MATCHED-SET PAIRED BOOTSTRAP
# =============================================================================

def bootstrap_one_backbone(
    backbone: str,
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
    margins: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_margin = margins[f"{backbone}__RAW"]
    msrr_margin = margins[f"{backbone}__MSRR"]

    raw_prob = sigmoid(raw_margin)
    msrr_prob = sigmoid(msrr_margin)

    raw_obs = metrics_from_full_oof(raw_margin, y_all, all_pair_rows)
    msrr_obs = metrics_from_full_oof(msrr_margin, y_all, all_pair_rows)

    raw_strict, raw_edge_count = pair_success_arrays(raw_margin, all_pair_rows)
    msrr_strict, msrr_edge_count = pair_success_arrays(msrr_margin, all_pair_rows)

    # Each sample row belongs to exactly one matched set.
    row_pair_pos = np.empty(len(y_all), dtype=np.int64)
    for pair_pos, tri in enumerate(all_pair_rows):
        row_pair_pos[np.asarray(tri, dtype=np.int64)] = pair_pos

    # Reset the RNG to the SAME seed for each backbone so all four backbones use
    # the same sequence of matched-set bootstrap draws.
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot_records: List[Dict[str, Any]] = []
    n_pairs = len(all_pair_rows)

    for b in range(B):
        sampled_pair_pos = rng.integers(0, n_pairs, size=n_pairs)
        pair_counts = np.bincount(sampled_pair_pos, minlength=n_pairs).astype(np.float64)
        row_weights = pair_counts[row_pair_pos]

        raw_auc = float(roc_auc_score(y_all, raw_prob, sample_weight=row_weights))
        msrr_auc = float(roc_auc_score(y_all, msrr_prob, sample_weight=row_weights))
        raw_ap = float(
            average_precision_score(y_all, raw_prob, sample_weight=row_weights)
        )
        msrr_ap = float(
            average_precision_score(y_all, msrr_prob, sample_weight=row_weights)
        )

        denom = float(pair_counts.sum())
        raw_sp = float(np.dot(pair_counts, raw_strict) / denom)
        msrr_sp = float(np.dot(pair_counts, msrr_strict) / denom)
        raw_edge = float(np.dot(pair_counts, raw_edge_count) / (2.0 * denom))
        msrr_edge = float(np.dot(pair_counts, msrr_edge_count) / (2.0 * denom))

        vals = {
            "AUROC": (raw_auc, msrr_auc),
            "AUPRC": (raw_ap, msrr_ap),
            "StrictPair": (raw_sp, msrr_sp),
            "Edge": (raw_edge, msrr_edge),
        }

        for metric, (rv, mv) in vals.items():
            boot_records.append(
                {
                    "bootstrap_id": b + 1,
                    "backbone": backbone,
                    "metric": metric,
                    "RAW": rv,
                    "MSRR": mv,
                    "delta": mv - rv,
                }
            )

        if (b + 1) % 1000 == 0:
            log(f"{backbone}: bootstrap {b + 1}/{B}")

    boot_df = pd.DataFrame(boot_records)
    summary_rows = []

    for metric in METRICS4:
        delta = boot_df.loc[boot_df["metric"] == metric, "delta"].to_numpy(np.float64)
        ci_low, ci_high = np.quantile(delta, [0.025, 0.975])

        p_lower = (np.sum(delta <= 0.0) + 1.0) / (len(delta) + 1.0)
        p_upper = (np.sum(delta >= 0.0) + 1.0) / (len(delta) + 1.0)
        p_two = min(1.0, 2.0 * min(p_lower, p_upper))

        summary_rows.append(
            {
                "backbone": backbone,
                "metric": metric,
                "RAW_OOF": raw_obs[metric],
                "MSRR_OOF": msrr_obs[metric],
                "OBSERVED_DELTA": msrr_obs[metric] - raw_obs[metric],
                "BOOTSTRAP_MEAN_DELTA": float(delta.mean()),
                "BOOTSTRAP_MEDIAN_DELTA": float(np.median(delta)),
                "CI95_LOW": float(ci_low),
                "CI95_HIGH": float(ci_high),
                "P_DELTA_GT_0": float(np.mean(delta > 0.0)),
                "TWO_SIDED_BOOTSTRAP_P": float(p_two),
                "STATISTICALLY_SUPPORTED": bool(ci_low > 0.0),
                "B": B,
                "SEED": BOOTSTRAP_SEED,
            }
        )

    return pd.DataFrame(summary_rows), boot_df


# =============================================================================
# 6. FIGURE
# =============================================================================

def make_forest_plot(summary: pd.DataFrame) -> None:
    order_b = {b: i for i, b in enumerate(BACKBONES)}
    order_m = {m: i for i, m in enumerate(METRICS4)}

    d = summary.copy()
    d["_b"] = d["backbone"].map(order_b)
    d["_m"] = d["metric"].map(order_m)
    d = d.sort_values(["_b", "_m"]).reset_index(drop=True)

    y = np.arange(len(d))
    x = d["OBSERVED_DELTA"].to_numpy(np.float64)
    lo = d["CI95_LOW"].to_numpy(np.float64)
    hi = d["CI95_HIGH"].to_numpy(np.float64)
    xerr = np.vstack([x - lo, hi - x])
    labels = [f"{r.backbone} — {r.metric}" for r in d.itertuples()]

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.2)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("MSRR − RAW metric improvement")
    ax.set_title("Paired matched-set bootstrap: MSRR vs RAW")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    fig.savefig(OUT / "121_PAIRED_BOOTSTRAP_FOREST.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "121_PAIRED_BOOTSTRAP_FOREST.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "121_RUN.log"
    if log_path.exists():
        log_path.unlink()

    log("=" * 120)
    log("121 RECOVER FROZEN 120B OOF + PAIRED BOOTSTRAP")
    log("=" * 120)
    log(f"SRC120B={SRC120B}")
    log(f"OUT={OUT}")
    log("MODEL_RETRAINING=NO")
    log("REPRESENTATION_REBUILD=NO")
    log("HYPERPARAMETER_RESELECTION=NO")
    log(f"BOOTSTRAP_UNIT=matched_set")
    log(f"B={B}")
    log(f"BOOTSTRAP_SEED={BOOTSTRAP_SEED}")

    if not SRC120B.exists():
        fail("FAIL_121_SOURCE_DIR_MISSING", str(SRC120B))

    oof_ref, fold_ref = load_reference_tables()
    bundle, runner, base, y_all, all_pair_rows, meta = load_formal_structure()

    margins, recovery_audit, margin_hashes = recover_margins(
        y_all,
        all_pair_rows,
        oof_ref,
    )

    row_oof = build_row_level_oof(
        y_all,
        meta,
        margins,
        fold_ref,
    )

    # Input/provenance manifest before bootstrap.
    manifest = {
        "experiment": "121_RECOVER_OOF_AND_PAIRED_BOOTSTRAP",
        "source_120b_dir": str(SRC120B),
        "source_120b_oof_results_sha256": sha256_file(SRC120B / "OOF_RESULTS.csv"),
        "source_120b_fold_metrics_sha256": sha256_file(SRC120B / "FOLD_METRICS.csv"),
        "source_margin_sha256": margin_hashes,
        "no_model_retraining": True,
        "no_representation_rebuild": True,
        "no_hyperparameter_reselection": True,
        "n_samples": len(y_all),
        "n_pairs": len(all_pair_rows),
        "combined_row_level_oof_rows": len(row_oof),
        "unit_id_source": meta["unit_id_source"],
        "pair_id_source": meta["pair_id_source"],
        "sample_role_source": meta["role_source"],
        "control_rank_source": meta["rank_source"],
        "bootstrap_B": B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "matched_set",
        "raw_msrr_same_bootstrap_draw": True,
        "same_draw_sequence_across_backbones": True,
    }
    jwrite(OUT / "121_RUN_MANIFEST.json", manifest)

    # Bootstrap.
    summaries = []
    replicates = []

    for backbone in BACKBONES:
        log("-" * 120)
        log(f"START_BOOTSTRAP={backbone}")
        s, r = bootstrap_one_backbone(
            backbone,
            y_all,
            all_pair_rows,
            margins,
        )
        summaries.append(s)
        replicates.append(r)

    summary = pd.concat(summaries, ignore_index=True)
    reps = pd.concat(replicates, ignore_index=True)

    summary.to_csv(
        OUT / "121_BOOTSTRAP_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reps.to_parquet(
        OUT / "121_BOOTSTRAP_ALL_REPLICATES.parquet",
        index=False,
    )

    supported_n = int(summary["STATISTICALLY_SUPPORTED"].sum())

    if supported_n == 16:
        gate = "STRONG_PAIRED_STATISTICAL_SUPPORT"
    else:
        per_backbone = (
            summary.groupby("backbone")["STATISTICALLY_SUPPORTED"]
            .sum()
            .to_dict()
        )
        if supported_n >= 12 and all(int(per_backbone.get(b, 0)) >= 3 for b in BACKBONES):
            gate = "CROSS_BACKBONE_STATISTICAL_SUPPORT"
        else:
            gate = "PARTIAL_OR_WEAK_STATISTICAL_SUPPORT"

    result = {
        "status": "PASS_121_PAIRED_BOOTSTRAP_COMPLETE",
        "gate121_decision": gate,
        "supported_comparisons": supported_n,
        "total_comparisons": 16,
        "bootstrap_B": B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "matched_set",
        "row_level_oof_recovered_from_frozen_120b_margin_arrays": True,
        "model_retraining_performed": False,
        "summary": summary.to_dict(orient="records"),
    }
    jwrite(OUT / "121_RESULT.json", result)

    make_forest_plot(summary)

    log("")
    log("=" * 120)
    log("PASS_121_PAIRED_BOOTSTRAP_COMPLETE")
    log(f"GATE121_DECISION={gate}")
    log(f"SUPPORTED_COMPARISONS={supported_n}/16")
    log("OOF_RECOVERY_SOURCE=FROZEN_120B_*_OOF_margin.npy")
    log("MODEL_RETRAINING=NO")
    log("=" * 120)

    for backbone in BACKBONES:
        log("")
        log(f"{backbone}:")
        sub = summary[summary["backbone"] == backbone]
        for metric in METRICS4:
            r = sub[sub["metric"] == metric].iloc[0]
            log(
                f"  {metric}: RAW={r['RAW_OOF']:.6f} "
                f"MSRR={r['MSRR_OOF']:.6f} "
                f"delta={r['OBSERVED_DELTA']:+.6f} "
                f"CI95=[{r['CI95_LOW']:+.6f},{r['CI95_HIGH']:+.6f}] "
                f"P(delta>0)={r['P_DELTA_GT_0']:.6f} "
                f"p={r['TWO_SIDED_BOOTSTRAP_P']:.6g}"
            )

    log("")
    log(f"ROW_LEVEL_OOF={OUT / '120B_RECOVERED_ROW_LEVEL_OOF.parquet'}")
    log(f"OUTPUT={OUT}")
    log("NEXT_STEP=123_MATCHED_CONTEXT_PLACEBO")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        jwrite(
            OUT / "121_FAILURE.json",
            {
                "status": "FAIL_121_RECOVER_OOF_AND_BOOTSTRAP",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(traceback.format_exc(), flush=True)
        raise
