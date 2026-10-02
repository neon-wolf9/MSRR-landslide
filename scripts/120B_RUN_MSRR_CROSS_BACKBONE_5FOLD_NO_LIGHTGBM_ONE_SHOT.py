#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py

Revised Cross-Backbone Validation after LightGBM native-runtime failure
======================================================================

Why 120B exists
---------------
The original 120 experiment was blocked before scientific evaluation because
LightGBM raised a native Windows access-violation on the formal project data.
A repaired clone could fit synthetic data but still failed on the formal
LightGBM fit. Therefore LightGBM is removed for a TECHNICAL reason, not because
of its predictive performance.

No LightGBM result from 120 is used.

The scientific question remains unchanged:

    Does the frozen MSRR representation improve different learner families,
    or is the gain specific to XGBoost?

Frozen four-backbone panel
--------------------------
1) ElasticNet-Logistic       linear learner
2) MLP                       neural learner
3) HistGradientBoosting      sklearn tree-boosting learner
4) XGBoost                   external tree-boosting learner

Each learner receives BOTH:
    RAW
    MSRR

=> 8 combinations.

Frozen representations
----------------------
RAW:
    92 static + full 70x10 dynamic sequence = 792 dims

MSRR:
    92 raw static
    + 92 signed static residual
    + 92 absolute static residual
    + 700 raw dynamic sequence
    + 700 signed dynamic residual
    + 700 absolute dynamic residual
    + 70 fixed residual temporal summaries
    = 2446 dims

This is exactly the A6_FULL_MSRR definition from experiment 119.
No representation change is allowed.

Evaluation protocol
-------------------
- Same formal matched P/C1/C2 benchmark.
- Same leakage-controlled spatial 5-fold protocol.
- Seed 7.
- Same train / validation / outer-test folds for all learners.
- Validation-only model/configuration selection.
- Outer-test metrics NEVER select hyperparameters.
- All four metrics are calculated from the same final score:
      AUROC, AUPRC, StrictPair, Edge.
- positive row weight = 1.0
- control row weight = 0.5

Cross-backbone gate
-------------------
A backbone supports MSRR if:
    MSRR > RAW on at least 3 of the 4 OOF metrics.

Decision:
    STRONG_MODEL_AGNOSTIC_SUPPORT
        4/4 backbones support MSRR.

    CROSS_BACKBONE_SUPPORT
        >=3/4 backbones support MSRR
        AND at least one supported backbone is non-tree.

    TREE_ONLY_SUPPORT
        both tree backbones support MSRR but broad cross-family support fails.

    WEAK_OR_NO_CROSS_BACKBONE_SUPPORT
        otherwise.

Important:
This replacement and gate are frozen BEFORE any completed 120 cross-backbone
scientific results exist. LightGBM is excluded only because its formal native
runtime was technically unusable.

Run
---
python ^
  <PROJECT_ROOT>\scripts\120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py ^
  --xgb-jobs 8

Output
------
<PROJECT_ROOT>\experiments\MSRR_CROSS_BACKBONE_5FOLD_V1B

Key outputs
-----------
OOF_RESULTS.csv
RAW_VS_MSRR_DELTAS.csv
FOLD_METRICS.csv
VALIDATION_SELECTION.csv
BACKBONE_SUPPORT_SUMMARY.csv
GATE120B_DECISION.json
GATE120B_REPORT.md
TECHNICAL_EXCLUSION_LIGHTGBM.json
*_OOF_margin.npy
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "MSRR_CROSS_BACKBONE_5FOLD_V1B"

SEED = 7
FOLDS = [1, 2, 3, 4, 5]
METRICS4 = ["AUROC", "AUPRC", "StrictPair", "Edge"]

BACKBONES = [
    "ElasticNet-Logistic",
    "MLP",
    "HistGradientBoosting",
    "XGBoost",
]
REPRESENTATIONS = ["RAW", "MSRR"]

LOGISTIC_C_GRID = [0.1, 1.0, 10.0]

HGB_CANDIDATES = [
    dict(
        config_id="H1",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
    ),
    dict(
        config_id="H2",
        learning_rate=0.035,
        max_iter=450,
        max_leaf_nodes=63,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
    ),
    dict(
        config_id="H3",
        learning_rate=0.025,
        max_iter=600,
        max_leaf_nodes=127,
        max_depth=10,
        min_samples_leaf=20,
        l2_regularization=2.0,
    ),
]

XGB_CANDIDATES = [
    dict(
        config_id="D4",
        n_estimators=450,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=2.0,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.05,
        gamma=0.0,
    ),
    dict(
        config_id="D6",
        n_estimators=650,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=2.0,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_lambda=5.0,
        reg_alpha=0.05,
        gamma=0.0,
    ),
    dict(
        config_id="D8",
        n_estimators=850,
        max_depth=8,
        learning_rate=0.02,
        min_child_weight=2.0,
        subsample=0.90,
        colsample_bytree=0.75,
        reg_lambda=7.0,
        reg_alpha=0.10,
        gamma=0.0,
    ),
]

MLP_HIDDEN = (256, 128)
MLP_DROPOUT = 0.20
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 256
MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 12

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.TRAIN_SEHC_V2_TEMPLATE import build_fold_loader, load_dataset_loader


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def jwrite(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def source_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return "UNAVAILABLE"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def prob_to_margin(p: np.ndarray) -> np.ndarray:
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    return np.log(p) - np.log1p(-p)


def class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    return np.where(
        y == 1,
        1.0,
        0.5,
    ).astype(np.float32)


def require_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except Exception as exc:
        raise RuntimeError(
            "xgboost unavailable in active environment"
        ) from exc


# ---------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------

def verify_pair_order(
    pair_rows: np.ndarray,
    y_all: np.ndarray,
    label: str,
) -> None:
    tri = np.asarray(pair_rows, dtype=np.int64)

    if tri.ndim != 2 or tri.shape[1] != 3:
        raise RuntimeError(
            f"{label}_PAIR_SHAPE_BAD={tri.shape}"
        )

    y3 = y_all[tri]

    if not np.all(y3[:, 0] == 1):
        raise RuntimeError(
            f"{label}_POSITIVE_NOT_SLOT0"
        )

    if not np.all(y3[:, 1:] == 0):
        raise RuntimeError(
            f"{label}_CONTROL_LABEL_BAD"
        )


# ---------------------------------------------------------------------
# Frozen representations
# ---------------------------------------------------------------------

def dynsum7(r_res_candidate: np.ndarray) -> np.ndarray:
    x = np.asarray(
        r_res_candidate,
        dtype=np.float32,
    )

    if x.ndim != 3 or x.shape[1:] != (70, 10):
        raise RuntimeError(
            f"DYNSUM_INPUT_BAD={x.shape}"
        )

    out = np.concatenate(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.max(axis=1),
            x.min(axis=1),
            x.sum(axis=1),
            x[:, -1, :],
            np.abs(x).max(axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if out.shape[1] != 70:
        raise RuntimeError(
            f"DYNSUM_OUTPUT_BAD={out.shape}"
        )

    return out


def raw_features(
    pair_rows: np.ndarray,
    static92: np.ndarray,
    rain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(
        pair_rows,
        dtype=np.int64,
    )
    anchors = tri.reshape(-1)

    s = static92[anchors]
    r = rain[anchors].reshape(
        len(anchors),
        -1,
    )

    x = np.concatenate(
        [s, r],
        axis=1,
    ).astype(np.float32, copy=False)

    if x.shape[1] != 792:
        raise RuntimeError(
            f"RAW_DIM_BAD={x.shape[1]} expected=792"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "NONFINITE_RAW_FEATURES"
        )

    return x, anchors


def msrr_features(
    pair_rows: np.ndarray,
    static92: np.ndarray,
    rain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(
        pair_rows,
        dtype=np.int64,
    )
    anchors = tri.reshape(-1)

    s_ctx = static92[tri].astype(
        np.float32,
        copy=False,
    )
    r_ctx = rain[tri].astype(
        np.float32,
        copy=False,
    )

    s_mean = s_ctx.mean(
        axis=1,
        keepdims=True,
    )
    r_mean = r_ctx.mean(
        axis=1,
        keepdims=True,
    )

    s_rel = s_ctx - s_mean
    r_rel = r_ctx - r_mean

    n = len(anchors)

    x = np.concatenate(
        [
            s_ctx.reshape(n, 92),
            s_rel.reshape(n, 92),
            np.abs(s_rel).reshape(n, 92),
            r_ctx.reshape(n, 700),
            r_rel.reshape(n, 700),
            np.abs(r_rel).reshape(n, 700),
            dynsum7(
                r_rel.reshape(n, 70, 10)
            ),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if x.shape[1] != 2446:
        raise RuntimeError(
            f"MSRR_DIM_BAD={x.shape[1]} expected=2446"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "NONFINITE_MSRR_FEATURES"
        )

    return x, anchors


# ---------------------------------------------------------------------
# Train-only standardization
# ---------------------------------------------------------------------

def fit_standardizer(
    xtr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = xtr.mean(
        axis=0,
        dtype=np.float64,
    )
    sd = xtr.std(
        axis=0,
        dtype=np.float64,
    )

    sd = np.where(
        sd < 1e-6,
        1.0,
        sd,
    )

    return (
        mu.astype(np.float32, copy=False),
        sd.astype(np.float32, copy=False),
    )


def apply_standardizer(
    x: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    z = (
        (x - mu) / sd
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(z).all():
        raise RuntimeError(
            "NONFINITE_STANDARDIZED_FEATURES"
        )

    return z


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def metrics_from_ordered_rows(
    margin: np.ndarray,
    y_rows: np.ndarray,
) -> Dict[str, float]:
    margin = np.asarray(
        margin,
        dtype=np.float64,
    )
    y_rows = np.asarray(
        y_rows,
        dtype=np.int8,
    )

    if margin.shape != y_rows.shape:
        raise RuntimeError(
            "MARGIN_LABEL_SHAPE_MISMATCH"
        )

    if len(margin) % 3 != 0:
        raise RuntimeError(
            "ROWS_NOT_DIVISIBLE_BY_3"
        )

    y3 = y_rows.reshape(-1, 3)

    if not np.all(y3[:, 0] == 1):
        raise RuntimeError(
            "METRIC_POSITIVE_NOT_SLOT0"
        )

    if not np.all(y3[:, 1:] == 0):
        raise RuntimeError(
            "METRIC_CONTROL_LABEL_BAD"
        )

    s3 = margin.reshape(-1, 3)
    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]

    p = sigmoid(margin)

    return {
        "AUROC": float(
            roc_auc_score(y_rows, p)
        ),
        "AUPRC": float(
            average_precision_score(y_rows, p)
        ),
        "StrictPair": float(
            (e1 & e2).mean()
        ),
        "Edge": float(
            np.stack(
                [e1, e2],
                axis=1,
            ).mean()
        ),
    }


def metrics_from_full_oof(
    margin_full: np.ndarray,
    y_all: np.ndarray,
    all_pair_rows: np.ndarray,
) -> Dict[str, float]:
    margin_full = np.asarray(
        margin_full,
        dtype=np.float64,
    )

    if not np.isfinite(
        margin_full
    ).all():
        raise RuntimeError(
            "INCOMPLETE_OOF_MARGIN"
        )

    s3 = margin_full[
        np.asarray(
            all_pair_rows,
            dtype=np.int64,
        )
    ]

    e1 = s3[:, 0] > s3[:, 1]
    e2 = s3[:, 0] > s3[:, 2]

    p = sigmoid(margin_full)

    return {
        "AUROC": float(
            roc_auc_score(
                y_all,
                p,
            )
        ),
        "AUPRC": float(
            average_precision_score(
                y_all,
                p,
            )
        ),
        "StrictPair": float(
            (e1 & e2).mean()
        ),
        "Edge": float(
            np.stack(
                [e1, e2],
                axis=1,
            ).mean()
        ),
    }


def val_scalar(
    m: Dict[str, float],
) -> float:
    return float(
        np.mean(
            [m[k] for k in METRICS4]
        )
    )


# ---------------------------------------------------------------------
# Generic deterministic validation selector
# ---------------------------------------------------------------------

def choose_best_index(
    rows: List[Dict[str, Any]],
) -> int:
    return sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["validation_score"],
            rows[i]["StrictPair"],
            rows[i]["Edge"],
            rows[i]["AUPRC"],
            rows[i]["AUROC"],
            -i,
        ),
        reverse=True,
    )[0]


# ---------------------------------------------------------------------
# ElasticNet Logistic
# ---------------------------------------------------------------------

def fit_elasticnet_select_predict(
    representation: str,
    fold: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    xte: np.ndarray,
    seed: int,
    jobs: int,
):
    rows = []
    models = []
    sw = class_weights(ytr)

    for ci, cval in enumerate(
        LOGISTIC_C_GRID
    ):
        cfg_id = f"C={cval:g}"

        log(
            f"FOLD{fold} ElasticNet {representation} "
            f"candidate={cfg_id}"
        )

        model = LogisticRegression(
            C=float(cval),
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=3000,
            tol=1e-4,
            random_state=seed + ci * 31,
            n_jobs=jobs,
        )

        t0 = time.perf_counter()

        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "always",
                ConvergenceWarning,
            )

            model.fit(
                xtr,
                ytr,
                sample_weight=sw,
            )

            conv = any(
                issubclass(
                    w.category,
                    ConvergenceWarning,
                )
                for w in caught
            )

        va_margin = np.asarray(
            model.decision_function(
                xva
            ),
            dtype=np.float64,
        )

        m = metrics_from_ordered_rows(
            va_margin,
            yva,
        )

        rows.append(
            {
                "human_fold": fold,
                "backbone": "ElasticNet-Logistic",
                "representation": representation,
                "config_id": cfg_id,
                **m,
                "validation_score": val_scalar(m),
                "fit_seconds": (
                    time.perf_counter() - t0
                ),
                "convergence_warning": bool(conv),
                "n_iter": int(
                    np.max(model.n_iter_)
                ),
            }
        )

        models.append(model)

    bi = choose_best_index(rows)
    best = models[bi]
    cfg = rows[bi]["config_id"]

    te_margin = np.asarray(
        best.decision_function(xte),
        dtype=np.float64,
    )

    log(
        f"FOLD{fold} ElasticNet {representation} "
        f"SELECTED={cfg}"
    )

    return (
        te_margin,
        pd.DataFrame(rows),
        cfg,
    )


# ---------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------

class FixedMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                MLP_HIDDEN[0],
            ),
            nn.ReLU(),
            nn.Dropout(
                MLP_DROPOUT
            ),
            nn.Linear(
                MLP_HIDDEN[0],
                MLP_HIDDEN[1],
            ),
            nn.ReLU(),
            nn.Dropout(
                MLP_DROPOUT
            ),
            nn.Linear(
                MLP_HIDDEN[1],
                1,
            ),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def mlp_predict_margin(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    pieces = []

    for start in range(
        0,
        len(x),
        batch_size,
    ):
        xb = torch.from_numpy(
            x[
                start:
                start + batch_size
            ]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        pieces.append(
            model(xb)
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        pieces
    ).astype(
        np.float64,
        copy=False,
    )


def fit_mlp_earlystop_predict(
    representation: str,
    fold: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    xte: np.ndarray,
    seed: int,
    device: torch.device,
):
    set_seed(seed)

    model = FixedMLP(
        xtr.shape[1]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )

    ds = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(
            ytr.astype(np.float32)
        ),
        torch.from_numpy(
            class_weights(ytr)
        ),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        ds,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    best_state = None
    best_tuple = None
    best_epoch = None
    wait = 0
    hist = []

    for epoch in range(
        1,
        MLP_MAX_EPOCHS + 1,
    ):
        model.train()

        for xb, yb, wb in loader:
            xb = xb.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            yb = yb.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            wb = wb.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            opt.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss_each = (
                torch.nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    reduction="none",
                )
            )

            loss = (
                (loss_each * wb).sum()
                / wb.sum().clamp_min(1e-8)
            )

            loss.backward()
            opt.step()

        va_margin = mlp_predict_margin(
            model,
            xva,
            device,
        )

        m = metrics_from_ordered_rows(
            va_margin,
            yva,
        )

        cur = (
            val_scalar(m),
            m["StrictPair"],
            m["Edge"],
            m["AUPRC"],
            m["AUROC"],
        )

        improved = (
            best_tuple is None
            or cur > best_tuple
        )

        if improved:
            best_tuple = cur
            best_epoch = epoch
            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        hist.append(
            {
                "human_fold": fold,
                "backbone": "MLP",
                "representation": representation,
                "config_id": "MLP_256_128",
                "epoch": epoch,
                **m,
                "validation_score": val_scalar(m),
                "is_best_so_far": improved,
            }
        )

        if epoch == 1 or epoch % 10 == 0 or improved:
            log(
                f"FOLD{fold} MLP {representation} "
                f"epoch={epoch} "
                f"AUC={m['AUROC']:.4f} "
                f"AP={m['AUPRC']:.4f} "
                f"SP={m['StrictPair']:.4f} "
                f"Edge={m['Edge']:.4f}"
            )

        if wait >= MLP_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            f"FOLD{fold}_MLP_{representation}_NO_BEST"
        )

    model.load_state_dict(
        best_state
    )

    te_margin = mlp_predict_margin(
        model,
        xte,
        device,
    )

    cfg = (
        f"MLP_256_128_best_epoch="
        f"{best_epoch}"
    )

    return (
        te_margin,
        pd.DataFrame(hist),
        cfg,
    )


# ---------------------------------------------------------------------
# sklearn HistGradientBoosting
# ---------------------------------------------------------------------

def fit_hgb_select_predict(
    representation: str,
    fold: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    xte: np.ndarray,
    seed: int,
):
    rows = []
    models = []
    sw = class_weights(ytr)

    for ci, cfg in enumerate(
        HGB_CANDIDATES
    ):
        cfg_id = cfg["config_id"]

        log(
            f"FOLD{fold} HistGradientBoosting "
            f"{representation} candidate={cfg_id}"
        )

        kwargs = {
            k: v
            for k, v in cfg.items()
            if k != "config_id"
        }

        model = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping=False,
            random_state=seed + ci * 43,
            **kwargs,
        )

        t0 = time.perf_counter()

        model.fit(
            xtr,
            ytr,
            sample_weight=sw,
        )

        va_margin = prob_to_margin(
            model.predict_proba(
                xva
            )[:, 1]
        )

        m = metrics_from_ordered_rows(
            va_margin,
            yva,
        )

        rows.append(
            {
                "human_fold": fold,
                "backbone": "HistGradientBoosting",
                "representation": representation,
                "config_id": cfg_id,
                **m,
                "validation_score": val_scalar(m),
                "fit_seconds": (
                    time.perf_counter() - t0
                ),
            }
        )

        models.append(model)

    bi = choose_best_index(rows)
    best = models[bi]
    cfg = rows[bi]["config_id"]

    te_margin = prob_to_margin(
        best.predict_proba(
            xte
        )[:, 1]
    )

    log(
        f"FOLD{fold} HistGradientBoosting "
        f"{representation} SELECTED={cfg}"
    )

    return (
        te_margin,
        pd.DataFrame(rows),
        cfg,
    )


# ---------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------

def make_xgb(
    cfg: Dict[str, Any],
    seed: int,
    jobs: int,
):
    xgb = require_xgboost()

    kwargs = {
        k: v
        for k, v in cfg.items()
        if k != "config_id"
    }

    kwargs.update(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        random_state=seed,
        n_jobs=jobs,
        verbosity=0,
    )

    return xgb.XGBClassifier(
        **kwargs
    )


def fit_xgb_select_predict(
    representation: str,
    fold: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    xte: np.ndarray,
    seed: int,
    jobs: int,
):
    rows = []
    models = []
    sw = class_weights(ytr)

    for ci, cfg in enumerate(
        XGB_CANDIDATES
    ):
        cfg_id = cfg["config_id"]

        log(
            f"FOLD{fold} XGBoost {representation} "
            f"candidate={cfg_id}"
        )

        model = make_xgb(
            cfg,
            seed + ci * 37,
            jobs,
        )

        t0 = time.perf_counter()

        model.fit(
            xtr,
            ytr,
            sample_weight=sw,
        )

        va_margin = prob_to_margin(
            model.predict_proba(
                xva
            )[:, 1]
        )

        m = metrics_from_ordered_rows(
            va_margin,
            yva,
        )

        rows.append(
            {
                "human_fold": fold,
                "backbone": "XGBoost",
                "representation": representation,
                "config_id": cfg_id,
                **m,
                "validation_score": val_scalar(m),
                "fit_seconds": (
                    time.perf_counter() - t0
                ),
            }
        )

        models.append(model)

    bi = choose_best_index(rows)
    best = models[bi]
    cfg = rows[bi]["config_id"]

    te_margin = prob_to_margin(
        best.predict_proba(
            xte
        )[:, 1]
    )

    log(
        f"FOLD{fold} XGBoost {representation} "
        f"SELECTED={cfg}"
    )

    return (
        te_margin,
        pd.DataFrame(rows),
        cfg,
    )


# ---------------------------------------------------------------------
# Fold comparison
# ---------------------------------------------------------------------

def fold_win_count(
    fold_df: pd.DataFrame,
    backbone: str,
    metric: str,
) -> int:
    msrr = (
        fold_df[
            (fold_df["backbone"] == backbone)
            &
            (fold_df["representation"] == "MSRR")
        ]
        .set_index("human_fold")[metric]
    )

    raw = (
        fold_df[
            (fold_df["backbone"] == backbone)
            &
            (fold_df["representation"] == "RAW")
        ]
        .set_index("human_fold")[metric]
    )

    common = sorted(
        set(msrr.index)
        & set(raw.index)
    )

    return int(
        sum(
            float(msrr.loc[f])
            >
            float(raw.loc[f])
            for f in common
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xgb-jobs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--mlp-device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args(argv)

    require_xgboost()
    set_seed(SEED)

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise FileExistsError(
            "Refusing to overwrite existing "
            f"non-empty output: {OUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.mlp_device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    if args.mlp_device == "cpu":
        device = torch.device("cpu")
    elif args.mlp_device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    technical_exclusion = {
        "excluded_backbone": "LightGBM",
        "reason": (
            "Native Windows access-violation on formal project data. "
            "Synthetic fit passed in repaired clone, but formal BASE-04 fit "
            "still failed. No completed LightGBM scientific result from 120 "
            "was available or used."
        ),
        "replacement": "sklearn HistGradientBoostingClassifier",
        "scientific_reason_for_replacement": (
            "Preserve a second tree-boosting family without relying on the "
            "technically unstable LightGBM native runtime."
        ),
        "performance_based_exclusion": False,
    }

    jwrite(
        OUT / "TECHNICAL_EXCLUSION_LIGHTGBM.json",
        technical_exclusion,
    )

    manifest = {
        "script": Path(__file__).name,
        "source_sha256": source_sha256(),
        "seed": SEED,
        "folds": FOLDS,
        "backbones": BACKBONES,
        "representations": REPRESENTATIONS,
        "raw_dim": 792,
        "msrr_dim": 2446,
        "lightgbm_excluded_for_technical_runtime_reason": True,
        "replacement_tree_learner": "HistGradientBoostingClassifier",
        "outer_test_used_for_selection": False,
        "post_result_rescue_allowed": False,
        "mlp_device": str(device),
        "logistic_C_grid": LOGISTIC_C_GRID,
        "hgb_candidates": HGB_CANDIDATES,
        "xgb_candidates": XGB_CANDIDATES,
    }

    jwrite(
        OUT / "RUN_MANIFEST.json",
        manifest,
    )

    log(
        "START 120B MSRR CROSS-BACKBONE "
        f"device={device} "
        f"sha256={manifest['source_sha256']}"
    )

    bundle, runner, base = (
        load_dataset_loader()
    )

    y_all = (
        bundle.sample.y_pair
        .to_numpy(np.int8)
    )

    all_pair_rows = np.asarray(
        bundle.pt,
        dtype=np.int64,
    )

    verify_pair_order(
        all_pair_rows,
        y_all,
        "GLOBAL",
    )

    methods = [
        f"{b}__{r}"
        for b in BACKBONES
        for r in REPRESENTATIONS
    ]

    oof = {
        m: np.full(
            int(base.N),
            np.nan,
            dtype=np.float64,
        )
        for m in methods
    }

    fold_rows: List[
        Dict[str, Any]
    ] = []

    selection_tables = []

    for hf in FOLDS:
        log(
            "=" * 116
        )
        log(
            f"START FOLD{hf}"
        )

        fold_seed = (
            SEED + hf * 1000
        )

        set_seed(
            fold_seed
        )

        fd = build_fold_loader(
            bundle,
            runner,
            base,
            hf,
            torch.device("cpu"),
        )

        static92 = as_numpy(
            fd.static92
        ).astype(
            np.float32,
            copy=False,
        )

        rain = as_numpy(
            fd.rain
        ).astype(
            np.float32,
            copy=False,
        )

        if (
            static92.ndim != 2
            or static92.shape[1] != 92
        ):
            raise RuntimeError(
                f"FOLD{hf}_STATIC_BAD="
                f"{static92.shape}"
            )

        if (
            rain.ndim != 3
            or rain.shape[1:] != (70, 10)
        ):
            raise RuntimeError(
                f"FOLD{hf}_RAIN_BAD="
                f"{rain.shape}"
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

        tr_pairs = all_pair_rows[
            tr_pair_ids
        ]
        va_pairs = all_pair_rows[
            va_pair_ids
        ]
        te_pairs = all_pair_rows[
            te_pair_ids
        ]

        verify_pair_order(
            tr_pairs,
            y_all,
            f"FOLD{hf}_TRAIN",
        )
        verify_pair_order(
            va_pairs,
            y_all,
            f"FOLD{hf}_VAL",
        )
        verify_pair_order(
            te_pairs,
            y_all,
            f"FOLD{hf}_TEST",
        )

        tr_idx = tr_pairs.reshape(-1)
        va_idx = va_pairs.reshape(-1)
        te_idx = te_pairs.reshape(-1)

        if np.intersect1d(
            tr_idx,
            va_idx,
        ).size:
            raise RuntimeError(
                f"FOLD{hf}_TRAIN_VAL_OVERLAP"
            )

        if np.intersect1d(
            tr_idx,
            te_idx,
        ).size:
            raise RuntimeError(
                f"FOLD{hf}_TRAIN_TEST_OVERLAP"
            )

        if np.intersect1d(
            va_idx,
            te_idx,
        ).size:
            raise RuntimeError(
                f"FOLD{hf}_VAL_TEST_OVERLAP"
            )

        ytr = y_all[tr_idx]
        yva = y_all[va_idx]
        yte = y_all[te_idx]

        for representation in REPRESENTATIONS:
            log(
                "-" * 96
            )
            log(
                f"FOLD{hf} BUILD {representation}"
            )

            if representation == "RAW":
                xtr, atr = raw_features(
                    tr_pairs,
                    static92,
                    rain,
                )
                xva, ava = raw_features(
                    va_pairs,
                    static92,
                    rain,
                )
                xte, ate = raw_features(
                    te_pairs,
                    static92,
                    rain,
                )
            else:
                xtr, atr = msrr_features(
                    tr_pairs,
                    static92,
                    rain,
                )
                xva, ava = msrr_features(
                    va_pairs,
                    static92,
                    rain,
                )
                xte, ate = msrr_features(
                    te_pairs,
                    static92,
                    rain,
                )

            if not np.array_equal(
                atr,
                tr_idx,
            ):
                raise RuntimeError(
                    f"FOLD{hf}_{representation}_TRAIN_ORDER"
                )

            if not np.array_equal(
                ava,
                va_idx,
            ):
                raise RuntimeError(
                    f"FOLD{hf}_{representation}_VAL_ORDER"
                )

            if not np.array_equal(
                ate,
                te_idx,
            ):
                raise RuntimeError(
                    f"FOLD{hf}_{representation}_TEST_ORDER"
                )

            # ---------------------------------------------------------
            # sklearn HistGradientBoosting
            # ---------------------------------------------------------
            hgb_margin, hgb_sel, hgb_cfg = (
                fit_hgb_select_predict(
                    representation,
                    hf,
                    xtr,
                    ytr,
                    xva,
                    yva,
                    xte,
                    fold_seed + 300,
                )
            )

            oof[
                f"HistGradientBoosting__{representation}"
            ][te_idx] = hgb_margin

            hgb_m = metrics_from_ordered_rows(
                hgb_margin,
                yte,
            )

            fold_rows.append(
                {
                    "human_fold": hf,
                    "backbone": "HistGradientBoosting",
                    "representation": representation,
                    "selected_config": hgb_cfg,
                    **hgb_m,
                }
            )

            selection_tables.append(
                hgb_sel
            )

            del hgb_margin
            gc.collect()

            # ---------------------------------------------------------
            # XGBoost
            # ---------------------------------------------------------
            xgb_margin, xgb_sel, xgb_cfg = (
                fit_xgb_select_predict(
                    representation,
                    hf,
                    xtr,
                    ytr,
                    xva,
                    yva,
                    xte,
                    fold_seed + 600,
                    args.xgb_jobs,
                )
            )

            oof[
                f"XGBoost__{representation}"
            ][te_idx] = xgb_margin

            xgb_m = metrics_from_ordered_rows(
                xgb_margin,
                yte,
            )

            fold_rows.append(
                {
                    "human_fold": hf,
                    "backbone": "XGBoost",
                    "representation": representation,
                    "selected_config": xgb_cfg,
                    **xgb_m,
                }
            )

            selection_tables.append(
                xgb_sel
            )

            del xgb_margin
            gc.collect()

            # ---------------------------------------------------------
            # Standardize only for linear / neural.
            # ---------------------------------------------------------
            mu, sd = fit_standardizer(
                xtr
            )

            xtr_z = apply_standardizer(
                xtr,
                mu,
                sd,
            )
            xva_z = apply_standardizer(
                xva,
                mu,
                sd,
            )
            xte_z = apply_standardizer(
                xte,
                mu,
                sd,
            )

            del xtr, xva, xte, mu, sd
            gc.collect()

            # ---------------------------------------------------------
            # ElasticNet Logistic
            # ---------------------------------------------------------
            log_margin, log_sel, log_cfg = (
                fit_elasticnet_select_predict(
                    representation,
                    hf,
                    xtr_z,
                    ytr,
                    xva_z,
                    yva,
                    xte_z,
                    fold_seed + 100,
                    args.xgb_jobs,
                )
            )

            oof[
                f"ElasticNet-Logistic__{representation}"
            ][te_idx] = log_margin

            log_m = metrics_from_ordered_rows(
                log_margin,
                yte,
            )

            fold_rows.append(
                {
                    "human_fold": hf,
                    "backbone": "ElasticNet-Logistic",
                    "representation": representation,
                    "selected_config": log_cfg,
                    **log_m,
                }
            )

            selection_tables.append(
                log_sel
            )

            del log_margin
            gc.collect()

            # ---------------------------------------------------------
            # MLP
            # ---------------------------------------------------------
            mlp_margin, mlp_hist, mlp_cfg = (
                fit_mlp_earlystop_predict(
                    representation,
                    hf,
                    xtr_z,
                    ytr,
                    xva_z,
                    yva,
                    xte_z,
                    fold_seed + 200,
                    device,
                )
            )

            oof[
                f"MLP__{representation}"
            ][te_idx] = mlp_margin

            mlp_m = metrics_from_ordered_rows(
                mlp_margin,
                yte,
            )

            fold_rows.append(
                {
                    "human_fold": hf,
                    "backbone": "MLP",
                    "representation": representation,
                    "selected_config": mlp_cfg,
                    **mlp_m,
                }
            )

            selection_tables.append(
                mlp_hist
            )

            del (
                mlp_margin,
                xtr_z,
                xva_z,
                xte_z,
                atr,
                ava,
                ate,
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del (
            fd,
            static92,
            rain,
        )
        gc.collect()

        log(
            f"COMPLETE FOLD{hf}"
        )

    # -----------------------------------------------------------------
    # Final OOF metrics
    # -----------------------------------------------------------------
    final = {}

    for method in methods:
        arr = oof[method]

        if not np.isfinite(
            arr
        ).all():
            raise RuntimeError(
                f"OOF_COVERAGE_FAILED "
                f"method={method} "
                f"missing={int((~np.isfinite(arr)).sum())}"
            )

        np.save(
            OUT
            / (
                method
                .replace("-", "_")
                .replace("__", "_")
                + "_OOF_margin.npy"
            ),
            arr,
        )

        final[method] = (
            metrics_from_full_oof(
                arr,
                y_all,
                all_pair_rows,
            )
        )

    oof_rows = []

    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            method = (
                f"{backbone}__{rep}"
            )

            oof_rows.append(
                {
                    "backbone": backbone,
                    "representation": rep,
                    **final[method],
                }
            )

    oof_df = pd.DataFrame(
        oof_rows
    )

    oof_df.to_csv(
        OUT / "OOF_RESULTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    fold_df.to_csv(
        OUT / "FOLD_METRICS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        selection_tables,
        ignore_index=True,
        sort=False,
    ).to_csv(
        OUT / "VALIDATION_SELECTION.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # Cross-backbone support
    # -----------------------------------------------------------------
    delta_rows = []
    support_rows = []
    supported = []

    for backbone in BACKBONES:
        raw = final[
            f"{backbone}__RAW"
        ]
        msrr = final[
            f"{backbone}__MSRR"
        ]

        delta = {
            m: float(
                msrr[m] - raw[m]
            )
            for m in METRICS4
        }

        metric_wins = int(
            sum(
                delta[m] > 0
                for m in METRICS4
            )
        )

        support = (
            metric_wins >= 3
        )

        if support:
            supported.append(
                backbone
            )

        row = {
            "backbone": backbone,
            **{
                f"delta_{m}":
                delta[m]
                for m in METRICS4
            },
            "OOF_metric_wins_out_of_4": metric_wins,
            "MSRR_SUPPORT": (
                "YES"
                if support
                else "NO"
            ),
        }

        for metric in METRICS4:
            row[
                f"fold_wins_{metric}_out_of_5"
            ] = fold_win_count(
                fold_df,
                backbone,
                metric,
            )

        delta_rows.append(
            row
        )

        family = (
            "linear"
            if backbone == "ElasticNet-Logistic"
            else
            "neural"
            if backbone == "MLP"
            else
            "tree"
        )

        support_rows.append(
            {
                "backbone": backbone,
                "family": family,
                "MSRR_SUPPORT": (
                    "YES"
                    if support
                    else "NO"
                ),
                "OOF_metric_wins_out_of_4": metric_wins,
                "AUROC_delta": delta["AUROC"],
                "AUPRC_delta": delta["AUPRC"],
                "StrictPair_delta": delta["StrictPair"],
                "Edge_delta": delta["Edge"],
            }
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    delta_df.to_csv(
        OUT / "RAW_VS_MSRR_DELTAS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    support_df = pd.DataFrame(
        support_rows
    )

    support_df.to_csv(
        OUT / "BACKBONE_SUPPORT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_supported = len(
        supported
    )

    non_tree_supported = any(
        b in supported
        for b in [
            "ElasticNet-Logistic",
            "MLP",
        ]
    )

    trees_supported = all(
        b in supported
        for b in [
            "HistGradientBoosting",
            "XGBoost",
        ]
    )

    if n_supported == 4:
        gate = (
            "STRONG_MODEL_AGNOSTIC_SUPPORT"
        )
    elif (
        n_supported >= 3
        and non_tree_supported
    ):
        gate = (
            "CROSS_BACKBONE_SUPPORT"
        )
    elif trees_supported:
        gate = (
            "TREE_ONLY_SUPPORT"
        )
    else:
        gate = (
            "WEAK_OR_NO_CROSS_BACKBONE_SUPPORT"
        )

    decision = {
        "status": "PASS_120B_CROSS_BACKBONE_COMPLETED",
        "gate120b_decision": gate,
        "supported_backbones": supported,
        "supported_backbone_count": n_supported,
        "total_backbones": 4,
        "definition_of_support": (
            "MSRR beats RAW on >=3/4 OOF metrics"
        ),
        "technical_lightgbm_exclusion": technical_exclusion,
        "results": final,
        "next_step": (
            "PAIRED_BOOTSTRAP_STATISTICAL_ANALYSIS"
            if n_supported >= 3
            else
            "NARROW_CLAIM_AND_RUN_FORMAL_TECHNICAL_AUDIT"
        ),
        "post_result_rescue_allowed": False,
    }

    jwrite(
        OUT / "GATE120B_DECISION.json",
        decision,
    )

    report = f"""# 120B MSRR Cross-Backbone Validation

## Technical note

LightGBM was excluded because its native runtime crashed on the formal project
data even after a repaired-environment clone passed synthetic fitting.
No completed LightGBM scientific result was available or used.

It was replaced before the cross-backbone experiment by sklearn
HistGradientBoostingClassifier.

## Decision

**{gate}**

Supported backbones: {n_supported}/4

{", ".join(supported) if supported else "None"}

## OOF results

```text
{oof_df.to_string(index=False)}
```

## MSRR - RAW deltas

```text
{delta_df.to_string(index=False)}
```

## Interpretation discipline

A backbone supports MSRR only if the frozen MSRR representation beats the
frozen RAW representation on at least 3 of the 4 OOF metrics.

- 4/4: strong model-agnostic support within this benchmark.
- >=3/4 including a non-tree learner: cross-family support.
- tree-only: narrow the manuscript claim to tree ensembles.
- otherwise: do not claim cross-backbone generality.

## Next step

{decision["next_step"]}
"""

    (
        OUT / "GATE120B_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "PASS_120B_MSRR_CROSS_BACKBONE_COMPLETE"
    )
    print(
        f"GATE120B_DECISION={gate}"
    )
    print(
        f"SUPPORTED_BACKBONES={n_supported}/4 "
        + (
            ",".join(supported)
            if supported
            else "NONE"
        )
    )

    for backbone in BACKBONES:
        raw = final[
            f"{backbone}__RAW"
        ]
        msrr = final[
            f"{backbone}__MSRR"
        ]

        print(
            f"{backbone} RAW: "
            f"AUROC={raw['AUROC']:.6f} "
            f"AUPRC={raw['AUPRC']:.6f} "
            f"StrictPair={raw['StrictPair']:.6f} "
            f"Edge={raw['Edge']:.6f}"
        )

        print(
            f"{backbone} MSRR: "
            f"AUROC={msrr['AUROC']:.6f} "
            f"AUPRC={msrr['AUPRC']:.6f} "
            f"StrictPair={msrr['StrictPair']:.6f} "
            f"Edge={msrr['Edge']:.6f}"
        )

    print(
        f"NEXT_STEP={decision['next_step']}"
    )
    print(
        f"MLP_DEVICE={device}"
    )
    print(
        f"OUTPUT={OUT}"
    )
    print(
        "=" * 120
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        OUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        jwrite(
            OUT / "FAILURE.json",
            {
                "status": "FAIL_120B_MSRR_CROSS_BACKBONE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "source_sha256": source_sha256(),
            },
        )

        print(
            traceback.format_exc(),
            flush=True,
        )
        raise
