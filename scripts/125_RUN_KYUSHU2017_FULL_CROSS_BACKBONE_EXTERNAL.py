#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py

Secondary confirmatory full external transfer panel
===================================================

Purpose
-------
The primary untouched external test (124D) has already been completed for
XGBoost RAW vs MSRR. 125 does NOT replace or reinterpret that primary result.

Instead, 125 asks a separate confirmatory question:

    Does the external RAW -> MSRR transfer advantage replicate across the
    other learner families that were already frozen in Experiment 120B?

Frozen learner panel
--------------------
1) ElasticNet-Logistic
2) MLP
3) HistGradientBoosting
4) XGBoost

For every learner:
    RAW  = 792 dimensions
    MSRR = 2446 dimensions

Scientific discipline
---------------------
- XGBoost external predictions are REUSED exactly from frozen 124D.
  They are not retrained and cannot change.
- The three NEW external learner families are trained on Hiroshima 2018 only.
- Their deployment hyperparameters are determined exclusively from the
  already-existing Hiroshima 120B validation results.
- Kyushu 2017 is never used for preprocessing fit, model/config selection,
  epoch selection, standardization, threshold selection, or tuning.
- The same full-Hiroshima-fitted preprocessing used by 124D is reconstructed
  from the unchanged authority scripts and applied unchanged to Kyushu.
- Linear and MLP representation-level standardization is fitted on complete
  Hiroshima only and applied unchanged to Kyushu.
- Kyushu labels are not parsed for the newly tested backbones until all new
  external predictions are fixed.
- Paired matched-set bootstrap: B=10,000; unit=P+C1+C2 matched set.

Important status
----------------
124D = PRIMARY UNTOUCHED EXTERNAL TEST (already observed)
125  = SECONDARY CONFIRMATORY CROSS-BACKBONE EXTERNAL REPLICATION

Therefore the 125 gate below is defined ONLY on the three previously unseen
external backbones (ElasticNet, MLP, HGB), so it does not exploit the already
known XGBoost external outcome.

Frozen 125 gate
---------------
For each newly tested backbone:
    observed_support = MSRR > RAW on >=3/4 external metrics
    bootstrap_support = CI95 lower bound >0 on >=3/4 external metrics

STRONG_SECONDARY_CROSS_BACKBONE_EXTERNAL_REPLICATION:
    all 3/3 new backbones have observed_support
    AND all 3/3 new backbones have bootstrap_support

CROSS_BACKBONE_EXTERNAL_REPLICATION:
    all 3/3 new backbones have observed_support
    AND at least 2/3 have bootstrap_support

PARTIAL_CROSS_BACKBONE_EXTERNAL_REPLICATION:
    at least 2/3 new backbones have observed_support

otherwise:
    CROSS_BACKBONE_EXTERNAL_REPLICATION_NOT_SUPPORTED

This script additionally creates an INTERNAL-vs-EXTERNAL table for all four
backbones. Only after this complete panel should we characterize whether
external absolute performance systematically decreases under event transfer.

Run
---
python ^
  <PROJECT_ROOT>\scripts\125_RUN_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL.py ^
  --jobs 8 --mlp-device auto

Output
------
<PROJECT_ROOT>\external\kyushu_2017_asakura_toho\
100_external_validation\MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 0. FROZEN PATHS / CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

KYUSHU = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "99_frozen_dataset"
)

OUT120B = (
    ROOT
    / "experiments"
    / "MSRR_CROSS_BACKBONE_5FOLD_V1B"
)

OUT124D = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "100_external_validation"
    / "MSRR_KYUSHU2017_EXTERNAL_V1D"
)

OUT = (
    ROOT
    / "external"
    / "kyushu_2017_asakura_toho"
    / "100_external_validation"
    / "MSRR_KYUSHU2017_EXTERNAL_FULL_PANEL_V2"
)

SEED = 7
FINAL_ELASTIC_SEED = SEED + 100
FINAL_MLP_SEED = SEED + 200
FINAL_HGB_SEED = SEED + 300

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260830

BACKBONES = [
    "ElasticNet-Logistic",
    "MLP",
    "HistGradientBoosting",
    "XGBoost",
]

NEW_BACKBONES = [
    "ElasticNet-Logistic",
    "MLP",
    "HistGradientBoosting",
]

REPRESENTATIONS = ["RAW", "MSRR"]
METRICS4 = ["AUROC", "AUPRC", "StrictPair", "Edge"]

EXPECTED_MATCHED_ROWS = 5076
EXPECTED_MATCHED_SETS = 1692

# 120B fixed MLP architecture/hyperparameters.
MLP_HIDDEN = (256, 128)
MLP_DROPOUT = 0.20
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 256


# =============================================================================
# 1. UTILITIES
# =============================================================================

def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "125_RUN.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(code: str, msg: str) -> None:
    log("")
    log("=" * 120)
    log(code)
    log(msg)
    log("=" * 120)
    raise RuntimeError(msg)


def to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def jwrite(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_builtin(obj),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def locate_script(preferred_name: str, pattern: str) -> Path:
    preferred = ROOT / "scripts" / preferred_name
    if preferred.exists():
        return preferred

    cands = sorted(
        (ROOT / "scripts").glob(pattern)
    )
    if not cands:
        raise FileNotFoundError(
            f"Cannot locate {preferred_name} under {ROOT / 'scripts'}"
        )
    return cands[0]


def import_script(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import script: {path}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_best_epoch(s: str) -> int:
    m = re.search(
        r"best_epoch\s*=\s*(\d+)",
        str(s),
    )
    if m is None:
        raise RuntimeError(
            f"Cannot parse MLP best_epoch from selected_config={s}"
        )
    return int(m.group(1))


# =============================================================================
# 2. LOAD 124D + FORMAL AUTHORITIES
# =============================================================================

def load_all_authorities():
    p124 = locate_script(
        "124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION.py",
        "124D_RUN_KYUSHU2017_FROZEN_EXTERNAL_VALIDATION*.py",
    )

    m124 = import_script(
        "external_124d_authority",
        p124,
    )

    # Redirect 124D helper output into 125.
    m124.OUT = OUT
    m124.log = log

    (
        p120,
        p26,
        m120,
        m26,
        bundle,
        runner,
        base,
    ) = m124.load_authorities()

    m120.OUT = OUT
    m120.log = log

    if hasattr(runner, "log"):
        runner.log = log

    return (
        p124,
        p120,
        p26,
        m124,
        m120,
        m26,
        bundle,
        runner,
        base,
    )


# =============================================================================
# 3. PRE-RESULT 125 PROTOCOL FREEZE
# =============================================================================

def freeze_protocol(
    p124: Path,
    p120: Path,
    p26: Path,
    runner: Any,
    base: Any,
    jobs: int,
    mlp_device: str,
) -> None:

    paths = {
        "124D_primary_external_authority": p124,
        "120B_cross_backbone_authority": p120,
        "26_static92_bridge": p26,
        "23D_preprocessor_authority": Path(runner.__file__),
        "20A_data_schema_authority": Path(base.__file__),
    }

    payload = {
        "experiment": "125_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL",
        "status": "FROZEN_BEFORE_NEW_BACKBONE_EXTERNAL_PREDICTIONS",
        "relationship_to_124D": (
            "124D is the already-observed primary untouched XGBoost external test. "
            "125 is a secondary confirmatory cross-backbone external replication."
        ),
        "development_event": "Hiroshima_2018",
        "external_event": "Kyushu_2017_Asakura_Toho",
        "backbones": BACKBONES,
        "new_unseen_external_backbones": NEW_BACKBONES,
        "xgboost_rule": (
            "Reuse exact frozen 124D RAW/MSRR XGBoost external predictions. "
            "Do not retrain or replace them."
        ),
        "representation_rule": {
            "RAW": 792,
            "MSRR": 2446,
        },
        "preprocessing_rule": (
            "Fit unchanged formal preprocessor once on all Hiroshima development samples; "
            "apply unchanged to Kyushu."
        ),
        "linear_mlp_standardization": (
            "Fit representation-level mean/std on complete Hiroshima only; apply unchanged to Kyushu."
        ),
        "deployment_config_rule": (
            "ElasticNet and HGB: select config by highest mean 120B Hiroshima validation_score "
            "across the five internal folds. MLP: fixed 256-128 architecture, epoch count = median "
            "of the five Hiroshima 120B validation-selected best epochs for that representation."
        ),
        "kyushu_used_for_preprocessing_fit": False,
        "kyushu_used_for_hyperparameter_selection": False,
        "kyushu_used_for_epoch_selection": False,
        "kyushu_used_for_standardization": False,
        "kyushu_used_for_threshold_selection": False,
        "kyushu_used_for_post_result_rescue": False,
        "post_result_rescue_allowed": False,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "matched pair-set; P+C1+C2 stay together",
        "support_definition_per_new_backbone": {
            "observed_support": "MSRR > RAW on >=3/4 external metrics",
            "bootstrap_support": "paired-bootstrap 95% CI lower bound >0 on >=3/4 external metrics",
        },
        "gate125": {
            "STRONG_SECONDARY_CROSS_BACKBONE_EXTERNAL_REPLICATION": (
                "3/3 new backbones have observed_support AND 3/3 have bootstrap_support"
            ),
            "CROSS_BACKBONE_EXTERNAL_REPLICATION": (
                "3/3 new backbones have observed_support AND >=2/3 have bootstrap_support"
            ),
            "PARTIAL_CROSS_BACKBONE_EXTERNAL_REPLICATION": (
                ">=2/3 new backbones have observed_support"
            ),
            "otherwise": "CROSS_BACKBONE_EXTERNAL_REPLICATION_NOT_SUPPORTED",
        },
        "jobs": int(jobs),
        "mlp_device": mlp_device,
        "authority_files": {
            k: {
                "path": str(v),
                "sha256": sha256(v),
            }
            for k, v in paths.items()
        },
    }

    jwrite(
        OUT / "125_PROTOCOL_FROZEN.json",
        payload,
    )


# =============================================================================
# 4. DEVELOPMENT-ONLY CONFIG SELECTION
# =============================================================================

def validation_aggregate_for_backbone(
    backbone: str,
    representation: str,
) -> pd.DataFrame:

    p = OUT120B / "VALIDATION_SELECTION.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    need = {
        "backbone",
        "representation",
        "config_id",
        "validation_score",
        "StrictPair",
        "Edge",
        "AUPRC",
        "AUROC",
    }

    if not need.issubset(df.columns):
        raise RuntimeError(
            "120B VALIDATION_SELECTION missing columns="
            + str(sorted(need - set(df.columns)))
        )

    sub = df[
        (df["backbone"].astype(str) == backbone)
        &
        (df["representation"].astype(str) == representation)
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No 120B validation rows for {backbone}/{representation}"
        )

    agg = (
        sub.groupby(
            "config_id",
            as_index=False,
        )
        .agg(
            mean_validation_score=("validation_score", "mean"),
            mean_StrictPair=("StrictPair", "mean"),
            mean_Edge=("Edge", "mean"),
            mean_AUPRC=("AUPRC", "mean"),
            mean_AUROC=("AUROC", "mean"),
            n_validation_rows=("validation_score", "size"),
        )
        .sort_values(
            [
                "mean_validation_score",
                "mean_StrictPair",
                "mean_Edge",
                "mean_AUPRC",
                "mean_AUROC",
                "config_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return agg


def select_deployment_configs() -> Tuple[pd.DataFrame, pd.DataFrame]:

    audit_frames = []
    rows = []

    for rep in REPRESENTATIONS:
        for backbone in [
            "ElasticNet-Logistic",
            "HistGradientBoosting",
        ]:
            agg = validation_aggregate_for_backbone(
                backbone,
                rep,
            )

            audit_frames.append(
                agg.assign(
                    backbone=backbone,
                    representation=rep,
                )
            )

            rows.append(
                {
                    "backbone": backbone,
                    "representation": rep,
                    "selected_config": str(
                        agg.iloc[0]["config_id"]
                    ),
                    "selection_source": (
                        "Hiroshima 120B mean validation_score across five folds only"
                    ),
                    "deployment_epochs": np.nan,
                }
            )

        # MLP: no model-config grid in 120B; early stopping selected an epoch.
        fp = OUT120B / "FOLD_METRICS.csv"
        if not fp.exists():
            raise FileNotFoundError(fp)

        fold = pd.read_csv(
            fp,
            low_memory=False,
        )

        sub = fold[
            (fold["backbone"].astype(str) == "MLP")
            &
            (fold["representation"].astype(str) == rep)
        ].copy()

        if len(sub) != 5:
            raise RuntimeError(
                f"Expected exactly 5 MLP fold rows for {rep}; got {len(sub)}"
            )

        epochs = [
            parse_best_epoch(x)
            for x in sub["selected_config"].astype(str)
        ]

        deploy_epoch = int(
            np.median(
                np.asarray(
                    epochs,
                    dtype=np.int64,
                )
            )
        )

        rows.append(
            {
                "backbone": "MLP",
                "representation": rep,
                "selected_config": "MLP_256_128",
                "selection_source": (
                    "median of five Hiroshima 120B validation-selected best epochs"
                ),
                "deployment_epochs": deploy_epoch,
                "fold_selected_epochs": ",".join(
                    str(x) for x in epochs
                ),
            }
        )

        # XGBoost is not reselected/retrained here.
        rows.append(
            {
                "backbone": "XGBoost",
                "representation": rep,
                "selected_config": "FROZEN_FROM_124D",
                "selection_source": (
                    "exact frozen 124D primary external prediction"
                ),
                "deployment_epochs": np.nan,
            }
        )

    return (
        pd.DataFrame(rows),
        pd.concat(
            audit_frames,
            ignore_index=True,
        ),
    )


# =============================================================================
# 5. FINAL FULL-HIROSHIMA MODEL FITS
# =============================================================================

def selected_row(
    configs: pd.DataFrame,
    backbone: str,
    representation: str,
) -> pd.Series:
    sub = configs[
        (configs["backbone"] == backbone)
        &
        (configs["representation"] == representation)
    ]
    if len(sub) != 1:
        raise RuntimeError(
            f"Deployment config row count={len(sub)} for {backbone}/{representation}"
        )
    return sub.iloc[0]


def logistic_c(cfg_id: str) -> float:
    m = re.search(
        r"C\s*=\s*([0-9.eE+-]+)",
        str(cfg_id),
    )
    if m is None:
        raise RuntimeError(
            f"Cannot parse ElasticNet C from {cfg_id}"
        )
    return float(m.group(1))


def hgb_cfg(m120: Any, cfg_id: str) -> Dict[str, Any]:
    for cfg in m120.HGB_CANDIDATES:
        if str(cfg["config_id"]) == str(cfg_id):
            return dict(cfg)
    raise RuntimeError(
        f"Unknown frozen HGB config={cfg_id}"
    )


def fit_predict_elasticnet(
    cfg_id: str,
    xh_z: np.ndarray,
    y_h: np.ndarray,
    xk_z: np.ndarray,
    jobs: int,
) -> np.ndarray:

    c = logistic_c(cfg_id)

    set_seed(FINAL_ELASTIC_SEED)

    model = LogisticRegression(
        C=c,
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        max_iter=3000,
        tol=1e-4,
        random_state=FINAL_ELASTIC_SEED,
        n_jobs=jobs,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        model.fit(
            xh_z,
            y_h,
            sample_weight=np.where(
                y_h == 1,
                1.0,
                0.5,
            ).astype(np.float32),
        )

        conv = any(
            issubclass(
                w.category,
                ConvergenceWarning,
            )
            for w in caught
        )

    if conv:
        log(
            "WARNING ElasticNet final full-Hiroshima fit emitted ConvergenceWarning"
        )

    margin = model.decision_function(
        xk_z
    )

    return np.asarray(
        margin,
        dtype=np.float64,
    )


def fit_predict_hgb(
    m120: Any,
    cfg_id: str,
    xh: np.ndarray,
    y_h: np.ndarray,
    xk: np.ndarray,
) -> np.ndarray:

    cfg = hgb_cfg(
        m120,
        cfg_id,
    )

    kwargs = {
        k: v
        for k, v in cfg.items()
        if k != "config_id"
    }

    set_seed(FINAL_HGB_SEED)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=FINAL_HGB_SEED,
        **kwargs,
    )

    model.fit(
        xh,
        y_h,
        sample_weight=np.where(
            y_h == 1,
            1.0,
            0.5,
        ).astype(np.float32),
    )

    p = model.predict_proba(
        xk
    )[:, 1]

    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(p) - np.log1p(-p)


class FixedMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                MLP_HIDDEN[0],
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                MLP_DROPOUT,
            ),
            torch.nn.Linear(
                MLP_HIDDEN[0],
                MLP_HIDDEN[1],
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                MLP_DROPOUT,
            ),
            torch.nn.Linear(
                MLP_HIDDEN[1],
                1,
            ),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def mlp_predict(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    model.eval()
    out = []

    for start in range(
        0,
        len(x),
        1024,
    ):
        xb = torch.from_numpy(
            x[start:start + 1024]
        ).to(
            device=device,
            dtype=torch.float32,
        )

        out.append(
            model(xb)
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out,
    ).astype(
        np.float64,
        copy=False,
    )


def fit_predict_mlp(
    xh_z: np.ndarray,
    y_h: np.ndarray,
    xk_z: np.ndarray,
    epochs: int,
    device: torch.device,
) -> np.ndarray:

    set_seed(
        FINAL_MLP_SEED,
    )

    model = FixedMLP(
        xh_z.shape[1],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )

    ds = TensorDataset(
        torch.from_numpy(
            xh_z.astype(
                np.float32,
                copy=False,
            )
        ),
        torch.from_numpy(
            y_h.astype(
                np.float32,
                copy=False,
            )
        ),
        torch.from_numpy(
            np.where(
                y_h == 1,
                1.0,
                0.5,
            ).astype(np.float32)
        ),
    )

    g = torch.Generator()
    g.manual_seed(
        FINAL_MLP_SEED,
    )

    loader = DataLoader(
        ds,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    for epoch in range(
        1,
        int(epochs) + 1,
    ):
        model.train()

        epoch_loss = 0.0
        n_batches = 0

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
                set_to_none=True,
            )

            logits = model(xb)

            each = (
                torch.nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    reduction="none",
                )
            )

            loss = (
                (each * wb).sum()
                / wb.sum().clamp_min(1e-8)
            )

            loss.backward()
            opt.step()

            epoch_loss += float(
                loss.detach().cpu()
            )
            n_batches += 1

        if (
            epoch == 1
            or epoch == epochs
            or epoch % 10 == 0
        ):
            log(
                f"MLP full-Hiroshima epoch={epoch}/{epochs} "
                f"mean_loss={epoch_loss/max(n_batches,1):.6f}"
            )

    return mlp_predict(
        model,
        xk_z,
        device,
    )


# =============================================================================
# 6. LOAD / VERIFY FROZEN 124D XGBOOST PREDICTIONS
# =============================================================================

def load_frozen_124d_xgb(
    model_index: pd.DataFrame,
) -> Dict[str, np.ndarray]:

    pred_path = (
        OUT124D
        / "124D_EXTERNAL_PREDICTIONS_FIXED_BEFORE_LABELS.parquet"
    )

    freeze_path = (
        OUT124D
        / "124D_PREDICTION_FREEZE.json"
    )

    if not pred_path.exists():
        raise FileNotFoundError(
            pred_path,
        )

    if not freeze_path.exists():
        raise FileNotFoundError(
            freeze_path,
        )

    pred = pd.read_parquet(
        pred_path,
    )

    freeze = json.loads(
        freeze_path.read_text(
            encoding="utf-8-sig",
        )
    )

    need = {
        "model_row",
        "unit_id",
        "pair_set_id",
        "backbone",
        "representation",
        "risk_score",
    }

    if not need.issubset(pred.columns):
        raise RuntimeError(
            f"124D frozen prediction file missing columns={sorted(need - set(pred.columns))}"
        )

    out = {}

    for rep in REPRESENTATIONS:
        sub = (
            pred[
                (pred["backbone"].astype(str) == "XGBoost")
                &
                (pred["representation"].astype(str) == rep)
            ]
            .sort_values(
                "model_row",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        if len(sub) != EXPECTED_MATCHED_ROWS:
            raise RuntimeError(
                f"124D frozen XGB {rep} rows={len(sub)}"
            )

        if not np.array_equal(
            sub["model_row"].to_numpy(np.int64),
            np.arange(
                EXPECTED_MATCHED_ROWS,
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                f"124D frozen XGB {rep} model_row coverage mismatch"
            )

        if not np.array_equal(
            sub["unit_id"].astype(str).to_numpy(),
            model_index["unit_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"124D frozen XGB {rep} unit_id order mismatch"
            )

        if not np.array_equal(
            sub["pair_set_id"].astype(str).to_numpy(),
            model_index["pair_set_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                f"124D frozen XGB {rep} pair_set_id order mismatch"
            )

        arr = sub["risk_score"].to_numpy(
            np.float64,
        )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"124D frozen XGB {rep} contains nonfinite scores"
            )

        key = (
            "raw_prediction_sha256"
            if rep == "RAW"
            else
            "msrr_prediction_sha256"
        )

        expected_hash = str(
            freeze[key]
        )

        actual_hash = hashlib.sha256(
            arr.tobytes()
        ).hexdigest()

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"124D frozen XGB {rep} hash mismatch"
            )

        out[rep] = arr

    return out


# =============================================================================
# 7. FIGURES
# =============================================================================

def make_figures(
    metrics_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(
            f"WARNING matplotlib unavailable: {exc}"
        )
        return

    # Figure 1: external metrics, all backbones.
    x = np.arange(
        len(METRICS4),
        dtype=float,
    )
    width = 0.10

    fig, ax = plt.subplots(
        figsize=(12, 6.5),
    )

    k = 0
    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            row = metrics_df[
                (metrics_df["backbone"] == backbone)
                &
                (metrics_df["representation"] == rep)
            ].iloc[0]

            vals = [
                float(row[m])
                for m in METRICS4
            ]

            offset = (
                k - 3.5
            ) * width

            ax.bar(
                x + offset,
                vals,
                width=width,
                label=f"{backbone} {rep}",
            )
            k += 1

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Kyushu 2017 external metric")
    ax.set_title(
        "Full cross-backbone external transfer: Hiroshima 2018 → Kyushu 2017"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "125_FULL_EXTERNAL_METRICS.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "125_FULL_EXTERNAL_METRICS.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: bootstrap forest, 16 rows.
    plot = boot_df.copy()

    labels = [
        f"{r.backbone} | {r.metric}"
        for r in plot.itertuples()
    ]

    yy = np.arange(
        len(plot),
        dtype=float,
    )

    xx = plot["OBSERVED_DELTA"].to_numpy(
        float,
    )
    lo = plot["CI95_LOW"].to_numpy(
        float,
    )
    hi = plot["CI95_HIGH"].to_numpy(
        float,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8.5),
    )

    ax.errorbar(
        xx,
        yy,
        xerr=np.vstack(
            [xx - lo, hi - xx]
        ),
        fmt="o",
        capsize=3,
    )
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    ax.set_yticks(yy)
    ax.set_yticklabels(
        labels,
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "External MSRR − RAW (95% paired matched-set bootstrap CI)"
    )
    ax.set_title(
        "Cross-backbone external representation gain"
    )
    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "125_EXTERNAL_BOOTSTRAP_FOREST.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "125_EXTERNAL_BOOTSTRAP_FOREST.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: internal vs external MSRR, backbone-level overview.
    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    plot_t = transfer_df[
        transfer_df["representation"] == "MSRR"
    ].copy()

    xx = np.arange(
        len(plot_t),
        dtype=float,
    )

    ax.plot(
        xx,
        plot_t["INTERNAL_AUROC"],
        marker="o",
        label="Hiroshima internal OOF AUROC",
    )
    ax.plot(
        xx,
        plot_t["EXTERNAL_AUROC"],
        marker="o",
        label="Kyushu external AUROC",
    )

    ax.set_xticks(xx)
    ax.set_xticklabels(
        plot_t["backbone"],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("AUROC")
    ax.set_title(
        "Internal-to-external MSRR AUROC transfer"
    )
    ax.legend(
        frameon=False,
    )
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "125_INTERNAL_VS_EXTERNAL_MSRR_AUROC.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "125_INTERNAL_VS_EXTERNAL_MSRR_AUROC.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 8. MAIN
# =============================================================================

def main(argv=None) -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
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

    if args.jobs < 1:
        raise ValueError(
            "--jobs must be >=1"
        )

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise FileExistsError(
            f"Refusing to overwrite non-empty 125 output: {OUT}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not OUT124D.exists():
        fail(
            "FAIL_125_124D_PRIMARY_RESULT_NOT_FOUND",
            str(OUT124D),
        )

    (
        p124,
        p120,
        p26,
        m124,
        m120,
        m26,
        bundle,
        runner,
        base,
    ) = load_all_authorities()

    if args.mlp_device == "cpu":
        device = torch.device(
            "cpu",
        )
    elif args.mlp_device == "cuda":
        if not torch.cuda.is_available():
            fail(
                "FAIL_125_CUDA_REQUESTED_BUT_UNAVAILABLE",
                "",
            )
        device = torch.device(
            "cuda",
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )

    freeze_protocol(
        p124,
        p120,
        p26,
        runner,
        base,
        args.jobs,
        str(device),
    )

    set_seed(
        SEED,
    )

    log("=" * 120)
    log("START 125 FULL CROSS-BACKBONE EXTERNAL TRANSFER")
    log("STATUS=SECONDARY_CONFIRMATORY_AFTER_PRIMARY_124D")
    log("XGBOOST_EXTERNAL_PREDICTIONS=FROZEN_124D_REUSE")
    log("NEW_EXTERNAL_BACKBONES=ElasticNet-Logistic,MLP,HistGradientBoosting")
    log("KYUSHU_USED_FOR_MODEL_SELECTION=NO")
    log("KYUSHU_USED_FOR_STANDARDIZATION=NO")
    log("POST_RESULT_RESCUE_ALLOWED=NO")
    log(f"MLP_DEVICE={device}")
    log("=" * 120)

    # -------------------------------------------------------------------------
    # A. Development labels + full-Hiroshima preprocessing.
    # -------------------------------------------------------------------------
    y_h_all = bundle.sample.y_pair.to_numpy(
        np.int8,
    )

    pair_h = np.asarray(
        bundle.pt,
        dtype=np.int64,
    )

    m120.verify_pair_order(
        pair_h,
        y_h_all,
        "HIROSHIMA_GLOBAL",
    )

    (
        static_h,
        rain_h,
        full_pre,
        hprep_info,
    ) = m124.fit_full_hiroshima_preprocessor(
        bundle,
        runner,
        base,
        m26,
    )

    xh_raw, ah_raw = m120.raw_features(
        pair_h,
        static_h,
        rain_h,
    )
    xh_msrr, ah_msrr = m120.msrr_features(
        pair_h,
        static_h,
        rain_h,
    )

    hidx = pair_h.reshape(-1)

    if not np.array_equal(
        ah_raw,
        hidx,
    ) or not np.array_equal(
        ah_msrr,
        hidx,
    ):
        fail(
            "FAIL_125_HIROSHIMA_FEATURE_ORDER",
            "",
        )

    y_h = y_h_all[
        hidx
    ]

    if xh_raw.shape != (
        base.N,
        792,
    ):
        fail(
            "FAIL_125_HIROSHIMA_RAW_SHAPE",
            str(xh_raw.shape),
        )

    if xh_msrr.shape != (
        base.N,
        2446,
    ):
        fail(
            "FAIL_125_HIROSHIMA_MSRR_SHAPE",
            str(xh_msrr.shape),
        )

    jwrite(
        OUT / "125_HIROSHIMA_FULL_PREPROCESSOR.json",
        full_pre,
    )

    jwrite(
        OUT / "125_HIROSHIMA_PREPROCESSING_AUDIT.json",
        hprep_info,
    )

    log(
        f"HIROSHIMA_FULL_PREPROCESSING_PASS "
        f"RAW={xh_raw.shape} MSRR={xh_msrr.shape}"
    )

    # -------------------------------------------------------------------------
    # B. Development-only configs BEFORE new external prediction.
    # -------------------------------------------------------------------------
    configs, config_audit = select_deployment_configs()

    configs.to_csv(
        OUT / "125_DEPLOYMENT_CONFIGS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config_audit.to_csv(
        OUT / "125_HIROSHIMA_VALIDATION_CONFIG_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for r in configs.itertuples():
        log(
            f"DEPLOYMENT_CONFIG {r.backbone} {r.representation} "
            f"config={r.selected_config} "
            f"epochs={getattr(r, 'deployment_epochs', np.nan)}"
        )

    # -------------------------------------------------------------------------
    # C. Reconstruct the exact same Kyushu model-ready inputs as 124D.
    # -------------------------------------------------------------------------
    inv = m124.inventory(
        KYUSHU,
    )

    inv.to_csv(
        OUT / "125_EXTERNAL_FILE_INVENTORY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_static_fields = (
        list(
            bundle.ss[
                "continuous_fields_in_order"
            ]
        )
        + [bundle.cat]
    )

    matched_path = m124.discover_by_schema(
        inv,
        [m124.EXPECTED["matched_rows"]],
        require_unit=True,
        require_pair=True,
        require_role_or_label=True,
        positive_name_tokens=[
            "matched",
            "pair",
            "triplet",
            "sample",
            "index",
            "formal",
        ],
        negative_name_tokens=[
            "dynamic",
            "rain",
            "192",
            "70",
            "prediction",
        ],
        label="MATCHED_INDEX",
    )

    static_path = m124.discover_by_schema(
        inv,
        [
            m124.EXPECTED["matched_rows"],
            m124.EXPECTED["master_grid_rows"],
        ],
        required_exact_cols=raw_static_fields,
        require_unit=True,
        positive_name_tokens=[
            "static",
            "feature",
            "raw",
            "92",
            "matched",
            "grid",
        ],
        negative_name_tokens=[
            "dynamic",
            "rain",
            "prediction",
            "result",
        ],
        label="RAW_STATIC_92",
    )

    dynamic70_path = m124.discover_by_schema(
        inv,
        [m124.EXPECTED["dynamic70_rows"]],
        required_exact_cols=list(
            base.RAIN_FIELDS,
        ),
        require_unit=True,
        positive_name_tokens=[
            "70",
            "dynamic",
            "rain",
            "view",
            "matched",
            "model",
        ],
        negative_name_tokens=[
            "192",
            "prediction",
            "result",
        ],
        label="DYNAMIC70_RAW_VIEW",
    )

    matched_raw = m124.read_table(
        matched_path,
    )
    static_ext_raw = m124.read_table(
        static_path,
    )
    dynamic70_raw = m124.read_table(
        dynamic70_path,
    )

    model_index, pair_k = (
        m124.build_external_model_index(
            matched_raw,
        )
    )

    (
        static_k,
        rain_k,
        kprep_audit,
    ) = m124.transform_external_with_hiroshima_preprocessor(
        static_ext_raw,
        dynamic70_raw,
        model_index,
        bundle,
        runner,
        m26,
        full_pre,
        base,
    )

    xk_raw, ak_raw = m120.raw_features(
        pair_k,
        static_k,
        rain_k,
    )
    xk_msrr, ak_msrr = m120.msrr_features(
        pair_k,
        static_k,
        rain_k,
    )

    kidx = pair_k.reshape(-1)

    if not np.array_equal(
        ak_raw,
        kidx,
    ) or not np.array_equal(
        ak_msrr,
        kidx,
    ):
        fail(
            "FAIL_125_KYUSHU_FEATURE_ORDER",
            "",
        )

    if xk_raw.shape != (
        EXPECTED_MATCHED_ROWS,
        792,
    ):
        fail(
            "FAIL_125_KYUSHU_RAW_SHAPE",
            str(xk_raw.shape),
        )

    if xk_msrr.shape != (
        EXPECTED_MATCHED_ROWS,
        2446,
    ):
        fail(
            "FAIL_125_KYUSHU_MSRR_SHAPE",
            str(xk_msrr.shape),
        )

    jwrite(
        OUT / "125_EXTERNAL_PREPROCESSING_AUDIT.json",
        {
            **kprep_audit,
            "matched_index_file": str(
                matched_path,
            ),
            "static_file": str(
                static_path,
            ),
            "dynamic70_file": str(
                dynamic70_path,
            ),
        },
    )

    log(
        f"PASS_125_EXTERNAL_MODEL_INPUT_AUDIT "
        f"RAW={xk_raw.shape} MSRR={xk_msrr.shape}"
    )

    # -------------------------------------------------------------------------
    # D. Representation-level standardization for ElasticNet / MLP.
    #    Hiroshima ONLY.
    # -------------------------------------------------------------------------
    raw_mu, raw_sd = m120.fit_standardizer(
        xh_raw,
    )
    msrr_mu, msrr_sd = m120.fit_standardizer(
        xh_msrr,
    )

    xh_raw_z = m120.apply_standardizer(
        xh_raw,
        raw_mu,
        raw_sd,
    )
    xk_raw_z = m120.apply_standardizer(
        xk_raw,
        raw_mu,
        raw_sd,
    )

    xh_msrr_z = m120.apply_standardizer(
        xh_msrr,
        msrr_mu,
        msrr_sd,
    )
    xk_msrr_z = m120.apply_standardizer(
        xk_msrr,
        msrr_mu,
        msrr_sd,
    )

    log(
        "LINEAR_MLP_STANDARDIZATION_FIT_SOURCE=HIROSHIMA_ONLY"
    )

    # -------------------------------------------------------------------------
    # E. Reuse frozen XGB and fit all three NEW external backbones.
    #    No Kyushu outcomes parsed yet for new models.
    # -------------------------------------------------------------------------
    scores: Dict[
        Tuple[str, str],
        np.ndarray,
    ] = {}

    xgb_frozen = load_frozen_124d_xgb(
        model_index,
    )

    scores[
        ("XGBoost", "RAW")
    ] = xgb_frozen["RAW"]

    scores[
        ("XGBoost", "MSRR")
    ] = xgb_frozen["MSRR"]

    log(
        "FROZEN_124D_XGBOOST_PREDICTIONS_HASH_VERIFIED=YES"
    )

    data = {
        "RAW": {
            "xh": xh_raw,
            "xk": xk_raw,
            "xh_z": xh_raw_z,
            "xk_z": xk_raw_z,
        },
        "MSRR": {
            "xh": xh_msrr,
            "xk": xk_msrr,
            "xh_z": xh_msrr_z,
            "xk_z": xk_msrr_z,
        },
    }

    timing_rows = []

    for rep in REPRESENTATIONS:
        d = data[rep]

        # ElasticNet
        row = selected_row(
            configs,
            "ElasticNet-Logistic",
            rep,
        )

        log(
            f"FIT_FINAL ElasticNet-Logistic {rep} "
            f"config={row['selected_config']}"
        )

        t0 = time.perf_counter()

        scores[
            ("ElasticNet-Logistic", rep)
        ] = fit_predict_elasticnet(
            str(
                row["selected_config"]
            ),
            d["xh_z"],
            y_h,
            d["xk_z"],
            args.jobs,
        )

        timing_rows.append(
            {
                "backbone": "ElasticNet-Logistic",
                "representation": rep,
                "seconds": (
                    time.perf_counter()
                    - t0
                ),
            }
        )

        # HGB
        row = selected_row(
            configs,
            "HistGradientBoosting",
            rep,
        )

        log(
            f"FIT_FINAL HistGradientBoosting {rep} "
            f"config={row['selected_config']}"
        )

        t0 = time.perf_counter()

        scores[
            ("HistGradientBoosting", rep)
        ] = fit_predict_hgb(
            m120,
            str(
                row["selected_config"]
            ),
            d["xh"],
            y_h,
            d["xk"],
        )

        timing_rows.append(
            {
                "backbone": "HistGradientBoosting",
                "representation": rep,
                "seconds": (
                    time.perf_counter()
                    - t0
                ),
            }
        )

        # MLP
        row = selected_row(
            configs,
            "MLP",
            rep,
        )

        epochs = int(
            row["deployment_epochs"]
        )

        log(
            f"FIT_FINAL MLP {rep} "
            f"epochs={epochs}"
        )

        t0 = time.perf_counter()

        scores[
            ("MLP", rep)
        ] = fit_predict_mlp(
            d["xh_z"],
            y_h,
            d["xk_z"],
            epochs,
            device,
        )

        timing_rows.append(
            {
                "backbone": "MLP",
                "representation": rep,
                "seconds": (
                    time.perf_counter()
                    - t0
                ),
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    pd.DataFrame(
        timing_rows,
    ).to_csv(
        OUT / "125_TRAIN_PREDICT_TIMING.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Coverage before outcome metrics.
    for key, arr in scores.items():
        if arr.shape != (
            EXPECTED_MATCHED_ROWS,
        ):
            fail(
                "FAIL_125_PREDICTION_SHAPE",
                f"{key}: {arr.shape}",
            )

        if not np.isfinite(
            arr,
        ).all():
            fail(
                "FAIL_125_PREDICTION_NONFINITE",
                str(key),
            )

    prelabel_frames = []

    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            prelabel_frames.append(
                pd.DataFrame(
                    {
                        "model_row": model_index[
                            "model_row"
                        ],
                        "unit_id": model_index[
                            "unit_id"
                        ],
                        "pair_set_id": model_index[
                            "pair_set_id"
                        ],
                        "backbone": backbone,
                        "representation": rep,
                        "risk_score": scores[
                            (backbone, rep)
                        ],
                    }
                )
            )

    prelabel = pd.concat(
        prelabel_frames,
        ignore_index=True,
    )

    prelabel.to_parquet(
        OUT / "125_ALL_BACKBONE_PREDICTIONS_FIXED_BEFORE_NEW_METRICS.parquet",
        index=False,
    )

    jwrite(
        OUT / "125_PREDICTION_FREEZE.json",
        {
            "status": "PASS_ALL_8_EXTERNAL_PREDICTION_VECTORS_FIXED",
            "xgboost_source": (
                "exact frozen 124D primary predictions"
            ),
            "new_backbones": NEW_BACKBONES,
            "prediction_rows_per_cell": EXPECTED_MATCHED_ROWS,
            "hashes": {
                f"{b}__{r}": hashlib.sha256(
                    scores[(b, r)].tobytes()
                ).hexdigest()
                for b in BACKBONES
                for r in REPRESENTATIONS
            },
            "new_backbone_kyushu_outcomes_used_for_fit_or_selection": False,
        },
    )

    log(
        "ALL_8_EXTERNAL_PREDICTION_VECTORS_FIXED=YES"
    )
    log(
        "BEGIN_FULL_PANEL_EXTERNAL_METRICS_AND_BOOTSTRAP_NOW"
    )

    # -------------------------------------------------------------------------
    # F. Outcome mapping and external metrics.
    # -------------------------------------------------------------------------
    eval_index, eval_rows = (
        m124.build_external_evaluation_order(
            matched_raw,
            model_index,
        )
    )

    if len(eval_rows) != EXPECTED_MATCHED_SETS:
        fail(
            "FAIL_125_EVAL_SET_COUNT",
            str(len(eval_rows)),
        )

    metric_rows = []

    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            m = m124.metrics_from_eval_rows(
                scores[(backbone, rep)],
                eval_rows,
            )

            metric_rows.append(
                {
                    "backbone": backbone,
                    "representation": rep,
                    **m,
                }
            )

    metrics_df = pd.DataFrame(
        metric_rows,
    )

    metrics_df.to_csv(
        OUT / "125_EXTERNAL_METRICS_FULL_PANEL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_rows = []

    for backbone in BACKBONES:
        raw = metrics_df[
            (metrics_df["backbone"] == backbone)
            &
            (metrics_df["representation"] == "RAW")
        ].iloc[0]

        msrr = metrics_df[
            (metrics_df["backbone"] == backbone)
            &
            (metrics_df["representation"] == "MSRR")
        ].iloc[0]

        delta = {
            m: float(
                msrr[m] - raw[m]
            )
            for m in METRICS4
        }

        wins = int(
            sum(
                delta[m] > 0
                for m in METRICS4
            )
        )

        delta_rows.append(
            {
                "backbone": backbone,
                **{
                    f"delta_{m}": delta[m]
                    for m in METRICS4
                },
                "positive_metric_deltas_out_of_4": wins,
                "observed_support": bool(
                    wins >= 3
                ),
            }
        )

    delta_df = pd.DataFrame(
        delta_rows,
    )

    delta_df.to_csv(
        OUT / "125_EXTERNAL_MSRR_MINUS_RAW_FULL_PANEL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # G. 10,000 matched-set paired bootstrap for ALL FOUR backbones.
    # -------------------------------------------------------------------------
    boot_summaries = []
    boot_reps = []

    for backbone in BACKBONES:
        log(
            f"BOOTSTRAP_START {backbone} B={BOOTSTRAP_B}"
        )

        # m124 uses the same frozen B and seed as 124D.
        bs, br = m124.paired_bootstrap(
            scores[(backbone, "RAW")],
            scores[(backbone, "MSRR")],
            eval_rows,
        )

        bs.insert(
            0,
            "backbone",
            backbone,
        )
        br.insert(
            0,
            "backbone",
            backbone,
        )

        boot_summaries.append(
            bs,
        )
        boot_reps.append(
            br,
        )

    boot_df = pd.concat(
        boot_summaries,
        ignore_index=True,
    )

    boot_rep_df = pd.concat(
        boot_reps,
        ignore_index=True,
    )

    boot_df.to_csv(
        OUT / "125_EXTERNAL_BOOTSTRAP_SUMMARY_FULL_PANEL.csv",
        index=False,
        encoding="utf-8-sig",
    )

    boot_rep_df.to_parquet(
        OUT / "125_EXTERNAL_BOOTSTRAP_REPLICATES_FULL_PANEL.parquet",
        index=False,
    )

    # -------------------------------------------------------------------------
    # H. 125 gate -- ONLY the three newly unseen external backbones.
    # -------------------------------------------------------------------------
    support_rows = []

    for backbone in BACKBONES:
        drow = delta_df[
            delta_df["backbone"] == backbone
        ].iloc[0]

        bsub = boot_df[
            boot_df["backbone"] == backbone
        ].copy()

        ci_positive = int(
            (
                bsub["CI95_LOW"] > 0
            ).sum()
        )

        observed_support = bool(
            int(
                drow[
                    "positive_metric_deltas_out_of_4"
                ]
            ) >= 3
        )

        bootstrap_support = bool(
            ci_positive >= 3
        )

        support_rows.append(
            {
                "backbone": backbone,
                "external_status": (
                    "ALREADY_OBSERVED_PRIMARY_124D"
                    if backbone == "XGBoost"
                    else
                    "NEW_125_CONFIRMATORY"
                ),
                "positive_metric_deltas_out_of_4": int(
                    drow[
                        "positive_metric_deltas_out_of_4"
                    ]
                ),
                "bootstrap_ci_low_gt0_out_of_4": ci_positive,
                "observed_support": observed_support,
                "bootstrap_support": bootstrap_support,
            }
        )

    support_df = pd.DataFrame(
        support_rows,
    )

    support_df.to_csv(
        OUT / "125_EXTERNAL_BACKBONE_SUPPORT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )

    new = support_df[
        support_df["backbone"].isin(
            NEW_BACKBONES
        )
    ]

    n_obs = int(
        new["observed_support"].sum()
    )
    n_boot = int(
        new["bootstrap_support"].sum()
    )

    if (
        n_obs == 3
        and n_boot == 3
    ):
        gate = (
            "STRONG_SECONDARY_CROSS_BACKBONE_EXTERNAL_REPLICATION"
        )

    elif (
        n_obs == 3
        and n_boot >= 2
    ):
        gate = (
            "CROSS_BACKBONE_EXTERNAL_REPLICATION"
        )

    elif n_obs >= 2:
        gate = (
            "PARTIAL_CROSS_BACKBONE_EXTERNAL_REPLICATION"
        )

    else:
        gate = (
            "CROSS_BACKBONE_EXTERNAL_REPLICATION_NOT_SUPPORTED"
        )

    # -------------------------------------------------------------------------
    # I. Internal 120B vs external 125 transfer table.
    # -------------------------------------------------------------------------
    internal_path = (
        OUT120B
        / "OOF_RESULTS.csv"
    )

    if not internal_path.exists():
        raise FileNotFoundError(
            internal_path,
        )

    internal = pd.read_csv(
        internal_path,
        low_memory=False,
    )

    transfer_rows = []

    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            ir = internal[
                (internal["backbone"].astype(str) == backbone)
                &
                (internal["representation"].astype(str) == rep)
            ]

            er = metrics_df[
                (metrics_df["backbone"] == backbone)
                &
                (metrics_df["representation"] == rep)
            ]

            if len(ir) != 1 or len(er) != 1:
                raise RuntimeError(
                    f"Internal/external row mismatch {backbone}/{rep}"
                )

            ir = ir.iloc[0]
            er = er.iloc[0]

            rec = {
                "backbone": backbone,
                "representation": rep,
            }

            for metric in METRICS4:
                rec[
                    f"INTERNAL_{metric}"
                ] = float(
                    ir[metric]
                )

                rec[
                    f"EXTERNAL_{metric}"
                ] = float(
                    er[metric]
                )

                rec[
                    f"EXTERNAL_MINUS_INTERNAL_{metric}"
                ] = float(
                    er[metric] - ir[metric]
                )

            transfer_rows.append(
                rec,
            )

    transfer_df = pd.DataFrame(
        transfer_rows,
    )

    transfer_df.to_csv(
        OUT / "125_INTERNAL_VS_EXTERNAL_TRANSFER_GAP.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # J. Final row-level file with outcomes.
    # -------------------------------------------------------------------------
    label_by_model_row = (
        eval_index.set_index(
            "model_row"
        )[
            [
                "unit_id",
                "pair_set_id",
                "sample_role",
                "control_rank",
                "y_true",
            ]
        ]
        .sort_index()
        .reset_index()
    )

    pred_frames = []

    for backbone in BACKBONES:
        for rep in REPRESENTATIONS:
            pred_frames.append(
                label_by_model_row.assign(
                    backbone=backbone,
                    representation=rep,
                    risk_score=scores[
                        (backbone, rep)
                    ],
                    score_sigmoid=m124.sigmoid(
                        scores[
                            (backbone, rep)
                        ]
                    ),
                )
            )

    pd.concat(
        pred_frames,
        ignore_index=True,
    ).to_parquet(
        OUT / "125_EXTERNAL_PREDICTIONS_FULL_PANEL_WITH_LABELS.parquet",
        index=False,
    )

    # -------------------------------------------------------------------------
    # K. Decision and figures.
    # -------------------------------------------------------------------------
    decision = {
        "status": "PASS_125_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL_COMPLETE",
        "gate125_decision": gate,
        "primary_untouched_xgboost_source": "124D frozen and hash-verified",
        "new_backbone_observed_support_count": n_obs,
        "new_backbone_bootstrap_support_count": n_boot,
        "new_backbone_total": 3,
        "all_four_backbone_support_summary": (
            support_df.to_dict(
                orient="records",
            )
        ),
        "post_result_rescue_allowed": False,
        "next_step": (
            "INTERPRET_FULL_EXTERNAL_TRANSFER_THEN_DOMAIN_SHIFT_AND_SPATIAL_ANALYSIS"
        ),
    }

    jwrite(
        OUT / "125_GATE_DECISION.json",
        decision,
    )

    make_figures(
        metrics_df,
        boot_df,
        transfer_df,
    )

    # -------------------------------------------------------------------------
    # L. Terminal summary
    # -------------------------------------------------------------------------
    log("")
    log("=" * 120)
    log(
        "PASS_125_KYUSHU2017_FULL_CROSS_BACKBONE_EXTERNAL_COMPLETE"
    )
    log(
        f"GATE125_DECISION={gate}"
    )
    log(
        f"NEW_BACKBONE_OBSERVED_SUPPORT={n_obs}/3"
    )
    log(
        f"NEW_BACKBONE_BOOTSTRAP_SUPPORT={n_boot}/3"
    )
    log("=" * 120)

    for backbone in BACKBONES:
        raw = metrics_df[
            (metrics_df["backbone"] == backbone)
            &
            (metrics_df["representation"] == "RAW")
        ].iloc[0]

        msrr = metrics_df[
            (metrics_df["backbone"] == backbone)
            &
            (metrics_df["representation"] == "MSRR")
        ].iloc[0]

        log(
            f"{backbone} RAW: "
            f"AUROC={raw['AUROC']:.6f} "
            f"AUPRC={raw['AUPRC']:.6f} "
            f"StrictPair={raw['StrictPair']:.6f} "
            f"Edge={raw['Edge']:.6f}"
        )

        log(
            f"{backbone} MSRR: "
            f"AUROC={msrr['AUROC']:.6f} "
            f"AUPRC={msrr['AUPRC']:.6f} "
            f"StrictPair={msrr['StrictPair']:.6f} "
            f"Edge={msrr['Edge']:.6f}"
        )

        d = delta_df[
            delta_df["backbone"] == backbone
        ].iloc[0]

        b = support_df[
            support_df["backbone"] == backbone
        ].iloc[0]

        log(
            f"{backbone} DELTA: "
            f"AUROC={d['delta_AUROC']:+.6f} "
            f"AUPRC={d['delta_AUPRC']:+.6f} "
            f"StrictPair={d['delta_StrictPair']:+.6f} "
            f"Edge={d['delta_Edge']:+.6f} "
            f"positive_metrics={int(b['positive_metric_deltas_out_of_4'])}/4 "
            f"CI_low_gt0={int(b['bootstrap_ci_low_gt0_out_of_4'])}/4"
        )

    log("")
    log(
        "FULL_EXTERNAL_BOOTSTRAP_SUMMARY:"
    )

    for r in boot_df.itertuples():
        log(
            f"  {r.backbone} {r.metric}: "
            f"delta={r.OBSERVED_DELTA:+.6f} "
            f"CI95=[{r.CI95_LOW:+.6f},{r.CI95_HIGH:+.6f}] "
            f"p={r.TWO_SIDED_BOOTSTRAP_P:.8f}"
        )

    log("")
    log(
        f"OUTPUT={OUT}"
    )
    log(
        "NEXT_STEP=INTERPRET_FULL_EXTERNAL_TRANSFER_THEN_DOMAIN_SHIFT_AND_SPATIAL_ANALYSIS"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except SystemExit:
        raise
    except Exception as exc:
        print("")
        print("=" * 120)
        print("125 FULL EXTERNAL PANEL FAILED")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 120)
        traceback.print_exc()
        raise SystemExit(1)
