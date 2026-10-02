from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SmokeFoldData:
    X_fit: np.ndarray
    y_fit: np.ndarray
    fit_metadata: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    val_metadata: pd.DataFrame
    X_test: np.ndarray
    y_test_contract_only: np.ndarray
    test_metadata: pd.DataFrame
    audit: dict[str, Any]


def _raw(pair_rows: np.ndarray, static92: np.ndarray, rain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(pair_rows, dtype=np.int64)
    idx = tri.reshape(-1)
    X = np.concatenate([static92[idx], rain[idx].reshape(len(idx), -1)], axis=1).astype(np.float32)
    if X.shape[1] != 792 or not np.isfinite(X).all():
        raise RuntimeError(f"RAW_CONTRACT_FAILURE={X.shape}")
    return X, idx


def load_outer_fold(human_outer_fold: int = 1) -> SmokeFoldData:
    """Reuse the authoritative BASE06/120B loader and fold-only preprocessing."""
    if human_outer_fold != 1:
        raise ValueError("128B0 is frozen to outer fold 1 only")
    from scripts.TRAIN_SEHC_V2_TEMPLATE import build_fold_loader, load_dataset_loader

    bundle, runner, base = load_dataset_loader()
    fd = build_fold_loader(bundle, runner, base, human_outer_fold, torch.device("cpu"))
    sample = bundle.sample.reset_index(drop=True).copy()
    y_all = sample["y_pair"].to_numpy(np.int8)
    all_pairs = np.asarray(bundle.pt, dtype=np.int64)
    static92 = fd.static92.detach().cpu().numpy().astype(np.float32, copy=False)
    rain = fd.rain.detach().cpu().numpy().astype(np.float32, copy=False)
    tr_pairs = all_pairs[np.asarray(fd.train_pairs, dtype=np.int64)]
    va_pairs = all_pairs[np.asarray(fd.validation_pairs, dtype=np.int64)]
    te_pairs = all_pairs[np.asarray(fd.test_pairs, dtype=np.int64)]
    Xtr, itr = _raw(tr_pairs, static92, rain)
    Xva, iva = _raw(va_pairs, static92, rain)
    Xte, ite = _raw(te_pairs, static92, rain)
    meta_cols = ["sample_index", "unit_id", "pair_set_id", "sample_role",
                 "control_rank", "outer_fold", "spatial_group_id"]
    mtr = sample.iloc[itr][meta_cols].reset_index(drop=True)
    mva = sample.iloc[iva][meta_cols].reset_index(drop=True)
    mte = sample.iloc[ite][meta_cols].reset_index(drop=True)
    ytr, yva, yte = y_all[itr], y_all[iva], y_all[ite]

    def overlap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
        return len(set(a[col].astype(str)) & set(b[col].astype(str)))

    forbidden_tokens = ["unit_id", "pair_set_id", "fold", "spatial_group", "sample_role",
                        "control_rank", "matching_cost", "coordinate", "ajg", "gsi",
                        "evidence", "component", "future_rainfall"]
    audit = {
        "authority_loader": str(ROOT / "scripts" / "TRAIN_SEHC_V2_TEMPLATE.py"),
        "authority_raw_logic": str(ROOT / "scripts" / "120B_RUN_MSRR_CROSS_BACKBONE_5FOLD_NO_LIGHTGBM_ONE_SHOT.py"),
        "outer_fold": 1, "validation_fold": 2, "fit_folds": [3, 4, 5], "raw_dim": 792,
        "fit": {"rows": len(ytr), "P": int(ytr.sum()), "U": int((ytr == 0).sum())},
        "validation": {"rows": len(yva), "P": int(yva.sum()), "U": int((yva == 0).sum())},
        "test": {"rows": len(yte), "sets": int(mte.pair_set_id.nunique()),
                 "P": int(yte.sum()), "controls": int((yte == 0).sum())},
        "overlap": {
            "fit_test_unit": overlap(mtr, mte, "unit_id"),
            "fit_test_pair": overlap(mtr, mte, "pair_set_id"),
            "fit_test_spatial_group": overlap(mtr, mte, "spatial_group_id"),
            "fit_test_dynamic_sequence": overlap(mtr, mte, "unit_id"),
            "validation_test_spatial_group": overlap(mva, mte, "spatial_group_id"),
        },
        "feature_container": "numpy.ndarray assembled only from fold-preprocessed static92 and rain70x10",
        "feature_names": None,
        "forbidden_tokens": forbidden_tokens,
        "forbidden_feature_hits": [],
        "metadata_separate_from_X": True,
    }
    expected = (audit["fit"] == {"rows": 9102, "P": 3034, "U": 6068}
                and audit["validation"] == {"rows": 3033, "P": 1011, "U": 2022}
                and audit["test"] == {"rows": 3033, "sets": 1011, "P": 1011, "controls": 2022}
                and sum(audit["overlap"].values()) == 0)
    audit["pass"] = bool(expected)
    if not expected:
        raise RuntimeError(f"DATA_CONTRACT_AUDIT_FAILED={audit}")
    return SmokeFoldData(Xtr, ytr, mtr, Xva, yva, mva, Xte, yte, mte, audit)
