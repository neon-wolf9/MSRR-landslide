from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS4 = ("AUROC", "AUPRC", "StrictPair", "Edge")


def validation_metrics(scores: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> dict[str, float]:
    """120B/121-aligned ranking metrics; score ties fail pair comparisons."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    if len(scores) != len(y) or len(metadata) != len(y) or not np.isfinite(scores).all():
        raise ValueError("VALIDATION_SCORE_ALIGNMENT_FAILURE")
    work = metadata[["pair_set_id"]].copy()
    work["y"] = y
    work["score"] = scores
    strict, edge = [], []
    for _, g in work.groupby("pair_set_id", sort=False):
        ps = g.loc[g.y.eq(1), "score"].to_numpy()
        us = g.loc[g.y.eq(0), "score"].to_numpy()
        if len(ps) != 1 or len(us) != 2:
            raise ValueError("PAIR_NOT_ONE_P_TWO_U")
        wins = ps[0] > us
        strict.append(float(wins.all()))
        edge.append(float(wins.mean()))
    out = {
        "AUROC": float(roc_auc_score(y, scores)),
        "AUPRC": float(average_precision_score(y, scores)),
        "StrictPair": float(np.mean(strict)),
        "Edge": float(np.mean(edge)),
    }
    out["validation_score"] = float(np.mean([out[k] for k in METRICS4]))
    return out


def selection_key(row: dict, index: int) -> tuple[float, ...]:
    return (row["validation_score"], row["StrictPair"], row["Edge"],
            row["AUPRC"], row["AUROC"], -index)


def array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()
