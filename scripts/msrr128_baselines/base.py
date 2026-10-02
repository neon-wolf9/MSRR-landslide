from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaselineAdapter(ABC):
    """Common 128 API. Label 0 means unlabeled, never verified negative."""

    @abstractmethod
    def fit(
        self,
        X_fit: np.ndarray,
        pu_label_fit: np.ndarray,
        X_val: np.ndarray,
        pu_label_val: np.ndarray,
        val_metadata: pd.DataFrame,
        seed: int,
    ) -> "BaselineAdapter":
        raise NotImplementedError

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_selected_config(self) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def check_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        if X.ndim != 2 or X.shape[1] != 792 or len(X) != len(y):
            raise ValueError(f"BAD_128_INPUT X={X.shape} y={y.shape}")
        if not np.isfinite(X).all() or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("NONFINITE_FEATURE_OR_BAD_PU_LABEL")
        if not np.any(y == 1) or not np.any(y == 0):
            raise ValueError("BOTH_P_AND_U_REQUIRED")
        return X, y

    @staticmethod
    def check_scores(score: np.ndarray, n: int) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if score.shape != (n,) or not np.isfinite(score).all():
            raise RuntimeError(f"BAD_SCORE_CONTRACT shape={score.shape} expected={(n,)}")
        return score
