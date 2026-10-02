from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .base import BaselineAdapter
from .metrics import validation_metrics


class PUBaggingDTAdapter(BaselineAdapter):
    """Paper-faithful benchmark reimplementation; U labels are temporary."""

    def __init__(self, *, n_estimators: int = 50, u_fraction: float = 0.70,
                 max_depth: int | None = None, min_samples_leaf: int = 1,
                 class_weight: str | None = "balanced",
                 progress_callback: Callable[[str], None] | None = None,
                 progress_every: int = 10):
        self.config = {"n_estimators": n_estimators, "u_fraction": u_fraction,
                       "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
                       "class_weight": class_weight}
        self.trees: list[DecisionTreeClassifier] = []
        self.validation_result: dict[str, float] | None = None
        self.progress_callback = progress_callback
        self.progress_every = max(1, int(progress_every))

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def fit(self, X_fit: np.ndarray, pu_label_fit: np.ndarray, X_val: np.ndarray,
            pu_label_val: np.ndarray, val_metadata: pd.DataFrame, seed: int) -> "PUBaggingDTAdapter":
        X_fit, y = self.check_xy(X_fit, pu_label_fit)
        X_val, yv = self.check_xy(X_val, pu_label_val)
        p = np.flatnonzero(y == 1)
        u = np.flatnonzero(y == 0)
        k = max(1, min(len(u), int(round(self.config["u_fraction"] * len(u)))))
        rng = np.random.default_rng(seed)
        self.trees = []
        ybag = np.r_[np.ones(len(p), np.int8), np.zeros(k, np.int8)]
        started = time.perf_counter()
        self._progress(f"PU_BAGGING START trees={self.config['n_estimators']} P={len(p)} U={len(u)} temporary_U={k}")
        for t in range(self.config["n_estimators"]):
            us = rng.choice(u, size=k, replace=False)
            tree = DecisionTreeClassifier(
                criterion="gini", max_depth=self.config["max_depth"],
                min_samples_leaf=self.config["min_samples_leaf"],
                max_features=None, class_weight=self.config["class_weight"],
                random_state=seed + t,
            )
            tree.fit(np.concatenate([X_fit[p], X_fit[us]], axis=0), ybag)
            self.trees.append(tree)
            done = t + 1
            if done == 1 or done % self.progress_every == 0 or done == self.config["n_estimators"]:
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (self.config["n_estimators"] - done)
                self._progress(f"PU_BAGGING PROGRESS trees={done}/{self.config['n_estimators']} elapsed_s={elapsed:.1f} eta_s={eta:.1f}")
        self.validation_result = validation_metrics(self.predict_score(X_val), yv, val_metadata)
        self.config = {**self.config, "temporary_U_size": k, "seed": seed,
                       "validation_metrics": self.validation_result}
        self._progress(f"PU_BAGGING COMPLETE validation_score={self.validation_result['validation_score']:.6f}")
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not self.trees:
            raise RuntimeError("ADAPTER_NOT_FIT")
        out = np.mean([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in self.trees], axis=0)
        return self.check_scores(out, len(X))

    def get_selected_config(self) -> dict[str, Any]:
        return dict(self.config)
