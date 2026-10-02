from __future__ import annotations

import random
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .base import BaselineAdapter
from .metrics import validation_metrics
from .pu_bagging_dt import PUBaggingDTAdapter


class _PUPullEncoder(nn.Module):
    """Official hidden/embedding widths; only first-layer input is adapted."""

    def __init__(self, input_dim: int = 792):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 50), nn.ReLU(),
                                 nn.Linear(50, 100), nn.ReLU(),
                                 nn.Linear(100, 50), nn.ReLU(),
                                 nn.Linear(50, 50), nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class PUPullBaggingDTAdapter(BaselineAdapter):
    """Official-code-informed leakage-safe smoke adaptation."""

    def __init__(self, *, epochs: int = 3, pu_trees: int = 20,
                 learning_rate: float = 1e-4,
                 progress_callback: Callable[[str], None] | None = None):
        self.epochs = epochs
        self.pu_trees = pu_trees
        self.learning_rate = learning_rate
        self.scaler: StandardScaler | None = None
        self.encoder: _PUPullEncoder | None = None
        self.pu_branch: PUBaggingDTAdapter | None = None
        self.pos_embedding: np.ndarray | None = None
        self.cmin = 0.0
        self.cmax = 1.0
        self.selected: dict[str, Any] = {}
        self.progress_callback = progress_callback

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def _embed(self, z: np.ndarray) -> np.ndarray:
        assert self.encoder is not None
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(torch.from_numpy(np.asarray(z, dtype=np.float32))).cpu().numpy()

    def _contrastive_raw(self, z: np.ndarray) -> np.ndarray:
        emb = self._embed(z)
        assert self.pos_embedding is not None
        return np.asarray(emb @ self.pos_embedding.mean(axis=0), dtype=np.float64)

    def _contrastive_score(self, z: np.ndarray) -> np.ndarray:
        raw = self._contrastive_raw(z)
        return np.clip((raw - self.cmin) / max(self.cmax - self.cmin, 1e-12), 0.0, 1.0)

    def fit(self, X_fit: np.ndarray, pu_label_fit: np.ndarray, X_val: np.ndarray,
            pu_label_val: np.ndarray, val_metadata: pd.DataFrame, seed: int) -> "PUPullBaggingDTAdapter":
        X_fit, y = self.check_xy(X_fit, pu_label_fit)
        X_val, yv = self.check_xy(X_val, pu_label_val)
        self._seed(seed)
        self.scaler = StandardScaler().fit(X_fit)
        zfit = self.scaler.transform(X_fit).astype(np.float32)
        zval = self.scaler.transform(X_val).astype(np.float32)
        p = torch.from_numpy(zfit[y == 1])
        u = torch.from_numpy(zfit[y == 0])
        self.encoder = _PUPullEncoder(792)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        losses = []
        started = time.perf_counter()
        self._progress(f"PU_PULL CONTRASTIVE START epochs={self.epochs} P={len(p)} U={len(u)} input_dim=792")
        for epoch in range(self.epochs):
            self.encoder.train()
            opt.zero_grad(set_to_none=True)
            ep = self.encoder(p)
            eu = self.encoder(u)
            pp = ep @ ep.T
            mask = ~torch.eye(len(ep), dtype=torch.bool)
            pos = torch.exp(pp[mask].reshape(len(ep), len(ep) - 1).mean(dim=1))
            neg = torch.exp(ep @ eu.T).sum(dim=1)
            loss = (-torch.log(pos / (pos + neg))).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            elapsed = time.perf_counter() - started
            done = epoch + 1
            eta = elapsed / done * (self.epochs - done)
            self._progress(f"PU_PULL EPOCH {done}/{self.epochs} loss={losses[-1]:.8f} elapsed_s={elapsed:.1f} eta_s={eta:.1f}")
        self.pos_embedding = self._embed(zfit[y == 1])
        ctr_fit = self._contrastive_raw(zfit)
        self.cmin, self.cmax = float(np.min(ctr_fit)), float(np.max(ctr_fit))
        self.pu_branch = PUBaggingDTAdapter(n_estimators=self.pu_trees, u_fraction=0.70,
                                            max_depth=None, min_samples_leaf=1,
                                            class_weight="balanced",
                                            progress_callback=self.progress_callback,
                                            progress_every=10)
        self._progress(f"PU_PULL PU_BRANCH START trees={self.pu_trees}")
        self.pu_branch.fit(zfit, y, zval, yv, val_metadata, seed + 1000)
        score = self._fused_score(zval)
        vm = validation_metrics(score, yv, val_metadata)
        self.selected = {"input_dim": 792, "hidden_widths": [50, 100, 50],
                         "embedding_dim": 50, "epochs": self.epochs,
                         "learning_rate": self.learning_rate, "pu_trees": self.pu_trees,
                         "losses": losses, "fold_local_scaler": True,
                         "contrastive_fit_min": self.cmin, "contrastive_fit_max": self.cmax,
                         "fusion": "0.5*contrastive_score + 0.5*PU_BaggingDT_score",
                         "validation_metrics": vm, "seed": seed}
        self._progress(f"PU_PULL COMPLETE validation_score={vm['validation_score']:.6f}")
        return self

    def _fused_score(self, z: np.ndarray) -> np.ndarray:
        assert self.pu_branch is not None
        return 0.5 * self._contrastive_score(z) + 0.5 * self.pu_branch.predict_score(z)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("ADAPTER_NOT_FIT")
        X = np.asarray(X, dtype=np.float32)
        z = self.scaler.transform(X).astype(np.float32)
        return self.check_scores(self._fused_score(z), len(X))

    def get_selected_config(self) -> dict[str, Any]:
        return dict(self.selected)
