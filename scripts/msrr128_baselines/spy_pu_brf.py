from __future__ import annotations

from itertools import product
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from .base import BaselineAdapter
from .metrics import selection_key, validation_metrics


class SpyPUBRFAdapter(BaselineAdapter):
    THRESHOLD_RULES = ("spy_min", "spy_q05", "spy_q10")

    def __init__(self, *, spy_fraction: float = 0.15, stage1_trees: int = 50,
                 brf_trees: int = 50,
                 threshold_rules: tuple[str, ...] | list[str] | None = None,
                 brf_search_space: dict[str, list[Any]] | None = None,
                 progress_callback: Callable[[str], None] | None = None):
        self.spy_fraction = spy_fraction
        self.stage1_trees = stage1_trees
        self.brf_trees = brf_trees
        self.threshold_rules = tuple(threshold_rules or self.THRESHOLD_RULES)
        unknown = sorted(set(self.threshold_rules) - set(self.THRESHOLD_RULES))
        if unknown:
            raise ValueError(f"UNKNOWN_SPY_THRESHOLD_RULES={unknown}")
        # None preserves the already-passed 128B0 fixed smoke behaviour.
        self.brf_search_space = brf_search_space
        self.model: BalancedRandomForestClassifier | None = None
        self.selected: dict[str, Any] = {}
        self.candidate_results: list[dict[str, Any]] = []
        self.progress_callback = progress_callback

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def fit(self, X_fit: np.ndarray, pu_label_fit: np.ndarray, X_val: np.ndarray,
            pu_label_val: np.ndarray, val_metadata: pd.DataFrame, seed: int) -> "SpyPUBRFAdapter":
        X_fit, y = self.check_xy(X_fit, pu_label_fit)
        X_val, yv = self.check_xy(X_val, pu_label_val)
        p = np.flatnonzero(y == 1)
        u = np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed)
        n_spy = max(1, int(round(self.spy_fraction * len(p))))
        spies = np.sort(rng.choice(p, size=n_spy, replace=False))
        remain_p = np.setdiff1d(p, spies, assume_unique=True)
        contaminated = np.r_[u, spies]
        self._progress(f"SPY STAGE1 START trees={self.stage1_trees} P_remaining={len(remain_p)} contaminated_U={len(contaminated)} spies={n_spy}")
        stage1_started = time.perf_counter()
        stage1 = RandomForestClassifier(n_estimators=self.stage1_trees, max_features="sqrt",
                                        class_weight="balanced", random_state=seed,
                                        n_jobs=1)
        stage1.fit(np.concatenate([X_fit[remain_p], X_fit[contaminated]]),
                   np.r_[np.ones(len(remain_p), np.int8), np.zeros(len(contaminated), np.int8)])
        spy_score = stage1.predict_proba(X_fit[spies])[:, list(stage1.classes_).index(1)]
        u_score = stage1.predict_proba(X_fit[u])[:, list(stage1.classes_).index(1)]
        thresholds = {"spy_min": float(np.min(spy_score)),
                      "spy_q05": float(np.quantile(spy_score, 0.05)),
                      "spy_q10": float(np.quantile(spy_score, 0.10))}
        self._progress(f"SPY STAGE1 COMPLETE elapsed_s={time.perf_counter()-stage1_started:.1f} thresholds={thresholds}")
        if self.brf_search_space is None:
            brf_configs = [{"n_estimators": self.brf_trees, "max_depth": None,
                            "min_samples_leaf": 1, "max_features": "sqrt",
                            "replacement": True}]
        else:
            required = {"n_estimators", "max_depth", "min_samples_leaf",
                        "max_features", "replacement"}
            if set(self.brf_search_space) != required:
                raise ValueError(f"BAD_BRF_SEARCH_KEYS={sorted(self.brf_search_space)}")
            brf_configs = [dict(zip(sorted(required), values)) for values in product(
                *(self.brf_search_space[k] for k in sorted(required)))]
        rows: list[dict[str, Any]] = []
        best_model: BalancedRandomForestClassifier | None = None
        best_row: dict[str, Any] | None = None
        best_i = -1
        candidate_i = 0
        total_candidates = len(self.threshold_rules) * len(brf_configs)
        search_started = time.perf_counter()
        for rule in self.threshold_rules:
            rn = u[u_score < thresholds[rule]]
            self._progress(f"SPY RN rule={rule} threshold={thresholds[rule]:.8g} n_RN={len(rn)}")
            if len(rn) == 0:
                rows.append({"config_id": rule, "threshold_rule": rule,
                             "threshold": thresholds[rule], "n_RN": 0,
                             "valid": False, "reason": "NO_RELIABLE_NEGATIVES"})
                continue
            X_stage2 = np.concatenate([X_fit[p], X_fit[rn]])
            y_stage2 = np.r_[np.ones(len(p), np.int8), np.zeros(len(rn), np.int8)]
            for bc in brf_configs:
                config_id = (f"{rule}__T{bc['n_estimators']}__D{bc['max_depth']}"
                             f"__L{bc['min_samples_leaf']}__F{bc['max_features']}"
                             f"__R{int(bc['replacement'])}")
                self._progress(f"SPY BRF START candidate={candidate_i+1}/{total_candidates} config={config_id} n_RN={len(rn)}")
                candidate_started = time.perf_counter()
                model = BalancedRandomForestClassifier(
                    n_estimators=int(bc["n_estimators"]), criterion="gini",
                    max_depth=bc["max_depth"],
                    min_samples_leaf=int(bc["min_samples_leaf"]),
                    max_features=bc["max_features"], sampling_strategy="all",
                    replacement=bool(bc["replacement"]), bootstrap=False,
                    random_state=seed + 100 + candidate_i, n_jobs=1,
                )
                model.fit(X_stage2, y_stage2)
                score = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
                m = validation_metrics(score, yv, val_metadata)
                row = {"config_id": config_id, "threshold_rule": rule,
                       "threshold": thresholds[rule], "n_RN": len(rn),
                       "valid": True, **bc, **m}
                rows.append(row)
                if best_row is None or selection_key(row, candidate_i) > selection_key(best_row, best_i):
                    best_model, best_row, best_i = model, row, candidate_i
                candidate_i += 1
                elapsed = time.perf_counter() - search_started
                eta = elapsed / candidate_i * (total_candidates - candidate_i)
                self._progress(f"SPY BRF DONE candidate={candidate_i}/{total_candidates} val={m['validation_score']:.6f} candidate_s={time.perf_counter()-candidate_started:.1f} elapsed_s={elapsed:.1f} eta_s={eta:.1f}")
        if best_model is None or best_row is None:
            raise RuntimeError("SPY_PRODUCED_NO_RN_FOR_ALL_FROZEN_RULES")
        self.model = best_model
        self.candidate_results = rows
        self.selected = {**best_row, "spy_fraction": self.spy_fraction, "n_spies": n_spy,
                         "stage1_trees": self.stage1_trees,
                         "brf_candidate_count": len(brf_configs),
                         "seed": seed}
        self._progress(f"SPY COMPLETE selected={self.selected['config_id']} n_RN={self.selected['n_RN']} validation_score={self.selected['validation_score']:.6f}")
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ADAPTER_NOT_FIT")
        X = np.asarray(X, dtype=np.float32)
        out = self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]
        return self.check_scores(out, len(X))

    def get_selected_config(self) -> dict[str, Any]:
        return dict(self.selected)
