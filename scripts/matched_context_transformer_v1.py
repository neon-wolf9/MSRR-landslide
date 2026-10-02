from __future__ import annotations

import numpy as np


class MatchedContextTransformerV1:
    """Frozen TRUE_MSRR transform for one unlabeled three-member matched set."""

    version = "TRUE_MSRR_120B_A6_V1"
    matched_set_size = 3
    static_feature_count = 92
    dynamic_slot_count = 70
    dynamic_feature_count = 10
    output_dim = 2446

    @staticmethod
    def _summary(residual: np.ndarray) -> np.ndarray:
        x = np.asarray(residual, dtype=np.float32)
        if x.shape != (3, 70, 10):
            raise ValueError(f"dynamic residual shape must be (3,70,10), got {x.shape}")
        return np.concatenate(
            [
                x.mean(axis=1), x.std(axis=1), x.max(axis=1), x.min(axis=1),
                x.sum(axis=1), x[:, -1, :], np.abs(x).max(axis=1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def transform_group(self, static92: np.ndarray, dynamic70x10: np.ndarray) -> np.ndarray:
        s = np.asarray(static92, dtype=np.float32)
        r = np.asarray(dynamic70x10, dtype=np.float32)
        if s.shape != (3, 92):
            raise ValueError(f"static shape must be (3,92), got {s.shape}")
        if r.shape != (3, 70, 10):
            raise ValueError(f"dynamic shape must be (3,70,10), got {r.shape}")
        if not np.isfinite(s).all() or not np.isfinite(r).all():
            raise ValueError("nonfinite matched-set input")
        s_rel = s - s.mean(axis=0, keepdims=True)
        r_rel = r - r.mean(axis=0, keepdims=True)
        out = np.concatenate(
            [
                s, s_rel, np.abs(s_rel), r.reshape(3, 700),
                r_rel.reshape(3, 700), np.abs(r_rel).reshape(3, 700),
                self._summary(r_rel),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if out.shape != (3, 2446):
            raise RuntimeError(f"TRUE_MSRR output shape mismatch: {out.shape}")
        return out

    def get_config(self) -> dict:
        return {
            "version": self.version,
            "inference_scope": "MATCHED_SET_ONLY",
            "matched_set_size": 3,
            "matched_set_composition": "1_POSITIVE_2_HARD_CONTROLS",
            "pointwise_full_grid_deployment": "NOT_CLAIMED",
            "primary_task": "WITHIN_MATCHED_SET_RISK_DISCRIMINATION",
            "static_feature_count": 92,
            "dynamic_shape": [70, 10],
            "output_dim": 2446,
            "context": "symmetric within-group arithmetic mean",
            "label_fields_read": 0,
            "sample_role_fields_read": 0,
        }
