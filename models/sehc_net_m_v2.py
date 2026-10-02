"""S-EHC-Net-M v2: ranking/sparse decoupled architecture.

This module defines the model only.  It deliberately reuses the validated v1
encoders and frozen 83-channel trigger-library implementation.  It does not
load data, create an optimizer, write checkpoints, or start training.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.sehc_net_m_v1 import (
    EVENT_DIM,
    RAIN_CHANNELS,
    RAIN_STEPS,
    STATIC_DIM,
    TRIGGER_COUNT,
    EventConditionedSparseTriggerGate,
    RainfallEventEncoder,
    StaticEncoder,
    TriggerLibrary83,
    _require_finite_float,
)


class SparseTriggerBranch(nn.Module):
    """83-trigger branch whose L1 path is isolated from the shared encoder.

    ``h.detach()`` is used only for gate conditioning.  Consequently the L1
    penalty on ``trigger_scores`` updates the gate parameters but cannot push
    the shared embedding in a direction that competes with RankLoss.  The
    global calibration loss can still train the sparse representation through
    ``sparse_embedding``.
    """

    def __init__(self, init_gate_logit: float = -1.5) -> None:
        super().__init__()
        self.trigger_library = TriggerLibrary83()
        self.trigger_gate = EventConditionedSparseTriggerGate(
            EVENT_DIM, init_gate_logit
        )
        self.trigger_encoder = nn.Sequential(
            nn.Linear(TRIGGER_COUNT, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )

    def forward(
        self,
        static_x: Tensor,
        rain_seq: Tensor,
        event_embedding: Tensor,
        h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        trigger_values = self.trigger_library(
            static_x, rain_seq, event_embedding
        )
        # Detaching the conditioning signal isolates SparseLoss from h.  The
        # library values remain available to the calibration representation.
        trigger_scores, sparse_activation = self.trigger_gate(
            trigger_values, h.detach()
        )
        sparse_embedding = self.trigger_encoder(sparse_activation)
        return trigger_scores, sparse_embedding


class SEHCNetMV2(nn.Module):
    """S-EHC-Net-M-v2 with separate ranking and sparse-calibration paths.

    Inputs:
        static_x: [B, 92]
        rain_seq: [B, 70, 10]

    Outputs are logits/scores.  In particular, ``global_score`` is a logit and
    should be passed directly to BCEWithLogitsLoss.
    """

    model_name = "S-EHC-Net-M-v2"
    trigger_count = TRIGGER_COUNT

    def __init__(self, dropout: float = 0.10, init_gate_logit: float = -1.5) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.static_encoder = StaticEncoder(dropout)
        self.rainfall_event_encoder = RainfallEventEncoder(dropout)
        self.shared_event_encoder = nn.Sequential(
            nn.Linear(2 * EVENT_DIM, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )

        # Branch 1: h only.  No trigger score, activation, or embedding enters.
        self.ranking_head = nn.Sequential(
            nn.Linear(EVENT_DIM, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Branch 2: validated 83-trigger library and event-conditioned gate.
        self.sparse_trigger_branch = SparseTriggerBranch(init_gate_logit)

        # Branch 3: sparse residual calibration over concat(h, sparse_embedding).
        self.sparse_residual_head = nn.Sequential(
            nn.Linear(2 * EVENT_DIM, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _validate_inputs(
        static_x: Tensor, rain_seq: Tensor, pair_role: object | None
    ) -> None:
        _require_finite_float(static_x, "static_x")
        _require_finite_float(rain_seq, "rain_seq")
        if static_x.ndim != 2 or static_x.shape[1] != STATIC_DIM:
            raise ValueError("static_x must have shape [B, 92]")
        if rain_seq.ndim != 3 or rain_seq.shape[1:] != (
            RAIN_STEPS,
            RAIN_CHANNELS,
        ):
            raise ValueError("rain_seq must have shape [B, 70, 10]")
        if len(static_x) != len(rain_seq):
            raise ValueError("static_x and rain_seq batch sizes differ")
        if pair_role is not None:
            try:
                role_count = len(pair_role)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError("pair_role must be a batch-length sequence") from exc
            if role_count != len(static_x):
                raise ValueError("pair_role length must match batch size")

    def forward(
        self,
        static_x: Tensor,
        rain_seq: Tensor,
        pair_role: object | None = None,
    ) -> dict[str, Tensor]:
        self._validate_inputs(static_x, rain_seq, pair_role)

        static_embedding = self.static_encoder(static_x)
        event_embedding, _ = self.rainfall_event_encoder(rain_seq)
        h = self.shared_event_encoder(
            torch.cat([static_embedding, event_embedding], dim=1)
        )

        rank_score = self.ranking_head(h).squeeze(-1)
        # Calibration/sparsity may learn their own branch parameters, but they
        # must not update either shared encoder through event_embedding or h.
        trigger_scores, sparse_embedding = self.sparse_trigger_branch(
            static_x, rain_seq, event_embedding.detach(), h.detach()
        )
        sparse_residual = self.sparse_residual_head(
            torch.cat([h.detach(), sparse_embedding], dim=1)
        ).squeeze(-1)
        # Preserve the requested additive forward value while preventing the
        # global BCE objective from updating the ranking head.
        global_score = rank_score.detach() + sparse_residual

        outputs = {
            "rank_score": rank_score,
            "global_score": global_score,
            "trigger_scores": trigger_scores,
            "sparse_embedding": sparse_embedding,
        }
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise FloatingPointError("model output contains NaN or Inf")
        return outputs


# Explicit compatibility alias for code that uses the v1-style class name.
SEHCNetM = SEHCNetMV2

__all__ = ["SEHCNetMV2", "SEHCNetM", "SparseTriggerBranch"]
