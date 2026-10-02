"""S-EHC-Net-M v1 architecture.

This module defines the model only. It does not load data, create an optimizer,
write checkpoints, or start training. Inputs are expected to be finite,
fold-preprocessed tensors from the repaired-V2 data contract.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


STATIC_DIM = 92
RAIN_STEPS = 70
RAIN_CHANNELS = 10
EVENT_DIM = 64
TRIGGER_COUNT = 83


def _require_finite_float(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(x):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")


class StaticEncoder(nn.Module):
    """Compact encoder for the requested 92-column static interface."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATIC_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )

    def forward(self, static_x: Tensor) -> Tensor:
        return self.network(static_x)


class CausalConv1d(nn.Module):
    """Left-padded convolution that cannot read future rainfall steps."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = EVENT_DIM,
        dilation: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x).transpose(1, 2)
        y = self.dropout(F.gelu(self.norm(y))).transpose(1, 2)
        return y + self.residual(x)


class RainfallEventEncoder(nn.Module):
    """Causal convolutional encoder with final-state and mean pooling.

    No Transformer, GNN, or additional attention mechanism is used.
    """

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(RAIN_CHANNELS, 32, 1)
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(32, EVENT_DIM, dilation=1, dropout=dropout),
                CausalResidualBlock(EVENT_DIM, EVENT_DIM, dilation=2, dropout=dropout),
                CausalResidualBlock(EVENT_DIM, EVENT_DIM, dilation=4, dropout=dropout),
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(2 * EVENT_DIM, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )

    def forward(self, rain_seq: Tensor) -> tuple[Tensor, Tensor]:
        sequence = self.input_projection(rain_seq.transpose(1, 2))
        for block in self.blocks:
            sequence = block(sequence)
        sequence = sequence.transpose(1, 2)
        event_embedding = self.output(
            torch.cat([sequence[:, -1], sequence.mean(dim=1)], dim=1)
        )
        return event_embedding, sequence


class TriggerLibrary83(nn.Module):
    """Structured 83-channel trigger representation.

    Channel groups follow the frozen library counts: 38 static, 10 rainfall,
    32 static-event interactions, and 3 nonlinear static responses. The layer
    is deliberately small. Exact scientific names/order must be bound to the
    frozen trigger schema by the future data/experiment adapter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.static_base = nn.Linear(STATIC_DIM, 38)
        self.rain_scale = nn.Parameter(torch.ones(RAIN_CHANNELS))
        self.rain_bias = nn.Parameter(torch.zeros(RAIN_CHANNELS))
        self.interaction_static = nn.Linear(STATIC_DIM, 32)
        self.interaction_event = nn.Linear(EVENT_DIM, 32)
        self.nonlinear_static = nn.Linear(STATIC_DIM, 3)

    def forward(
        self, static_x: Tensor, rain_seq: Tensor, event_embedding: Tensor
    ) -> Tensor:
        static_terms = self.static_base(static_x)
        rainfall_terms = rain_seq[:, -1] * self.rain_scale + self.rain_bias
        interaction_terms = self.interaction_static(static_x) * torch.tanh(
            self.interaction_event(event_embedding)
        )
        nonlinear_terms = self.nonlinear_static(static_x).square()
        terms = torch.cat(
            [static_terms, rainfall_terms, interaction_terms, nonlinear_terms], dim=1
        )
        if terms.shape[1] != TRIGGER_COUNT:
            raise RuntimeError("trigger library must produce exactly 83 channels")
        return terms


class EventConditionedSparseTriggerGate(nn.Module):
    """Sample-specific sparse gate conditioned on static/event context."""

    def __init__(self, context_dim: int = EVENT_DIM, init_gate_logit: float = -1.5) -> None:
        super().__init__()
        self.base_gate_logits = nn.Parameter(
            torch.full((TRIGGER_COUNT,), float(init_gate_logit))
        )
        self.context_to_gate = nn.Linear(context_dim, TRIGGER_COUNT)

    def forward(self, trigger_values: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        trigger_weights = torch.sigmoid(
            self.base_gate_logits.unsqueeze(0) + self.context_to_gate(context)
        )
        sparse_activation = trigger_values * trigger_weights
        return trigger_weights, sparse_activation


class SEHCNetM(nn.Module):
    """S-EHC-Net-M v1 with event-conditioned sparse triggers and two heads."""

    model_name = "S-EHC-Net-M-v1"
    trigger_count = TRIGGER_COUNT

    def __init__(self, dropout: float = 0.10, init_gate_logit: float = -1.5) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.static_encoder = StaticEncoder(dropout)
        self.rainfall_event_encoder = RainfallEventEncoder(dropout)
        self.base_fusion = nn.Sequential(
            nn.Linear(2 * EVENT_DIM, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )
        self.trigger_library = TriggerLibrary83()
        self.trigger_gate = EventConditionedSparseTriggerGate(
            EVENT_DIM, init_gate_logit
        )
        self.trigger_encoder = nn.Sequential(
            nn.Linear(TRIGGER_COUNT, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(2 * EVENT_DIM, EVENT_DIM),
            nn.LayerNorm(EVENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(EVENT_DIM, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.global_risk_head = nn.Sequential(
            nn.Linear(EVENT_DIM, 32),
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
        context = self.base_fusion(
            torch.cat([static_embedding, event_embedding], dim=1)
        )

        trigger_values = self.trigger_library(
            static_x, rain_seq, event_embedding
        )
        trigger_weights, sparse_activation = self.trigger_gate(
            trigger_values, context
        )
        trigger_representation = self.trigger_encoder(sparse_activation)
        fused = self.final_fusion(
            torch.cat([context, trigger_representation], dim=1)
        )

        pair_score = self.ranking_head(fused).squeeze(-1)
        global_logit = self.global_risk_head(fused).squeeze(-1)
        global_risk = torch.sigmoid(global_logit)

        outputs = {
            "pair_score": pair_score,
            "global_risk": global_risk,
            "trigger_weights": trigger_weights,
            "event_embedding": event_embedding,
            "sparse_activation": sparse_activation,
            # The logit is retained for numerically stable BCEWithLogitsLoss.
            "global_logit": global_logit,
        }
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise FloatingPointError("model output contains NaN or Inf")
        return outputs


__all__ = ["SEHCNetM"]
