"""Decoupled objective for S-EHC-Net-M-v2.

All coefficients are non-negative and all terms are added under minimization:

    L = lambda_rank * RankLoss(rank_score)
      + lambda_global * BCEGlobalLoss(global_score)
      + lambda_sparse * SparseLoss(trigger_scores)
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive_role_mask(
    pair_role: Tensor | Sequence[object] | None,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if pair_role is None:
        if batch_size % 3:
            raise ValueError("ranking loss requires complete 3-row pairs")
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        mask[::3] = True
        return mask
    if isinstance(pair_role, Tensor):
        if pair_role.numel() != batch_size:
            raise ValueError("pair_role length must match batch size")
        role = pair_role.reshape(-1).to(device=device)
        return role if role.dtype == torch.bool else role > 0
    if len(pair_role) != batch_size:
        raise ValueError("pair_role length must match batch size")
    positive_tokens = {"1", "POSITIVE", "MATCHED_POSITIVE", "POS"}
    return torch.tensor(
        [str(value).strip().upper() in positive_tokens for value in pair_role],
        dtype=torch.bool,
        device=device,
    )


class SEHCV2Loss(nn.Module):
    """Rank-only rank branch, global BCE, and trigger-only L1 sparsity."""

    def __init__(
        self,
        lambda_rank: float = 1.0,
        lambda_global: float = 2.0,
        lambda_sparse: float = 0.005,
        rank_temperature: float = 1.0,
        positive_role_weight: float = 1.0,
        control_role_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_rank = _nonnegative_finite(lambda_rank, "lambda_rank")
        self.lambda_global = _nonnegative_finite(lambda_global, "lambda_global")
        self.lambda_sparse = _nonnegative_finite(lambda_sparse, "lambda_sparse")
        self.positive_role_weight = _nonnegative_finite(
            positive_role_weight, "positive_role_weight"
        )
        self.control_role_weight = _nonnegative_finite(
            control_role_weight, "control_role_weight"
        )
        self.rank_temperature = float(rank_temperature)
        if not math.isfinite(self.rank_temperature) or self.rank_temperature <= 0:
            raise ValueError("rank_temperature must be finite and positive")

    def forward(
        self,
        outputs: dict[str, Tensor],
        labels: Tensor,
        pair_role: Tensor | Sequence[object] | None = None,
    ) -> dict[str, Tensor]:
        required = {
            "rank_score",
            "global_score",
            "trigger_scores",
            "sparse_embedding",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"model outputs missing keys: {sorted(missing)}")

        # RankLoss reads rank_score only.
        rank_score = outputs["rank_score"].reshape(-1)
        labels = torch.as_tensor(
            labels, dtype=rank_score.dtype, device=rank_score.device
        ).reshape(-1)
        if labels.numel() != rank_score.numel():
            raise ValueError("labels length must match model batch size")
        if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels must contain finite binary values")
        if rank_score.numel() % 3:
            raise ValueError("ranking loss requires complete 3-row pairs")
        positive = _positive_role_mask(
            pair_role, rank_score.numel(), rank_score.device
        )
        positive_by_pair = positive.reshape(-1, 3)
        if not positive_by_pair.sum(dim=1).eq(1).all():
            raise ValueError("each consecutive triplet must contain one positive role")
        if not labels.bool().eq(positive).all():
            raise ValueError("labels and pair_role/positive-first ordering disagree")
        scores_by_pair = rank_score.reshape(-1, 3)
        positive_scores = scores_by_pair[positive_by_pair].reshape(-1, 1)
        control_scores = scores_by_pair[~positive_by_pair].reshape(-1, 2)
        margins = positive_scores - control_scores
        rank_loss = F.softplus(-margins / self.rank_temperature).mean()

        # BCEGlobalLoss reads global_score only; it is a logit.
        global_score = outputs["global_score"].reshape(-1)
        if global_score.numel() != labels.numel():
            raise ValueError("global_score length must match labels")
        bce_per_sample = F.binary_cross_entropy_with_logits(
            global_score, labels, reduction="none"
        )
        role_weights = torch.where(
            positive,
            labels.new_tensor(self.positive_role_weight),
            labels.new_tensor(self.control_role_weight),
        )
        global_loss = (bce_per_sample * role_weights).sum() / role_weights.sum().clamp_min(
            1e-12
        )

        # SparseLoss reads trigger_scores only.  It cannot inspect rank/global.
        trigger_scores = outputs["trigger_scores"]
        if trigger_scores.ndim != 2:
            raise ValueError("trigger_scores must have shape [B, triggers]")
        if trigger_scores.shape[0] != labels.numel():
            raise ValueError("trigger_scores batch size must match labels")
        sparse_loss = trigger_scores.abs().mean()

        total = (
            self.lambda_rank * rank_loss
            + self.lambda_global * global_loss
            + self.lambda_sparse * sparse_loss
        )
        losses = {
            "loss_total": total,
            "loss_rank": rank_loss,
            "loss_global": global_loss,
            "loss_sparse": sparse_loss,
            "mean_pair_margin": margins.mean(),
        }
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("loss output contains NaN or Inf")
        return losses


__all__ = ["SEHCV2Loss"]
