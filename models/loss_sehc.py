"""Positive-weight minimization objective for S-EHC-Net-M v1."""
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


def _role_mask(
    pair_role: Tensor | Sequence[object] | None,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Return a positive-role mask while preserving complete triplet groups."""
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
        if role.dtype == torch.bool:
            return role
        return role > 0

    if len(pair_role) != batch_size:
        raise ValueError("pair_role length must match batch size")
    positive_tokens = {"1", "POSITIVE", "MATCHED_POSITIVE", "POS"}
    values = [str(value).strip().upper() in positive_tokens for value in pair_role]
    return torch.tensor(values, dtype=torch.bool, device=device)


class SEHCNetLoss(nn.Module):
    """RankNet + global BCE + L1 sparse-gate objective.

    All coefficients are non-negative and all components are added under
    minimization:

        L = lambda_rank * L_rank
          + lambda_bce * L_bce
          + lambda_sparse * L_sparse
    """

    def __init__(
        self,
        lambda_rank: float = 1.0,
        lambda_bce: float = 0.3,
        lambda_sparse: float = 1e-3,
        rank_temperature: float = 1.0,
        positive_role_weight: float = 1.0,
        control_role_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_rank = _nonnegative_finite(lambda_rank, "lambda_rank")
        self.lambda_bce = _nonnegative_finite(lambda_bce, "lambda_bce")
        self.lambda_sparse = _nonnegative_finite(
            lambda_sparse, "lambda_sparse"
        )
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
            "pair_score",
            "global_risk",
            "trigger_weights",
            "sparse_activation",
        }
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"model outputs missing keys: {sorted(missing)}")

        pair_score = outputs["pair_score"].reshape(-1)
        labels = torch.as_tensor(
            labels, dtype=pair_score.dtype, device=pair_score.device
        ).reshape(-1)
        if labels.numel() != pair_score.numel():
            raise ValueError("labels length must match model batch size")
        if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("labels must contain finite binary values")
        if pair_score.numel() % 3:
            raise ValueError("ranking loss requires complete 3-row pairs")

        positive = _role_mask(
            pair_role, pair_score.numel(), pair_score.device
        )
        positive_by_pair = positive.reshape(-1, 3)
        if not positive_by_pair.sum(dim=1).eq(1).all():
            raise ValueError("each consecutive triplet must contain one positive role")
        if not labels.bool().eq(positive).all():
            raise ValueError("labels and pair_role/positive-first ordering disagree")

        scores_by_pair = pair_score.reshape(-1, 3)
        positive_scores = scores_by_pair[positive_by_pair].reshape(-1, 1)
        control_scores = scores_by_pair[~positive_by_pair].reshape(-1, 2)
        margins = positive_scores - control_scores
        rank_loss = F.softplus(-margins / self.rank_temperature).mean()

        if "global_logit" in outputs:
            global_logit = outputs["global_logit"].reshape(-1)
            if global_logit.numel() != labels.numel():
                raise ValueError("global_logit length must match labels")
            bce_per_sample = F.binary_cross_entropy_with_logits(
                global_logit, labels, reduction="none"
            )
        else:
            probability = outputs["global_risk"].reshape(-1)
            if probability.numel() != labels.numel():
                raise ValueError("global_risk length must match labels")
            bce_per_sample = F.binary_cross_entropy(
                probability.clamp(1e-7, 1.0 - 1e-7), labels, reduction="none"
            )

        role_weights = torch.where(
            positive,
            labels.new_tensor(self.positive_role_weight),
            labels.new_tensor(self.control_role_weight),
        )
        bce_loss = (bce_per_sample * role_weights).sum() / role_weights.sum().clamp_min(
            1e-12
        )

        trigger_weights = outputs["trigger_weights"]
        if trigger_weights.ndim != 2 or trigger_weights.shape != outputs[
            "sparse_activation"
        ].shape:
            raise ValueError(
                "trigger_weights and sparse_activation must share shape [B, triggers]"
            )
        sparse_loss = trigger_weights.abs().mean()

        total = (
            self.lambda_rank * rank_loss
            + self.lambda_bce * bce_loss
            + self.lambda_sparse * sparse_loss
        )
        losses = {
            "loss_total": total,
            "loss_rank": rank_loss,
            "loss_bce": bce_loss,
            "loss_sparse": sparse_loss,
            "mean_pair_margin": margins.mean(),
        }
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("loss output contains NaN or Inf")
        return losses


class LDARRankingLoss(nn.Module):
    """Learnability-aware difficulty adaptive pairwise logistic loss.

    ``difficulty`` and ``learnability`` are raw, pair-level values.  Fixed
    normalization bounds should be supplied by the training entry point so the
    normalization is stable across mini-batches.  A one-dimensional pair
    weight is broadcast over that pair's control scores.
    """

    def __init__(
        self,
        difficulty_min: float | None = None,
        difficulty_max: float | None = None,
        learnability_min: float | None = None,
        learnability_max: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.difficulty_bounds = self._validate_bounds(
            difficulty_min, difficulty_max, "difficulty"
        )
        self.learnability_bounds = self._validate_bounds(
            learnability_min, learnability_max, "learnability"
        )
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")

    @staticmethod
    def _validate_bounds(
        lower: float | None,
        upper: float | None,
        name: str,
    ) -> tuple[float, float] | None:
        if lower is None and upper is None:
            return None
        if lower is None or upper is None:
            raise ValueError(f"{name}_min and {name}_max must be supplied together")
        lower_value, upper_value = float(lower), float(upper)
        if not math.isfinite(lower_value) or not math.isfinite(upper_value):
            raise ValueError(f"{name} bounds must be finite")
        if upper_value < lower_value:
            raise ValueError(f"{name}_max must be greater than or equal to {name}_min")
        return lower_value, upper_value

    def _normalize(
        self,
        values: Tensor,
        bounds: tuple[float, float] | None,
    ) -> Tensor:
        if not torch.is_floating_point(values):
            values = values.float()
        if not torch.isfinite(values).all():
            raise ValueError("LDAR weights must be finite")
        if bounds is None:
            lower, upper = values.detach().amin(), values.detach().amax()
        else:
            lower = values.new_tensor(bounds[0])
            upper = values.new_tensor(bounds[1])
        scale = upper - lower
        if float(scale.detach().cpu()) <= self.eps:
            return torch.ones_like(values)
        return ((values - lower) / scale).clamp(0.0, 1.0)

    @staticmethod
    def _align_pair_weight(weight: Tensor, scores: Tensor, name: str) -> Tensor:
        if weight.shape == scores.shape:
            return weight
        if scores.ndim >= 1 and weight.shape == scores.shape[:-1]:
            return weight.unsqueeze(-1).expand_as(scores)
        if weight.numel() == 1:
            return weight.reshape([1] * scores.ndim).expand_as(scores)
        raise ValueError(
            f"{name} shape {tuple(weight.shape)} cannot broadcast to scores "
            f"shape {tuple(scores.shape)}"
        )

    def forward(
        self,
        pos_score: Tensor,
        control_score: Tensor,
        difficulty: Tensor,
        learnability: Tensor,
    ) -> Tensor:
        if pos_score.shape != control_score.shape:
            raise ValueError("pos_score and control_score must have identical shapes")
        if not torch.is_floating_point(pos_score) or not torch.is_floating_point(
            control_score
        ):
            raise TypeError("ranking scores must use a floating-point dtype")
        if not torch.isfinite(pos_score).all() or not torch.isfinite(control_score).all():
            raise ValueError("ranking scores must be finite")

        difficulty_tensor = torch.as_tensor(
            difficulty, dtype=pos_score.dtype, device=pos_score.device
        )
        learnability_tensor = torch.as_tensor(
            learnability, dtype=pos_score.dtype, device=pos_score.device
        )
        difficulty_norm = self._align_pair_weight(
            self._normalize(difficulty_tensor, self.difficulty_bounds),
            pos_score,
            "difficulty",
        )
        learnability_norm = self._align_pair_weight(
            self._normalize(learnability_tensor, self.learnability_bounds),
            pos_score,
            "learnability",
        )
        base_loss = F.softplus(-(pos_score - control_score))
        final_weight = difficulty_norm * learnability_norm
        loss = (final_weight * base_loss).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("LDAR ranking loss contains NaN or Inf")
        return loss


__all__ = ["SEHCNetLoss", "LDARRankingLoss"]
