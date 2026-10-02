#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Import-safe BASE06-compatible training interface for S-EHC-Net-M-v2.

This is intentionally a template: importing or invoking it does not train,
create a checkpoint, or write an experiment result.  Its helpers bind v2 to
the validated BASE06 loader, fold-only preprocessing, pair traversal, metrics,
and output contract for a future smoke runner.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
BASE06_RUNNER = PROJECT / "scripts" / "23D_RUN_NEURAL_BASELINES_REPAIRED_V2.py"
V1_FORMAL_RUNNER = PROJECT / "scripts" / "26_RUN_SEHC_NET_M_V1_FORMAL.py"
DEFAULT_OUTPUT = PROJECT / "experiments" / "SEHC_V2_DESIGN" / "SMOKE_OUTPUT"
DATA_PROTOCOL = "SAME_AS_BASE06"
SPLIT = "SAME_SPATIAL_5FOLD"
ALLOWED_SEEDS = (7, 11, 21)

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from losses.sehc_v2_loss import SEHCV2Loss
from models.sehc_net_m_v2 import SEHCNetMV2


@dataclass(frozen=True)
class FoldInterface:
    """Prepared tensors and pair rows under the BASE06 fold contract."""

    outer_fold: int
    validation_fold: int
    training_folds: tuple[int, ...]
    train_pairs: np.ndarray
    validation_pairs: np.ndarray
    test_pairs: np.ndarray
    static92: torch.Tensor
    rain: torch.Tensor


def _import_file(module_name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import authority module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset_loader() -> tuple[Any, Any, Any]:
    """Return (bundle, BASE06 runner, BASE06 data/metric interface)."""
    runner = _import_file("sehc_v2_base06_authority", BASE06_RUNNER)
    base = runner.base
    bundle = base.preflight()
    return bundle, runner, base


def build_fold_loader(
    bundle: Any,
    runner: Any,
    base: Any,
    human_fold: int,
    device: torch.device,
) -> FoldInterface:
    """Apply the unchanged BASE06 spatial split and fold-only preprocessing."""
    if human_fold not in {1, 2, 3, 4, 5}:
        raise ValueError("human_fold must be in 1..5")
    outer_fold = human_fold - 1
    validation_fold = (outer_fold + 1) % 5
    training_folds = tuple(sorted(set(range(5)) - {outer_fold, validation_fold}))
    train_indices = base.srows(bundle, list(training_folds))
    preprocessor = runner.fit_preprocessor(bundle, train_indices)
    xs91, category_index, rain_np = runner.transform_all(bundle, preprocessor)
    v1_interface = _import_file("sehc_v2_v1_interface", V1_FORMAL_RUNNER)
    static92_np = v1_interface.build_static_92(xs91, category_index)
    return FoldInterface(
        outer_fold=outer_fold,
        validation_fold=validation_fold,
        training_folds=training_folds,
        train_pairs=base.prows(bundle, list(training_folds)),
        validation_pairs=base.prows(bundle, [validation_fold]),
        test_pairs=base.prows(bundle, [outer_fold]),
        static92=torch.as_tensor(static92_np, dtype=torch.float32, device=device),
        rain=torch.as_tensor(rain_np, dtype=torch.float32, device=device),
    )


def make_model_and_loss(
    device: torch.device,
    *,
    lambda_rank: float = 1.0,
    lambda_global: float = 2.0,
    lambda_sparse: float = 0.005,
) -> tuple[SEHCNetMV2, SEHCV2Loss]:
    model = SEHCNetMV2(dropout=0.10, init_gate_logit=-1.5).to(device)
    criterion = SEHCV2Loss(
        lambda_rank=lambda_rank,
        lambda_global=lambda_global,
        lambda_sparse=lambda_sparse,
        rank_temperature=1.0,
    )
    return model, criterion


def forward_complete_pairs(
    model: SEHCNetMV2,
    flat_indices: torch.Tensor,
    fold_data: FoldInterface,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Preserve BASE06 positive-first three-row pair ordering."""
    labels = torch.zeros(
        len(flat_indices), dtype=torch.float32, device=flat_indices.device
    )
    labels[::3] = 1.0
    roles = labels.bool()
    outputs = model(
        fold_data.static92[flat_indices],
        fold_data.rain[flat_indices],
        pair_role=roles,
    )
    return outputs, labels, roles


def calculate_metrics(
    base: Any,
    bundle: Any,
    pair_rows: np.ndarray,
    rank_scores: np.ndarray,
    global_scores: np.ndarray,
    seed: int,
    outer_fold: int,
) -> tuple[Any, Any, dict[str, float]]:
    """Use rank_score for StrictPair/Edge and global_score for AUROC/AUPRC."""
    pair_frame, edge_frame = base.pair_tables(
        bundle, pair_rows, rank_scores, "SEHC_NET_M_V2", seed, outer_fold
    )
    used = bundle.pt[pair_rows].reshape(-1)
    labels = bundle.sample.y_pair.to_numpy(np.int8)[used]
    logits = np.asarray(global_scores[used], dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    predicted = (probability >= 0.5).astype(np.int8)
    metrics = base.metrics(
        pair_frame, edge_frame, labels, logits, probability, predicted
    )
    return pair_frame, edge_frame, metrics


def write_experiment_output(path: Path, payload: dict[str, Any]) -> None:
    """Future-run output adapter; never called during import or template main."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_protocol", default=DATA_PROTOCOL, choices=[DATA_PROTOCOL])
    parser.add_argument("--split", default=SPLIT, choices=[SPLIT])
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "TEMPLATE_ONLY_NO_TRAINING "
        f"data_protocol={args.data_protocol} split={args.split} "
        f"fold={args.fold} seed={args.seed} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FoldInterface",
    "load_dataset_loader",
    "build_fold_loader",
    "make_model_and_loss",
    "forward_complete_pairs",
    "calculate_metrics",
    "write_experiment_output",
]
